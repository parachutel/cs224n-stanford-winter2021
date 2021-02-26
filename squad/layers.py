"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

import qanet_config
import math

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


#####################################################################################
# QANet #############################################################################
#####################################################################################

d_model = qanet_config.d_model
n_head = qanet_config.num_heads
d_word = qanet_config.glove_dim
d_char = qanet_config.char_dim
batch_size = qanet_config.batch_size
dropout = qanet_config.dropout
dropout_char = qanet_config.dropout_char

d_k = d_model // n_head
d_cq = d_model * 4
len_c = qanet_config.para_limit + 1 # + 1 for not answerable class
len_q = qanet_config.ques_limit + 1 # + 1 for not answerable class


def mask_logits(target, mask):
    return target * (1 - mask) + mask * (-1e30)


class PosEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        freqs = torch.Tensor([
            10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) 
            for i in range(d_model)
        ]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        # print('PosEncoder', x.shape, self.pos_encoding.shape)
        x = x + self.pos_encoding
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        x = x.transpose(1, 2)
        return x


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        Wo = torch.empty(d_model, d_k * n_head)
        Wqs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wks = [torch.empty(d_model, d_k) for _ in range(n_head)]
        Wvs = [torch.empty(d_model, d_k) for _ in range(n_head)]
        nn.init.kaiming_uniform_(Wo)
        for i in range(n_head):
            nn.init.xavier_uniform_(Wqs[i])
            nn.init.xavier_uniform_(Wks[i])
            nn.init.xavier_uniform_(Wvs[i])
        self.Wo = nn.Parameter(Wo)
        self.Wqs = nn.ParameterList([nn.Parameter(X) for X in Wqs])
        self.Wks = nn.ParameterList([nn.Parameter(X) for X in Wks])
        self.Wvs = nn.ParameterList([nn.Parameter(X) for X in Wvs])

    def forward(self, x, mask):
        WQs, WKs, WVs = [], [], []
        sqrt_d_k_inv = 1 / math.sqrt(d_k)
        x = x.transpose(1, 2)
        hmask = mask.unsqueeze(1)
        vmask = mask.unsqueeze(2)
        for i in range(n_head):
            WQs.append(torch.matmul(x, self.Wqs[i]))
            WKs.append(torch.matmul(x, self.Wks[i]))
            WVs.append(torch.matmul(x, self.Wvs[i]))
        heads = []
        for i in range(n_head):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
            out = torch.mul(out, sqrt_d_k_inv)
            # not sure... I think `dim` should be 2 since it weighted each column of `WVs[i]`
            out = mask_logits(out, hmask)
            out = F.softmax(out, dim=2) * vmask
            headi = torch.bmm(out, WVs[i])
            heads.append(headi)
        head = torch.cat(heads, dim=2)
        out = torch.matmul(head, self.Wo)
        return out.transpose(1, 2)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.a = 1 / math.sqrt(d_k)

    def forward(self, x, mask):
        bs, _, l_x = x.size()
        x = x.transpose(1,2)
        k = self.k_linear(x).view(bs, l_x, n_head, d_k)
        q = self.q_linear(x).view(bs, l_x, n_head, d_k)
        v = self.v_linear(x).view(bs, l_x, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(n_head, 1, 1)
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.a
        attn = mask_logits(attn, mask)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)
            
        out = torch.bmm(attn, v)
        out = out.view(n_head, bs, l_x, d_k).permute(1,2,0,3).contiguous().view(bs, l_x, d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out.transpose(1,2)

class QANetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = DepthwiseSeparableConv(d_char, d_char, 5, dim=2)
        self.high = Highway(2, d_word+d_char)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()
        wd_emb = F.dropout(wd_emb, p=dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = MultiHeadAttention()
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = PosEncoder(length)
        # self.norm = nn.LayerNorm([d_model, length])
        self.normb = nn.LayerNorm([d_model, length])
        self.norms = nn.ModuleList([nn.LayerNorm([d_model, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([d_model, length])
        self.L = conv_num

    def forward(self, x, mask):
        out = self.pos(x)
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        # print("Before attention: {}".format(out.size()))
        out = self.self_att(out, mask)
        # print("After attention: {}".format(out.size()))
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        return out


class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.empty(d_model * 3)
        lim = 1 / d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        ss = []
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)
        
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        S = torch.cat([Ct, Qt, CQ], dim=3)
        S = torch.matmul(S, self.w)
        S1 = F.softmax(mask_logits(S, qmask), dim=2)
        S2 = F.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=dropout, training=self.training)
        return out.transpose(1, 2)


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        lim = 3 / (2 * d_model)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = torch.matmul(self.w1, X1)
        Y2 = torch.matmul(self.w2, X2)
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)
        return p1, p2
