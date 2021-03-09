import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, 
                 dropatt=0.1, pre_lnorm=True):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, mask, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        w = w.permute(2, 0, 1)
        #print(w.size())
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        sizes = mask.size()
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        mask = mask.permute(0, 2, 3, 1)
        attn_score = attn_score.permute(2, 3, 1, 0)
        attn_score = mask_logits(attn_score, mask)
        attn_score = attn_score.permute(3, 2, 0, 1)
                
        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            #output = w + attn_out
            output = attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)
            output = self.layer_norm(attn_out)

        return output

class HighwayEncoder(nn.Module):
    """
    Edits: An dropout layer with p=0.1 was added

    Encode an input sequence using a highway network.
!    Based on the paper:
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
            # Shapes of g, t, and x are all (batch_size, hidden_size, seq_len)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            t = F.dropout(t, p=0.1, training=self.training)
            x = g * t + (1 - g) * x

        return x

class PositionWiseFF(nn.Module):
    """
    Output of the encoder/decoder units in a transformer
    Dropout added

    See reference here: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, d_model, d_inner, dropout, p_norm = False):
        """
        Args:
             d_model (int): hidden_size of model embedding
             d_inner (int): hidden size of the two layer neural net 
             p_norm (bool): lnorm inputs prior to inputting to the FF net
        """
        
        super(PositionWiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.lnorm = p_norm

        #Layer instantiation
        self.FF = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp):
        return self.FF(inp)
    
class Initialized_Conv1d(nn.Module):
    """
    Wrapper Function
    Initializes nn.conv1d and adds a relu output.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super(Initialized_Conv1d, self).__init__()
        
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class Embedding(nn.Module):
    """
    Embedding layer specified by QANet. 
    Concatenation of 300-dimensional (p1) pre-trained GloVe word vectors and 200-dimensional (p2) trainable char-vectors 
    Char-vectors have a set length of 16 via truncation or padding. Max value is taken by each row/char? 
    To obtain a vector of (p1 + p2) long word vector 
    Uses two-layer highway network (Srivastava 2015) 

    Note: Dropout was used on character_word embeddings and between layers, specified as 0.1 and 0.05 respectively

    Question: Linear/Conv1d layer before or after highway?
    """

    def __init__(self, p1, p2, hidden_size, dropout_w = 0.1, dropout_c = 0.05):
        super(Embedding, self).__init__()
        self.conv2d = nn.Conv2d(p2, hidden_size, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(p1 + hidden_size, hidden_size, bias=False)
        self.high = HighwayEncoder(2, hidden_size)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c


    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb).transpose(1,2)
        #Emb: shape [batch_size * seq_len * hidden_size]
        #print(emb.size())
        emb = self.high(emb).transpose(1,2)
        return emb

class DepthwiseSeperableConv(nn.Module):
    """
    Performs a depthwise seperable convolution
    First you should only convolve over each input channel individually, afterwards you convolve the input channels via inx1x1 to get the number of output channels
    This method conserves memory
    
    For clarification see the following: 
    https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
    https://arxiv.org/abs/1706.03059


    Args:
         in_channel (int): input channel
         out_channel (int): output channel
         k (int): kernel size

    Question: Padding in depthwise_convolution
    """
    def __init__(self, in_channel, out_channel, k, bias=True):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size = k, groups = in_channel, padding = k//2, bias = False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size = 1, bias=bias)

    def forward(self, input):
        return F.relu(self.pointwise_conv(self.depthwise_conv(input)))


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]
    
    
class Encoder(nn.Module):
    """
    Encoder structure specified in the QANet implementation

    Args:
         num_conv (int): number of depthwise convlutional layers
         d_model (int): size of model embedding
         num_head (int): number of attention-heads
         k (int): kernel size for convolutional layers
         dropout (float): layer dropout probability
    """

    def __init__(self, num_conv, n_head, d_model, d_head, d_inner, k, dropout = 0.1):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList([DepthwiseSeperableConv(d_model, d_model, k) for _ in range(num_conv)])
        self.conv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv)])
        
        self.att = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, pre_lnorm=True)
        #self.att = SelfAttention(d_model, n_head, dropout = dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)        
        self.FF = PositionWiseFF(d_model, d_model * 4, dropout, p_norm = True) #NEEDS TO BE UPDATED inner_model
        #self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        #self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.num_conv = num_conv
        self.dropout = dropout
        #self.pos_emb = PositionalEmbedding(d_model)

    def forward(self, x, mask, l, blks, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        """
        dropout probability: uses stochastic depth survival probability = 1 - (l/L)*pL, 
        reference here: https://arxiv.org/pdf/1603.09382.pdf 
        """
        total_layers = (self.num_conv + 1) * blks
        bsz, d_model, seq_len = x.size()
        
        out = x
        dropout = self.dropout

        for i, conv in enumerate(self.convs):
            res = out
            out = self.conv_norms[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.res_drop(out, res, dropout*float(l)/total_layers)
            l += 1

        res = out
        #print(out.size())
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        #print(out)
        
        out = self.att(out, mask, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        out = out.permute(1,2, 0)
        #out = self.att(out, mask)
        out = self.res_drop(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out
        
        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = out.transpose(1,2)
        #out = self.FFN_1(out)
        #out = self.FFN_2(out)
        out = self.FF(out)
        out = out.transpose(1,2)
        out = self.res_drop(out, res, dropout*float(l)/total_layers)
        return out

        
    def res_drop(self, x, res, drop):
        """
        Layer-dropout with residual addition
        """
        if self.training == True:
           if torch.empty(1).uniform_(0,1) < drop:
               return res
           else:
               return F.dropout(x, drop, training=self.training) + res
        else:
            return x + res

class CQAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        w4C = torch.empty(d_model, 1)
        w4Q = torch.empty(d_model, 1)
        w4mlu = torch.empty(1, 1, d_model)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, d_model = C.shape
        batch_size, Lq, d_model = Q.shape
        dropout = self.dropout
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res        

class QAOutput(nn.Module):
    def __init__(self, hidden_size):
        super(QAOutput, self).__init__()
        self.w1 = Initialized_Conv1d(hidden_size*2, 1)
        self.w2 = Initialized_Conv1d(hidden_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        p1 = F.log_softmax(Y1, dim=1)
        p2 = F.log_softmax(Y2, dim=1)        
        return p1, p2
        
