"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

import qanet_config

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat, n_encoder_blocks=7):
        super().__init__()
        self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=qanet_config.pretrained_char)
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat))
        self.emb = layers.QANetEmbedding()
        self.context_conv = layers.DepthwiseSeparableConv(layers.d_word + layers.d_char, layers.d_model, 5)
        self.question_conv = layers.DepthwiseSeparableConv(layers.d_word + layers.d_char, layers.d_model, 5)
        self.c_emb_enc = layers.EncoderBlock(conv_num=4, ch_num=layers.d_model, k=7, length=layers.len_c)
        self.q_emb_enc = layers.EncoderBlock(conv_num=4, ch_num=layers.d_model, k=7, length=layers.len_q)
        self.cq_att = layers.CQAttention()
        self.cq_resizer = layers.DepthwiseSeparableConv(layers.d_model * 4, layers.d_model, 5)
        enc_blk = layers.EncoderBlock(conv_num=2, ch_num=layers.d_model, k=5, length=layers.len_c)
        self.model_enc_blks = nn.ModuleList([enc_blk] * n_encoder_blocks)
        self.out = layers.Pointer()

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        cmask = (torch.zeros_like(Cwid) == Cwid).float()
        qmask = (torch.zeros_like(Qwid) == Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        C = self.context_conv(C)
        Q = self.question_conv(Q)
        Ce = self.c_emb_enc(C, cmask)
        Qe = self.q_emb_enc(Q, qmask)
        
        X = self.cq_att(Ce, Qe, cmask, qmask)
        M1 = self.cq_resizer(X)
        for enc in self.model_enc_blks: 
            M1 = enc(M1, cmask)
        M2 = M1
        for enc in self.model_enc_blks: 
            M2 = enc(M2, cmask)
        M3 = M2
        for enc in self.model_enc_blks: 
            M3 = enc(M3, cmask)
        p1, p2 = self.out(M1, M2, M3, cmask)
        return p1, p2