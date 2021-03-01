"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import baseline_modules as layers
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args
import qanet_modules


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

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
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
    def __init__(self, word_mat, char_mat, n_encoder_blocks=7, n_head=4):
        super().__init__()
        D = qanet_modules.D
        self.Lc = None # deprecated
        self.Lq = None # deprecated
        self.n_model_enc_blks = n_encoder_blocks
        if args.use_pretrained_char:
            print('Using pretrained character embeddings.')
            self.char_emb = nn.Embedding.from_pretrained(
                torch.Tensor(char_mat), freeze=True)
        else:
            char_mat = torch.Tensor(char_mat)
            self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
        self.word_emb = nn.Embedding.from_pretrained(
            torch.Tensor(word_mat), freeze=True)
        self.emb = qanet_modules.Embedding()
        self.emb_enc = qanet_modules.EncoderBlock(
            conv_num=4, ch_num=D, k=7, n_head=n_head)
        self.cq_att = qanet_modules.CQAttention()
        self.cq_resizer = qanet_modules.Initialized_Conv1d(4 * D, D)
        self.model_enc_blks = nn.ModuleList([
            qanet_modules.EncoderBlock(conv_num=2, ch_num=D, k=5, n_head=n_head) 
            for _ in range(n_encoder_blocks)
        ])
        self.out = qanet_modules.Pointer()

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        # (bs, ctxt_len, word_emb_dim=300), (bs, ctxt_len, char_lim, char_emb_dim=64)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        # (bs, ques_len, word_emb_dim=300), (bs, ctxt_len, ques_len, char_emb_dim=64)
        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        Ce = self.emb_enc(C, maskC, 1, 1) # (bs, d_model, ctxt_len)
        Qe = self.emb_enc(Q, maskQ, 1, 1) # (bs, d_model, ques_len)
        X = self.cq_att(Ce, Qe, maskC, maskQ) # (bs, 4 * d_model, ctxt_len)
        M0 = self.cq_resizer(X) # (bs, d_model, ctxt_len)
        M0 = F.dropout(M0, p=args.qanet_dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M2 = M0
        M0 = F.dropout(M0, p=args.qanet_dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2