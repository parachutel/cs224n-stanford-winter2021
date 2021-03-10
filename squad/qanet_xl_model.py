import qanet_xl_modules as layers
import qanet_modules as qanet
import torch
import torch.nn as nn
import torch.nn.functional as F

class QANetXL(nn.Module):

    def __init__(self, word_vectors, char_vectors,
                 d_model, d_head, mem_len=80, same_length=False, clamp_len=-1, 
                 train_cemb=False, pad=0, dropout=0.1, num_head=8,
                 n_encoder_blocks=7):
        super().__init__()
        if train_cemb:
            self.char_emb = nn.Embedding.from_pretrained(char_vectors, freeze=False)
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_vectors)

        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.num_head = num_head
        self.pad = pad
        self.dropout = dropout
        self.mem_len = mem_len
        self.d_head = d_head
        self.d_model = d_model
        self.num_head = num_head
        self.n_encoder_blocks = n_encoder_blocks
        self.same_length = same_length
        self.clamp_len = clamp_len
        self.ext_len = 0
        
        wemb_dim = word_vectors.size()[1]
        cemb_dim = char_vectors.size()[1]

        #Layer Declarations
        self.emb = layers.Embedding(wemb_dim, cemb_dim, d_model)
        self.qanet_emb_enc = qanet.EncoderBlock(
            conv_num=4, ch_num=d_model, k=7, n_head=num_head)
        self.emb_enc = layers.Encoder(4, num_head, d_model, d_head, 
            d_inner=d_model * 4, k=7, dropout=0.1) #Hard coded
        self.cq_att = layers.CQAttention(d_model=d_model)
        self.cq_resizer = layers.Initialized_Conv1d(d_model * 4, d_model) 
        #Foward layer to reduce dimension of cq_att output back to d_dim
        self.model_enc_blks = nn.ModuleList([
            layers.Encoder(2, num_head, d_model, d_head, 
                d_inner=d_model * 4, k=5, dropout=0.1) 
            for _ in range(self.n_encoder_blocks)
        ])
        self.out = layers.QAOutput(d_model)
        self.drop = nn.Dropout(dropout)

        self._create_parameters()

    def _create_parameters(self):
        self.pos_emb = layers.PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_head, self.d_head))        

    def init_mems(self, n_layers):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(n_layers):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i].permute(2, 0, 1)], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

            return new_mems

    def _forwardEmb(self, word_emb, mask, mems=None):
        bsz, d_model, qlen = word_emb.size()
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        
        if not self.training:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        # pos_emb = self.drop(pos_emb)
        
        hids = []
        hids.append(core_out)
        mems_i = None if mems is None else mems[1]
        core_out = self.emb_enc(core_out, mask, 1, 1, pos_emb, self.r_w_bias, 
            self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        hids.append(core_out)
        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def _forwardEnc(self, word_emb, mask, mems=None):
        bsz, d_model, qlen = word_emb.size()

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if not self.training: #same_length
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                               dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        # pos_emb = self.drop(pos_emb)
        
        hids = []
        hids.append(core_out)
        for i, layer in enumerate(self.model_enc_blks):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, mask, i*(2+2)+1, 7, pos_emb, self.r_w_bias, 
                self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        return core_out, new_mems    
        
    def forward(self, Cword, Cchar, Qword, Qchar):
        maskC = (torch.ones_like(Cword) * self.pad != Cword).float()
        maskQ = (torch.ones_like(Qword) * self.pad != Qword).float()

        Qw, Qc = self.word_emb(Qword), self.char_emb(Qchar)
        Q = self.emb(Qc, Qw)
        Qe = self.qanet_emb_enc(Q, maskQ, 1, 1) # (bs, d_model, ques_len)

        bs, seq_len = Cword.shape[0], Cword.shape[1]

        n_segments = seq_len // self.mem_len + 1 * (seq_len % self.mem_len > 0)

        M1_collection = []
        M2_collection = []
        M3_collection = []

        memsC = self.init_mems(n_layers=1 + 1)
        mems0 = self.init_mems(n_layers=self.n_encoder_blocks + 1)
        mems1 = self.init_mems(n_layers=self.n_encoder_blocks + 1)
        mems2 = self.init_mems(n_layers=self.n_encoder_blocks + 1)
        
        for i_seg in range(n_segments):

            Cword_seg = Cword[:, i_seg * self.mem_len : (i_seg + 1) * self.mem_len]
            Cchar_seg = Cchar[:, i_seg * self.mem_len : (i_seg + 1) * self.mem_len]
            maskC_seg = (torch.ones_like(Cword_seg) * self.pad != Cword_seg).float()

            Cw_seg, Cc_seg = self.word_emb(Cword_seg), self.char_emb(Cchar_seg)
            C_seg = self.emb(Cc_seg, Cw_seg)

            Ce_seg, memsC  = self._forwardEmb(C_seg, maskC_seg,  mems=memsC)
            X = self.cq_att(Ce_seg, Qe, maskC_seg, maskQ)
            
            M0 = self.cq_resizer(X)
            M0 = F.dropout(M0, p=self.dropout, training=self.training)

            M1, mems0 = self._forwardEnc(M0, maskC_seg, mems=mems0)
            M1_collection.append(M1)
            M2, mems1 = self._forwardEnc(M1, maskC_seg, mems=mems1)
            M2_collection.append(M2)
            M3, mems2 = self._forwardEnc(M2, maskC_seg, mems=mems2)
            M3_collection.append(M3)
        
        M1 = torch.cat(M1_collection, dim=-1)
        M2 = torch.cat(M2_collection, dim=-1)
        M3 = torch.cat(M3_collection, dim=-1)

        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2
