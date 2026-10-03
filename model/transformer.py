import sys
from pathlib import Path

import torch
import torch.nn as nn

# make the given starter/ folder importable, wherever this file is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
from embeddings import TokenEmbedding, InputLayer          # given by the assignment

from model.layers import Encoder, Decoder


# ---------------------------------------------------------------- masks (Task 2.4)
def make_pad_mask(ids, pad_id=0):
    """(B, L) token ids -> (B, 1, 1, L) bool. True where the token is a real token."""
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length, device=None):
    """(1, 1, L, L) bool. Row t is True for columns 0..t only (no peeking ahead)."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt_ids, pad_id=0):
    """padding mask AND causal mask -> (B, 1, T, T)"""
    pad = make_pad_mask(tgt_ids, pad_id)
    causal = make_causal_mask(tgt_ids.size(1), tgt_ids.device)
    return pad & causal



class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, h=4, n_layers=3, d_ff=1024,
                 dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id

        shared = TokenEmbedding(vocab_size, d_model, pad_id)      # ONE embedding table
        self.src_in = InputLayer(shared, d_model, dropout=dropout)
        self.tgt_in = InputLayer(shared, d_model, dropout=dropout)

        self.encoder = Encoder(n_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, h, d_ff, dropout)

        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.out_proj.weight = shared.emb.weight                  # weight sharing

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.src_in.tok.emb.weight[self.pad_id].zero_()

    def encode(self, src):
        src_mask = make_pad_mask(src, self.pad_id)
        memory = self.encoder(self.src_in(src), src_mask)
        return memory, src_mask

    def decode(self, tgt_in, memory, src_mask):
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)
        hidden = self.decoder(self.tgt_in(tgt_in), memory, tgt_mask, src_mask)
        return self.out_proj(hidden)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)