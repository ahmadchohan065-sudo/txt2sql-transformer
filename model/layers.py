import torch.nn as nn
from model.attention import MultiHeadAttention
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))
class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, src_mask):
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x
class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.cross_attn_weights = None
    def forward(self, x, memory, tgt_mask, src_mask):
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop(attn_out))
        cross_out, cross_w = self.cross_attn(x, memory, memory, src_mask)
        self.cross_attn_weights = cross_w.detach()
        x = self.norm2(x + self.drop(cross_out))
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x
class Encoder(nn.Module):
    def __init__(self, n_layers=3, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)]
        )
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
class Decoder(nn.Module):
    def __init__(self, n_layers=3, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)]
        )
    def forward(self, x, memory, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x
