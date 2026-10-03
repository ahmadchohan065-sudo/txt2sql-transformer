import math
import torch
import torch.nn as nn
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output, weights
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, h=4):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by the number of heads"
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    def _split_heads(self, x):
        batch, length, _ = x.shape
        x = x.view(batch, length, self.h, self.d_k)
        return x.transpose(1, 2)
    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))
        out, weights = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.w_o(out), weights
