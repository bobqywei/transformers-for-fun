from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, dim_model: int, dim_kq: int, dim_v: int):
        super(Attention, self).__init__()
        self.dim_kq = dim_kq
        self.dim_v = dim_v

        self.proj_k = nn.Linear(dim_model, dim_kq)
        self.proj_q = nn.Linear(dim_model, dim_kq)
        self.proj_v = nn.Linear(dim_model, dim_v)
        self.scale_logits = 1 / np.sqrt(float(self.dim_kq))

    def forward(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Input kqv dims: [b, s, m], Mask dim: [s, s], Output dim: [b, s, v]."""

        K = self.proj_k(k)
        Q = self.proj_q(q)
        V = self.proj_v(v)
        logits = torch.bmm(K, torch.transpose(Q, 1, 2)) * self.scale_logits
        if not attn_mask is None:
            logits.masked_fill_(attn_mask, float('-inf'))
        weights = F.softmax(logits, dim=2)
        outputs = torch.bmm(weights, V)

        return outputs


class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim_model: int, num_heads: int, dim_kq: int, dim_v: int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_v = dim_v
        self.dim_model = dim_model

        self.heads = nn.ModuleList(
            [Attention(dim_model, dim_kq, dim_v) for _ in range(num_heads)])
        self.proj_out = nn.Linear(num_heads * dim_v, dim_model)

    def forward(self, k: torch.Tensor, q: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Input kqv dims: [b, s, m], Mask dim: [s, s], Output dim: [b, s, m]."""

        attention_outputs = [head(k, q, v, attn_mask) for head in self.heads]
        attention_outputs = torch.cat(attention_outputs, dim=2)  # [b, s, h * v]
        mha_output = self.proj_out(attention_outputs)

        return mha_output


class FFN(nn.Module):
    
    def __init__(self, dim_model: int, dim_inner: int):
        super(FFN, self).__init__()

        self.linear1 = nn.Linear(dim_model, dim_inner)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_inner, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear2(self.relu(self.linear1(x)))

        return out


class LayerNormalization(nn.Module):

    def __init__(self, normalized_shape: List[int], epsilon: float = 1e-5):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.reduce_dims = [-i for i in range(1, len(normalized_shape) + 1)]

        self.w = torch.nn.Parameter(torch.ones(normalized_shape, requires_grad=True, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(normalized_shape, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=self.reduce_dims, keepdim=True)
        var = torch.square(x - mean).mean(dim=self.reduce_dims, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        x = x * self.w + self.b

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, dim_model: int, dropout: float):
        super(PositionalEncoding, self).__init__()

        positional_encoding = torch.zeros((max_seq_len, dim_model), dtype=torch.float32)
        pos = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, dim_model, 2, dtype=torch.float32)
        divisor = torch.exp(-1 * np.log(10000.) * i / dim_model) 
        positional_encoding[:, 0::2] = torch.sin(pos * divisor)
        positional_encoding[:, 1::2] = torch.cos(pos * divisor)
        self.register_buffer("positional_encoding", positional_encoding)

        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        out = x + self.positional_encoding[:seq_len, :]
        out = self.dropout(out)

        return out
