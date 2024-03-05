''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from models_zero.archs.transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


class MultiHeadAttention4(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)

        self.w_qzs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_kzs = nn.Linear(d_model, n_head * d_k, bias=False)

        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc1 = nn.Linear(n_head * d_v, d_model, bias=False)
        self.fc2 = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention_SA = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.attention_ZA = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, qz, kz, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = v

        # print(q.shape)
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        qz = self.layer_norm(qz)
        kz = self.layer_norm(kz)

        v = self.layer_norm(v)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        qz = self.w_qs(qz).view(sz_b, len_q, n_head, d_k)
        kz = self.w_ks(kz).view(sz_b, len_k, n_head, d_k)

        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        qz, kz = qz.transpose(1, 2), kz.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        v1 = self.attention_SA(q, k, v, mask=mask)
        v2 = self.attention_ZA(qz, kz, v, mask=mask)

        # print(attn.shape, '2')
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v1 = v1.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v1 = self.dropout(self.fc1(v1))
        v1 = v1 + residual

        v2 = v2.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v2 = self.dropout(self.fc2(v2))
        v2 = v2 + residual
        v = v1 + v2

        # q = self.layer_norm(q)
        return v


class PositionwiseFeedForward4(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


