import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import PAD_ID


# LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, ids, lengths):
        x = self.emb_drop(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))

        mask = (ids != PAD_ID).unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(~mask, -1e4).max(1).values
        return self.head(torch.cat([mean, mx], dim=-1))


# Transformer
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = self.attn_drop(scores.softmax(dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(ctx)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = x + self.drop(self.attn(self.ln1(x), pad_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, n_heads=4, n_layers=2,
                 d_ff=256, dropout=0.2, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, ids, lengths=None):
        pad_mask = ids == PAD_ID
        x = self.embedding(ids) * math.sqrt(self.d_model)
        x = self.drop(self.pos(x))
        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.ln_f(x)

        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.classifier(pooled)


DEFAULT_CONFIGS = {
    "lstm": dict(emb_dim=128, hidden_dim=128, num_layers=1, dropout=0.4),
    "transformer": dict(d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.2, max_len=1024),
}


def build_model(model_type, vocab_size, num_classes, **cfg):
    if model_type == "lstm":
        return LSTMClassifier(vocab_size, num_classes, **cfg)
    if model_type == "transformer":
        return TransformerClassifier(vocab_size, num_classes, **cfg)
    raise ValueError(f"Error in model name: {model_type}")