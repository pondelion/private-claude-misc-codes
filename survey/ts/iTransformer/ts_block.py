"""Inverted-variate Transformer encoder with a same-length task adapter.

The paper's encoder boundary is ``forward_features``: [B,T,C] -> [B,C,D].
``forward`` adds a same-length, learnable readout [B,T,Cout], not the paper's
fixed-horizon forecasting head. Every input channel is a separate token;
self-attention is the first cross-channel fusion operation.
"""
from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class VariateAttention(nn.Module):
    """Official full noncausal attention over C variate tokens [B,C,D]."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:[B,C,D] -> q/k/v:[B,H,C,D/H]. No time-causal mask applies here.
        q = rearrange(self.query(x), "b c (h d) -> b h c d", h=self.heads)
        k = rearrange(self.key(x), "b c (h d) -> b h c d", h=self.heads)
        v = rearrange(self.value(x), "b c (h d) -> b h c d", h=self.heads)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        weights = self.attention_dropout(scores.softmax(dim=-1))
        # [B,H,C,C] @ [B,H,C,D/H] -> [B,C,D].
        fused = rearrange(weights @ v, "b h c d -> b c (h d)")
        return self.output(fused)


class InvertedEncoderLayer(nn.Module):
    """Official post-norm attention + 1x1-conv FFN, [B,C,D] -> same."""

    def __init__(self, d_model: int, d_ff: int, heads: int,
                 dropout: float, activation: str) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")
        self.attention = VariateAttention(d_model, heads, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        # Conv1d acts on feature D, independently at each variate token C.
        y = rearrange(x, "b c d -> b d c")
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(rearrange(self.conv2(y), "b d c -> b c d"))
        return self.norm2(x + y)


class ITransformerBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout]; fixed T, arbitrary batch size.

    ``forward_features`` exposes the exact official encoder boundary [B,Cin,D]
    for verification and alternative readouts. The default ``forward`` readout
    maps each token back to T values and always fuses channels via Linear.
    """

    def __init__(self, in_dim: int, out_dim: int, seq_len: int,
                 d_model: int = 64, d_ff: int = 128, heads: int = 4,
                 layers: int = 2, dropout: float = 0.0,
                 activation: str = "gelu", use_norm: bool = True) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, d_model, d_ff, heads, layers) < 1:
            raise ValueError("all dimensions and layers must be positive")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.use_norm = use_norm
        self.embedding = nn.Linear(seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            InvertedEncoderLayer(d_model, d_ff, heads, dropout, activation)
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        # This is the reusable adapter, separate from the official H-step head.
        self.time_readout = nn.Linear(d_model, seq_len)
        self.output = nn.Linear(in_dim, out_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError(f"expected [B,{self.seq_len},{self.in_dim}]")
        if self.use_norm:
            # Official instance normalization: detached mean, population var.
            mean = x.mean(1, keepdim=True).detach()
            centered = x - mean
            std = torch.sqrt(centered.var(1, keepdim=True,
                                          unbiased=False) + 1e-5)
            x = centered / std
        # One whole series per token: [B,T,C] -> [B,C,T] -> [B,C,D].
        tokens = self.embedding_dropout(
            self.embedding(rearrange(x, "b t c -> b c t")))
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,D] -> [B,C,T] -> [B,T,C] -> [B,T,Cout]. The output Linear
        # deliberately mixes variates even when Cin == Cout.
        tokens = self.forward_features(x)
        series = rearrange(self.time_readout(tokens), "b c t -> b t c")
        return self.output(series)


if __name__ == "__main__":
    model = ITransformerBlock(5, 7, seq_len=32, d_model=32)
    sample = torch.randn(3, 32, 5, requires_grad=True)
    encoded = model(sample)
    assert encoded.shape == (3, 32, 7)
    encoded.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
