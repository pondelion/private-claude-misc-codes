"""TimeXer multivariate encoder (released ``features='M'`` path).

For each endogenous channel, nonoverlapping temporal patches plus one global
token attend to all exogenous channel tokens. The official encoder boundary
is [B,C,D,N+1]; a same-length task readout follows it here.
"""
from __future__ import annotations

import math

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F


class FullAttention(nn.Module):
    """Official noncausal attention [B,Lq,D] × [B,Lk,D] -> [B,Lq,D]."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """Q:[B,Lq,D], K/V:[B,Lk,D] -> [B,Lq,D]."""
        q = rearrange(self.query(query), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.key(key), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.value(value), "b n (h d) -> b h n d", h=self.heads)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        # [B,H,Lq,Lk]; the last axis enumerates candidate key tokens.
        attended = self.dropout(scores.softmax(dim=-1)) @ v
        # [B,H,Lq,D/H] -> combine heads [B,Lq,D].
        return self.output(rearrange(attended, "b h n d -> b n (h d)"))


class TimeXerLayer(nn.Module):
    """Self-attention, global-only cross-attention, and post-norm FFN."""

    def __init__(self, dim: int, d_ff: int, heads: int,
                 dropout: float, activation: str) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")
        self.self_attention = FullAttention(dim, heads, dropout)
        self.cross_attention = FullAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, exogenous: torch.Tensor) -> torch.Tensor:
        """[B*Cin,N+1,D] and [B*Cin,Cin,D] -> [B*Cin,N+1,D]."""
        # x:[B*C,N+1,D], exogenous:[B*C,C,D]. First, patches+global
        # communicate within each endogenous channel.
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x)))
        global_token = x[:, -1:, :]  # [B*C,1,D]
        global_token = self.norm2(global_token + self.dropout(
            self.cross_attention(global_token, exogenous, exogenous)))
        x = torch.cat((x[:, :-1, :], global_token), dim=1)
        # Keep N patch tokens, replace global token: [B*Cin,N+1,D].
        y = rearrange(x, "b n d -> b d n")  # [B*Cin,D,N+1]
        y = self.dropout(self.activation(self.conv1(y)))
        # Pointwise Conv1d: [B*Cin,Dff,N+1].
        y = self.dropout(rearrange(self.conv2(y), "b d n -> b n d"))
        # [B*Cin,D,N+1] -> [B*Cin,N+1,D].
        return self.norm3(x + y)


class TimeXerBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout] with explicit O(Cin²) cross-attention.

    This reproduces the released multivariate path, not a linear-complexity
    approximation. At Cin≈1000 the global-to-exogenous attention is expensive;
    reduce/select variables upstream only if changing the modeling contract
    is acceptable for the downstream task.
    """

    def __init__(self, in_dim: int, out_dim: int, seq_len: int,
                 patch_len: int = 8, d_model: int = 64,
                 d_ff: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.0, activation: str = "gelu",
                 use_norm: bool = True) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, patch_len, d_model,
               d_ff, heads, layers) < 1 or seq_len < patch_len:
            raise ValueError("invalid dimensions or patch length")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_count = seq_len // patch_len
        self.use_norm = use_norm
        self.patch_embed = nn.Linear(patch_len, d_model, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, in_dim, 1, d_model))
        self.exogenous_embed = nn.Linear(seq_len, d_model)
        self.en_dropout = nn.Dropout(dropout)
        self.ex_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            TimeXerLayer(d_model, d_ff, heads, dropout, activation)
            for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        # Forecast head is replaced by an H=T readout and output fusion.
        self.time_readout = nn.Linear(d_model * (self.patch_count + 1),
                                      seq_len)
        self.output = nn.Linear(in_dim, out_dim)

    def _position(self, device: torch.device, dtype: torch.dtype,
                  dim: int) -> torch.Tensor:
        """Return sinusoidal patch positions [N,D] for N patch tokens."""
        # Official PositionalEmbedding, added only to temporal patches.
        positions = torch.arange(self.patch_count, device=device,
                                 dtype=dtype)[:, None]
        rates = torch.exp(torch.arange(0, dim, 2, device=device,
                                       dtype=dtype) * (-math.log(10000.0) / dim))
        result = torch.zeros(self.patch_count, dim, device=device, dtype=dtype)
        result[:, 0::2] = torch.sin(positions * rates)
        result[:, 1::2] = torch.cos(positions * rates[:result[:, 1::2].shape[1]])
        return result

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return official pre-head tokens [B,C,D,N+1]."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError(f"expected [B,{self.seq_len},{self.in_dim}]")
        if self.use_norm:
            mean = x.mean(1, keepdim=True).detach()  # [B,1,Cin]
            centered = x - mean
            std = torch.sqrt(centered.var(1, keepdim=True,
                                          unbiased=False) + 1e-5)
            x = centered / std  # [B,T,Cin], std:[B,1,Cin]
        # Endogenous: each channel -> N patches + its own global token.
        series = rearrange(x, "b t c -> b c t")  # [B,Cin,T]
        patches = series.unfold(-1, self.patch_len, self.patch_len)
        # [B,Cin,T] -> [B,Cin,N,P], N=floor(T/P); trailing remainder is unused.
        embedded = self.patch_embed(patches)  # [B,Cin,N,D]
        embedded = embedded + self._position(
            x.device, x.dtype, embedded.shape[-1])
        globals_ = self.global_token.expand(x.shape[0], -1, -1, -1)
        # [1,Cin,1,D] -> [B,Cin,1,D] (variable-specific global token).
        endogenous = self.en_dropout(rearrange(
            torch.cat((embedded, globals_), dim=2),
            "b c n d -> (b c) n d"))
        # [B,Cin,N+1,D] -> [B*Cin,N+1,D].
        # Official 'M' path uses all C input series as exogenous tokens,
        # including the current endogenous series itself.
        exogenous = self.ex_dropout(self.exogenous_embed(series))
        # Whole-series embedding: [B,Cin,T] -> [B,Cin,D].
        exogenous = repeat(exogenous, "b c d -> (b endogenous) c d",
                           endogenous=self.in_dim)
        # Duplicate candidates for each endogenous target: [B*Cin,Cin,D].
        for layer in self.layers:
            endogenous = layer(endogenous, exogenous)
        encoded = self.norm(endogenous)
        # [B*Cin,N+1,D] -> official boundary [B,Cin,D,N+1].
        return rearrange(encoded, "(b c) n d -> b c d n", c=self.in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> same-length downstream features [B,T,Cout]."""
        features = self.forward_features(x)  # [B,C,D,N+1]
        # [B,Cin,D,N+1] -> [B,Cin,D*(N+1)] -> [B,Cin,T].
        series = self.time_readout(rearrange(features,
                                             "b c d n -> b c (d n)"))
        # [B,Cin,T] -> [B,T,Cin] -> [B,T,Cout].
        return self.output(rearrange(series, "b c t -> b t c"))


if __name__ == "__main__":
    block = TimeXerBlock(4, 6, seq_len=32, patch_len=8, d_model=32)
    sample = torch.randn(2, 32, 4, requires_grad=True)
    output = block(sample)
    assert output.shape == (2, 32, 6)
    output.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
