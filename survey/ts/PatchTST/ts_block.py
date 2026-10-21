"""PatchTST channel-independent patch encoder and same-length adapter.

``forward_features`` returns the official backbone boundary [B,C,D,N].
The final Linear(Cin,Cout) is deliberately added for downstream fusion.
"""
from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class ResidualPatchAttention(nn.Module):
    """PatchTST/RealFormer attention, optionally carrying logits by layer."""

    def __init__(self, dim: int, heads: int, attn_dropout: float,
                 dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.attention_dropout = nn.Dropout(attn_dropout)
        self.register_buffer("scale", torch.tensor((dim // heads) ** -0.5))

    def forward(self, x: torch.Tensor,
                previous: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patch tokens [B*Cin,N,D] -> values [B*Cin,N,D], logits
        [B*Cin,H,N,N]; H=head count, N=patch positions.
        """
        # [B*C,N,D] -> [B*C,H,N,D/H]; attention only within each channel.
        q = rearrange(self.query(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.key(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.value(x), "b n (h d) -> b h n d", h=self.heads)
        logits = (q @ k.transpose(-2, -1)) * self.scale
        # [B*Cin,H,N,N]; previous layer logits have the same axes.
        if previous is not None:
            logits = logits + previous
        weights = self.attention_dropout(logits.softmax(dim=-1))
        # [B*Cin,H,N,N] @ [B*Cin,H,N,D/H] -> [B*Cin,N,D].
        value = rearrange(weights @ v, "b h n d -> b n (h d)")
        return self.output(value), logits


class PatchEncoderLayer(nn.Module):
    """Official post-norm patch Transformer layer [B*C,N,D] -> same."""

    def __init__(self, dim: int, d_ff: int, heads: int,
                 dropout: float, attn_dropout: float,
                 norm: str = "BatchNorm") -> None:
        super().__init__()
        self.attention = ResidualPatchAttention(dim, heads, attn_dropout,
                                                dropout)
        self.norm_attn = (nn.BatchNorm1d(dim) if norm == "BatchNorm"
                          else nn.LayerNorm(dim))
        self.norm_ffn = (nn.BatchNorm1d(dim) if norm == "BatchNorm"
                         else nn.LayerNorm(dim))
        self.ff = nn.Sequential(nn.Linear(dim, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, dim))
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    @staticmethod
    def _norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(norm, nn.BatchNorm1d):
            # BatchNorm1d expects [B*Cin,D,N], not token-last [B*Cin,N,D].
            return rearrange(norm(rearrange(x, "b n d -> b d n")),
                             "b d n -> b n d")
        return norm(x)

    def forward(self, x: torch.Tensor, previous: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B*Cin,N,D] -> same; carry score [B*Cin,H,N,N]."""
        attended, scores = self.attention(x, previous)
        x = self._norm(self.norm_attn, x + self.dropout_attn(attended))
        x = self._norm(self.norm_ffn, x + self.dropout_ffn(self.ff(x)))
        return x, scores


class PatchTSTBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout]; fixed T, no cross-channel mixing until output.

    Set ``padding_patch='end'`` to append ``stride`` replicated samples as
    the released code does; otherwise a partial final patch is discarded.
    """

    def __init__(self, in_dim: int, out_dim: int, seq_len: int,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, d_ff: int = 128, heads: int = 4,
                 layers: int = 2, dropout: float = 0.0,
                 attn_dropout: float = 0.0, norm: str = "BatchNorm",
                 padding_patch: str | None = "end", revin: bool = True,
                 affine: bool = True) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, patch_len, stride, d_model,
               d_ff, heads, layers) < 1 or seq_len < patch_len:
            raise ValueError("invalid dimensions or patch length")
        if padding_patch not in {None, "end"}:
            raise ValueError("padding_patch must be None or 'end'")
        if norm not in {"BatchNorm", "LayerNorm"}:
            raise ValueError("norm must be BatchNorm or LayerNorm")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.affine = affine
        count = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":
            count += 1
        self.patch_count = count
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(in_dim))
            self.affine_bias = nn.Parameter(torch.zeros(in_dim))
        self.embedding = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(count, d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            PatchEncoderLayer(d_model, d_ff, heads, dropout, attn_dropout,
                              norm) for _ in range(layers))
        # Official flatten head with H=T, then mandatory cross-series fusion.
        self.time_readout = nn.Linear(d_model * count, seq_len)
        self.output = nn.Linear(in_dim, out_dim)

    def _normalize(self, x: torch.Tensor) -> tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor]:
        """[B,T,Cin] -> normalized [B,T,Cin], mean/std [B,1,Cin]."""
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False)
                         + 1e-5).detach()
        x = (x - mean) / std
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x, mean, std

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> channel-independent patch tokens [B,Cin,D,N]."""
        # [B,T,Cin] -> [B,Cin,T]; channels are separate processing streams.
        x = rearrange(x, "b t c -> b c t")
        if self.padding_patch == "end":
            x = F.pad(x, (0, self.stride), mode="replicate")
            # [B,Cin,T] -> [B,Cin,T+stride].
        patches = x.unfold(-1, self.patch_len, self.stride)
        # [B,Cin,T(+stride)] -> [B,Cin,N,P]; N=patch count, P=patch_len.
        embedded = self.embedding(patches)  # [B,Cin,N,D]
        tokens = rearrange(embedded, "b c n d -> (b c) n d")
        # [B,Cin,N,D] -> [B*Cin,N,D]; no cross-series attention.
        tokens = self.dropout(tokens + self.position)
        scores = None
        for layer in self.layers:
            tokens, scores = layer(tokens, scores)
        # [B*Cin,N,D] -> official backbone boundary [B,Cin,D,N].
        return rearrange(tokens, "(b c) n d -> b c d n", c=self.in_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B,C,D,N] at the official pre-forecast-head boundary."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError(f"expected [B,{self.seq_len},{self.in_dim}]")
        if self.revin:
            x, _, _ = self._normalize(x)
        return self._encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> same-length task features [B,T,Cout]."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError(f"expected [B,{self.seq_len},{self.in_dim}]")
        if self.revin:
            normalized, mean, std = self._normalize(x)
        else:
            normalized = x
        features = self._encode(normalized)  # [B,C,D,N]
        # [B,Cin,D,N] -> [B,Cin,D*N] -> [B,Cin,T].
        values = self.time_readout(rearrange(features,
                                             "b c d n -> b c (d n)"))
        values = rearrange(values, "b c t -> b t c")  # [B,T,Cin]
        if self.revin:
            if self.affine:
                values = (values - self.affine_bias) / (
                    self.affine_weight + 1e-10)
            values = values * std + mean
        return self.output(values)  # [B,T,Cout], even when Cin=Cout.


if __name__ == "__main__":
    block = PatchTSTBlock(4, 6, seq_len=32, patch_len=8, stride=4)
    sample = torch.randn(2, 32, 4, requires_grad=True)
    result = block(sample)
    assert result.shape == (2, 32, 6)
    result.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
