"""TimeBridge patch/integrated/downsampled/cointegrated encoder.

The released paper's forecast decoder is excluded. ``forward_features``
returns the source encoder boundary [B,C+4,M,D] (four default time markers).
``forward`` reads data channels back to [B,T,Cout] and fuses channels.
"""
from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def period_norm(x: torch.Tensor, length: int) -> torch.Tensor:
    """Official no-grad trailing mean on feature axis, [B*C,N,D] -> same."""
    if length < 1 or length > x.shape[-1]:
        raise ValueError("stable_len must be within feature width")
    # [batch,patch,D] -> [batch,patch,D-length+1,length]
    windows = x.unfold(-1, length, 1)
    means = F.pad(windows.mean(-1), (length - 1, 0), mode="replicate")
    return x - means


class TSMixer(nn.Module):
    """Released Q/K/V projection, softmax attention, output [B,Q,D]."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.q_proj(q), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.k_proj(k), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.v_proj(v), "b n (h d) -> b h n d", h=self.heads)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        weighted = self.dropout(scores.softmax(dim=-1)) @ v
        return self.out(rearrange(weighted, "b h n d -> b n (h d)"))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, d_ff: int, dropout: float,
                 activation: str) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")
        self.fc1 = nn.Linear(dim, d_ff)
        self.fc2 = nn.Linear(d_ff, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(attended))
        y = self.dropout(self.activation(self.fc1(x)))
        return self.norm2(x + self.dropout(self.fc2(y)))


class IntegratedAttention(_FeedForward):
    """Within-series patch attention with stationary Q,K, raw V."""

    def __init__(self, dim: int, d_ff: int, heads: int,
                 stable_len: int, dropout: float, activation: str) -> None:
        super().__init__(dim, d_ff, dropout, activation)
        self.attention = TSMixer(dim, heads, dropout)
        self.stable_len = stable_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, dim = x.shape
        flat = rearrange(x, "b c n d -> (b c) n d")
        with torch.no_grad():
            stable = period_norm(flat, self.stable_len)
        attended = self.attention(stable, stable, flat)
        return self.finish(x, rearrange(attended,
                                        "(b c) n d -> b c n d", b=batch,
                                        c=channels))


class PatchSampling(_FeedForward):
    """Learn Q from input patches, cross-attend original K,V [B,C,N,D]."""

    def __init__(self, dim: int, d_ff: int, heads: int,
                 in_p: int, out_p: int, dropout: float,
                 activation: str) -> None:
        super().__init__(dim, d_ff, dropout, activation)
        if not 1 <= out_p <= in_p:
            raise ValueError("out_p must be in [1,in_p]")
        self.conv1 = nn.Conv1d(in_p, out_p, 1, bias=False)
        self.conv2 = nn.Conv1d(out_p + 1, out_p, 1, bias=False)
        self.attention = TSMixer(dim, heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        flat = rearrange(x, "b c n d -> (b c) n d")
        reduced = self.conv1(flat)
        reduced = self.conv2(torch.cat((reduced, flat.mean(-2,
                                    keepdim=True)), dim=-2)) + reduced
        # Released down_attn adds dropout(reduced) to attention output.
        attended = self.attention(reduced, flat, flat) + self.dropout(reduced)
        attended = rearrange(attended, "(b c) n d -> b c n d",
                             b=batch, c=channels)
        # PatchSampling does not add the original x: patch count changed.
        y = self.norm1(attended)
        z = self.dropout(self.activation(self.fc1(y)))
        return self.norm2(y + self.dropout(self.fc2(z)))


class CointegratedAttention(_FeedForward):
    """Released two-axis variable attention after patch downsampling."""

    def __init__(self, dim: int, d_ff: int, heads: int,
                 in_dim: int, dropout: float, activation: str,
                 axial: bool = True) -> None:
        super().__init__(dim, d_ff, dropout, activation)
        self.axial = axial
        self.num_rc = math.ceil(math.sqrt(in_dim + 4))
        self.attention1 = TSMixer(dim, heads, dropout)
        self.attention2 = TSMixer(dim, heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, patches, dim = x.shape
        flat = rearrange(x, "b c n d -> (b n) c d")
        if self.axial:
            # Pad C+4 channels to R². The source partitions padded tokens
            # into R rows, attends within rows, transposes, then attends rows.
            side = self.num_rc
            padded = F.pad(flat.transpose(-1, -2),
                           (0, side * side - channels)).transpose(-1, -2)
            rows = rearrange(padded, "bn (r c) d -> (bn r) c d", r=side)
            rows = self.attention1(rows, rows, rows)
            cols = rearrange(rows, "(bn r) c d -> (bn c) r d", r=side)
            cols = self.attention2(cols, cols, cols) + cols
            attended = rearrange(cols,
                                  "(b n c) r d -> b (r c) n d",
                                  b=batch, n=patches)[:, :channels]
        else:
            attended = self.attention1(flat, flat, flat)
            attended = rearrange(attended, "(b n) c d -> b c n d",
                                  b=batch, n=patches)
        return self.finish(x, attended)


class TimeBridgeBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout], fixed T and fixed channel count."""

    def __init__(self, in_dim: int, out_dim: int, seq_len: int,
                 period: int = 8, num_p: int | None = None,
                 d_model: int = 64, d_ff: int = 128, heads: int = 4,
                 ia_layers: int = 1, pd_layers: int = 1,
                 ca_layers: int = 1, stable_len: int = 8,
                 dropout: float = 0.0, activation: str = "gelu",
                 axial: bool = True, revin: bool = True) -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, period, d_model,
               d_ff, heads, stable_len) < 1:
            raise ValueError("dimensions must be positive")
        if seq_len % period:
            raise ValueError("official patching requires T divisible by period")
        if min(ia_layers, pd_layers, ca_layers) < 0:
            raise ValueError("layer counts must be nonnegative")
        patches = seq_len // period
        num_p = patches if num_p is None else num_p
        if pd_layers and not 1 <= num_p <= patches:
            raise ValueError("num_p must be in [1,T/period]")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.period = period
        self.revin = revin
        self.embedding = nn.Linear(period, d_model, bias=False)
        self.embedding_dropout = nn.Dropout(dropout)
        blocks: list[nn.Module] = []
        blocks.extend(IntegratedAttention(d_model, d_ff, heads, stable_len,
                                          dropout, activation)
                      for _ in range(ia_layers))
        blocks.extend(PatchSampling(d_model, d_ff, heads,
                                    patches if index == 0 else num_p,
                                    num_p, dropout, activation)
                      for index in range(pd_layers))
        blocks.extend(CointegratedAttention(d_model, d_ff, heads, in_dim,
                                            dropout, activation, axial)
                      for _ in range(ca_layers))
        self.layers = nn.ModuleList(blocks)
        final_patches = num_p if pd_layers else patches
        self.time_readout = nn.Linear(final_patches * d_model, seq_len,
                                      bias=False)
        self.output = nn.Linear(in_dim, out_dim)

    def forward_features(self, x: torch.Tensor,
                         markers: torch.Tensor | None = None) -> torch.Tensor:
        """Return [B,Cin+4,M,D] after unchanged source encoder stages."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError(f"expected [B,{self.seq_len},{self.in_dim}]")
        if markers is None:
            markers = x.new_zeros(x.shape[0], self.seq_len, 4)
        if markers.shape != (x.shape[0], self.seq_len, 4):
            raise ValueError("markers must be [B,T,4]")
        if self.revin:
            mean = x.mean(1, keepdim=True).detach()
            std = x.std(1, keepdim=True).detach()
            x = (x - mean) / (std + 1e-5)
        joined = torch.cat((x, markers), dim=-1)
        patches = rearrange(joined, "b (n p) c -> b c n p", p=self.period)
        tokens = self.embedding_dropout(self.embedding(patches))
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens

    def forward(self, x: torch.Tensor,
                markers: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.forward_features(x, markers)[:, :self.in_dim]
        flattened = rearrange(encoded, "b c n d -> b c (n d)")
        series = rearrange(self.time_readout(flattened),
                           "b c t -> b t c")
        return self.output(series)


if __name__ == "__main__":
    block = TimeBridgeBlock(4, 6, 32, period=8, num_p=2,
                            d_model=32, stable_len=4)
    sample = torch.randn(2, 32, 4, requires_grad=True)
    result = block(sample)
    assert result.shape == (2, 32, 6)
    result.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
