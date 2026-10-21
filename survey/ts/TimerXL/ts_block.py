"""Timer-XL causal TimeAttention encoder and same-length feature readout.

Encoder boundary: [B,C*N,D], where N=T/P nonoverlapping patch tokens per
variable. Official next-patch forecast head is intentionally separate from
the reusable ``forward`` adapter [B,T,Cout].
"""
from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class TimeAttention(nn.Module):
    """Official non-flash TimeAttention: causal over patch time, not channels.

    Token order is ``(channel, patch)``. At query patch t, keys from *all*
    channels with patch index <= t are visible in multivariate mode.
    """

    def __init__(self, dim: int, heads: int, dropout: float,
                 covariate: bool = False) -> None:
        super().__init__()
        if dim % heads or (dim // heads) % 4:
            raise ValueError("d_model / heads must be divisible by 4")
        self.heads = heads
        self.covariate = covariate
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        # Same-/different-variate binary learnable bias, one per head.
        self.binary_bias = nn.Embedding(2, heads)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        """Rotate adjacent pairs without changing the input tensor shape."""
        pairs = rearrange(x, "... (d two) -> ... d two", two=2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
        return rearrange(rotated, "... d two -> ... (d two)")

    def _partial_rotary(self, x: torch.Tensor) -> torch.Tensor:
        """[B,H,C*N,D/H] -> same; rotate first half of head width."""
        # Official QK projection rotates the first HALF of each head's D/H.
        width = x.shape[-1] // 2
        rotating, unchanged = x[..., :width], x[..., width:]
        # Both keep [B,H,C*N,*]; widths sum to the original D/H.
        positions = torch.arange(x.shape[-2], device=x.device,
                                 dtype=torch.float32)
        theta = 1.0 / torch.pow(
            torch.tensor(10000.0, device=x.device),
            torch.arange(0, width, 2, device=x.device,
                         dtype=torch.float32) / width)
        angles = positions[:, None] * theta[None, :]
        # [C*N,width/2] -> [C*N,width], matching adjacent feature pairs.
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        rotated = (rotating * angles.cos()[None, None] +
                   self._rotate(rotating) * angles.sin()[None, None])
        return torch.cat((rotated, unchanged), dim=-1)

    def _mask(self, channels: int, patches: int,
              device: torch.device) -> torch.Tensor:
        """Return forbidden key positions [C*N,C*N] for each query."""
        ids = torch.arange(channels * patches, device=device)
        var = ids // patches
        time = ids % patches
        if self.covariate:
            # Released TimerCovariateMask: within-variable causal history,
            # and every history patch of covariate variables for last var.
            allowed = (var[:, None] == var[None, :]) & (
                time[:, None] >= time[None, :])
            allowed[-patches:, :-patches] = True
        else:
            allowed = time[:, None] >= time[None, :]
        return ~allowed

    def forward(self, x: torch.Tensor, channels: int,
                patches: int) -> torch.Tensor:
        """[B,C*N,D] -> [B,C*N,D], attending to allowed patch times."""
        if x.shape[1] != channels * patches:
            raise ValueError("token count must equal channels * patches")
        # [B,C*N,D] -> [B,H,C*N,D/H]. RoPE uses flattened token index,
        # matching OpenLTM's QueryKeyProjection, not patch index alone.
        q = rearrange(self.query(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.key(x), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.value(x), "b n (h d) -> b h n d", h=self.heads)
        q = self._partial_rotary(q)
        k = self._partial_rotary(k)
        var_id = torch.arange(channels, device=x.device).repeat_interleave(
            patches)
        same = (var_id[:, None] == var_id[None, :]).long()
        bias = rearrange(self.binary_bias(same), "q k h -> h q k")
        # same:[C*N,C*N], bias:[H,C*N,C*N].
        # OpenLTM non-flash code scales scores+binary bias together.
        scores = (q @ k.transpose(-2, -1) + bias[None]) * (
            q.shape[-1] ** -0.5)
        scores = scores.masked_fill(
            self._mask(channels, patches, x.device)[None, None],
            float("-inf"))
        weights = self.attention_dropout(scores.softmax(dim=-1))
        # weights:[B,H,C*N,C*N] × V:[B,H,C*N,D/H].
        fused = rearrange(weights @ v, "b h n d -> b n (h d)")
        # [B,H,C*N,D/H] -> [B,C*N,D].
        return self.output(fused)


class TimerLayer(nn.Module):
    """Official post-norm causal attention + pointwise FFN [B,C*N,D]."""

    def __init__(self, dim: int, d_ff: int, heads: int,
                 dropout: float, activation: str,
                 covariate: bool) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")
        self.attention = TimeAttention(dim, heads, dropout, covariate)
        self.conv1 = nn.Conv1d(dim, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, channels: int,
                patches: int) -> torch.Tensor:
        """[B,C*N,D] -> same after attention and per-token FFN."""
        x = self.norm1(x + self.dropout(
            self.attention(x, channels, patches)))
        y = rearrange(x, "b n d -> b d n")  # [B,D,C*N]
        y = self.dropout(self.activation(self.conv1(y)))
        # Conv1d kernel=1: [B,D,C*N] -> [B,Dff,C*N].
        y = self.dropout(rearrange(self.conv2(y), "b d n -> b n d"))
        # Conv2 returns [B,D,C*N], rearranged to [B,C*N,D].
        return self.norm2(x + y)


class TimerXLBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout]; T must divide patch_len.

    The self-attention is causal in patch time, but the optional input-window
    normalization uses all T observed points. For online *per-step* use,
    recompute from a prefix instead of exposing a future point in the window.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 patch_len: int = 16, d_model: int = 64,
                 d_ff: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.0, activation: str = "gelu",
                 use_norm: bool = True, covariate: bool = False) -> None:
        super().__init__()
        if min(in_dim, out_dim, patch_len, d_model,
               d_ff, heads, layers) < 1:
            raise ValueError("dimensions and layers must be positive")
        self.in_dim = in_dim
        self.patch_len = patch_len
        self.use_norm = use_norm
        self.embedding = nn.Linear(patch_len, d_model)
        self.layers = nn.ModuleList(
            TimerLayer(d_model, d_ff, heads, dropout, activation, covariate)
            for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        # Official next-patch head is D -> output_token_len. Here D -> P
        # reconstructs same-length features, followed by cross-series fusion.
        self.time_readout = nn.Linear(d_model, patch_len)
        self.output = nn.Linear(in_dim, out_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B,Cin*(T/P),D] before the forecast/readout head."""
        if x.ndim != 3 or x.shape[-1] != self.in_dim or x.shape[1] % self.patch_len:
            raise ValueError("expected [B,T,Cin] with T divisible by patch_len")
        if self.use_norm:
            mean = x.mean(1, keepdim=True).detach()  # [B,1,Cin]
            centered = x - mean
            std = torch.sqrt(centered.var(1, keepdim=True,
                                          unbiased=False) + 1e-5)
            x = centered / std  # [B,T,Cin], std:[B,1,Cin]
        patches = rearrange(x, "b (n p) c -> b c n p", p=self.patch_len)
        # [B,T,Cin] -> [B,Cin,N,P], N=T/P, P=patch length.
        count = patches.shape[2]
        tokens = rearrange(self.embedding(patches),
                           "b c n d -> b (c n) d")
        # Shared Linear(P,D): [B,Cin,N,D] -> [B,Cin*N,D].
        for layer in self.layers:
            tokens = layer(tokens, self.in_dim, count)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> downstream features [B,T,Cout]."""
        tokens = self.forward_features(x)
        count = x.shape[1] // self.patch_len
        patches = rearrange(self.time_readout(tokens),
                            "b (c n) p -> b (n p) c", c=self.in_dim,
                            n=count)
        # [B,Cin*N,D] -> [B,Cin*N,P] -> [B,T,Cin].
        return self.output(patches)  # [B,T,Cout].


if __name__ == "__main__":
    block = TimerXLBlock(4, 6, patch_len=8, d_model=32, heads=4)
    sample = torch.randn(2, 32, 4, requires_grad=True)
    result = block(sample)
    assert result.shape == (2, 32, 6)
    result.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
