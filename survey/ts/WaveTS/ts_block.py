"""WaveTS WDT -> per-order FRU -> inverse WDT -> time projection.

The original FRU changes coefficient length from L to L+H. Here H=0 so the
block is reusable as [B,L,Cin] -> [B,L,Cout]. Coefficient-domain *linear*
maps and final time projection match the released operation types.
Only Haar/db1/bior1.1 is implemented without a wavelet dependency. Odd-length
boundaries use the official ``reflect`` rule (append the penultimate value).
"""
from __future__ import annotations

import math
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _haar_forward(x: torch.Tensor, levels: int):
    """[B,C,T] -> low [B,C,ceil(T/2**J)], J highs and original lengths."""
    highs, lengths = [], []
    low = x
    for _ in range(levels):
        lengths.append(low.size(-1))
        if low.size(-1) % 2:
            # pytorch_wavelets mode='reflect' extends an odd band with its
            # penultimate sample; replicate would change boundary coefficients.
            if low.size(-1) < 2:
                raise ValueError("reflect Haar requires at least two samples per level")
            low = F.pad(low, (0, 1), mode="reflect")
        # Split adjacent time samples: [B,C,T] -> two [B,C,T/2] bands.
        even, odd = low[..., ::2], low[..., 1::2]
        highs.append((even - odd) / math.sqrt(2))
        low = (even + odd) / math.sqrt(2)
    return low, highs, lengths


def _haar_inverse(low: torch.Tensor, highs: list[torch.Tensor],
                  lengths: list[int]):
    """Inverse Haar, trimming only forward-transform boundary padding."""
    for high, length in zip(reversed(highs), reversed(lengths)):
        # Reconstruct even/odd samples and interleave [B,C,T/2,2] -> [B,C,T].
        even = (low + high) / math.sqrt(2)
        odd = (low - high) / math.sqrt(2)
        low = rearrange(torch.stack((even, odd), -1),
                        "b c half pair -> b c (half pair)")[..., :length]
    return low


class FRU(nn.Module):
    """Official frequency-resolution unit specialized to output length L.

    Each approximation/detail band has an independent Linear(L_j,L_j),
    shared across batch and channels exactly as the released FRU does.
    """

    def __init__(self, seq_len: int, levels: int):
        super().__init__()
        lengths = []
        n = seq_len
        for _ in range(levels):
            n = math.ceil(n / 2)
            lengths.append(n)
        self.low = nn.Linear(lengths[-1], lengths[-1])
        self.highs = nn.ModuleList(nn.Linear(k, k) for k in lengths)

    def forward(self, low: torch.Tensor, highs: list[torch.Tensor]):
        # low:[B,C,L_J]; high_j:[B,C,L_j] -> same shapes.
        return self.low(low), [layer(h) for layer, h in zip(self.highs, highs)]


class WaveTSBlock(nn.Module):
    """Fixed-window WaveTS block, [B,seq_len,Cin] -> [B,seq_len,Cout].

    `seq_len` is required because official FRU and time projection have
    learned matrices whose shapes depend on input window length. A single
    checkpoint cannot cover arbitrary T without changing that model.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 seq_len: int,
                 levels: int = 3,
                 orders: tuple[int, ...] = (1, 2)):
        super().__init__()
        if min(in_dim, out_dim, seq_len, levels) < 1 or not orders:
            raise ValueError(
                'positive dimensions/levels and nonempty orders required')
        if any(not isinstance(k, int) or k < 1 for k in orders):
            raise ValueError('orders must be positive integers')
        self.in_dim, self.out_dim, self.seq_len, self.levels = in_dim, out_dim, seq_len, levels
        self.orders = tuple(orders)
        self.frus = nn.ModuleList(FRU(seq_len, levels) for _ in orders)
        # Official projection_time mixes the order axis after concatenating
        # reconstructed sequences in time: [B,C,O*L] -> [B,C,L].
        self.projection_time = nn.Linear(len(orders) * seq_len, seq_len)
        # Always learn cross-series fusion, even when Cin == Cout. The
        # wavelet/FRU backbone itself applies shared weights per series.
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[
                2] != self.in_dim:
            raise ValueError(f'expected [B,{self.seq_len},{self.in_dim}]')
        # Official non-stationary normalization: mean is detached, variance
        # uses population denominator and remains on the autograd path.
        mean = x.mean(1, keepdim=True).detach()
        centered = x - mean
        std = torch.sqrt(centered.var(1, keepdim=True, unbiased=False) + 1e-5)
        normed = (centered / std).transpose(1, 2)  # [B,C,L]
        branches = []
        for order, fru in zip(self.orders, self.frus):
            # Each order has its own learned FRU; wavelet analysis is shared.
            low, highs, lengths = _haar_forward(normed, self.levels)
            low_factor = (-1)**order * 2.0**(order * self.levels)
            factors = [(-1)**order * 2.0**(order * (j + 1))
                       for j in range(self.levels)]
            low, highs = fru(low * low_factor,
                             [h * f for h, f in zip(highs, factors)])
            reconstruction = _haar_inverse(
                low / low_factor, [h / f for h, f in zip(highs, factors)],
                lengths)
            branches.append(reconstruction.transpose(1, 2))  # each [B,L,C]
        ordered = torch.cat(branches, dim=1).transpose(1, 2)  # [B,C,O*L]
        prediction = self.projection_time(ordered).transpose(1, 2)  # [B,L,C]
        return self.output(prediction * std + mean)


if __name__ == '__main__':
    model = WaveTSBlock(4, 6, seq_len=32, levels=3)
    x = torch.randn(2, 32, 4, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert y.shape == (2, 32, 6) and x.grad is not None
