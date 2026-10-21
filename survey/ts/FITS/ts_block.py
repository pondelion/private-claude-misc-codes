"""FITS low-frequency complex mixing as a same-length sequence encoder.

The original forecasting head increases FFT length.  Here the target length
is the input length, so this module can precede arbitrary task heads.
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class FITSBlock(nn.Module):
    """Map [B,T,Cin] to [B,T,Cout] through a learned low-pass spectrum.

    Args:
        cut_freq: Number of retained rFFT bins, including DC.  Every input
            must satisfy floor(T/2)+1 >= cut_freq.
        individual: Give each input channel independent complex weights.
        residual: Preserve the input's high frequencies via an optional skip.
            This is an encoder adaptation; the official FITS path uses False.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 cut_freq: int = 16,
                 individual: bool = False,
                 residual: bool = False,
                 eps: float = 1e-5,
                 variance_unbiased: bool = True) -> None:
        super().__init__()
        if min(in_dim, out_dim, cut_freq) < 1:
            raise ValueError("dimensions and cut_freq must be positive")
        self.in_dim = in_dim
        self.cut_freq = cut_freq
        self.individual = individual
        self.residual = residual
        self.eps = eps
        self.variance_unbiased = variance_unbiased
        if individual:
            self.freq_upsampler = nn.ModuleList([
                nn.Linear(cut_freq, cut_freq, dtype=torch.cfloat)
                for _ in range(in_dim)
            ])
        else:
            self.freq_upsampler = nn.Linear(cut_freq,
                                            cut_freq,
                                            dtype=torch.cfloat)
        self.output = nn.Linear(in_dim, out_dim)
        if in_dim == out_dim:
            # Keep a trainable channel-fusion adapter while making its initial
            # mapping neutral for a direct official-core comparison.
            nn.init.eye_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> [B,T,Cout]; T is even, F=T//2+1 FFT bins."""
        if x.ndim != 3 or x.size(-1) != self.in_dim:
            raise ValueError("x must have shape [B,T,in_dim]")
        batch, length, channels = x.shape
        if length % 2:
            raise ValueError(
                "official FITS irfft returns T-1 for odd T; this same-length "
                "encoder supports even T only")
        bins = length // 2 + 1
        if bins < self.cut_freq:
            raise ValueError(
                f"T={length} has {bins} rFFT bins, fewer than cut_freq={self.cut_freq}"
            )

        # Instance-wise reversible normalization: mean, scale [B,1,Cin].
        mean = x.mean(dim=1, keepdim=True)
        scale = (x.var(dim=1, keepdim=True, unbiased=self.variance_unbiased) +
                 self.eps).sqrt()
        normalized = (x - mean) / scale  # [B,T,Cin]
        spectrum = torch.fft.rfft(normalized,
                                  dim=1)  # [B,F,Cin], F=floor(T/2)+1
        # Keep K=cut_freq low bins and make frequency the Linear input axis:
        # [B,F,Cin] -> [B,Cin,K]. No series fusion occurs in this mixer.
        low = rearrange(spectrum[:, :self.cut_freq], "b f c -> b c f")
        if self.individual:
            mixed = torch.stack([
                layer(low[:, channel])
                for channel, layer in enumerate(self.freq_upsampler)
            ],
                                dim=1)  # [B,Cin,cut_freq]
        else:
            mixed = self.freq_upsampler(low)  # [B,Cin,cut_freq]
        mixed = rearrange(mixed, "b c f -> b f c")  # [B,K,Cin]

        # Zero-fill omitted high frequencies; unlike forecasting FITS the FFT
        # length is unchanged, so the original length-ratio correction is 1.
        full = mixed.new_zeros((batch, bins, channels))  # [B,F,Cin]
        full[:, :self.cut_freq] = mixed
        decoded = torch.fft.irfft(full, n=length, dim=1)  # [B,T,Cin]
        if self.residual:
            decoded = decoded + normalized
        restored = decoded * scale + mean  # [B,T,Cin]
        return self.output(restored)  # [B,T,Cout]
