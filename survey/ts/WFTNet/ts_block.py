"""WFTNet Fourier-period/CWT local-period encoder.

The official wavelet path calls ``ptwt.cwt(..., 'morl')``; this block calls
the same library operation. The forecasting-only time extension is omitted:
WFT features stay aligned to the observed T positions for encoder use.
"""
from __future__ import annotations

import math
import inspect
import numpy as np
import ptwt

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class InceptionBlock(nn.Module):
    """Mean of odd-kernel 2D convolutions, [B,D,H,W]->[B,E,H,W]."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_kernels: int = 3) -> None:
        super().__init__()
        if num_kernels < 1:
            raise ValueError("num_kernels must be positive")
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, 2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel.weight,
                                    mode="fan_out",
                                    nonlinearity="relu")
            nn.init.zeros_(kernel.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,D,H,W] -> [B,E,H,W]; average parallel 2D kernels."""
        # Stack K kernel outputs: [B,E,H,W,K] -> mean over K.
        return torch.stack([kernel(x) for kernel in self.kernels],
                           dim=-1).mean(dim=-1)


class MorletCWT(nn.Module):
    """Official eight-scale ``ptwt.cwt``: [B,D,T] -> [B,D,8,T]."""

    def __init__(self, wavelet_scale: float = 4.0) -> None:
        super().__init__()
        if wavelet_scale < -1:
            raise ValueError("wavelet_scale must be >= -1")
        # Match the released ``2 ** np.linspace(-1, scale, 8)`` exactly.
        self.scales = 2 ** np.linspace(-1.0, wavelet_scale, 8)
        # The official requirements pin ptwt 0.1.6, whose CWT uses precision
        # 10 internally. ptwt 1.x changed its default to 12; request 10 when
        # the keyword exists, retaining compatibility with the pinned 0.1.6.
        self.cwt_kwargs = ({"precision": 10}
                           if "precision" in inspect.signature(ptwt.cwt).parameters
                           else {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,D,T] -> Morlet coefficients [B,D,8,T]."""
        if x.ndim != 3:
            raise ValueError("x must be [B,D,T]")
        coefficients, _ = ptwt.cwt(x, self.scales, "morl", **self.cwt_kwargs)
        # ptwt returns [8,B,D,T] in float64 for NumPy float64 scales;
        # released Wavelet.forward permutes it and explicitly calls .float().
        return rearrange(coefficients, "s b d t -> b d s t").float()


class WFTBlock(nn.Module):
    """One WFT fusion block, [B,T,D] -> [B,T,D]."""

    def __init__(self,
                 d_model: int,
                 d_ff: int = 64,
                 top_k: int = 3,
                 period_coeff: float = 0.8,
                 wavelet_scale: float = 4.0,
                 num_kernels: int = 3,
                 pwc_mode: str = "official_fixed",
                 pwc_bins: int = 8,
                 seq_len: int | None = None,
                 pred_len: int | None = None) -> None:
        super().__init__()
        if min(d_model, d_ff) < 1 or top_k < 0:
            raise ValueError("invalid dimensions or top_k")
        if not 0.0 <= period_coeff <= 1.0:
            raise ValueError("period_coeff must lie in [0,1]")
        if pwc_mode not in {"official_fixed", "paper_adaptive"}:
            raise ValueError(
                "pwc_mode must be official_fixed or paper_adaptive")
        if pwc_bins < 1:
            raise ValueError("pwc_bins must be positive")
        self.top_k = top_k
        self.period_mix = period_coeff**10
        self.pwc_mode = pwc_mode
        self.pwc_bins = pwc_bins
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cwt = MorletCWT(wavelet_scale)
        self.period_conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, num_kernels), nn.GELU(),
            InceptionBlock(d_ff, d_model, num_kernels))
        self.scale_conv = nn.Conv2d(d_model, d_model, (8, 1), groups=d_model)
        self.projection = (nn.Linear(seq_len + pred_len, pred_len)
                           if seq_len is not None and pred_len is not None
                           else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,D] -> [B,T,D], combining FFT periods and wavelet scales."""
        _, length, _ = x.shape
        fft = torch.fft.rfft(x, dim=1)  # [B,F,D]
        if self.top_k > fft.size(1) - 1:
            raise ValueError("top_k exceeds the number of non-DC rFFT bins")

        # Global top-k bins define periods; per-sample amplitudes weight them.
        period_outputs = []
        if self.top_k:
            energy = fft.abs().mean(dim=(0, 2))  # [F]
            energy = energy.clone()
            energy[0] = -torch.inf
            indices = torch.topk(energy, self.top_k).indices  # [K]
            weights = torch.softmax(fft.abs().mean(-1)[:, indices], dim=-1)
            # [B,K] sample-specific weights for K selected frequency bins.
            for index in indices.tolist():
                period = max(1, length // index)
                padded_length = math.ceil(length / period) * period
                padded = F.pad(rearrange(x, "b t d -> b d t"),
                               (0, padded_length - length))
                # [B,T,D] -> [B,D,T] -> [B,D,Tpad].
                grid = rearrange(padded, "b d (n p) -> b d n p", p=period)
                # [B,D,Tpad] -> [B,D,Nperiod,period].
                changed = self.period_conv(grid)  # [B,D,n,period]
                series = rearrange(changed, "b d n p -> b (n p) d")
                # [B,D,Nperiod,period] -> [B,Tpad,D] -> trim to [B,T,D].
                period_outputs.append(series[:, :length])
            stacked = torch.stack(period_outputs, dim=-1)  # [B,T,D,K]
            global_path = (stacked * weights[:, None, None, :]).sum(dim=-1)
            # [B,T,D,K] weighted over K -> [B,T,D].

        coeffs = self.cwt(rearrange(x, "b t d -> b d t"))
        # [B,T,D] -> [B,D,T] -> CWT [B,D,8,T].
        local_grid = self.period_conv(coeffs)  # [B,D,8,T]
        local_path = self.scale_conv(local_grid).squeeze(2)
        # Depthwise scale Conv: [B,D,8,T] -> [B,D,1,T] -> [B,D,T].
        local_path = rearrange(local_path, "b d t -> b t d")  # [B,T,D]
        if self.pwc_mode == "paper_adaptive" and self.top_k:
            # Paper's per-example PWC: mean over channels of max spectral
            # power / total power among the first m non-DC bins. [B,1,1]
            power = fft[:, 1:1 + self.pwc_bins].abs().square()
            ratio = power.max(dim=1).values / power.sum(dim=1).clamp_min(1e-12)
            mix = ratio.mean(dim=-1)[:, None, None].pow(10)
        else:
            # The released implementation actually uses a fixed config scalar.
            mix = self.period_mix
        fused = ((1.0 - mix) * local_path +
                 mix * global_path) if self.top_k else local_path
        result = x + fused  # [B,T,D]
        if not self.top_k and self.projection is not None:
            return rearrange(
                self.projection(rearrange(result, "b t d -> b d t")),
                "b d t -> b t d")
        return result


class WFTNetBlock(nn.Module):
    """Stacked WFTBlocks and channel projection, [B,T,Cin]->[B,T,Cout]."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 d_model: int = 32,
                 d_ff: int = 64,
                 layers: int = 1,
                 top_k: int = 3,
                 period_coeff: float = 0.8,
                 wavelet_scale: float = 4.0,
                 num_kernels: int = 3,
                 pwc_mode: str = "official_fixed",
                 pwc_bins: int = 8,
                 seq_len: int | None = None) -> None:
        super().__init__()
        if min(in_dim, out_dim, d_model, layers) < 1:
            raise ValueError("dimensions and layers must be positive")
        if seq_len is None or seq_len < 1:
            raise ValueError("seq_len is required for the released time path")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.input = nn.Conv1d(in_dim,
                               d_model,
                               3,
                               padding=1,
                               padding_mode="circular",
                               bias=False)
        self.layers = nn.ModuleList([
            WFTBlock(d_model, d_ff, top_k, period_coeff, wavelet_scale,
                     num_kernels, pwc_mode, pwc_bins,
                     seq_len=seq_len, pred_len=None)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        position = torch.arange(seq_len, dtype=torch.float32)[:, None]
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model))
        positional = torch.zeros(seq_len, d_model)
        positional[:, 0::2] = torch.sin(position * frequency)
        positional[:, 1::2] = torch.cos(position * frequency[:d_model // 2])
        self.register_buffer("positional", positional[None])
        # Forecast-only T->T+H expansion is not needed for an encoder. Decode
        # every observed-position D-vector directly, without D->Cin bottleneck.
        self.output = nn.Linear(d_model, out_dim)
        # Normalization discards level/scale from the WFT path. This learned
        # raw-input skip preserves access to that information downstream.
        self.input_skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> [B,T,Cout] with no temporal extension or cropping."""
        if x.ndim != 3 or x.size(-1) != self.in_dim:
            raise ValueError("x must have shape [B,T,in_dim]")
        if x.size(1) != self.seq_len:
            raise ValueError("input length differs from seq_len")
        mean = x.mean(dim=1, keepdim=True).detach()  # [B,1,Cin]
        centered = x - mean
        scale = (centered.var(dim=1, keepdim=True,
                              unbiased=False) + 1e-5).sqrt()
        normalized = centered / scale  # [B,T,Cin]
        embedded = self.input(rearrange(normalized, "b t c -> b c t"))
        # [B,T,Cin] -> [B,Cin,T] -> Conv1d [B,D,T]; early series fusion.
        hidden = rearrange(embedded, "b d t -> b t d")  # [B,T,D]
        hidden = hidden + self.positional  # official DataEmbedding
        # Keep all T observed positions. WFT's FFT/CWT/period Conv process
        # [B,T,D] and return [B,T,D]; no artificial future positions exist.
        for layer in self.layers:
            hidden = self.norm(layer(hidden))  # [B,T,D]
        # Learned per-time readout [B,T,D] -> [B,T,Cout]. Unlike the former
        # D->Cin->Cout path, no intermediate Cin bottleneck is introduced.
        # The raw skip [B,T,Cin] -> [B,T,Cout] retains level/scale cues that
        # instance normalization removes from the WFT branch.
        return self.output(hidden) + self.input_skip(x)
