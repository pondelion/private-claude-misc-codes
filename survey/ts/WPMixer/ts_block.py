"""WPMixer multi-resolution Haar patch mixer for downstream encoders.

The released forecasting model uses ``pytorch_wavelets`` and offers several
wavelet families.  This dependency-free block implements the exact reversible
Haar/db1 filter bank only.  It keeps the published per-band patch/embedding
mixers and returns the same time length instead of a forecasting horizon.
"""
from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def haar_decompose(x: torch.Tensor, levels: int):
    """[B,C,T] -> low [B,C,ceil(T/2**J)], details fine-to-coarse.

    The official Haar/db1 backend defaults to zero boundary extension.
    ``lengths`` records the original sizes for the exact inverse check.
    """
    low, details, lengths = x, [], []
    root2 = math.sqrt(2.0)
    for _ in range(levels):
        lengths.append(low.size(-1))
        if low.size(-1) % 2:
            low = F.pad(low, (0, 1), mode="constant", value=0.0)
            # Odd coefficient length L -> L+1 before even/odd splitting.
        even, odd = low[..., ::2], low[..., 1::2]
        # Each [B,C,ceil(L/2)]; low/detail retain independent frequency bands.
        details.append((even - odd) / root2)
        low = (even + odd) / root2
    return low, details, lengths


def haar_reconstruct(low: torch.Tensor, details: list[torch.Tensor],
                     lengths: list[int]) -> torch.Tensor:
    """Invert fine-to-coarse bands: [B,C,L_J] -> [B,C,T]."""
    root2 = math.sqrt(2.0)
    for detail, original_length in zip(reversed(details), reversed(lengths)):
        even = (low + detail) / root2
        odd = (low - detail) / root2
        low = rearrange(torch.stack((even, odd), dim=-1),
                        "b c t pair -> b c (t pair)")
        # [B,C,L,2] -> [B,C,2L] -> original length after boundary trim.
        low = low[..., :original_length]
    return low


def haar_reconstruct_official(low: torch.Tensor,
                              details: list[torch.Tensor]) -> torch.Tensor:
    """Mirror DWT1DInverse's coefficient alignment before final time slice.

    The official inverse only trims an intermediate low band if it is one
    sample longer than the matching detail. It does not crop the final even
    length; WPMixerCore takes the *last* requested output samples afterward.
    """
    root2 = math.sqrt(2.0)
    for detail in reversed(details):
        if low.size(-1) > detail.size(-1):
            low = low[..., :-1]
        even = (low + detail) / root2
        odd = (low - detail) / root2
        low = rearrange(torch.stack((even, odd), dim=-1),
                        "b c t pair -> b c (t pair)")
        # [B,C,L,2] -> [B,C,2L]; caller keeps requested trailing T points.
    return low


class _Mixer(nn.Module):
    """Published token mixer followed by residual embedding mixer."""

    def __init__(self, channels: int, patches: int, d_model: int,
                 token_factor: int, embed_factor: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.token = nn.Sequential(nn.Linear(patches, patches * token_factor),
                                   nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(patches * token_factor, patches))
        self.norm2 = nn.BatchNorm2d(channels)
        self.embedding = nn.Sequential(
            nn.Linear(d_model, d_model * embed_factor), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * embed_factor, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,N,D] -> [B,C,N,D]; N=patches, D=embedding width."""
        # [B,C,N,D] -> [B,D,C,N] so Linear mixes patch positions N.
        tokens = rearrange(self.norm1(x), "b c n d -> b d c n")
        tokens = self.token(tokens)  # [B,D,C,N] -> same; mixes patch N.
        x = rearrange(self.dropout(tokens), "b d c n -> b c n d")
        x = self.norm2(x)  # [B,C,N,D]
        return x + self.dropout(self.embedding(x))  # [B,C,N,D]


class ResolutionBranch(nn.Module):
    """One WPMixer coefficient branch, [B,C,L] -> [B,C,L]."""

    def __init__(self, channels: int, length: int, patch_len: int,
                 patch_stride: int, d_model: int, token_factor: int,
                 embed_factor: int, dropout: float) -> None:
        super().__init__()
        self.length = length
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        padded_length = max(length, patch_len) + patch_stride
        self.patch_count = (padded_length - patch_len) // patch_stride + 1
        self.patch_norm = nn.BatchNorm2d(channels)
        # Each official ResolutionBranch owns its own affine RevIN. Keeping
        # these parameters separate matters even though all start at 1/0.
        self.affine_weight = nn.Parameter(torch.ones(channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.mixer1 = _Mixer(channels, self.patch_count, d_model, token_factor,
                             embed_factor, dropout)
        self.mixer2 = _Mixer(channels, self.patch_count, d_model, token_factor,
                             embed_factor, dropout)
        self.norm = nn.BatchNorm2d(channels)
        self.readout = nn.Linear(self.patch_count * d_model, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One wavelet band [B,C,L] -> same length after patch mixers."""
        if x.size(-1) != self.length:
            raise ValueError("coefficient length differs from construction")
        # Per-band RevIN, following the official ResolutionBranch.  Statistics
        # [B,C,1] are inverted after the learned patch readout.
        mean = x.mean(dim=-1, keepdim=True).detach()
        scale = (x.var(dim=-1, keepdim=True, unbiased=False) +
                 1e-5).sqrt().detach()
        normalized = (x - mean) / scale  # [B,C,L]
        normalized = (normalized * self.affine_weight[None, :, None] +
                      self.affine_bias[None, :, None])
        if self.length < self.patch_len:
            normalized = F.pad(normalized, (0, self.patch_len - self.length),
                               mode="replicate")
            # [B,C,L] -> [B,C,P] if the coefficient band is shorter than P.
        normalized = F.pad(normalized, (0, self.patch_stride),
                           mode="replicate")
        # [B,C,max(L,P)] -> [B,C,max(L,P)+stride].
        patches = normalized.unfold(-1, self.patch_len, self.patch_stride)
        # [B,C,padded_length] -> [B,C,N,P], N=self.patch_count.
        # [B,C,N,P] -> [B,C,N,D]; BN acts across batch and patch axes.
        embedded = self.embed_dropout(
            self.patch_embed(self.patch_norm(patches)))
        out = self.mixer1(embedded)
        out = self.norm(out + self.mixer2(out))  # [B,C,N,D]
        flat = rearrange(out, "b c n d -> b c (n d)")
        # [B,C,N,D] -> [B,C,N*D] for per-band time readout.
        predicted = self.readout(flat)  # [B,C,L]
        predicted = ((predicted - self.affine_bias[None, :, None]) /
                     (self.affine_weight[None, :, None] + 1e-10))
        return predicted * scale + mean  # [B,C,L]


class WPMixerBlock(nn.Module):
    """Wavelet-patch encoder, [B,seq_len,Cin] -> [B,seq_len,Cout]."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 seq_len: int,
                 levels: int = 2,
                 patch_len: int = 8,
                 patch_stride: int = 4,
                 d_model: int = 16,
                 token_factor: int = 2,
                 embed_factor: int = 2,
                 dropout: float = 0.0,
                 wavelet: str = "haar") -> None:
        super().__init__()
        if min(in_dim, out_dim, seq_len, patch_len, patch_stride, d_model,
               token_factor, embed_factor) < 1 or levels < 1:
            raise ValueError("all dimensions and levels must be positive")
        if wavelet not in {"haar", "db1"}:
            raise ValueError("this portable block supports Haar/db1 only")
        if seq_len < 2**levels:
            raise ValueError("seq_len must be at least 2**levels")
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.levels = levels
        lengths = [seq_len]
        for _ in range(levels):
            lengths.append((lengths[-1] + 1) // 2)
        # pytorch_wavelets returns detail bands fine-to-coarse: D1,...,DJ.
        # Branch indices therefore match the official [AJ,D1,...,DJ] order.
        band_lengths = [lengths[-1], *lengths[1:]]
        self.branches = nn.ModuleList([
            ResolutionBranch(in_dim, length, patch_len, patch_stride, d_model,
                             token_factor, embed_factor, dropout)
            for length in band_lengths
        ])
        self.input_affine_weight = nn.Parameter(torch.ones(in_dim))
        self.input_affine_bias = nn.Parameter(torch.zeros(in_dim))
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,Cin] -> same-length [B,T,Cout] with late series fusion."""
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.in_dim):
            raise ValueError("x must have shape [B,seq_len,in_dim]")
        mean = x.mean(dim=1, keepdim=True).detach()  # [B,1,Cin]
        scale = (x.var(dim=1, keepdim=True, unbiased=False) +
                 1e-5).sqrt().detach()
        normalized = (x - mean) / scale
        normalized = (normalized * self.input_affine_weight +
                      self.input_affine_bias)  # [B,T,Cin]
        low, details, _ = haar_decompose(
            rearrange(normalized, "b t c -> b c t"), self.levels)
        # [B,T,Cin] -> [B,Cin,T] -> low [B,Cin,L_J], details L_1..L_J.
        processed_low = self.branches[0](low)
        processed_details = [
            branch(detail)
            for branch, detail in zip(self.branches[1:], details)
        ]
        reconstructed = haar_reconstruct_official(
            processed_low, processed_details)  # [B,Cin,T or T+1]
        reconstructed = reconstructed[..., -self.seq_len:]  # [B,Cin,T]
        reconstructed = rearrange(reconstructed, "b c t -> b t c")
        # [B,Cin,T] -> [B,T,Cin] before inverse normalization/output Linear.
        # Inverse of the affine RevIN stage, then of instance normalization.
        restored = ((reconstructed - self.input_affine_bias) /
                    (self.input_affine_weight + 1e-10))
        restored = restored * scale + mean
        return self.output(restored)  # [B,T,Cout]
