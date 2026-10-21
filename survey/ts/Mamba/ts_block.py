"""Shared (B,T,Cin)->(B,T,Cout) adapter for three distinct Mamba mixers.

All portable version-specific mixers live together in ``mamba_versions.py``.
Official fused implementations are selected when available on CUDA.
"""
from __future__ import annotations

import math
import warnings
import torch
from torch import nn
from mamba_versions import TorchMamba, TorchMamba2, TorchMamba3


class MambaBlock(nn.Module):
    """Pre-norm residual block: [B,T,Cin] -> [B,T,Cout].

    `version` selects *different* official architectures, not a speed preset.
    backend='auto' uses official mamba_ssm when installed on CUDA; 'torch' is
    portable but slower on long sequences.  No model version is substituted.
    """

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 d_model: int = 64,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 depth: int = 1,
                 dropout: float = 0.0,
                 backend: str = "auto",
                 version: int = 1,
                 headdim: int | None = None,
                 ngroups: int = 1,
                 is_mimo: bool = False,
                 mimo_rank: int = 4,
                 rope_fraction: float = 0.5,
                 is_outproj_norm: bool = False):
        super().__init__()
        if depth < 1 or backend not in {"auto", "official", "torch"
                                        } or version not in (1, 2, 3):
            raise ValueError("invalid depth, backend, or version")
        if version == 1 and (is_mimo or is_outproj_norm):
            raise ValueError("MIMO/outproj norm are Mamba-3-only options")
        if version == 2 and (is_mimo or is_outproj_norm):
            raise ValueError("MIMO/outproj norm are Mamba-3-only options")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.version = version
        self.input = nn.Linear(in_dim, d_model)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(depth))
        self.mixers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # Defaults for small off-the-shelf blocks; official module defaults
        # (128 for v2, 64 for v3) often do not divide small `expand*d_model`.
        if version in (2, 3) and headdim is None:
            headdim = math.gcd(d_model * expand, 64 if version == 3 else 128)
        for _ in range(depth):
            official = None
            if backend == "official" or (backend == "auto"
                                         and torch.cuda.is_available()):
                try:
                    if version == 1:
                        from mamba_ssm import Mamba
                        official = Mamba(d_model=d_model,
                                         d_state=d_state,
                                         d_conv=d_conv,
                                         expand=expand)
                    elif version == 2:
                        from mamba_ssm import Mamba2
                        official = Mamba2(d_model=d_model,
                                          d_state=d_state,
                                          d_conv=d_conv,
                                          expand=expand,
                                          headdim=headdim,
                                          ngroups=ngroups)
                    else:
                        from mamba_ssm import Mamba3
                        official = Mamba3(d_model=d_model,
                                          d_state=d_state,
                                          expand=expand,
                                          headdim=headdim,
                                          ngroups=ngroups,
                                          is_mimo=is_mimo,
                                          mimo_rank=mimo_rank,
                                          rope_fraction=rope_fraction,
                                          is_outproj_norm=is_outproj_norm)
                except (ImportError, AssertionError) as exc:
                    if backend == "official":
                        raise RuntimeError(
                            "backend='official' needs the selected version and its kernels"
                        ) from exc
                    warnings.warn(
                        f"Mamba-{version} official mixer unavailable ({exc}); using its PyTorch reference",
                        RuntimeWarning,
                        stacklevel=2)
            if official is not None:
                mixer = official
            elif version == 1:
                mixer = TorchMamba(d_model, d_state, d_conv, expand)
            elif version == 2:
                mixer = TorchMamba2(d_model, d_state, d_conv, expand, headdim,
                                    ngroups)
            else:
                mixer = TorchMamba3(d_model, d_state, expand, headdim, ngroups,
                                    rope_fraction, is_mimo, mimo_rank,
                                    is_outproj_norm)
            self.mixers.append(mixer)
        self.output = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.in_dim or x.size(1) < 1:
            raise ValueError(f"expected [B,T,{self.in_dim}]")
        # Lift input features: [B,T,Cin] -> [B,T,D].
        h = self.input(x)
        for norm, mixer in zip(self.norms, self.mixers):
            # Each causal mixer preserves [B,T,D]; residual keeps the signal.
            h = h + self.dropout(mixer(norm(h)))
        # The official mixer has width D; this readout gives the requested
        # Cout without adding a second raw-input bypass around the mixer.
        return self.output(h)


if __name__ == "__main__":
    for version in (1, 2, 3):
        block = MambaBlock(5,
                           7,
                           d_model=16,
                           d_state=4,
                           backend="torch",
                           version=version)
        x = torch.randn(2, 12, 5, requires_grad=True)
        y = block(x)
        y.square().mean().backward()
        assert y.shape == (2, 12, 7) and x.grad is not None and torch.isfinite(
            x.grad).all()
