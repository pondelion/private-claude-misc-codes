"""ModernTCN large-kernel, two-stage grouped-convolution block.

The output is aligned to the input length; this interpolation/readout is an
adaptation of the paper's task-specific heads. Convolutions are noncausal.
"""
from __future__ import annotations

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def _fuse(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale[:, None, None]
    bias = bn.bias + (torch.zeros_like(bn.running_mean) if conv.bias is None
                      else conv.bias - bn.running_mean) * scale
    if conv.bias is None:
        bias = bn.bias - bn.running_mean * scale
    return weight, bias


class ReparamDepthwise(nn.Module):
    """Training: large+small Conv/BN; evaluation: one fused depthwise Conv."""

    def __init__(self,
                 channels: int,
                 large_kernel: int = 15,
                 small_kernel: int = 3):
        super().__init__()
        if large_kernel % 2 != 1 or small_kernel % 2 != 1 or small_kernel > large_kernel:
            raise ValueError(
                "kernels must be odd and small_kernel <= large_kernel")
        self.large_kernel = large_kernel
        self.large = nn.Sequential(
            nn.Conv1d(channels,
                      channels,
                      large_kernel,
                      padding=large_kernel // 2,
                      groups=channels,
                      bias=False), nn.BatchNorm1d(channels))
        self.small = nn.Sequential(
            nn.Conv1d(channels,
                      channels,
                      small_kernel,
                      padding=small_kernel // 2,
                      groups=channels,
                      bias=False), nn.BatchNorm1d(channels))
        self.fused: nn.Conv1d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            return self.fused(x)
        return self.large(x) + self.small(x)

    @torch.no_grad()
    def merge_kernel(self) -> None:
        """Call after training and model.eval(); do not resume training afterward."""
        if self.fused is not None:
            return
        if self.training:
            raise RuntimeError("call model.eval() before merging BN branches")
        w1, b1 = _fuse(*self.large)
        w2, b2 = _fuse(*self.small)
        pad = (self.large_kernel - w2.size(-1)) // 2
        merged = nn.Conv1d(w1.size(0),
                           w1.size(0),
                           self.large_kernel,
                           padding=self.large_kernel // 2,
                           groups=w1.size(0),
                           bias=True,
                           device=w1.device,
                           dtype=w1.dtype)
        merged.weight.copy_(w1 + F.pad(w2, (pad, pad)))
        merged.bias.copy_(b1 + b2)
        self.fused = merged
        del self.large, self.small


class ModernTCNUnit(nn.Module):
    """[B,N,D,P] -> same. N=variables, D=latent/variable, P=patches."""

    def __init__(self, nvars: int, dim: int, expansion: int, large: int,
                 small: int, dropout: float):
        super().__init__()
        hidden = dim * expansion
        self.nvars, self.dim = nvars, dim
        self.dw = ReparamDepthwise(nvars * dim, large, small)
        self.bn = nn.BatchNorm1d(dim)
        # Group by N: mix D within each variable independently.
        # Released Block: PWConv -> Dropout -> GELU -> PWConv -> Dropout.
        self.ffn_var = nn.Sequential(
            nn.Conv1d(nvars * dim, nvars * hidden, 1, groups=nvars),
            nn.Dropout(dropout), nn.GELU(),
            nn.Conv1d(nvars * hidden, nvars * dim, 1, groups=nvars),
            nn.Dropout(dropout))
        # Reorder to [B,D,N,P], then group by D: mix all N variables per latent coordinate.
        self.ffn_cross = nn.Sequential(
            nn.Conv1d(dim * nvars, dim * nvars * expansion, 1, groups=dim),
            nn.Dropout(dropout), nn.GELU(),
            nn.Conv1d(dim * nvars * expansion, dim * nvars, 1, groups=dim),
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d, p = x.shape
        # Depthwise temporal convolution: [B,N,D,P] -> [B,N*D,P].
        h = self.dw(rearrange(x, "b n d p -> b (n d) p"))
        # BN shares its D channel statistics across the B*N variable streams.
        h = rearrange(h, "b (n d) p -> (b n) d p", n=n, d=d)
        h = self.bn(h)
        h = rearrange(h, "(b n) d p -> b (n d) p", b=b, n=n)
        # Variable-wise FFN mixes D but does not mix distinct N variables.
        h = self.ffn_var(h)
        h = rearrange(h, "b (n d) p -> b n d p", n=n, d=d)
        # Cross-variable FFN groups by D and mixes the N variable axis.
        h = self.ffn_cross(rearrange(h, "b n d p -> b (d n) p"))
        h = rearrange(h, "b (d n) p -> b n d p", d=d, n=n)
        return x + h


class ModernTCNBlock(nn.Module):
    """[B,T,Cin] -> [B,T,Cout], length-preserving practical adapter."""

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 d_model: int = 32,
                 patch_size: int = 8,
                 patch_stride: int = 4,
                 depth: int = 2,
                 large_kernel: int = 15,
                 small_kernel: int = 3,
                 expansion: int = 2,
                 dropout: float = 0.0,
                 stage_dims: tuple[int, ...] | None = None,
                 stage_depths: tuple[int, ...] | None = None,
                 downsample_ratio: int = 2):
        super().__init__()
        if min(in_dim, out_dim, d_model, patch_size, patch_stride, depth,
               expansion, downsample_ratio) < 1:
            raise ValueError("dimensions must be positive")
        if patch_stride > patch_size:
            raise ValueError(
                "patch_stride must not exceed patch_size; otherwise samples are uncovered"
            )
        self.in_dim = in_dim
        self.patch_size, self.patch_stride = patch_size, patch_stride
        dims = (d_model, ) if stage_dims is None else tuple(stage_dims)
        depths = (depth, ) * len(dims) if stage_depths is None else tuple(
            stage_depths)
        if not dims or len(dims) != len(depths) or min(*dims, *depths) < 1:
            raise ValueError(
                "stage_dims and stage_depths must have equal nonzero length")
        self.stage_dims, self.downsample_ratio = dims, downsample_ratio
        self.stem = nn.Conv1d(1, dims[0], patch_size, stride=patch_stride)
        self.stem_bn = nn.BatchNorm1d(dims[0])
        # Official ModernTCN: each stage contains its own stack of identical
        # Block architecture, with BN -> strided Conv between stages.
        self.stages = nn.ModuleList(
            nn.ModuleList(
                ModernTCNUnit(in_dim, dim, expansion, large_kernel,
                              small_kernel, dropout) for _ in range(count))
            for dim, count in zip(dims, depths))
        self.downsamples = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i],
                          dims[i + 1],
                          downsample_ratio,
                          stride=downsample_ratio))
            for i in range(len(dims) - 1))
        self.readout = nn.Linear(dims[-1], 1)
        self.output = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.in_dim or x.size(1) < 1:
            raise ValueError(f"expected [B,T,{self.in_dim}] with T>=1")
        b, t, n = x.shape
        # Treat each variable as an independent patch-embedding stream.
        xc = rearrange(x, "b t n -> (b n) 1 t")
        # Official forward_feature pads exactly K-S samples when K != S.
        # Only windows shorter than K need an additional generic-adapter pad.
        right = (self.patch_size - self.patch_stride
                 if self.patch_size != self.patch_stride else 0)
        right = max(right, self.patch_size - t)
        xc = F.pad(xc, (0, right), mode="replicate")
        z = self.stem_bn(self.stem(xc))
        p = z.size(-1)
        z = rearrange(z, "(b n) d p -> b n d p", b=b, n=n)
        for stage_idx, units in enumerate(self.stages):
            if stage_idx:
                # Official forward_feature copies the final `right` patches
                # (not just the last patch) before each downsampling stage.
                flat = rearrange(z, "b n d p -> (b n) d p")
                right = (-flat.size(-1)) % self.downsample_ratio
                if right:
                    flat = torch.cat((flat, flat[..., -right:]), dim=-1)
                flat = self.downsamples[stage_idx - 1](flat)
                z = rearrange(flat, "(b n) d p -> b n d p", b=b, n=n)
            for unit in units:
                z = unit(z)
        per_var = self.readout(z.permute(0, 3, 1, 2)).squeeze(-1)  # [B,P,N]
        aligned = F.interpolate(per_var.transpose(1, 2),
                                size=t,
                                mode="linear",
                                align_corners=False)
        # The original fixed-horizon head is replaced only to return one
        # feature vector per input time step for arbitrary downstream tasks.
        return self.output(aligned.transpose(1, 2))

    def merge_kernels(self) -> None:
        """Reduce each large+small+BN pair to one convolution for inference."""
        if self.training:
            raise RuntimeError("call eval() before merge_kernels()")
        for stage in self.stages:
            for unit in stage:
                unit.dw.merge_kernel()


if __name__ == "__main__":
    model = ModernTCNBlock(3, 5, d_model=8, depth=2)
    x = torch.randn(2, 33, 3, requires_grad=True)
    y = model(x)
    y.square().mean().backward()
    assert y.shape == (2, 33, 5) and x.grad is not None
    model.eval()
    with torch.no_grad():
        before = model(x.detach())
        model.merge_kernels()
        after = model(x.detach())
        torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
