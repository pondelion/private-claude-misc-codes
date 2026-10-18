"""Batched filtering with per-sample kernels (grouped convolution).

``cv2.filter2D`` / ``cv2.blur`` / ``cv2.sepFilter2D`` / ``cv2.GaussianBlur`` use BORDER_REFLECT_101 by default, which is
torch's ``"reflect"`` padding. Kernels of different sizes in one batch are zero-padded (centered) to a common size,
which leaves the result unchanged.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

Border = Literal["reflect", "replicate", "constant"]


def _conv2d(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """conv2d in full fp32 precision: cuDNN uses TF32 by default on Ampere+ GPUs, which gives ~1e-3 relative error
    (measured 1.5e-4 absolute on smoothed displacement fields vs. cv2). The caller's other cuDNN flags are kept."""
    cudnn = torch.backends.cudnn
    with cudnn.flags(enabled=cudnn.enabled, benchmark=cudnn.benchmark, deterministic=cudnn.deterministic, allow_tf32=False):
        return F.conv2d(x, weight, groups=groups)

# cv2.getGaussianKernel uses these fixed kernels when sigma <= 0 and ksize <= 7
_CV2_SMALL_GAUSSIAN = {
    1: [1.0],
    3: [0.25, 0.5, 0.25],
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
}


def _pad(x: torch.Tensor, ph: int, pw: int, border: Border) -> torch.Tensor:
    if ph == 0 and pw == 0:
        return x
    if border == "reflect":
        h, w = x.shape[-2:]
        if ph >= h or pw >= w:  # torch reflect needs pad < size; fall back to coordinate folding via repeated pads
            while ph > 0 or pw > 0:
                sh, sw = min(ph, x.shape[-2] - 1), min(pw, x.shape[-1] - 1)
                x = F.pad(x, (sw, sw, sh, sh), mode="reflect")
                ph, pw = ph - sh, pw - sw
            return x
        return F.pad(x, (pw, pw, ph, ph), mode="reflect")
    if border == "replicate":
        return F.pad(x, (pw, pw, ph, ph), mode="replicate")
    return F.pad(x, (pw, pw, ph, ph), mode="constant", value=0.0)


def filter2d(img: torch.Tensor, kernels: torch.Tensor, border: Border = "reflect") -> torch.Tensor:
    """Per-sample 2D correlation (cv2.filter2D semantics, no kernel flip).
    img (n, C, H, W) float, kernels (n, kh, kw) or (kh, kw) shared. Odd kernel sizes."""
    n, c, h, w = img.shape
    if kernels.ndim == 2:
        kernels = kernels.expand(n, *kernels.shape)
    kh, kw = kernels.shape[-2:]
    x = _pad(img, kh // 2, kw // 2, border)
    weight = kernels.to(img.dtype).repeat_interleave(c, dim=0)[:, None]  # (n*C, 1, kh, kw)
    out = _conv2d(x.reshape(1, n * c, *x.shape[-2:]), weight, groups=n * c)
    return out.view(n, c, h, w)


def separable_filter(img: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, border: Border = "reflect") -> torch.Tensor:
    """Per-sample separable filter. kx, ky: (n, k) or (k,) shared."""
    n, c, h, w = img.shape
    if kx.ndim == 1:
        kx = kx.expand(n, kx.shape[0])
    if ky.ndim == 1:
        ky = ky.expand(n, ky.shape[0])
    x = _pad(img, 0, kx.shape[-1] // 2, border)
    wx = kx.to(img.dtype).repeat_interleave(c, dim=0)[:, None, None, :]
    x = _conv2d(x.reshape(1, n * c, h, x.shape[-1]), wx, groups=n * c)
    x = _pad(x.view(n, c, h, w), ky.shape[-1] // 2, 0, border)
    wy = ky.to(img.dtype).repeat_interleave(c, dim=0)[:, None, :, None]
    x = _conv2d(x.reshape(1, n * c, x.shape[-2], w), wy, groups=n * c)
    return x.view(n, c, h, w)


def embed_kernels(kernels: list[torch.Tensor] | torch.Tensor, size: int) -> torch.Tensor:
    """Center-embed square kernels of (odd) sizes <= size into (n, size, size)."""
    if torch.is_tensor(kernels) and kernels.ndim == 3 and kernels.shape[-1] == size:
        return kernels
    out = []
    for k in kernels:
        p = (size - k.shape[-1]) // 2
        out.append(F.pad(k, (p, p, p, p)))
    return torch.stack(out)


def box_kernels(ksizes: torch.Tensor) -> torch.Tensor:
    """(n,) odd sizes -> (n, K, K) normalized box kernels centered in K = max size."""
    kmax = int(ksizes.max())
    r = torch.arange(kmax, device=ksizes.device) - kmax // 2
    half = (ksizes // 2)[:, None]
    inside = r.abs()[None] <= half  # (n, K)
    k2 = (inside[:, :, None] & inside[:, None, :]).float()
    return k2 / k2.sum(dim=(1, 2), keepdim=True)


def cv2_gaussian_kernel_1d(ksize: int, sigma: float, device: torch.device) -> torch.Tensor:
    """``cv2.getGaussianKernel(ksize, sigma)`` (float64 math, returned as float32)."""
    if sigma <= 0 and ksize in _CV2_SMALL_GAUSSIAN:
        return torch.tensor(_CV2_SMALL_GAUSSIAN[ksize], dtype=torch.float32, device=device)
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    x = torch.arange(ksize, dtype=torch.float64, device=device) - (ksize - 1) / 2
    k = torch.exp(-(x**2) / (2 * sigma * sigma))
    return (k / k.sum()).float()


def cv2_gaussian_kernels_1d(ksizes: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Per-sample ``cv2.getGaussianKernel`` centered in K = max ksize -> (n, K). sigma <= 0 handled like cv2."""
    kmax = int(ksizes.max())
    dev = ksizes.device
    rows = []
    for ks, sg in zip(ksizes.tolist(), sigmas.tolist()):
        k = cv2_gaussian_kernel_1d(int(ks), float(sg), dev)
        p = (kmax - k.numel()) // 2
        rows.append(F.pad(k, (p, p)))
    return torch.stack(rows)


def albu_gaussian_kernels_1d(sigmas: torch.Tensor, ksizes: torch.Tensor) -> torch.Tensor:
    """albumentations ``create_gaussian_kernel_1d`` per sample: size = ksize (0 -> int(sigma*3.5)*2+1), even -> +1,
    x = linspace(-(size//2), size//2, size), normalized exp(-x^2 / (2 sigma^2)). Returns (n, K) centered."""
    sizes = torch.where(ksizes == 0, (sigmas * 3.5).long() * 2 + 1, ksizes.long())
    sizes = sizes + (sizes % 2 == 0).long()
    kmax = int(sizes.max())
    r = (torch.arange(kmax, device=sigmas.device) - kmax // 2).float()[None]
    k = torch.exp(-0.5 * (r / sigmas[:, None]) ** 2)
    k = torch.where(r.abs() <= (sizes // 2)[:, None], k, torch.zeros_like(k))
    return k / k.sum(dim=1, keepdim=True)


def median_filter(img: torch.Tensor, ksizes: torch.Tensor) -> torch.Tensor:
    """Per-sample ``cv2.medianBlur`` (BORDER_REPLICATE). ksizes: (n,) odd. Processed per distinct kernel size."""
    out = torch.empty_like(img)
    n, c, h, w = img.shape
    for ks in torch.unique(ksizes).tolist():
        sel = torch.nonzero(ksizes == ks).squeeze(1)
        r = ks // 2
        # process in chunks to bound the memory of the unfolded tensor (k^2 x image)
        chunk = max(1, int(2**28 // max(1, c * h * w * ks * ks)))
        for start in range(0, sel.numel(), chunk):
            idx = sel[start : start + chunk]
            x = F.pad(img[idx], (r, r, r, r), mode="replicate")
            patches = x.unfold(2, ks, 1).unfold(3, ks, 1)  # (m, C, H, W, k, k)
            out[idx] = patches.reshape(*patches.shape[:4], ks * ks).median(dim=-1).values
    return out
