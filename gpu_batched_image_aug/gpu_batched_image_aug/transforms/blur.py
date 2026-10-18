"""Blur transforms (image only). Kernel sizes are drawn per sample; samples are filtered in one grouped convolution
(BORDER_REFLECT_101 like cv2.filter2D / cv2.blur, BORDER_REPLICATE for the median like cv2.medianBlur)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ..core.data import AugBatch
from ..core.random import Rng
from ..core.transform import ImageOnlyTransform
from ..core.utils import check_bounds, check_nondecreasing, int_range, odd_blur_range, to_range
from ..functional import geometric as fgeo
from ..functional.filters import (
    albu_gaussian_kernels_1d,
    box_kernels,
    cv2_gaussian_kernel_1d,
    filter2d,
    median_filter,
    separable_filter,
)

__all__ = ["Blur", "Defocus", "GaussianBlur", "MedianBlur", "MotionBlur", "ZoomBlur"]
# motion_kernels is importable from this module for testing / custom use


def _sample_odd(rng: Rng, low: int, high: int, n: int) -> torch.Tensor:
    """albumentations ``sample_odd_from_range`` (min 3)."""
    low = max(3, low + (low % 2 == 0))
    high = max(3, high + (high % 2 == 0))
    high = max(high, low)
    num = (high - low) // 2 + 1
    return low + 2 * rng.randint(0, num - 1, n)


def motion_kernels(
    ksize: torch.Tensor, angle: torch.Tensor, direction: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor
) -> torch.Tensor:
    """albumentations ``create_motion_kernel`` (batched, normalized): a line through the kernel center at ``angle``
    degrees, sampled at ksize points of np.linspace(t_start, t_end, ksize) (direction in [-1, 1] biases the line to
    one side), shifted by (shift_x, shift_y) * (ksize // 2) / 2 (shift values in [-1, 1]; 0 = not shifted).
    Returns (n, K, K) kernels centered in K = max(ksize)."""
    n = ksize.shape[0]
    dev = ksize.device
    direction = direction.clamp(-1, 1)
    kmax = int(ksize.max())
    center = (ksize // 2).float()
    line = (ksize // 2).float()
    bias = direction.double().abs()
    t_start = torch.where(direction > 0, -line * (1 - bias), -line.double())
    t_end = torch.where(direction < 0, line * (1 - bias), line.double())
    # float64 like albumentations: points landing exactly on .5 must round the same way (half to even)
    center, line, t_start, t_end = center.double(), line.double(), t_start.double(), t_end.double()
    j = torch.arange(kmax, device=dev, dtype=torch.float64)[None]
    denom = (ksize - 1).clamp(min=1).double()[:, None]
    t = t_start[:, None] + (t_end - t_start)[:, None] * j / denom
    rad = torch.deg2rad(angle.double())
    x = center[:, None] + torch.cos(rad)[:, None] * t + (shift_x.double() * line / 2)[:, None]
    y = center[:, None] + torch.sin(rad)[:, None] * t + (shift_y.double() * line / 2)[:, None]
    hi = (ksize - 1)[:, None].double()
    xi = torch.minimum(torch.round(x).clamp(min=0), hi).long()
    yi = torch.minimum(torch.round(y).clamp(min=0), hi).long()
    used = j < ksize[:, None]
    kern = torch.zeros(n, kmax * kmax, device=dev)
    kern.scatter_(1, torch.where(used, yi * kmax + xi, torch.zeros_like(yi)), used.float())
    kern = kern.view(n, kmax, kmax)
    # kernels of size k were built in the top-left k x k corner; move them to the center of the K x K canvas
    shift = (kmax - ksize) // 2
    out = torch.zeros_like(kern)
    for s_ in torch.unique(shift).tolist():
        sel = shift == s_
        out[sel] = torch.roll(kern[sel], shifts=(s_, s_), dims=(1, 2))
    empty = out.sum(dim=(1, 2)) == 0
    if bool(empty.any()):
        out[empty, kmax // 2, kmax // 2] = 1.0
    return out / out.sum(dim=(1, 2), keepdim=True)


class Blur(ImageOnlyTransform):
    """Box filter, kernel size odd ~ U{blur_limit} (>= 3)."""

    def __init__(self, blur_limit: int | Sequence[int] = (3, 7), p: float = 0.5):
        super().__init__(p)
        self.blur_limit = odd_blur_range(blur_limit, 3, "blur_limit")

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        return {"ksize": _sample_odd(rng, *self.blur_limit, batch.batch_size)}

    def apply(self, img, ksize, **params):
        return filter2d(img, box_kernels(ksize))


class MedianBlur(Blur):
    """Median filter, kernel size odd ~ U{blur_limit} (>= 3)."""

    def apply(self, img, ksize, **params):
        return median_filter(img, ksize)


class GaussianBlur(ImageOnlyTransform):
    """albumentations ``GaussianBlur``: sigma ~ U(sigma_limit), ksize ~ randint(blur_limit) (0 -> int(sigma*3.5)*2+1,
    even -> +1), separable kernel built like albumentations' ``create_gaussian_kernel_1d``."""

    def __init__(self, blur_limit: int | Sequence[int] = 0, sigma_limit: float | Sequence[float] = (0.5, 3.0), p: float = 0.5):
        super().__init__(p)
        self.blur_limit = check_nondecreasing(int_range(blur_limit, 0, "blur_limit"), "blur_limit")
        self.sigma_limit = check_nondecreasing(to_range(sigma_limit, 0.0, "sigma_limit"), "sigma_limit")

    def get_params(self, batch, rng):
        n = batch.batch_size
        sigma = rng.uniform(*self.sigma_limit, n).clamp(min=1e-6)
        ksize = rng.randint(self.blur_limit[0], self.blur_limit[1], n)
        return {"kernel": albu_gaussian_kernels_1d(sigma, ksize)}

    def apply(self, img, kernel, **params):
        return separable_filter(img, kernel, kernel)


class MotionBlur(ImageOnlyTransform):
    """albumentations ``MotionBlur``: a line of length ksize at angle ~ U(angle_range) degrees, biased by direction ~
    U(direction_range) in [-1, 1], optionally shifted (``allow_shifted``); normalized."""

    def __init__(
        self,
        blur_limit: int | Sequence[int] = (3, 7),
        allow_shifted: bool = True,
        angle_range: Sequence[float] = (0, 360),
        direction_range: Sequence[float] = (-1.0, 1.0),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.blur_limit = odd_blur_range(blur_limit, 3, "blur_limit")
        self.allow_shifted = allow_shifted
        self.angle_range = check_bounds((float(angle_range[0]), float(angle_range[1])), 0, 360, "angle_range")
        self.direction_range = check_bounds((float(direction_range[0]), float(direction_range[1])), -1, 1, "direction_range")

    def get_params(self, batch, rng):
        n = batch.batch_size
        ksize = _sample_odd(rng, *self.blur_limit, n)
        angle = rng.uniform(*self.angle_range, n)
        direction = rng.uniform(*self.direction_range, n)
        if self.allow_shifted:
            shift_x, shift_y = rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n)
        else:
            shift_x = shift_y = torch.zeros(n, device=batch.device)
        return {"kernel": motion_kernels(ksize, angle, direction, shift_x, shift_y)}

    def apply(self, img, kernel, **params):
        return filter2d(img, kernel)


class Defocus(ImageOnlyTransform):
    """albumentations ``Defocus``: disk of radius ~ randint(radius) on a (2*max(8, r)+1)^2 grid, smoothed by a Gaussian
    (3x3 if r <= 8 else 5x5, sigma ~ U(alias_blur), BORDER_REFLECT_101), normalized, then convolved."""

    def __init__(self, radius: int | Sequence[int] = (3, 10), alias_blur: float | Sequence[float] = (0.1, 0.5), p: float = 0.5):
        super().__init__(p)
        self.radius = check_nondecreasing(int_range(radius, 1, "radius"), "radius")
        self.alias_blur = check_nondecreasing(to_range(alias_blur, 0.0, "alias_blur"), "alias_blur")

    def get_params(self, batch, rng):
        n = batch.batch_size
        r = rng.randint(self.radius[0], self.radius[1], n)
        sigma = rng.uniform(*self.alias_blur, n)
        kernels = []
        for ri, si in zip(r.tolist(), sigma.tolist()):
            m = max(8, ri)
            ax = torch.arange(-m, m + 1, device=batch.device, dtype=torch.float32)
            disk = ((ax[None] ** 2 + ax[:, None] ** 2) <= ri * ri).float()
            disk = disk / disk.sum()
            k = 3 if ri <= 8 else 5
            g = cv2_gaussian_kernel_1d(k, si, batch.device)
            kernels.append(separable_filter(disk[None, None], g, g)[0, 0])
        size = max(k.shape[-1] for k in kernels)
        out = torch.stack([F.pad(k, [(size - k.shape[-1]) // 2] * 4) for k in kernels])
        return {"kernel": out}

    def apply(self, img, kernel, **params):
        return filter2d(img, kernel)


class ZoomBlur(ImageOnlyTransform):
    """albumentations ``ZoomBlur``: average of the image and center zooms by factors arange(1, max_factor, step).
    Each zoom reproduces albumentations' ``central_zoom`` exactly: center crop of ceil(size / z), bilinear resize to
    int(crop * z) (cv2.resize half-pixel mapping) and center trim back to the original size."""

    def __init__(self, max_factor: float | Sequence[float] = (1, 1.31), step_factor: float | Sequence[float] = (0.01, 0.03), p: float = 0.5):
        super().__init__(p)
        self.max_factor = check_nondecreasing(to_range(max_factor, 1.0, "max_factor"), "max_factor")
        self.step_factor = check_nondecreasing(to_range(step_factor, 0.0, "step_factor"), "step_factor")

    def get_params(self, batch, rng):
        n = batch.batch_size
        step = rng.uniform(*self.step_factor, n).double()
        mx = torch.maximum(1 + step, rng.uniform(*self.max_factor, n).double())
        count = torch.ceil((mx - 1.0) / step).long().clamp(min=1)  # len(np.arange(1, mx, step))
        return {"step": step, "count": count}

    def apply(self, img, step, count, **params):
        n, c, h, w = img.shape
        acc = torch.zeros_like(img)
        cmax = int(count.max())
        dev = img.device

        def axis(size: int, z: torch.Tensor) -> torch.Tensor:
            ch = torch.ceil(size / z)  # crop size
            start = torch.floor((size - ch) / 2)
            new = torch.floor(ch * z)  # resized crop size
            trim = torch.floor((new - size) / 2)
            d = torch.arange(size, device=dev, dtype=torch.float64)[None]
            src = (((d + trim[:, None]) + 0.5) * (ch / new)[:, None] - 0.5).clamp(min=0)
            return (torch.minimum(src, (ch - 1)[:, None]) + start[:, None]).float()

        for i in range(cmax):
            z = 1.0 + step * i  # float64 like np.arange
            active = (i < count).float().view(n, 1, 1, 1)
            sx = axis(w, z)[:, None, :].expand(n, h, w)
            sy = axis(h, z)[:, :, None].expand(n, h, w)
            acc = acc + active * fgeo.sample(img, sx, sy, 1, 1)
        return (img + acc) / (count.view(n, 1, 1, 1).float() + 1)
