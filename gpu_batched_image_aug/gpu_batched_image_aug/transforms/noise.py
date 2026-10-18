"""Noise transforms (image only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..core.data import AugBatch
from ..core.random import Rng
from ..core.transform import ImageOnlyTransform
from ..core.utils import check_bounds, check_nondecreasing
from ..functional import color as fcolor

__all__ = ["GaussNoise", "ISONoise", "MultiplicativeNoise", "SaltAndPepper", "ShotNoise"]


def _col(v: torch.Tensor) -> torch.Tensor:
    return v.view(-1, 1, 1, 1)


class GaussNoise(ImageOnlyTransform):
    """albumentations ``GaussNoise``: per image sigma ~ U(std_range), mean ~ U(mean_range) (both as fractions of the
    max value); noise independent per pixel and channel (``per_channel``) or shared across channels.
    ``noise_scale_factor`` < 1 generates the noise at reduced resolution and upsamples it (nearest)."""

    def __init__(
        self,
        std_range: Sequence[float] = (0.2, 0.44),
        mean_range: Sequence[float] = (0.0, 0.0),
        per_channel: bool = True,
        noise_scale_factor: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.std_range = check_bounds(check_nondecreasing((float(std_range[0]), float(std_range[1])), "std_range"), 0, 1, "std_range")
        self.mean_range = check_bounds(check_nondecreasing((float(mean_range[0]), float(mean_range[1])), "mean_range"), -1, 1, "mean_range")
        self.per_channel = per_channel
        if not 0 < noise_scale_factor <= 1:
            raise ValueError("noise_scale_factor must be in (0, 1]")
        self.noise_scale_factor = float(noise_scale_factor)

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, c, h, w = batch.image.shape
        sigma = rng.uniform(*self.std_range, n)
        mean = rng.uniform(*self.mean_range, n)
        nc = c if self.per_channel else 1
        if self.noise_scale_factor < 1:
            sh, sw = max(1, int(h * self.noise_scale_factor)), max(1, int(w * self.noise_scale_factor))
            noise = rng.randn((n, nc, sh, sw))
            noise = torch.nn.functional.interpolate(noise, size=(h, w), mode="nearest")
        else:
            noise = rng.randn((n, nc, h, w))
        return {"noise": noise * _col(sigma) + _col(mean)}

    def apply(self, img, noise, **params):
        return (img + noise).clamp_(0, 1)


class MultiplicativeNoise(ImageOnlyTransform):
    """albumentations ``MultiplicativeNoise``: multiplier ~ U(multiplier), one per image, per channel
    (``per_channel``), per pixel (``elementwise``) or per pixel and channel (both)."""

    def __init__(
        self,
        multiplier: float | Sequence[float] = (0.9, 1.1),
        per_channel: bool = False,
        elementwise: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p)
        m = (float(multiplier), float(multiplier)) if isinstance(multiplier, (int, float)) else (float(multiplier[0]), float(multiplier[1]))
        self.multiplier = check_nondecreasing(m, "multiplier")
        self.per_channel = per_channel
        self.elementwise = elementwise

    def get_params(self, batch, rng):
        n, c, h, w = batch.image.shape
        cc = c if self.per_channel else 1
        shape = (n, cc, h, w) if self.elementwise else (n, cc, 1, 1)
        return {"mult": rng.uniform(*self.multiplier, shape)}

    def apply(self, img, mult, **params):
        return (img * mult).clamp_(0, 1)


class ISONoise(ImageOnlyTransform):
    """albumentations ``ISONoise`` (RGB): in HLS space, H += N(0, color_shift * intensity) (degrees, as in
    albumentations' float path) and L += Poisson(std(L) * intensity) * intensity * (1 - L)."""

    def __init__(self, color_shift: Sequence[float] = (0.01, 0.05), intensity: Sequence[float] = (0.1, 0.5), p: float = 0.5):
        super().__init__(p)
        self.color_shift = (float(color_shift[0]), float(color_shift[1]))
        self.intensity = (float(intensity[0]), float(intensity[1]))

    def get_params(self, batch, rng):
        n, c, h, w = batch.image.shape
        if c != 3:
            raise ValueError("ISONoise expects RGB images")
        cs = rng.uniform(*self.color_shift, n)
        it = rng.uniform(*self.intensity, n)
        hls = fcolor.rgb_to_hls(batch.image)
        std_l = hls[:, 1].flatten(1).std(dim=1, unbiased=False)
        lum = rng.poisson((std_l * it).view(n, 1, 1).expand(n, h, w).contiguous())
        color = rng.randn((n, h, w)) * (cs * it).view(n, 1, 1)
        return {"lum_noise": lum, "color_noise": color, "intensity": it}

    def apply(self, img, lum_noise, color_noise, intensity, **params):
        hls = fcolor.rgb_to_hls(img)
        hls[:, 0] = hls[:, 0] + color_noise
        hls[:, 1] = (hls[:, 1] + lum_noise * intensity.view(-1, 1, 1) * (1.0 - hls[:, 1])).clamp(0, 1)
        return fcolor.hls_to_rgb(hls).clamp_(0, 1)


class SaltAndPepper(ImageOnlyTransform):
    """albumentations ``SaltAndPepper``: exactly int(H*W*amount) pixels (without replacement), a fraction
    salt_vs_pepper of them set to max, the rest to 0 (all channels)."""

    def __init__(self, amount: Sequence[float] = (0.01, 0.06), salt_vs_pepper: Sequence[float] = (0.4, 0.6), p: float = 0.5):
        super().__init__(p)
        self.amount = (float(amount[0]), float(amount[1]))
        self.salt_vs_pepper = (float(salt_vs_pepper[0]), float(salt_vs_pepper[1]))

    def get_params(self, batch, rng):
        n, c, h, w = batch.image.shape
        area = h * w
        amount = rng.uniform(*self.amount, n)
        ratio = rng.uniform(*self.salt_vs_pepper, n)
        num = (area * amount).long()
        num_salt = (num * ratio).long()
        keys = rng.rand((n, area))
        sorted_keys, _ = torch.sort(keys, dim=1)
        # thresholds such that exactly `num` (resp. `num_salt`) keys are <= threshold
        thr_all = torch.where(num > 0, sorted_keys.gather(1, (num - 1).clamp(min=0)[:, None]).squeeze(1), torch.full_like(amount, -1.0))
        thr_salt = torch.where(num_salt > 0, sorted_keys.gather(1, (num_salt - 1).clamp(min=0)[:, None]).squeeze(1), torch.full_like(amount, -1.0))
        selected = keys <= thr_all[:, None]
        salt = keys <= thr_salt[:, None]
        return {"salt": salt.view(n, 1, h, w), "pepper": (selected & ~salt).view(n, 1, h, w)}

    def apply(self, img, salt, pepper, **params):
        return torch.where(salt, torch.ones_like(img), torch.where(pepper, torch.zeros_like(img), img))


class ShotNoise(ImageOnlyTransform):
    """albumentations ``ShotNoise``: Poisson noise in linear light: x = img^2.2, out = (Poisson((x + s*1e-6) / s) * s)
    clipped, then ^(1/2.2); scale s ~ U(scale_range)."""

    def __init__(self, scale_range: Sequence[float] = (0.1, 0.3), p: float = 0.5):
        super().__init__(p)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))

    def get_params(self, batch, rng):
        n = batch.batch_size
        s = rng.uniform(*self.scale_range, n)
        lin = torch.pow(batch.image.clamp(min=0), 2.2)
        rate = (lin + _col(s) * 1e-6) / _col(s)
        return {"noisy": (rng.poisson(rate) * _col(s)).clamp(0, 1)}

    def apply(self, img, noisy, **params):
        return torch.pow(noisy, 1 / 2.2)
