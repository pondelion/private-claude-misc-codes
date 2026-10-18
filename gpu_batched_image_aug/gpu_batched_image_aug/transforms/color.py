"""Color / tone transforms (image only). Images are float [0, 1]; results are clipped to [0, 1] like
albumentations' uint8 saturation (albumentations itself does not always clip float images; clipping keeps the
documented [0, 1] invariant of this library)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch

from ..core.data import AugBatch
from ..core.random import Rng
from ..core.transform import ImageOnlyTransform
from ..core.utils import check_bounds, check_nondecreasing, symmetric_range, to_range
from ..functional import color as fcolor
from ..functional.filters import cv2_gaussian_kernel_1d, filter2d, separable_filter

__all__ = [
    "ChannelShuffle",
    "ColorJitter",
    "Emboss",
    "Equalize",
    "FancyPCA",
    "HueSaturationValue",
    "InvertImg",
    "Normalize",
    "Posterize",
    "RGBShift",
    "RandomBrightnessContrast",
    "RandomGamma",
    "RandomToneCurve",
    "Sharpen",
    "Solarize",
    "ToGray",
    "ToSepia",
    "UnsharpMask",
]


def _clip(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_(0.0, 1.0)


def _col(v: torch.Tensor) -> torch.Tensor:
    return v.view(-1, 1, 1, 1)


class RandomBrightnessContrast(ImageOnlyTransform):
    """albumentations ``RandomBrightnessContrast``: out = img * (1 + c) + b', c ~ U(contrast_limit), b ~
    U(brightness_limit), b' = b (brightness_by_max) or b * mean(img). ``ensure_safe_range`` is not supported."""

    def __init__(
        self,
        brightness_limit: float | Sequence[float] = (-0.2, 0.2),
        contrast_limit: float | Sequence[float] = (-0.2, 0.2),
        brightness_by_max: bool = True,
        ensure_safe_range: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p)
        if ensure_safe_range:
            raise NotImplementedError("ensure_safe_range is not supported")
        self.brightness_limit = symmetric_range(brightness_limit, "brightness_limit")
        self.contrast_limit = symmetric_range(contrast_limit, "contrast_limit")
        self.brightness_by_max = brightness_by_max
        self.ensure_safe_range = ensure_safe_range

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n = batch.batch_size
        return {"alpha": 1.0 + rng.uniform(*self.contrast_limit, n), "beta": rng.uniform(*self.brightness_limit, n)}

    def apply(self, img, alpha, beta, **params):
        b = _col(beta)
        if not self.brightness_by_max:
            b = b * img.mean(dim=(1, 2, 3), keepdim=True)
        return _clip(img * _col(alpha) + b)


def _cj_range(value, name: str, is_hue: bool) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"{name} must be non negative")
        if is_hue:
            return (-float(value), float(value))
        return (max(1 - float(value), 0.0), 1 + float(value))
    r = (float(value[0]), float(value[1]))
    return check_bounds(r, -0.5, 0.5, name) if is_hue else check_bounds(r, 0, float("inf"), name)


class ColorJitter(ImageOnlyTransform):
    """albumentations ``ColorJitter`` (torchvision semantics): brightness / contrast / saturation / hue applied in a
    random order per sample. A scalar b means (max(0, 1-b), 1+b) (hue: (-h, h), h <= 0.5)."""

    def __init__(
        self,
        brightness: float | Sequence[float] = (0.8, 1.2),
        contrast: float | Sequence[float] = (0.8, 1.2),
        saturation: float | Sequence[float] = (0.8, 1.2),
        hue: float | Sequence[float] = (-0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.brightness = _cj_range(brightness, "brightness", False)
        self.contrast = _cj_range(contrast, "contrast", False)
        self.saturation = _cj_range(saturation, "saturation", False)
        self.hue = _cj_range(hue, "hue", True)

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n = batch.batch_size
        return {
            "brightness": rng.uniform(*self.brightness, n),
            "contrast": rng.uniform(*self.contrast, n),
            "saturation": rng.uniform(*self.saturation, n),
            "hue": rng.uniform(*self.hue, n),
            "order": rng.permutation(n, 4),
        }

    @staticmethod
    def _op(img: torch.Tensor, i: int, f: torch.Tensor) -> torch.Tensor:
        gray_img = img.shape[1] == 1
        if i == 0:  # brightness
            return _clip(img * _col(f))
        if i == 1:  # contrast: blend with the mean of the grayscale image
            mean = (img if gray_img else fcolor.rgb_to_gray(img)).mean(dim=(1, 2, 3), keepdim=True)
            return _clip(img * _col(f) + mean * (1 - _col(f)))
        if gray_img:
            return img
        if i == 2:  # saturation: blend with grayscale
            gray = fcolor.rgb_to_gray(img)
            return _clip(img * _col(f) + gray * (1 - _col(f)))
        hsv = fcolor.rgb_to_hsv(img)  # hue
        hsv[:, 0] = torch.remainder(hsv[:, 0] + f.view(-1, 1, 1) * 360.0, 360.0)
        return _clip(fcolor.hsv_to_rgb(hsv))

    def apply(self, img, brightness, contrast, saturation, hue, order, **params):
        factors = [brightness, contrast, saturation, hue]
        for pos in range(4):
            for op in range(4):
                sel = order[:, pos] == op
                if bool(sel.all()):
                    img = self._op(img, op, factors[op])
                elif bool(sel.any()):
                    img[sel] = self._op(img[sel], op, factors[op][sel])
        return img


class HueSaturationValue(ImageOnlyTransform):
    """albumentations ``HueSaturationValue`` with its (dtype independent) units: hue shift in OpenCV uint8 hue units
    (1 unit = 2 degrees), saturation / value shifts in uint8 units (1 unit = 1/255). Pixels with zero saturation keep
    zero saturation. Grayscale images only get the value shift."""

    def __init__(
        self,
        hue_shift_limit: float | Sequence[float] = (-20, 20),
        sat_shift_limit: float | Sequence[float] = (-30, 30),
        val_shift_limit: float | Sequence[float] = (-20, 20),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.hue_shift_limit = symmetric_range(hue_shift_limit, "hue_shift_limit")
        self.sat_shift_limit = symmetric_range(sat_shift_limit, "sat_shift_limit")
        self.val_shift_limit = symmetric_range(val_shift_limit, "val_shift_limit")

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n = batch.batch_size
        return {
            "hue": rng.uniform(*self.hue_shift_limit, n),
            "sat": rng.uniform(*self.sat_shift_limit, n),
            "val": rng.uniform(*self.val_shift_limit, n),
        }

    def apply(self, img, hue, sat, val, **params):
        if img.shape[1] == 1:
            return _clip(img + _col(val) / 255.0)
        hsv = fcolor.rgb_to_hsv(img)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        h = torch.remainder(h + hue.view(-1, 1, 1) * 2.0, 360.0)
        s = torch.where(s == 0, s, (s + sat.view(-1, 1, 1) / 255.0).clamp(0, 1))
        v = (v + val.view(-1, 1, 1) / 255.0).clamp(0, 1)
        return _clip(fcolor.hsv_to_rgb(torch.stack([h, s, v], dim=1)))


class RGBShift(ImageOnlyTransform):
    """albumentations ``RGBShift``: per-channel constant shift ~ U(limit). Limits with |value| > 1 are in uint8 units
    (divided by 255, as albumentations does), otherwise fractions of the max value."""

    def __init__(
        self,
        r_shift_limit: float | Sequence[float] = (-20, 20),
        g_shift_limit: float | Sequence[float] = (-20, 20),
        b_shift_limit: float | Sequence[float] = (-20, 20),
        p: float = 0.5,
    ):
        super().__init__(p)

        def norm(limit):
            r = symmetric_range(limit, "shift_limit")
            return (r[0] / 255.0, r[1] / 255.0) if abs(r[0]) > 1 or abs(r[1]) > 1 else r

        self.r_shift_limit = symmetric_range(r_shift_limit, "r_shift_limit")
        self.g_shift_limit = symmetric_range(g_shift_limit, "g_shift_limit")
        self.b_shift_limit = symmetric_range(b_shift_limit, "b_shift_limit")
        self._ranges = [norm(r_shift_limit), norm(g_shift_limit), norm(b_shift_limit)]

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, c = batch.image.shape[:2]
        if c != 3:
            raise ValueError("RGBShift expects 3-channel images")
        shifts = torch.stack([rng.uniform(lo, hi, n) for lo, hi in self._ranges], dim=1)
        return {"shift": shifts}

    def apply(self, img, shift, **params):
        return _clip(img + shift[:, :, None, None])


class RandomGamma(ImageOnlyTransform):
    """albumentations ``RandomGamma``: img ** (U(gamma_limit) / 100)."""

    def __init__(self, gamma_limit: float | Sequence[float] = (80, 120), p: float = 0.5):
        super().__init__(p)
        self.gamma_limit = check_nondecreasing(to_range(gamma_limit, 1.0, "gamma_limit"), "gamma_limit")

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        return {"gamma": rng.uniform(*self.gamma_limit, batch.batch_size) / 100.0}

    def apply(self, img, gamma, **params):
        return torch.pow(img.clamp(min=0), _col(gamma))


class ChannelShuffle(ImageOnlyTransform):
    """Random permutation of the channels per sample."""

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        return {"perm": rng.permutation(batch.batch_size, batch.image.shape[1])}

    def apply(self, img, perm, **params):
        n, c, h, w = img.shape
        return torch.gather(img, 1, perm[:, :, None, None].expand(n, c, h, w))


class ToGray(ImageOnlyTransform):
    """albumentations ``ToGray`` for RGB images. Methods: weighted_average (cv2 weights), desaturation
    ((max + min) / 2), average, max, from_lab (L* / 100, as for uint8 images). "pca" is not supported."""

    def __init__(
        self,
        num_output_channels: int = 3,
        method: Literal["weighted_average", "from_lab", "desaturation", "average", "max"] = "weighted_average",
        p: float = 0.5,
    ):
        super().__init__(p)
        if method not in ("weighted_average", "from_lab", "desaturation", "average", "max"):
            raise NotImplementedError(f"method={method!r} is not supported")
        self.num_output_channels = int(num_output_channels)
        self.method = method

    def apply_batch(self, batch, params):
        if self.num_output_channels != batch.image.shape[1]:
            raise ValueError("num_output_channels must equal the number of image channels (a batch has one channel count)")
        return super().apply_batch(batch, params)

    def apply(self, img, **params):
        if img.shape[1] == 1:
            return img
        if self.method == "weighted_average":
            g = fcolor.rgb_to_gray(img)
        elif self.method == "desaturation":
            g = (img.amax(1, keepdim=True) + img.amin(1, keepdim=True)) / 2
        elif self.method == "average":
            g = img.mean(1, keepdim=True)
        elif self.method == "max":
            g = img.amax(1, keepdim=True)
        else:
            g = _lab_lightness(img) / 100.0
        return g.expand(-1, self.num_output_channels, -1, -1).contiguous()


def _lab_lightness(img: torch.Tensor) -> torch.Tensor:
    """CIE L* (0..100) of sRGB-gamma RGB as in cv2.COLOR_RGB2LAB (D65)."""

    def lin(c):
        return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    r, g, b = lin(img[:, 0:1]), lin(img[:, 1:2]), lin(img[:, 2:3])
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    return torch.where(y > 0.008856, 116.0 * y.clamp(min=0) ** (1 / 3) - 16.0, 903.3 * y)


class ToSepia(ImageOnlyTransform):
    _M = ((0.393, 0.769, 0.189), (0.349, 0.686, 0.168), (0.272, 0.534, 0.131))

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, img, **params):
        if img.shape[1] == 1:
            return img
        m = torch.tensor(self._M, dtype=img.dtype, device=img.device)
        return _clip(torch.einsum("oc,nchw->nohw", m, img))


class InvertImg(ImageOnlyTransform):
    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, img, **params):
        return 1.0 - img


class Solarize(ImageOnlyTransform):
    """Invert pixels >= threshold, threshold ~ U(threshold_range) (fraction of the max value)."""

    def __init__(self, threshold_range: Sequence[float] = (0.5, 0.5), p: float = 0.5):
        super().__init__(p)
        self.threshold_range = check_bounds(check_nondecreasing((float(threshold_range[0]), float(threshold_range[1])), "threshold_range"), 0, 1, "threshold_range")

    def get_params(self, batch, rng):
        return {"threshold": rng.uniform(*self.threshold_range, batch.batch_size)}

    def apply(self, img, threshold, **params):
        return torch.where(img >= _col(threshold), 1.0 - img, img)


class Posterize(ImageOnlyTransform):
    """Keep ``num_bits`` most significant bits of the uint8 levels. num_bits: int, (min, max) or a list of per-channel
    (min, max) ranges."""

    def __init__(self, num_bits: int | Sequence[int] | Sequence[Sequence[int]] = 4, p: float = 0.5):
        super().__init__(p)
        if isinstance(num_bits, int):
            self.num_bits: Any = (num_bits, num_bits)
        elif len(num_bits) > 2:
            self.num_bits = [(int(b), int(b)) if isinstance(b, int) else (int(b[0]), int(b[1])) for b in num_bits]
        else:
            self.num_bits = (int(num_bits[0]), int(num_bits[1]))

    def get_params(self, batch, rng):
        n, c = batch.image.shape[:2]
        if isinstance(self.num_bits, list):
            if len(self.num_bits) != c:
                raise ValueError("num_bits list length must match the number of channels")
            bits = torch.stack([rng.randint(lo, hi, n) for lo, hi in self.num_bits], dim=1)
        else:
            bits = rng.randint(self.num_bits[0], self.num_bits[1], n)[:, None].expand(n, c)
        return {"bits": bits}

    def apply(self, img, bits, **params):
        levels = fcolor.to_uint8_levels(img)
        mask = (255 - ((1 << (8 - bits)) - 1))[:, :, None, None]
        return (levels & mask).float() / 255.0


class Equalize(ImageOnlyTransform):
    """cv2.equalizeHist on the uint8 levels of each channel (``by_channels``) or of the Y channel of YCrCb.
    Only mode="cv" without mask is supported."""

    def __init__(self, mode: Literal["cv", "pil"] = "cv", by_channels: bool = True, p: float = 0.5):
        super().__init__(p)
        if mode != "cv":
            raise NotImplementedError("Only mode='cv' is supported")
        self.mode = mode
        self.by_channels = by_channels

    @staticmethod
    def _equalize(levels: torch.Tensor) -> torch.Tensor:
        """levels (n, C, H, W) long -> equalized float [0, 1]"""
        n, c, h, w = levels.shape
        flat = levels.reshape(n * c, h * w)
        hist = torch.zeros(n * c, 256, dtype=torch.int64, device=levels.device)
        hist.scatter_add_(1, flat, torch.ones_like(flat))
        i0 = torch.argmax((hist > 0).int(), dim=1)
        total = hist.sum(1)
        h_i0 = hist.gather(1, i0[:, None]).squeeze(1)
        cdf = hist.cumsum(1)
        # float32 arithmetic and round-half-to-even exactly like cv2.equalizeHist (sum * scale, cvRound)
        scale = (255.0 / (total - h_i0).clamp(min=1).float())
        lut = ((cdf - cdf.gather(1, i0[:, None])).float() * scale[:, None]).round().clamp(0, 255)
        single = (total - h_i0) == 0  # one gray level only: cv2 keeps the image
        lut = torch.where(single[:, None], torch.arange(256, device=levels.device, dtype=lut.dtype).expand_as(lut), lut)
        out = torch.gather(lut, 1, flat).view(n, c, h, w)
        return (out / 255.0).float()

    def apply(self, img, **params):
        if img.shape[1] == 1 or self.by_channels:
            return self._equalize(fcolor.to_uint8_levels(img))
        # uint8 pipeline like albumentations: exact cv2 integer YCrCb conversion, equalize Y, convert back
        ycc = fcolor.rgb_to_ycrcb_u8(fcolor.to_uint8_levels(img))
        y_eq = (self._equalize(ycc[:, 0:1]) * 255).round().long()
        ycc = torch.cat([y_eq, ycc[:, 1:]], dim=1)
        return fcolor.ycrcb_to_rgb_u8(ycc).float() / 255.0


class FancyPCA(ImageOnlyTransform):
    """albumentations ``FancyPCA``: per image, eigen decomposition of the channel covariance (eigenvalues
    descending), alpha_i ~ N(0, alpha), and per pixel noise = V @ diag(alpha_i * lambda_i) @ (x - mean) (albumentations'
    formulation), clipped to [0, 1]."""

    def __init__(self, alpha: float = 0.1, p: float = 0.5):
        super().__init__(p)
        self.alpha = float(alpha)

    def get_params(self, batch, rng):
        return {"alpha_vector": rng.normal(0.0, self.alpha, batch.image.shape[:2])}

    def apply(self, img, alpha_vector, **params):
        n, c, h, w = img.shape
        x = img.reshape(n, c, h * w)
        centered = x - x.mean(dim=2, keepdim=True)
        if c == 1:
            std = centered.std(dim=2, unbiased=False, keepdim=True)
            noise = alpha_vector[:, :, None] * std * centered
        else:
            cov = centered.double() @ centered.double().transpose(1, 2) / (h * w - 1)
            # albumentations' formula depends on the eigenvector signs, which differ between LAPACK (numpy, torch CPU)
            # and cuSOLVER; the tiny CxC problems are solved on the CPU to match numpy exactly
            evals, evecs = torch.linalg.eigh(cov.cpu())
            evals, evecs = evals.flip(-1).float().to(img.device), evecs.flip(-1).float().to(img.device)  # descending
            m = evecs @ torch.diag_embed(alpha_vector * evals)
            noise = m @ centered
        return _clip((x + noise).view(n, c, h, w))


class RandomToneCurve(ImageOnlyTransform):
    """albumentations ``RandomToneCurve``: cubic Bezier tone curve through (0,0),(0.25,low_y),(0.75,high_y),(1,1),
    low_y ~ clip(N(0.25, scale)), high_y ~ clip(N(0.75, scale)) (per channel with ``per_channel``), applied as a
    uint8 LUT."""

    def __init__(self, scale: float = 0.1, per_channel: bool = False, p: float = 0.5):
        super().__init__(p)
        self.scale = float(scale)
        self.per_channel = per_channel

    def get_params(self, batch, rng):
        n, c = batch.image.shape[:2]
        k = c if self.per_channel and c != 1 else 1
        return {
            "low_y": rng.normal(0.25, self.scale, (n, k)).clamp(0, 1),
            "high_y": rng.normal(0.75, self.scale, (n, k)).clamp(0, 1),
        }

    def apply(self, img, low_y, high_y, **params):
        t = torch.linspace(0.0, 1.0, 256, device=img.device, dtype=torch.float64)[None, None]
        lo, hi = low_y.double()[..., None], high_y.double()[..., None]
        omt = 1 - t
        lut = torch.round((3 * omt**2 * t * lo + 3 * omt * t**2 * hi + t**3) * 255).clamp(0, 255) / 255.0
        return fcolor.apply_lut(fcolor.to_uint8_levels(img), lut.float())


class Emboss(ImageOnlyTransform):
    """albumentations ``Emboss``: 3x3 kernel (1-alpha) * identity + alpha * emboss(strength)."""

    def __init__(self, alpha: Sequence[float] = (0.2, 0.5), strength: Sequence[float] = (0.2, 0.7), p: float = 0.5):
        super().__init__(p)
        self.alpha = (float(alpha[0]), float(alpha[1]))
        self.strength = (float(strength[0]), float(strength[1]))

    def get_params(self, batch, rng):
        n = batch.batch_size
        a = rng.uniform(*self.alpha, n).view(n, 1, 1)
        s = rng.uniform(*self.strength, n).view(n, 1, 1)
        dev = batch.device
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32, device=dev)
        base = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=dev)
        sgn = torch.tensor([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=torch.float32, device=dev)
        effect = base + s * sgn
        return {"kernel": (1 - a) * ident + a * effect}

    def apply(self, img, kernel, **params):
        return _clip(filter2d(img, kernel))


class Sharpen(ImageOnlyTransform):
    """albumentations ``Sharpen``. method="kernel": 3x3 (1-alpha) * identity + alpha * [[-1,-1,-1],[-1,8+l,-1],
    [-1,-1,-1]]; method="gaussian": img + alpha * (img - GaussianBlur(img, kernel_size, sigma))."""

    def __init__(
        self,
        alpha: Sequence[float] = (0.2, 0.5),
        lightness: Sequence[float] = (0.5, 1.0),
        method: Literal["kernel", "gaussian"] = "kernel",
        kernel_size: int = 5,
        sigma: float = 1.0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.alpha = (float(alpha[0]), float(alpha[1]))
        self.lightness = (float(lightness[0]), float(lightness[1]))
        self.method = method
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)

    def get_params(self, batch, rng):
        n = batch.batch_size
        a = rng.uniform(*self.alpha, n)
        if self.method == "gaussian":
            return {"alpha": a}
        light = rng.uniform(*self.lightness, n).view(n, 1, 1)
        dev = batch.device
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32, device=dev)
        effect = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=dev).expand(n, 3, 3).clone()
        effect[:, 1, 1] += light[:, 0, 0]
        a3 = a.view(n, 1, 1)
        return {"alpha": a, "kernel": (1 - a3) * ident + a3 * effect}

    def apply(self, img, alpha, kernel=None, **params):
        if self.method == "kernel":
            return _clip(filter2d(img, kernel))
        k = cv2_gaussian_kernel_1d(self.kernel_size, self.sigma, img.device)
        blurred = separable_filter(img, k, k)
        return _clip(img + _col(alpha) * (img - blurred))


class UnsharpMask(ImageOnlyTransform):
    """albumentations ``UnsharpMask``: ksize odd ~ U{blur_limit}, sigma ~ U(sigma_limit) (0 -> derived from ksize as in
    cv2), alpha ~ U(alpha); residual = img - blur; sharpened only where |residual| * 255 > threshold (soft mask)."""

    def __init__(
        self,
        blur_limit: int | Sequence[int] = (3, 7),
        sigma_limit: float | Sequence[float] = 0.0,
        alpha: float | Sequence[float] = (0.2, 0.5),
        threshold: int = 10,
        p: float = 0.5,
    ):
        super().__init__(p)
        from ..core.utils import odd_blur_range

        self.blur_limit = odd_blur_range(blur_limit, 3, "blur_limit")
        self.sigma_limit = to_range(sigma_limit, 0.0, "sigma_limit")
        self.alpha = to_range(alpha, 0.0, "alpha") if isinstance(alpha, (int, float)) else (float(alpha[0]), float(alpha[1]))
        self.threshold = int(threshold)

    def get_params(self, batch, rng):
        n = batch.batch_size
        num = (self.blur_limit[1] - self.blur_limit[0]) // 2 + 1
        ksize = self.blur_limit[0] + 2 * rng.randint(0, num - 1, n)
        return {"ksize": ksize, "sigma": rng.uniform(*self.sigma_limit, n), "alpha": rng.uniform(*self.alpha, n)}

    def apply(self, img, ksize, sigma, alpha, **params):
        from ..functional.filters import cv2_gaussian_kernels_1d

        k = cv2_gaussian_kernels_1d(ksize, sigma)
        blur = separable_filter(img, k, k)
        residual = img - blur
        mask = (residual.abs() * 255 > self.threshold).float()
        sharp = (img + _col(alpha) * residual).clamp(0, 1)
        soft = separable_filter(mask, k, k)
        return _clip(sharp * soft + img * (1 - soft))


class Normalize(ImageOnlyTransform):
    """(img - mean) / std with mean/std as fractions of the max value (albumentations defaults: ImageNet).
    Output is no longer in [0, 1]: put it last, and use float input/output (uint8 output would be clipped)."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        p: float = 1.0,
    ):
        super().__init__(p)
        self.mean = tuple(float(m) for m in mean)
        self.std = tuple(float(s) for s in std)

    def apply(self, img, **params):
        m = torch.tensor(self.mean, dtype=img.dtype, device=img.device).view(1, -1, 1, 1)
        s = torch.tensor(self.std, dtype=img.dtype, device=img.device).view(1, -1, 1, 1)
        return (img - m) / s
