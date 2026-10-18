"""Batched color-space conversions with OpenCV float semantics.

Images are (n, 3, H, W) float RGB in [0, 1].
- HSV: H in degrees [0, 360), S, V in [0, 1]  (cv2.COLOR_RGB2HSV on float32)
- HLS: H in degrees [0, 360), L, S in [0, 1]  (cv2.COLOR_RGB2HLS on float32)
- gray: 0.299 R + 0.587 G + 0.114 B (cv2.COLOR_RGB2GRAY)
"""

from __future__ import annotations

import torch

GRAY_WEIGHTS = (0.299, 0.587, 0.114)


def rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    """(n, 3, H, W) -> (n, 1, H, W)."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return GRAY_WEIGHTS[0] * r + GRAY_WEIGHTS[1] * g + GRAY_WEIGHTS[2] * b


def rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    v, _ = img.max(dim=1)
    mn, _ = img.min(dim=1)
    diff = v - mn
    s = torch.where(v > 0, diff / v.clamp(min=1e-12), torch.zeros_like(v))
    safe = diff.clamp(min=1e-12)
    h = torch.where(
        v == r,
        60.0 * (g - b) / safe,
        torch.where(v == g, 120.0 + 60.0 * (b - r) / safe, 240.0 + 60.0 * (r - g) / safe),
    )
    h = torch.where(diff > 0, h, torch.zeros_like(h))
    h = torch.where(h < 0, h + 360.0, h)
    return torch.stack([h, s, v], dim=1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    h = torch.remainder(h, 360.0) / 60.0
    i = torch.floor(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], dim=1).gather(1, i[:, None]).squeeze(1)
    g = torch.stack([t, v, v, q, p, p], dim=1).gather(1, i[:, None]).squeeze(1)
    b = torch.stack([p, p, t, v, v, q], dim=1).gather(1, i[:, None]).squeeze(1)
    return torch.stack([r, g, b], dim=1)


def rgb_to_hls(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    vmax, _ = img.max(dim=1)
    vmin, _ = img.min(dim=1)
    diff = vmax - vmin
    l = (vmax + vmin) / 2
    denom = torch.where(l < 0.5, vmax + vmin, 2.0 - vmax - vmin).clamp(min=1e-12)
    s = torch.where(diff > 0, diff / denom, torch.zeros_like(l))
    safe = diff.clamp(min=1e-12)
    h = torch.where(
        vmax == r,
        60.0 * (g - b) / safe,
        torch.where(vmax == g, 120.0 + 60.0 * (b - r) / safe, 240.0 + 60.0 * (r - g) / safe),
    )
    h = torch.where(diff > 0, h, torch.zeros_like(h))
    h = torch.where(h < 0, h + 360.0, h)
    return torch.stack([h, l, s], dim=1)


def hls_to_rgb(hls: torch.Tensor) -> torch.Tensor:
    h, l, s = hls[:, 0], hls[:, 1], hls[:, 2]
    q = torch.where(l <= 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q
    hk = torch.remainder(h, 360.0) / 360.0

    def channel(t: torch.Tensor) -> torch.Tensor:
        t = torch.remainder(t, 1.0)
        return torch.where(
            t < 1 / 6, p + (q - p) * 6 * t, torch.where(t < 0.5, q, torch.where(t < 2 / 3, p + (q - p) * (2 / 3 - t) * 6, p))
        )

    r = channel(hk + 1 / 3)
    g = channel(hk)
    b = channel(hk - 1 / 3)
    gray = s == 0
    r, g, b = (torch.where(gray, l, c) for c in (r, g, b))
    return torch.stack([r, g, b], dim=1)


def rgb_to_ycrcb(img: torch.Tensor) -> torch.Tensor:
    """cv2.COLOR_RGB2YCrCb (float, delta = 0.5)."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return torch.stack([y, cr, cb], dim=1)


def ycrcb_to_rgb(ycc: torch.Tensor) -> torch.Tensor:
    y, cr, cb = ycc[:, 0], ycc[:, 1] - 0.5, ycc[:, 2] - 0.5
    r = y + 1.403 * cr
    g = y - 0.714 * cr - 0.344 * cb
    b = y + 1.773 * cb
    return torch.stack([r, g, b], dim=1)


def to_uint8_levels(img: torch.Tensor) -> torch.Tensor:
    """Quantize [0, 1] floats to the 256 uint8 levels (as integer tensor)."""
    return (img.clamp(0, 1) * 255.0).round().long()


def apply_lut(levels: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    """levels: (n, C, H, W) long in [0, 255]; luts: (n, C, 256) or (n, 1, 256) float -> (n, C, H, W) float."""
    n, c, h, w = levels.shape
    if luts.shape[1] == 1 and c > 1:
        luts = luts.expand(n, c, 256)
    return torch.gather(luts, 2, levels.reshape(n, c, h * w)).view(n, c, h, w)


_SHIFT = 14
_HALF = 1 << (_SHIFT - 1)


def rgb_to_ycrcb_u8(levels: torch.Tensor) -> torch.Tensor:
    """Exact cv2.COLOR_RGB2YCrCb for uint8 data (14-bit fixed point). levels (n, 3, H, W) long -> long."""
    r, g, b = levels[:, 0], levels[:, 1], levels[:, 2]
    y = (r * 4899 + g * 9617 + b * 1868 + _HALF) >> _SHIFT
    cr = ((r - y) * 11682 + (128 << _SHIFT) + _HALF) >> _SHIFT
    cb = ((b - y) * 9241 + (128 << _SHIFT) + _HALF) >> _SHIFT
    return torch.stack([y, cr, cb], dim=1).clamp(0, 255)


def ycrcb_to_rgb_u8(levels: torch.Tensor) -> torch.Tensor:
    """Exact cv2.COLOR_YCrCb2RGB for uint8 data (14-bit fixed point). levels (n, 3, H, W) long -> long."""
    y, cr, cb = levels[:, 0], levels[:, 1] - 128, levels[:, 2] - 128
    r = y + ((cr * 22987 + _HALF) >> _SHIFT)
    g = y + ((cb * -5636 + cr * -11698 + _HALF) >> _SHIFT)
    b = y + ((cb * 29049 + _HALF) >> _SHIFT)
    return torch.stack([r, g, b], dim=1).clamp(0, 255)
