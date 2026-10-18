"""Shared test helpers. Tests compare against albumentations / OpenCV, so both must be installed."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEVICE = torch.device(os.environ.get("GBIA_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def rng_np(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def natural_image(h: int = 96, w: int = 128, seed: int = 0) -> np.ndarray:
    """Smooth random RGB uint8 image (HWC) with edges, closer to real images than white noise."""
    g = rng_np(seed)
    small = g.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3)).astype(np.float32)
    import cv2

    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img[h // 3 : h // 3 + 10, :, :] = 255
    img[:, w // 2 : w // 2 + 6, 1] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


def to_t(img_hwc: np.ndarray) -> torch.Tensor:
    """HWC (uint8 or float) -> (1, C, H, W) float32 [0, 1] on DEVICE."""
    x = torch.from_numpy(np.ascontiguousarray(img_hwc)).to(DEVICE)
    if x.ndim == 2:
        x = x[..., None]
    x = x.permute(2, 0, 1)[None].float()
    return x / 255.0 if img_hwc.dtype == np.uint8 else x


def to_np(t: torch.Tensor, as_uint8: bool = True) -> np.ndarray:
    """(1, C, H, W) or (C, H, W) [0, 1] -> HWC numpy (uint8 rounded or float32)."""
    if t.ndim == 4:
        t = t[0]
    x = t.permute(1, 2, 0).detach().float().cpu().numpy()
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8) if as_uint8 else x


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if m == 0 else float(10 * np.log10(255.0**2 / m))


def interior(a: np.ndarray, margin: int = 2) -> np.ndarray:
    return a[margin:-margin, margin:-margin]
