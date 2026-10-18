"""Compression / resolution degradation (image only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F

from ..core.constants import INTER_AREA, INTER_CUBIC, INTER_NEAREST, resolve_interpolation
from ..core.data import AugBatch
from ..core.random import Rng
from ..core.transform import ImageOnlyTransform
from ..core.utils import check_bounds, check_nondecreasing
from ..functional import geometric as fgeo
from ..functional.jpeg import jpeg

__all__ = ["Downscale", "ImageCompression"]


class ImageCompression(ImageOnlyTransform):
    """JPEG compression artifacts, quality ~ randint(quality_range) per sample (simulated, see functional.jpeg).
    ``treat_as_bgr=True`` reproduces albumentations, which feeds RGB arrays to cv2.imencode (BGR) so the chroma
    channels are computed from swapped red/blue; set False for physically correct JPEG of RGB input.
    compression_type="webp" is not supported."""

    def __init__(
        self,
        compression_type: Literal["jpeg", "webp"] = "jpeg",
        quality_range: Sequence[int] = (99, 100),
        treat_as_bgr: bool = True,
        p: float = 0.5,
    ):
        super().__init__(p)
        if compression_type != "jpeg":
            raise NotImplementedError("Only compression_type='jpeg' is supported")
        self.compression_type = compression_type
        self.quality_range = check_bounds(check_nondecreasing((int(quality_range[0]), int(quality_range[1])), "quality_range"), 1, 100, "quality_range")
        self.treat_as_bgr = treat_as_bgr

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        return {"quality": rng.randint(self.quality_range[0], self.quality_range[1], batch.batch_size)}

    def apply(self, img, quality, **params):
        c = img.shape[1]
        if c in (1, 3):
            return jpeg(img, quality, self.treat_as_bgr)
        # albumentations: first 3 channels as color JPEG, extra channels as grayscale JPEGs
        out = [jpeg(img[:, :3], quality, self.treat_as_bgr)] if c >= 3 else []
        start = 3 if c >= 3 else 0
        out += [jpeg(img[:, i : i + 1], quality) for i in range(start, c)]
        return torch.cat(out, dim=1)


def _axis_coords(dst_size: int, src_size: torch.Tensor, interpolation: int, n: int, dev: torch.device) -> torch.Tensor:
    """cv2.resize source coordinates along one axis for per-sample source sizes -> (n, dst_size)."""
    d = torch.arange(dst_size, device=dev, dtype=torch.float32)[None]
    s = src_size.float()[:, None]
    scale = s / dst_size
    if interpolation == INTER_NEAREST:
        return torch.minimum(torch.floor(d * scale), s - 1)
    return torch.minimum(((d + 0.5) * scale - 0.5).clamp(min=0), s - 1)


class Downscale(ImageOnlyTransform):
    """albumentations ``Downscale``: resize by scale ~ U(scale_range) (size = round(size * scale)), then back to the
    original size. interpolation_pair: {"downscale": ..., "upscale": ...} with nearest / linear / cubic (and area for
    downscale)."""

    def __init__(
        self,
        scale_range: Sequence[float] = (0.25, 0.25),
        interpolation_pair: dict | None = None,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.scale_range = check_bounds(check_nondecreasing((float(scale_range[0]), float(scale_range[1])), "scale_range"), 0, 1, "scale_range")
        pair = interpolation_pair or {"upscale": INTER_NEAREST, "downscale": INTER_NEAREST}
        self.interpolation_pair = {
            "downscale": resolve_interpolation(pair["downscale"], allow_area=True),
            "upscale": resolve_interpolation(pair["upscale"]),
        }

    def get_params(self, batch, rng):
        n, h, w = batch.batch_size, batch.height, batch.width
        s = rng.uniform(*self.scale_range, n)
        dh = torch.round(h * s).clamp(min=1).long()
        dw = torch.round(w * s).clamp(min=1).long()
        return {"dh": dh, "dw": dw, "scale": s}

    def apply(self, img, dh, dw, scale, **params):
        n, c, h, w = img.shape
        dev = img.device
        hmax, wmax = int(dh.max()), int(dw.max())
        down_i = self.interpolation_pair["downscale"]
        if down_i == INTER_AREA:
            small = torch.zeros(n, c, hmax, wmax, device=dev, dtype=img.dtype)
            for i in range(n):
                hi, wi = int(dh[i]), int(dw[i])
                small[i, :, :hi, :wi] = F.interpolate(img[i : i + 1], size=(hi, wi), mode="area")[0]
        else:
            # sample the original at the per-sample downscaled grid (embedded top-left in an hmax x wmax canvas)
            yy = torch.arange(hmax, device=dev, dtype=torch.float32)[None]
            xx = torch.arange(wmax, device=dev, dtype=torch.float32)[None]
            # cv2.resize(fx=fy=scale) maps destination -> source with the inverse of the *given* factor (1 / scale),
            # not with src_size / dst_size
            sy_scale = sx_scale = (1.0 / scale.float())[:, None]
            if down_i == INTER_NEAREST:
                sy = torch.floor(yy * sy_scale).clamp(max=h - 1)
                sx = torch.floor(xx * sx_scale).clamp(max=w - 1)
            else:
                sy = ((yy + 0.5) * sy_scale - 0.5).clamp(0, h - 1)
                sx = ((xx + 0.5) * sx_scale - 0.5).clamp(0, w - 1)
            small = fgeo.sample(img, sx[:, None, :].expand(n, hmax, wmax), sy[:, :, None].expand(n, hmax, wmax), down_i, 1)
        # replicate each sample's last valid row/column into the unused part of the canvas so that interpolation at the
        # edge of the valid region (cubic reads 2 neighbours) sees replicated borders like cv2.resize
        ry = torch.minimum(torch.arange(hmax, device=dev)[None], dh[:, None] - 1)
        rx = torch.minimum(torch.arange(wmax, device=dev)[None], dw[:, None] - 1)
        small = torch.gather(small, 2, ry[:, None, :, None].expand(n, c, hmax, wmax))
        small = torch.gather(small, 3, rx[:, None, None, :].expand(n, c, hmax, wmax))
        up_i = self.interpolation_pair["upscale"]
        cy = _axis_coords(h, dh, up_i, n, dev)  # coords inside each sample's valid region
        cx = _axis_coords(w, dw, up_i, n, dev)
        out = fgeo.sample(small, cx[:, None, :].expand(n, h, w), cy[:, :, None].expand(n, h, w), up_i, 1)
        return out.clamp_(0, 1) if up_i == INTER_CUBIC else out

