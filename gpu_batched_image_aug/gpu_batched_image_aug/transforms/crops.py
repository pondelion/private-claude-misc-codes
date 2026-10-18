"""Crops / resizes. They change the spatial size, so they must be applied to the whole batch (p=1, outside
OneOf/SomeOf); every sample still gets its own random crop."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ..core.constants import INTER_AREA, INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, resolve_interpolation
from ..core.data import AugBatch
from ..core.random import Rng
from ..core.transform import DualTransform
from ..functional import geometric as fgeo

__all__ = ["CenterCrop", "Crop", "RandomCrop", "RandomResizedCrop", "Resize"]


def _gather_crop(x: torch.Tensor, x1: torch.Tensor, y1: torch.Tensor, h: int, w: int) -> torch.Tensor:
    n = x.shape[0]
    rows = y1[:, None] + torch.arange(h, device=x.device)  # (n, h)
    cols = x1[:, None] + torch.arange(w, device=x.device)  # (n, w)
    idx_n = torch.arange(n, device=x.device)[:, None, None]
    out = x[idx_n, :, rows[:, :, None], cols[:, None, :]]  # (n, h, w, C)
    return out.permute(0, 3, 1, 2).contiguous()


class _FixedSizeCrop(DualTransform):
    changes_shape = True
    is_geometric = True

    def __init__(self, height: int, width: int, p: float = 1.0):
        super().__init__(p)
        self.height = int(height)
        self.width = int(width)

    def _check(self, batch: AugBatch) -> None:
        if self.height > batch.height or self.width > batch.width:
            raise ValueError(
                f"Crop size {(self.height, self.width)} exceeds image size {(batch.height, batch.width)} "
                "(pad_if_needed is not supported)."
            )

    def apply(self, img: torch.Tensor, x1: torch.Tensor, y1: torch.Tensor, **params: Any) -> torch.Tensor:
        return _gather_crop(img, x1, y1, self.height, self.width)

    def apply_to_mask(self, mask: torch.Tensor, x1: torch.Tensor, y1: torch.Tensor, **params: Any) -> torch.Tensor:
        return _gather_crop(mask, x1, y1, self.height, self.width)

    def apply_to_bboxes(self, bboxes, valid, height, width, x1=None, y1=None, **params):
        off = torch.stack([x1, y1, x1, y1], dim=-1).to(bboxes.dtype)[:, None, :]
        return bboxes - off, valid

    def apply_to_keypoints(self, keypoints, valid, height, width, x1=None, y1=None, **params):
        out = keypoints.clone()
        out[..., 0] -= x1.to(out.dtype)[:, None]
        out[..., 1] -= y1.to(out.dtype)[:, None]
        return out, valid


def _scale_keypoints(kps: torch.Tensor, sx: torch.Tensor | float, sy: torch.Tensor | float) -> torch.Tensor:
    """albumentations ``keypoints_scale``: x *= sx, y *= sy, scale *= max(sx, sy). sx/sy: floats or (n,)."""
    sx = torch.as_tensor(sx, dtype=kps.dtype, device=kps.device).reshape(-1, 1)
    sy = torch.as_tensor(sy, dtype=kps.dtype, device=kps.device).reshape(-1, 1)
    x, y, a, s = kps.unbind(-1)
    return torch.stack([x * sx, y * sy, a, s * torch.maximum(sx, sy)], dim=-1)


class RandomCrop(_FixedSizeCrop):
    """albumentations ``RandomCrop`` (pad_if_needed is not supported): top-left = int((size - crop + 1) * U[0, 1))."""

    def __init__(self, height: int, width: int, pad_if_needed: bool = False, p: float = 1.0):
        if pad_if_needed:
            raise NotImplementedError("pad_if_needed is not supported")
        super().__init__(height, width, p)
        self.pad_if_needed = pad_if_needed

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        self._check(batch)
        n = batch.batch_size
        y1 = (rng.rand(n) * (batch.height - self.height + 1)).long()
        x1 = (rng.rand(n) * (batch.width - self.width + 1)).long()
        return {"x1": x1, "y1": y1}


class CenterCrop(_FixedSizeCrop):
    def __init__(self, height: int, width: int, pad_if_needed: bool = False, p: float = 1.0):
        if pad_if_needed:
            raise NotImplementedError("pad_if_needed is not supported")
        super().__init__(height, width, p)
        self.pad_if_needed = pad_if_needed

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        self._check(batch)
        n = batch.batch_size
        y1 = torch.full((n,), (batch.height - self.height) // 2, device=batch.device, dtype=torch.long)
        x1 = torch.full((n,), (batch.width - self.width) // 2, device=batch.device, dtype=torch.long)
        return {"x1": x1, "y1": y1}


class Crop(_FixedSizeCrop):
    """Fixed region crop [x_min, x_max) x [y_min, y_max) (albumentations ``Crop`` without padding)."""

    def __init__(self, x_min: int = 0, y_min: int = 0, x_max: int = 1024, y_max: int = 1024, p: float = 1.0):
        super().__init__(y_max - y_min, x_max - x_min, p)
        self.x_min, self.y_min, self.x_max, self.y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        if self.x_max > batch.width or self.y_max > batch.height:
            raise ValueError("Crop region exceeds the image")
        n = batch.batch_size
        return {
            "x1": torch.full((n,), self.x_min, device=batch.device, dtype=torch.long),
            "y1": torch.full((n,), self.y_min, device=batch.device, dtype=torch.long),
        }


def _resize(x: torch.Tensor, h: int, w: int, interpolation: int) -> torch.Tensor:
    dtype = x.dtype
    y = x if dtype.is_floating_point else x.float()
    if interpolation == INTER_NEAREST:
        y = F.interpolate(y, size=(h, w), mode="nearest")
    elif interpolation == INTER_LINEAR:
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False, antialias=False)
    elif interpolation == INTER_CUBIC:
        y = F.interpolate(y, size=(h, w), mode="bicubic", align_corners=False, antialias=False)
    elif interpolation == INTER_AREA:
        y = F.interpolate(y, size=(h, w), mode="area")
    else:
        raise ValueError(interpolation)
    if dtype.is_floating_point:
        return y.to(dtype)
    if dtype == torch.bool:
        return y > 0.5
    return y.round().to(dtype)


class Resize(DualTransform):
    """Resize every image to (height, width). Matches cv2.resize for nearest / linear / cubic (half-pixel mapping, no
    antialiasing). ``INTER_AREA`` uses adaptive average pooling (exact for integer downscale factors)."""

    changes_shape = True
    is_geometric = True

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int | str = INTER_LINEAR,
        mask_interpolation: int | str = INTER_NEAREST,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.height, self.width = int(height), int(width)
        self.interpolation = resolve_interpolation(interpolation, allow_area=True)
        self.mask_interpolation = resolve_interpolation(mask_interpolation, allow_area=True)

    def apply(self, img: torch.Tensor, **params: Any) -> torch.Tensor:
        out = _resize(img, self.height, self.width, self.interpolation)
        return out.clamp(0, 1) if self.interpolation == INTER_CUBIC else out

    def apply_to_mask(self, mask: torch.Tensor, **params: Any) -> torch.Tensor:
        return _resize(mask, self.height, self.width, self.mask_interpolation)

    def apply_to_bboxes(self, bboxes, valid, height, width, **params):
        s = bboxes.new_tensor([self.width / width, self.height / height, self.width / width, self.height / height])
        return bboxes * s, valid

    def apply_to_keypoints(self, keypoints, valid, height, width, **params):
        return _scale_keypoints(keypoints, self.width / width, self.height / height), valid


class RandomResizedCrop(DualTransform):
    """albumentations ``RandomResizedCrop``: up to 10 attempts of area ~ U(scale) * H * W and log-uniform aspect
    ratio; the first attempt that fits is used, otherwise a central crop clamped to ``ratio``. The crop is resized to
    ``size`` = (height, width) with cv2.resize semantics."""

    changes_shape = True
    is_geometric = True

    def __init__(
        self,
        size: Sequence[int],
        scale: Sequence[float] = (0.08, 1.0),
        ratio: Sequence[float] = (0.75, 4 / 3),
        interpolation: int | str = INTER_LINEAR,
        mask_interpolation: int | str = INTER_NEAREST,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.size = (int(size[0]), int(size[1]))
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))
        self.interpolation = resolve_interpolation(interpolation)
        self.mask_interpolation = resolve_interpolation(mask_interpolation)

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, ih, iw = batch.batch_size, batch.height, batch.width
        area = ih * iw
        attempts = 10
        target = rng.uniform(self.scale[0] * area, self.scale[1] * area, (n, attempts)).double()
        log_r = rng.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]), (n, attempts)).double()
        ar = torch.exp(log_r)
        w = torch.round(torch.sqrt(target * ar))
        h = torch.round(torch.sqrt(target / ar))
        ok = (w > 0) & (w <= iw) & (h > 0) & (h <= ih)
        first = torch.argmax(ok.int(), dim=1)  # first valid attempt (0 if none)
        any_ok = ok.any(dim=1)
        w_sel = w.gather(1, first[:, None]).squeeze(1)
        h_sel = h.gather(1, first[:, None]).squeeze(1)
        hs, ws = rng.rand(n).double(), rng.rand(n).double()
        y1 = torch.floor((ih - h_sel + 1) * hs)
        x1 = torch.floor((iw - w_sel + 1) * ws)

        # fallback: central crop clamped to the ratio range
        in_ratio = iw / ih
        if in_ratio < self.ratio[0]:
            fw, fh = iw, round(iw / self.ratio[0])
        elif in_ratio > self.ratio[1]:
            fh, fw = ih, round(ih * self.ratio[1])
        else:
            fw, fh = iw, ih
        fx1, fy1 = (iw - fw) // 2, (ih - fh) // 2
        x1 = torch.where(any_ok, x1, torch.full_like(x1, fx1))
        y1 = torch.where(any_ok, y1, torch.full_like(y1, fy1))
        w_sel = torch.where(any_ok, w_sel, torch.full_like(w_sel, fw))
        h_sel = torch.where(any_ok, h_sel, torch.full_like(h_sel, fh))
        crop = torch.stack([x1, y1, x1 + w_sel, y1 + h_sel], dim=1).long()
        return {"crop": crop}

    def _apply(self, x: torch.Tensor, crop: torch.Tensor, interpolation: int) -> torch.Tensor:
        oh, ow = self.size
        cx, cy = fgeo.crop_resize_coords(crop, oh, ow, interpolation)
        return fgeo.sample(x, cx, cy, interpolation, border_mode=1)  # coords are inside the crop

    def apply(self, img: torch.Tensor, crop: torch.Tensor, **params: Any) -> torch.Tensor:
        out = self._apply(img, crop, self.interpolation)
        return out.clamp(0, 1) if self.interpolation == INTER_CUBIC else out

    def apply_to_mask(self, mask: torch.Tensor, crop: torch.Tensor, **params: Any) -> torch.Tensor:
        return self._apply(mask, crop, self.mask_interpolation)

    def apply_to_bboxes(self, bboxes, valid, height, width, crop=None, **params):
        oh, ow = self.size
        c = crop.to(bboxes.dtype)
        cw, ch = (c[:, 2] - c[:, 0]), (c[:, 3] - c[:, 1])
        off = torch.stack([c[:, 0], c[:, 1], c[:, 0], c[:, 1]], dim=-1)[:, None]
        s = torch.stack([ow / cw, oh / ch, ow / cw, oh / ch], dim=-1)[:, None]
        return (bboxes - off) * s, valid

    def apply_to_keypoints(self, keypoints, valid, height, width, crop=None, **params):
        oh, ow = self.size
        c = crop.to(keypoints.dtype)
        out = keypoints.clone()
        out[..., 0] -= c[:, 0, None]
        out[..., 1] -= c[:, 1, None]
        return _scale_keypoints(out, ow / (c[:, 2] - c[:, 0]), oh / (c[:, 3] - c[:, 1])), valid
