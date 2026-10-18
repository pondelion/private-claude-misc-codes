"""Dropout transforms (CoarseDropout / GridDropout / XYMasking / PixelDropout / ChannelDropout)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch

from ..core.bbox import boxes_visible_extent_after_holes
from ..core.data import AugBatch
from ..core.random import Rng
from ..core.transform import DualTransform, ImageOnlyTransform
from ..core.utils import check_nondecreasing, int_range

__all__ = ["ChannelDropout", "CoarseDropout", "GridDropout", "PixelDropout", "XYMasking"]

FillType = float | Sequence[float] | Literal["random", "random_uniform"]


def _holes_to_index_map(holes: torch.Tensor, valid: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """holes (n, K, 4) int [x1, y1, x2, y2), valid (n, K) -> (n, H, W) long: index of the last hole covering the
    pixel (later holes overwrite earlier ones, like sequential filling), -1 where no hole."""
    n, k = holes.shape[:2]
    dev = holes.device
    out = torch.full((n, h, w), -1, dtype=torch.long, device=dev)
    ys = torch.arange(h, device=dev)
    xs = torch.arange(w, device=dev)
    for i in range(k):
        x1, y1, x2, y2 = (holes[:, i, j][:, None] for j in range(4))
        inside = ((ys >= y1) & (ys < y2))[:, :, None] & ((xs >= x1) & (xs < x2))[:, None, :]
        inside &= valid[:, i].view(n, 1, 1)
        out = torch.where(inside, torch.full_like(out, i), out)
    return out


class _BaseDropout(DualTransform):
    """Holes are rectangles [x1, x2) x [y1, y2) in pixels.

    - fill: value in the input value scale (0..255 for uint8 input, 0..1 for float input), a per-channel tuple,
      "random" (uniform random value per pixel and channel) or "random_uniform" (one random color per hole).
      albumentations' "inpaint_telea" / "inpaint_ns" are not supported on GPU.
    - fill_mask: None (mask unchanged) or a value / per-channel tuple written into the holes of the mask.
    Boxes: albumentations ``filter_bboxes_by_holes``: boxes whose remaining visible area < min_area, visible ratio
    < min_visibility, or nothing visible are invalidated; the others are shrunk to their visible rows/columns.
    Keypoints: with ``remove_invisible`` (albumentations ``filter_keypoints_in_holes``) keypoints with
    x1 <= x < x2 and y1 <= y < y2 for any hole are invalidated.
    """

    def __init__(self, fill: FillType = 0, fill_mask: float | Sequence[float] | None = None, p: float = 0.5):
        super().__init__(p)
        if isinstance(fill, str) and fill not in ("random", "random_uniform"):
            raise NotImplementedError(f"fill={fill!r} is not supported (use a value, 'random' or 'random_uniform')")
        self.fill = fill
        self.fill_mask = fill_mask

    def make_holes(self, batch: AugBatch, rng: Rng) -> tuple[torch.Tensor, torch.Tensor]:
        """-> holes (n, K, 4) long, valid (n, K) bool"""
        raise NotImplementedError

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        holes, valid = self.make_holes(batch, rng)
        index_map = _holes_to_index_map(holes, valid, batch.height, batch.width)
        params: dict[str, Any] = {"index_map": index_map, "holes": holes, "hole_valid": valid}
        n, c = batch.batch_size, batch.image.shape[1]
        if self.fill == "random":
            params["random_fill"] = rng.rand((n, c, batch.height, batch.width))
        elif self.fill == "random_uniform":
            params["random_fill"] = rng.rand((n, max(1, holes.shape[1]), c))
        return params

    def apply(self, img: torch.Tensor, index_map: torch.Tensor, max_value: float = 1.0, random_fill=None, **params: Any) -> torch.Tensor:
        hole = (index_map >= 0)[:, None]
        if self.fill == "random":
            return torch.where(hole, random_fill, img)
        if self.fill == "random_uniform":
            colors = torch.gather(
                random_fill, 1, index_map.clamp(min=0).flatten(1)[..., None].expand(-1, -1, img.shape[1])
            )  # (n, H*W, C)
            colors = colors.transpose(1, 2).reshape(img.shape)
            return torch.where(hole, colors, img)
        f = torch.as_tensor(self.fill, dtype=img.dtype, device=img.device).flatten() / max_value
        f = f.view(1, -1, 1, 1) if f.numel() > 1 else f.view(1, 1, 1, 1)
        return torch.where(hole, f.expand_as(img), img)

    def apply_to_mask(self, mask: torch.Tensor, index_map: torch.Tensor, **params: Any) -> torch.Tensor:
        if self.fill_mask is None:
            return mask
        f = torch.as_tensor(self.fill_mask, device=mask.device).flatten().to(mask.dtype)
        f = f.view(1, -1, 1, 1) if f.numel() > 1 else f.view(1, 1, 1, 1)
        return torch.where((index_map >= 0)[:, None], f.expand_as(mask), mask)

    def apply_to_bboxes(self, bboxes, valid, height, width, index_map=None, bbox_params=None, **params):
        hole = index_map >= 0
        if bboxes.shape[1] == 0 or not bool(hole.any()):
            return bboxes, valid
        min_area = bbox_params.min_area if bbox_params is not None else 0.0
        min_vis = bbox_params.min_visibility if bbox_params is not None else 0.0
        remaining, area, shrunk = boxes_visible_extent_after_holes(bboxes, hole)
        ratio = remaining.float() / area.clamp(min=1).float()
        keep = (remaining >= min_area) & (ratio >= min_vis) & (remaining > 0)
        return torch.where(keep[..., None], shrunk, bboxes), valid & keep

    def apply_to_keypoints(self, keypoints, valid, height, width, holes=None, hole_valid=None, keypoint_params=None, **params):
        if keypoint_params is None or not keypoint_params.remove_invisible or holes.shape[1] == 0:
            return keypoints, valid
        x = keypoints[..., 0][:, :, None]  # (n, K, 1)
        y = keypoints[..., 1][:, :, None]
        h = holes.to(keypoints.dtype)[:, None]  # (n, 1, holes, 4)
        inside = (x >= h[..., 0]) & (x < h[..., 2]) & (y >= h[..., 1]) & (y < h[..., 3]) & hole_valid[:, None]
        return keypoints, valid & ~inside.any(dim=-1)


def _hole_sizes(rng_range: tuple[float, float], size: int, n: int, k: int, rng: Rng) -> torch.Tensor:
    if rng_range[1] >= 1:
        lo, hi = int(rng_range[0]), int(min(rng_range[1], size))
        return rng.randint(lo, hi, (n, k))
    return (size * rng.uniform(rng_range[0], rng_range[1], (n, k))).long()


class CoarseDropout(_BaseDropout):
    """albumentations ``CoarseDropout``: num_holes ~ randint(num_holes_range); hole height/width in pixels (ints, if
    range[1] >= 1) or as a fraction of the image size (floats in [0, 1)); top-left uniform so the hole fits."""

    def __init__(
        self,
        num_holes_range: Sequence[int] = (1, 2),
        hole_height_range: Sequence[float] = (0.1, 0.2),
        hole_width_range: Sequence[float] = (0.1, 0.2),
        fill: FillType = 0,
        fill_mask: float | Sequence[float] | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill, fill_mask, p)
        self.num_holes_range = check_nondecreasing(int_range(num_holes_range, None, "num_holes_range"), "num_holes_range")
        if self.num_holes_range[0] < 1:
            raise ValueError("num_holes_range must be >= 1")
        self.hole_height_range = check_nondecreasing((float(hole_height_range[0]), float(hole_height_range[1])), "hole_height_range")
        self.hole_width_range = check_nondecreasing((float(hole_width_range[0]), float(hole_width_range[1])), "hole_width_range")

    def make_holes(self, batch: AugBatch, rng: Rng) -> tuple[torch.Tensor, torch.Tensor]:
        n, h, w = batch.batch_size, batch.height, batch.width
        kmax = self.num_holes_range[1]
        num = rng.randint(self.num_holes_range[0], kmax, n)
        hh = _hole_sizes(self.hole_height_range, h, n, kmax, rng)
        hw = _hole_sizes(self.hole_width_range, w, n, kmax, rng)
        y1 = rng.randint(0, h - hh, (n, kmax))
        x1 = rng.randint(0, w - hw, (n, kmax))
        holes = torch.stack([x1, y1, x1 + hw, y1 + hh], dim=-1)
        valid = torch.arange(kmax, device=batch.device)[None] < num[:, None]
        return holes, valid


class GridDropout(_BaseDropout):
    """albumentations ``GridDropout``: the image is split into a grid of cells (sizes almost equal, in random order
    per axis); in each cell a hole of ``ratio`` x cell size is placed at a random offset (``random_offset``) or at
    ``shift_xy``. Grid from ``holes_number_xy`` (columns, rows), ``unit_size_range`` (square unit size in pixels,
    random per sample) or the default (size // 10)."""

    def __init__(
        self,
        ratio: float = 0.5,
        random_offset: bool = True,
        unit_size_range: Sequence[int] | None = None,
        holes_number_xy: Sequence[int] | None = None,
        shift_xy: Sequence[int] = (0, 0),
        fill: FillType = 0,
        fill_mask: float | Sequence[float] | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill, fill_mask, p)
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = float(ratio)
        self.random_offset = random_offset
        self.unit_size_range = None if unit_size_range is None else (int(unit_size_range[0]), int(unit_size_range[1]))
        self.holes_number_xy = None if holes_number_xy is None else (int(holes_number_xy[0]), int(holes_number_xy[1]))
        self.shift_xy = (int(shift_xy[0]), int(shift_xy[1]))

    @staticmethod
    def _shuffled_splits(size: int, parts: torch.Tensor, rng: Rng) -> torch.Tensor:
        """(n,) parts -> (n, P+1) split positions (cells of almost equal size in random order); entries beyond a
        sample's part count repeat ``size``."""
        n = parts.shape[0]
        pmax = int(parts.max())
        j = torch.arange(pmax, device=parts.device)[None]
        base = size // parts[:, None]
        rem = size % parts[:, None]
        sizes = torch.where(j < rem, base + 1, base)
        sizes = torch.where(j < parts[:, None], sizes, torch.zeros_like(sizes))
        keys = torch.where(j < parts[:, None], rng.rand((n, pmax)), torch.full((n, pmax), 2.0, device=parts.device))
        order = torch.argsort(keys, dim=1)
        sizes = torch.gather(sizes, 1, order)
        return torch.cat([torch.zeros(n, 1, dtype=sizes.dtype, device=sizes.device), sizes.cumsum(1)], dim=1)

    def make_holes(self, batch: AugBatch, rng: Rng) -> tuple[torch.Tensor, torch.Tensor]:
        n, h, w = batch.batch_size, batch.height, batch.width
        dev = batch.device
        if self.holes_number_xy:
            rows = torch.full((n,), self.holes_number_xy[1], device=dev, dtype=torch.long)
            cols = torch.full((n,), self.holes_number_xy[0], device=dev, dtype=torch.long)
        else:
            if self.unit_size_range is not None:
                if self.unit_size_range[1] > min(h, w):
                    raise ValueError("unit_size_range must be within the shortest image edge")
                unit = rng.randint(self.unit_size_range[0], self.unit_size_range[1] - 1, n)  # numpy integers: high exclusive
                uh = uw = unit
            else:
                uh = torch.full((n,), max(2, h // 10), device=dev, dtype=torch.long)
                uw = torch.full((n,), max(2, w // 10), device=dev, dtype=torch.long)
            rows, cols = h // uh, w // uw
        ys = self._shuffled_splits(h, rows, rng)  # (n, R+1)
        xs = self._shuffled_splits(w, cols, rng)  # (n, Cc+1)
        rmax, cmax = ys.shape[1] - 1, xs.shape[1] - 1
        cy1 = ys[:, :-1][:, :, None].expand(n, rmax, cmax)
        cy2 = ys[:, 1:][:, :, None].expand(n, rmax, cmax)
        cx1 = xs[:, :-1][:, None, :].expand(n, rmax, cmax)
        cx2 = xs[:, 1:][:, None, :].expand(n, rmax, cmax)
        valid = (torch.arange(rmax, device=dev)[None, :, None] < rows[:, None, None]) & (
            torch.arange(cmax, device=dev)[None, None, :] < cols[:, None, None]
        )
        ch, cw = (cy2 - cy1), (cx2 - cx1)
        hh = torch.minimum(torch.maximum((ch * self.ratio), torch.ones_like(ch, dtype=torch.float32)), (ch - 1).float()).long()
        hw = torch.minimum(torch.maximum((cw * self.ratio), torch.ones_like(cw, dtype=torch.float32)), (cw - 1).float()).long()
        if self.random_offset:
            oy = rng.randint(0, (ch - hh).clamp(min=0), (n, rmax, cmax))
            ox = rng.randint(0, (cw - hw).clamp(min=0), (n, rmax, cmax))
        else:
            oy = torch.full_like(ch, self.shift_xy[1])
            ox = torch.full_like(cw, self.shift_xy[0])
        x1 = torch.minimum((cx1 + ox).clamp(min=0), w - hw)
        y1 = torch.minimum((cy1 + oy).clamp(min=0), h - hh)
        x2 = torch.clamp(x1 + hw, max=w)
        y2 = torch.clamp(y1 + hh, max=h)
        holes = torch.stack([x1, y1, x2, y2], dim=-1).view(n, rmax * cmax, 4)
        return holes, valid.view(n, rmax * cmax)


class XYMasking(_BaseDropout):
    """albumentations ``XYMasking``: vertical stripes (full height) of width ~ randint(mask_x_length) and horizontal
    stripes (full width) of height ~ randint(mask_y_length); counts ~ randint(num_masks_x / num_masks_y) (an int is a
    fixed count)."""

    def __init__(
        self,
        num_masks_x: int | Sequence[int] = 0,
        num_masks_y: int | Sequence[int] = 0,
        mask_x_length: int | Sequence[int] = 0,
        mask_y_length: int | Sequence[int] = 0,
        fill: FillType = 0,
        fill_mask: float | Sequence[float] | None = None,
        p: float = 0.5,
    ):
        super().__init__(fill, fill_mask, p)

        def as_range(v):
            return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))

        self.num_masks_x, self.num_masks_y = as_range(num_masks_x), as_range(num_masks_y)
        self.mask_x_length, self.mask_y_length = as_range(mask_x_length), as_range(mask_y_length)
        if self.num_masks_x[1] == 0 and self.num_masks_y[1] == 0:
            raise ValueError("At least one of num_masks_x / num_masks_y must be positive")

    def _stripes(self, num_r, len_r, size, n, rng, axis):
        kmax = num_r[1]
        if kmax == 0 or len_r[1] == 0:
            return None
        if len_r[1] > size:
            raise ValueError(f"mask length {len_r} exceeds the image size {size}")
        num = rng.randint(num_r[0], kmax, n)
        length = rng.randint(len_r[0], len_r[1], (n, kmax))
        start = rng.randint(0, size - length, (n, kmax))
        valid = torch.arange(kmax, device=rng.device)[None] < num[:, None]
        return start, start + length, valid

    def make_holes(self, batch: AugBatch, rng: Rng) -> tuple[torch.Tensor, torch.Tensor]:
        n, h, w = batch.batch_size, batch.height, batch.width
        holes, valids = [], []
        sx = self._stripes(self.num_masks_x, self.mask_x_length, w, n, rng, "x")
        if sx is not None:
            a, b, v = sx
            holes.append(torch.stack([a, torch.zeros_like(a), b, torch.full_like(a, h)], dim=-1))
            valids.append(v)
        sy = self._stripes(self.num_masks_y, self.mask_y_length, h, n, rng, "y")
        if sy is not None:
            a, b, v = sy
            holes.append(torch.stack([torch.zeros_like(a), a, torch.full_like(a, w), b], dim=-1))
            valids.append(v)
        if not holes:
            return torch.zeros(n, 0, 4, dtype=torch.long, device=batch.device), torch.zeros(n, 0, dtype=torch.bool, device=batch.device)
        return torch.cat(holes, dim=1), torch.cat(valids, dim=1)


class PixelDropout(DualTransform):
    """albumentations ``PixelDropout``: each pixel (or pixel-channel with ``per_channel``) is dropped with
    ``dropout_prob`` and replaced by ``drop_value`` (input value scale; None -> one random value per channel).
    With ``mask_drop_value`` the mask gets an independent drop pattern. Boxes (when not per_channel) are invalidated
    if their visible fraction falls below min_visibility / visible area below min_area (``mask_dropout_bboxes``)."""

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: float | Sequence[float] | None = 0,
        mask_drop_value: float | Sequence[float] | None = None,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.dropout_prob = float(dropout_prob)
        self.per_channel = per_channel
        self.drop_value = drop_value
        self.mask_drop_value = mask_drop_value

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, c, h, w = batch.image.shape
        shape = (n, c, h, w) if self.per_channel else (n, 1, h, w)
        params: dict[str, Any] = {"drop_mask": rng.bernoulli(self.dropout_prob, shape)}
        if self.drop_value is None:
            params["random_values"] = rng.rand((n, c, 1, 1))
        if self.mask_drop_value is not None and batch.mask is not None:
            mc = batch.mask.shape[1]
            mshape = (n, mc, h, w) if self.per_channel else (n, 1, h, w)
            params["mask_drop_mask"] = rng.bernoulli(self.dropout_prob, mshape)
        return params

    def apply(self, img, drop_mask, max_value: float = 1.0, random_values=None, **params):
        if self.drop_value is None:
            values = random_values
        else:
            values = (torch.as_tensor(self.drop_value, dtype=img.dtype, device=img.device).flatten() / max_value).view(1, -1, 1, 1)
        return torch.where(drop_mask, values.expand_as(img) if values.numel() > 1 else values, img)

    def apply_to_mask(self, mask, mask_drop_mask=None, **params):
        if self.mask_drop_value is None or mask_drop_mask is None:
            return mask
        v = torch.as_tensor(self.mask_drop_value, device=mask.device).flatten().to(mask.dtype).view(1, -1, 1, 1)
        return torch.where(mask_drop_mask, v.expand_as(mask) if v.numel() > 1 else v, mask)

    def apply_to_bboxes(self, bboxes, valid, height, width, drop_mask=None, bbox_params=None, **params):
        if self.per_channel or bboxes.shape[1] == 0 or bbox_params is None:
            return bboxes, valid
        dropped = drop_mask[:, 0]  # (n, H, W)
        # albumentations mask_dropout_bboxes: inclusive pixel test x1 <= x <= x2, y1 <= y <= y2
        n = dropped.shape[0]
        integ = torch.zeros(n, height + 1, width + 1, dtype=torch.int32, device=dropped.device)
        integ[:, 1:, 1:] = (~dropped).to(torch.int32).cumsum(1).cumsum(2)
        xa = torch.ceil(bboxes[..., 0]).clamp(0, width).long()
        xb = (torch.floor(bboxes[..., 2]) + 1).clamp(0, width).long()
        ya = torch.ceil(bboxes[..., 1]).clamp(0, height).long()
        yb = (torch.floor(bboxes[..., 3]) + 1).clamp(0, height).long()
        idx = torch.arange(n, device=dropped.device)[:, None]
        visible = (integ[idx, yb, xb] - integ[idx, ya, xb] - integ[idx, yb, xa] + integ[idx, ya, xa]).float()
        area = (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])
        keep = (visible >= bbox_params.min_area) & (visible / area.clamp(min=1e-7) >= bbox_params.min_visibility)
        return bboxes, valid & keep


class ChannelDropout(ImageOnlyTransform):
    """albumentations ``ChannelDropout``: randint(channel_drop_range) random channels are set to ``fill``."""

    def __init__(self, channel_drop_range: Sequence[int] = (1, 1), fill: float = 0, p: float = 0.5):
        super().__init__(p)
        self.channel_drop_range = (int(channel_drop_range[0]), int(channel_drop_range[1]))
        self.fill = fill

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, c = batch.image.shape[:2]
        if c == 1:
            raise NotImplementedError("ChannelDropout is not defined for 1-channel images")
        if self.channel_drop_range[1] >= c:
            raise ValueError("Can not drop all channels in ChannelDropout")
        k = rng.randint(self.channel_drop_range[0], self.channel_drop_range[1], n)
        rank = torch.argsort(rng.permutation(n, c), dim=1)  # random rank of each channel
        return {"drop": rank < k[:, None]}

    def apply(self, img, drop, max_value: float = 1.0, **params):
        return torch.where(drop[:, :, None, None], torch.full_like(img, float(self.fill) / max_value), img)
