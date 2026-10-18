"""Geometric transforms (image / mask / bboxes / keypoints)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import torch

from ..core.bbox import boxes_to_masks, masks_to_boxes, pixel_center_to_edge_matrix, transform_boxes_by_matrix
from ..core.constants import BORDER_CONSTANT, INTER_LINEAR, INTER_NEAREST, resolve_border_mode, resolve_interpolation
from ..core.data import AugBatch
from ..core.keypoints import transform_points_by_matrix, wrap_angle
from ..core.random import Rng
from ..core.transform import DualTransform
from ..core.utils import check_bounds, check_nondecreasing, fixed_or_range, symmetric_range, to_range
from ..functional import geometric as fgeo

__all__ = [
    "Affine",
    "D4",
    "ElasticTransform",
    "GridDistortion",
    "HorizontalFlip",
    "OpticalDistortion",
    "Perspective",
    "RandomRotate90",
    "Rotate",
    "ShiftScaleRotate",
    "Transpose",
    "VerticalFlip",
]


def _require_square(batch: AugBatch, name: str) -> None:
    if batch.height != batch.width:
        raise ValueError(f"{name} changes the image shape for non-square images, which a batch tensor cannot hold.")


# ----------------------------------------------------------------------------------------------------------------
# exact flips / rotations
# ----------------------------------------------------------------------------------------------------------------


class HorizontalFlip(DualTransform):
    is_geometric = True

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, img: torch.Tensor, **params: Any) -> torch.Tensor:
        return img.flip(-1)

    def apply_to_mask(self, mask: torch.Tensor, **params: Any) -> torch.Tensor:
        return mask.flip(-1)

    def apply_to_bboxes(self, bboxes, valid, height, width, **params):
        x1, y1, x2, y2 = bboxes.unbind(-1)
        return torch.stack([width - x2, y1, width - x1, y2], dim=-1), valid

    def apply_to_keypoints(self, keypoints, valid, height, width, **params):
        return _kp_hflip(keypoints, width), valid


class VerticalFlip(DualTransform):
    is_geometric = True

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, img: torch.Tensor, **params: Any) -> torch.Tensor:
        return img.flip(-2)

    def apply_to_mask(self, mask: torch.Tensor, **params: Any) -> torch.Tensor:
        return mask.flip(-2)

    def apply_to_bboxes(self, bboxes, valid, height, width, **params):
        x1, y1, x2, y2 = bboxes.unbind(-1)
        return torch.stack([x1, height - y2, x2, height - y1], dim=-1), valid

    def apply_to_keypoints(self, keypoints, valid, height, width, **params):
        return _kp_vflip(keypoints, height), valid


class Transpose(DualTransform):
    """Swap rows and columns (main-diagonal reflection). Requires square images (a non-square batch would change
    shape only for the applied samples)."""

    is_geometric = True

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply_batch(self, batch: AugBatch, params: dict[str, Any]) -> AugBatch:
        _require_square(batch, "Transpose")
        return super().apply_batch(batch, params)

    def apply(self, img: torch.Tensor, **params: Any) -> torch.Tensor:
        return img.transpose(-1, -2)

    def apply_to_mask(self, mask: torch.Tensor, **params: Any) -> torch.Tensor:
        return mask.transpose(-1, -2)

    def apply_to_bboxes(self, bboxes, valid, height, width, **params):
        x1, y1, x2, y2 = bboxes.unbind(-1)
        return torch.stack([y1, x1, y2, x2], dim=-1), valid

    def apply_to_keypoints(self, keypoints, valid, height, width, **params):
        return _kp_transpose(keypoints), valid


# keypoint versions of the exact flips / rotations (albumentations keypoints_hflip / vflip / transpose / rot90; the
# angle is wrapped to [0, 2*pi) after each elementary operation like albumentations' angle_2pi_range)


def _kp(x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.stack([x, y, wrap_angle(a), s], dim=-1)


def _kp_hflip(kps: torch.Tensor, width: int) -> torch.Tensor:
    x, y, a, s = kps.unbind(-1)
    return _kp((width - 1) - x, y, math.pi - a, s)


def _kp_vflip(kps: torch.Tensor, height: int) -> torch.Tensor:
    x, y, a, s = kps.unbind(-1)
    return _kp(x, (height - 1) - y, -a, s)


def _kp_transpose(kps: torch.Tensor) -> torch.Tensor:
    x, y, a, s = kps.unbind(-1)
    a = wrap_angle(a)
    return _kp(y, x, torch.where(a <= math.pi, math.pi / 2 - a, 3 * math.pi / 2 - a), s)


def _kp_rot90(kps: torch.Tensor, k: int, height: int, width: int) -> torch.Tensor:
    """Counter-clockwise rotation by k*90 degrees (fixed k)."""
    x, y, a, s = kps.unbind(-1)
    k = k % 4
    if k == 1:
        return _kp(y, (width - 1) - x, a - math.pi / 2, s)
    if k == 2:
        return _kp((width - 1) - x, (height - 1) - y, a - math.pi, s)
    if k == 3:
        return _kp((height - 1) - y, x, a + math.pi / 2, s)
    return kps


def _select_per_sample(index: torch.Tensor, candidates: list[torch.Tensor]) -> torch.Tensor:
    """index (n,), candidates: list of (n, K, F) -> (n, K, F) picking candidates[index[i]] for sample i."""
    out = candidates[0]
    e = index.view(-1, 1, 1)
    for i in range(1, len(candidates)):
        out = torch.where(e == i, candidates[i], out)
    return out


def _rot90_boxes(boxes: torch.Tensor, k: torch.Tensor, size: int) -> torch.Tensor:
    """Counter-clockwise rotation by k*90 degrees of boxes in a size x size image (albumentations bboxes_rot90)."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    s = size
    r1 = torch.stack([y1, s - x2, y2, s - x1], dim=-1)
    r2 = torch.stack([s - x2, s - y2, s - x1, s - y1], dim=-1)
    r3 = torch.stack([s - y2, x1, s - y1, x2], dim=-1)
    kk = k.view(-1, 1, 1)
    out = torch.where(kk == 1, r1, boxes)
    out = torch.where(kk == 2, r2, out)
    return torch.where(kk == 3, r3, out)


class RandomRotate90(DualTransform):
    """Rotate by k * 90 degrees counter-clockwise, k uniform in {0, 1, 2, 3}. Requires square images."""

    is_geometric = True

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        return {"factor": rng.randint(0, 3, batch.batch_size)}

    def apply_batch(self, batch: AugBatch, params: dict[str, Any]) -> AugBatch:
        _require_square(batch, "RandomRotate90")
        return super().apply_batch(batch, params)

    def apply(self, img: torch.Tensor, factor: torch.Tensor, **params: Any) -> torch.Tensor:
        return fgeo.rot90(img, factor)

    def apply_to_mask(self, mask: torch.Tensor, factor: torch.Tensor, **params: Any) -> torch.Tensor:
        return fgeo.rot90(mask, factor)

    def apply_to_bboxes(self, bboxes, valid, height, width, factor=None, **params):
        return _rot90_boxes(bboxes, factor, width), valid

    def apply_to_keypoints(self, keypoints, valid, height, width, factor=None, **params):
        return _select_per_sample(factor, [_kp_rot90(keypoints, k, height, width) for k in range(4)]), valid


# D4 group elements in albumentations' order; each is expressed as (k rotations, transpose-after?)
_D4_ELEMENTS = ("e", "r90", "r180", "r270", "v", "hvt", "h", "t")


class D4(DualTransform):
    """One of the 8 symmetries of the square chosen uniformly (albumentations D4). Requires square images.
    Elements: e (identity), r90/r180/r270 (counter-clockwise rotations), v (vertical flip), h (horizontal flip),
    t (transpose), hvt (anti-transpose)."""

    is_geometric = True

    def __init__(self, p: float = 1.0):
        super().__init__(p)

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        return {"element": rng.randint(0, 7, batch.batch_size)}

    def apply_batch(self, batch: AugBatch, params: dict[str, Any]) -> AugBatch:
        _require_square(batch, "D4")
        return super().apply_batch(batch, params)

    @staticmethod
    def _apply_elem(x: torch.Tensor, name: str) -> torch.Tensor:
        if name == "e":
            return x
        if name == "r90":
            return torch.rot90(x, 1, dims=(-2, -1))
        if name == "r180":
            return torch.rot90(x, 2, dims=(-2, -1))
        if name == "r270":
            return torch.rot90(x, 3, dims=(-2, -1))
        if name == "v":
            return x.flip(-2)
        if name == "h":
            return x.flip(-1)
        if name == "t":
            return x.transpose(-1, -2)
        if name == "hvt":
            return x.flip(-1).flip(-2).transpose(-1, -2)
        raise ValueError(name)

    def _apply_all(self, x: torch.Tensor, element: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        for i, name in enumerate(_D4_ELEMENTS):
            sel = element == i
            if i > 0 and bool(sel.any()):
                out[sel] = self._apply_elem(x[sel], name)
        return out

    def apply(self, img: torch.Tensor, element: torch.Tensor, **params: Any) -> torch.Tensor:
        return self._apply_all(img, element)

    def apply_to_mask(self, mask: torch.Tensor, element: torch.Tensor, **params: Any) -> torch.Tensor:
        return self._apply_all(mask, element)

    def apply_to_bboxes(self, bboxes, valid, height, width, element=None, **params):
        s = width
        x1, y1, x2, y2 = bboxes.unbind(-1)
        cand = {
            "e": bboxes,
            "r90": torch.stack([y1, s - x2, y2, s - x1], -1),
            "r180": torch.stack([s - x2, s - y2, s - x1, s - y1], -1),
            "r270": torch.stack([s - y2, x1, s - y1, x2], -1),
            "v": torch.stack([x1, s - y2, x2, s - y1], -1),
            "h": torch.stack([s - x2, y1, s - x1, y2], -1),
            "t": torch.stack([y1, x1, y2, x2], -1),
            "hvt": torch.stack([s - y2, s - x2, s - y1, s - x1], -1),
        }
        out = bboxes
        e = element.view(-1, 1, 1)
        for i, name in enumerate(_D4_ELEMENTS):
            out = torch.where(e == i, cand[name], out)
        return out, valid

    def apply_to_keypoints(self, keypoints, valid, height, width, element=None, **params):
        k, h, w = keypoints, height, width
        cand = {
            "e": k,
            "r90": _kp_rot90(k, 1, h, w),
            "r180": _kp_rot90(k, 2, h, w),
            "r270": _kp_rot90(k, 3, h, w),
            "v": _kp_vflip(k, h),
            "hvt": _kp_transpose(_kp_rot90(k, 2, h, w)),
            "h": _kp_hflip(k, w),
            "t": _kp_transpose(k),
        }
        return _select_per_sample(element, [cand[name] for name in _D4_ELEMENTS]), valid


# ----------------------------------------------------------------------------------------------------------------
# matrix based warps
# ----------------------------------------------------------------------------------------------------------------


class _WarpBase(DualTransform):
    """Common options of the resampling transforms (albumentations names)."""

    is_geometric = True

    def __init__(
        self,
        interpolation: int | str = INTER_LINEAR,
        mask_interpolation: int | str = INTER_NEAREST,
        border_mode: int | str = BORDER_CONSTANT,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.interpolation = resolve_interpolation(interpolation)
        self.mask_interpolation = resolve_interpolation(mask_interpolation)
        self.border_mode = resolve_border_mode(border_mode)
        self.fill = fill
        self.fill_mask = fill_mask

    def _fill_image(self, max_value: float) -> list[float]:
        """``fill`` is in the value scale of the input images like albumentations (0..255 for uint8 input, 0..1 for
        float input)."""
        return (torch.as_tensor(self.fill, dtype=torch.float32).flatten() / max_value).tolist()


class _MatrixWarp(_WarpBase):
    rotate_method: str = "largest_box"

    def apply(self, img: torch.Tensor, matrix: torch.Tensor, max_value: float = 1.0, **params: Any) -> torch.Tensor:
        return fgeo.warp(img, matrix, None, self.interpolation, self.border_mode, self._fill_image(max_value))

    def apply_to_mask(self, mask: torch.Tensor, matrix: torch.Tensor, **params: Any) -> torch.Tensor:
        return fgeo.warp(mask, matrix, None, self.mask_interpolation, self.border_mode, self.fill_mask)

    def apply_to_bboxes(self, bboxes, valid, height, width, matrix=None, **params):
        m = pixel_center_to_edge_matrix(matrix)
        return transform_boxes_by_matrix(bboxes, m, self.rotate_method), valid

    def apply_to_keypoints(self, keypoints, valid, height, width, matrix=None, scale=None, **params):
        """albumentations ``keypoints_affine``: points through the (pixel-center) matrix, angle += atan2(-m10, m00),
        scale *= max(scale_x, scale_y) of the sampled scale (1 for Rotate). Not reproduced: the extra reflected
        keypoints albumentations generates for reflect border modes."""
        m = matrix.to(torch.float64)
        xy = transform_points_by_matrix(keypoints[..., :2], m)
        da = torch.atan2(-m[:, 1, 0], m[:, 0, 0]).to(keypoints.dtype)[:, None]
        ds = scale.amax(dim=1).to(keypoints.dtype)[:, None] if scale is not None else 1.0
        out = torch.cat([xy, (keypoints[..., 2] + da)[..., None], (keypoints[..., 3] * ds)[..., None]], dim=-1)
        # albumentations returns the keypoints untouched for an identity matrix
        eye = torch.eye(3, dtype=m.dtype, device=m.device)
        identity = torch.isclose(m, eye.expand_as(m)).flatten(1).all(dim=1)
        return torch.where(identity.view(-1, 1, 1), keypoints, out), valid


def _parse_xy(value: Any, default: float, name: str) -> dict[str, tuple[float, float]]:
    """albumentations Affine ``_handle_dict_arg``: float -> fixed for both axes, pair -> same range for both axes,
    dict {"x": ..., "y": ...} -> per axis (missing axis -> default)."""
    if isinstance(value, dict):
        if "x" not in value and "y" not in value:
            raise ValueError(f'{name} dict must contain "x" and/or "y"')
        return {k: fixed_or_range(value.get(k, default), name) for k in ("x", "y")}
    r = fixed_or_range(value, name)
    return {"x": r, "y": r}


class Affine(_MatrixWarp):
    """albumentations ``Affine`` (same parameters and parameter distributions). ``fit_output=True`` is not supported
    (it changes the output size per sample).

    - scale: float (fixed), (min, max), or {"x": ..., "y": ...}
    - translate_percent / translate_px: same forms; percentage is truncated to whole pixels like albumentations
    - rotate: degrees, float (fixed) or (min, max)
    - shear: degrees, float, (min, max) or dict per axis
    - keep_ratio: use the x scale for both axes; balanced_scale: pick the <1 or >1 sub-interval with equal probability
    - rotate_method: "largest_box" or "ellipse" for boxes
    """

    def __init__(
        self,
        scale: float | Sequence[float] | dict = (1.0, 1.0),
        translate_percent: float | Sequence[float] | dict | None = None,
        translate_px: int | Sequence[int] | dict | None = None,
        rotate: float | Sequence[float] = 0.0,
        shear: float | Sequence[float] | dict = (0.0, 0.0),
        interpolation: int | str = INTER_LINEAR,
        mask_interpolation: int | str = INTER_NEAREST,
        fit_output: bool = False,
        keep_ratio: bool = False,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        balanced_scale: bool = False,
        border_mode: int | str = BORDER_CONSTANT,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        super().__init__(interpolation, mask_interpolation, border_mode, fill, fill_mask, p)
        if fit_output:
            raise NotImplementedError("fit_output=True changes the output size per sample and is not supported.")
        if translate_percent is not None and translate_px is not None:
            raise ValueError("Use either translate_percent or translate_px, not both.")
        self.scale = _parse_xy(scale, 1.0, "scale")
        self.translate_percent = None if translate_percent is None else _parse_xy(translate_percent, 0.0, "translate_percent")
        self.translate_px = _parse_xy(translate_px if translate_px is not None else 0, 0, "translate_px") if translate_percent is None else None
        self.rotate = fixed_or_range(rotate, "rotate")
        self.shear = _parse_xy(shear, 0.0, "shear")
        self.fit_output = fit_output
        self.keep_ratio = keep_ratio
        self.rotate_method = rotate_method
        self.balanced_scale = balanced_scale

    def _sample_scale(self, rng: Rng, n: int) -> torch.Tensor:
        out = []
        for key in ("x", "y"):
            lo, hi = self.scale[key]
            if self.balanced_scale and lo != hi:
                lower = (lo, 1.0) if lo < 1 else None
                upper = (1.0, hi) if hi > 1 else None
                if lower and upper:
                    pick_upper = rng.bernoulli(0.5, n)
                    a = torch.where(pick_upper, torch.full((n,), 1.0, device=rng.device), torch.full((n,), lo, device=rng.device))
                    b = torch.where(pick_upper, torch.full((n,), hi, device=rng.device), torch.full((n,), 1.0, device=rng.device))
                    out.append(rng.uniform(a, b, n))
                    continue
                if lower or upper:
                    a, b = lower or upper
                    out.append(rng.uniform(a, b, n))
                    continue
                out.append(torch.ones(n, device=rng.device))
                continue
            out.append(rng.uniform(lo, hi, n))
        s = torch.stack(out, dim=1)
        if self.keep_ratio:
            s[:, 1] = s[:, 0]
        return s

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, h, w = batch.batch_size, batch.height, batch.width
        if self.translate_px is not None:
            tx = rng.randint(int(self.translate_px["x"][0]), int(self.translate_px["x"][1]), n)
            ty = rng.randint(int(self.translate_px["y"][0]), int(self.translate_px["y"][1]), n)
        else:
            tx = torch.trunc(rng.uniform(*self.translate_percent["x"], n) * w)
            ty = torch.trunc(rng.uniform(*self.translate_percent["y"], n) * h)
        translate = torch.stack([tx, ty], dim=1).float()
        shear = -torch.stack([rng.uniform(*self.shear["x"], n), rng.uniform(*self.shear["y"], n)], dim=1)
        scale = self._sample_scale(rng, n)
        rotate = rng.uniform(*self.rotate, n)
        matrix = fgeo.affine_matrix(translate, shear, scale, rotate, (w / 2 - 0.5, h / 2 - 0.5))
        return {"matrix": matrix, "scale": scale}


class ShiftScaleRotate(Affine):
    """albumentations ``ShiftScaleRotate`` (a special case of Affine): shift as a fraction of the size, scale as
    1 + scale_limit, rotation in degrees. Scalars mean symmetric ranges."""

    def __init__(
        self,
        shift_limit: float | Sequence[float] = (-0.0625, 0.0625),
        scale_limit: float | Sequence[float] = (-0.1, 0.1),
        rotate_limit: float | Sequence[float] = (-45, 45),
        interpolation: int | str = INTER_LINEAR,
        border_mode: int | str = BORDER_CONSTANT,
        shift_limit_x: float | Sequence[float] | None = None,
        shift_limit_y: float | Sequence[float] | None = None,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        mask_interpolation: int | str = INTER_NEAREST,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        sl = symmetric_range(shift_limit, "shift_limit")
        sx = symmetric_range(shift_limit_x, "shift_limit_x") if shift_limit_x is not None else sl
        sy = symmetric_range(shift_limit_y, "shift_limit_y") if shift_limit_y is not None else sl
        check_bounds(sx, -1, 1, "shift_limit_x")
        check_bounds(sy, -1, 1, "shift_limit_y")
        sc = symmetric_range(scale_limit, "scale_limit")
        super().__init__(
            scale=(1 + sc[0], 1 + sc[1]),
            translate_percent={"x": sx, "y": sy},
            rotate=symmetric_range(rotate_limit, "rotate_limit"),
            shear=(0, 0),
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            border_mode=border_mode,
            fill=fill,
            fill_mask=fill_mask,
            rotate_method=rotate_method,
            p=p,
        )
        self.shift_limit_x, self.shift_limit_y = sx, sy
        self.scale_limit = sc
        self.rotate_limit = symmetric_range(rotate_limit, "rotate_limit")


class Rotate(_MatrixWarp):
    """albumentations ``Rotate``: angle ~ U(limit) degrees around the image center. ``crop_border=True`` is not
    supported (it changes the output size per sample)."""

    def __init__(
        self,
        limit: float | Sequence[float] = (-90, 90),
        interpolation: int | str = INTER_LINEAR,
        border_mode: int | str = BORDER_CONSTANT,
        rotate_method: Literal["largest_box", "ellipse"] = "largest_box",
        crop_border: bool = False,
        mask_interpolation: int | str = INTER_NEAREST,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        super().__init__(interpolation, mask_interpolation, border_mode, fill, fill_mask, p)
        if crop_border:
            raise NotImplementedError("crop_border=True changes the output size per sample and is not supported.")
        self.limit = symmetric_range(limit, "limit")
        self.rotate_method = rotate_method
        self.crop_border = crop_border

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, h, w = batch.batch_size, batch.height, batch.width
        angle = rng.uniform(*self.limit, n)
        zeros = torch.zeros(n, 2, device=batch.device)
        ones = torch.ones(n, 2, device=batch.device)
        return {"matrix": fgeo.affine_matrix(zeros, zeros, ones, angle, (w / 2 - 0.5, h / 2 - 0.5))}


class Perspective(_MatrixWarp):
    """albumentations ``Perspective`` with ``keep_size=True``, ``fit_output=False`` (the other settings change the
    output size per sample and are not supported). Corner jitter ~ |N(0, scale)| mod 0.32 of the image size."""

    def __init__(
        self,
        scale: float | Sequence[float] = (0.05, 0.1),
        keep_size: bool = True,
        fit_output: bool = False,
        interpolation: int | str = INTER_LINEAR,
        mask_interpolation: int | str = INTER_NEAREST,
        border_mode: int | str = BORDER_CONSTANT,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        super().__init__(interpolation, mask_interpolation, border_mode, fill, fill_mask, p)
        if not keep_size or fit_output:
            raise NotImplementedError("Only keep_size=True, fit_output=False is supported (fixed output size).")
        self.scale = check_nondecreasing(to_range(scale, 0.0, "scale"), "scale")
        self.keep_size = keep_size
        self.fit_output = fit_output

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, h, w = batch.batch_size, batch.height, batch.width
        dev = batch.device
        scale = rng.uniform(*self.scale, n)
        pts = torch.remainder(rng.normal(0.0, scale.view(n, 1, 1).double(), (n, 4, 2)).abs(), 0.32)
        pts[:, 1, 0] = 1.0 - pts[:, 1, 0]
        pts[:, 2] = 1.0 - pts[:, 2]
        pts[:, 3, 1] = 1.0 - pts[:, 3, 1]
        pts = pts * torch.tensor([w, h], dtype=pts.dtype, device=dev)
        pts = _order_points(pts)
        tl, tr, br, bl = pts.unbind(1)

        def dist(a, b):
            return torch.sqrt(((a - b) ** 2).sum(-1)).clamp(min=2.0)

        max_w = torch.maximum(dist(tr, tl), dist(br, bl)).floor()
        max_h = torch.maximum(dist(br, tr), dist(bl, tl)).floor()
        dst = torch.tensor([[0, 0], [w, 0], [w, h], [0, h]], dtype=pts.dtype, device=dev).expand(n, 4, 2)
        m = fgeo.perspective_transform_from_points(pts, dst)
        s = torch.zeros(n, 3, 3, dtype=m.dtype, device=dev)
        s[:, 0, 0] = w / max_w
        s[:, 1, 1] = h / max_h
        s[:, 2, 2] = 1.0
        return {"matrix": s @ m, "bbox_matrix": s @ m, "kp_matrix": m, "kp_resize": torch.stack([s[:, 0, 0], s[:, 1, 1]], 1)}

    def apply_to_bboxes(self, bboxes, valid, height, width, matrix=None, bbox_matrix=None, **params):
        # albumentations applies the same matrix to box (pixel-edge) coordinates for Perspective
        return transform_boxes_by_matrix(bboxes, bbox_matrix, "largest_box"), valid

    def apply_to_keypoints(self, keypoints, valid, height, width, kp_matrix=None, kp_resize=None, **params):
        """albumentations ``perspective_keypoints``: points through M, angle += atan2(m10, m00),
        scale *= max(sign(m00) |row0|, sign(m11) |row1|), then the keep_size rescale (x, y, scale)."""
        m = kp_matrix.to(torch.float64)
        xy = transform_points_by_matrix(keypoints[..., :2], m)
        r = kp_resize.to(torch.float64)
        xy = xy * r[:, None, :].to(xy.dtype)
        da = torch.atan2(m[:, 1, 0], m[:, 0, 0])
        sx = torch.sign(m[:, 0, 0]) * torch.sqrt(m[:, 0, 0] ** 2 + m[:, 0, 1] ** 2)
        sy = torch.sign(m[:, 1, 1]) * torch.sqrt(m[:, 1, 0] ** 2 + m[:, 1, 1] ** 2)
        ds = torch.maximum(sx, sy) * r.amax(dim=1)
        a = keypoints[..., 2] + da.to(keypoints.dtype)[:, None]
        s = keypoints[..., 3] * ds.to(keypoints.dtype)[:, None]
        return torch.cat([xy, a[..., None], s[..., None]], dim=-1), valid


def _order_points(pts: torch.Tensor) -> torch.Tensor:
    """albumentations ``order_points`` (batched): sort by x; left pair -> (tl, bl) by y, right pair -> (tr, br) by y."""
    order = torch.argsort(pts[..., 0], dim=1, stable=True)
    s = torch.gather(pts, 1, order[..., None].expand(-1, -1, 2))
    left, right = s[:, :2], s[:, 2:]
    l_swap = left[:, 0, 1] >= left[:, 1, 1]
    r_swap = right[:, 0, 1] >= right[:, 1, 1]
    tl = torch.where(l_swap[:, None], left[:, 1], left[:, 0])
    bl = torch.where(l_swap[:, None], left[:, 0], left[:, 1])
    tr = torch.where(r_swap[:, None], right[:, 1], right[:, 0])
    br = torch.where(r_swap[:, None], right[:, 0], right[:, 1])
    return torch.stack([tl, tr, br, bl], dim=1)


# ----------------------------------------------------------------------------------------------------------------
# dense distortions (remap)
# ----------------------------------------------------------------------------------------------------------------


class _RemapBase(_WarpBase):
    """Transforms defined by per-pixel source maps. Boxes are transformed like albumentations
    (``keypoint_remapping_method="mask"``): each box is rasterized to a mask, remapped with nearest interpolation and
    the extent of the remaining pixels is the new box. Difference: a box whose mask vanishes completely is marked
    invalid (albumentations keeps the original coordinates).

    Keypoints follow albumentations' default ``keypoint_remapping_method="mask"``: each keypoint is drawn as a radius-1
    disk (center + 4 neighbours) at its rounded position into a label image (later keypoints overwrite earlier ones),
    the label image is remapped with nearest interpolation and the first pixel (row-major) carrying the label is the
    new integer position; keypoints that do not survive are marked invalid (albumentations drops them, regardless of
    ``remove_invisible``). ``keypoint_remapping_method="direct"`` is not supported."""

    #: limit of (boxes x H x W) elements processed at once when remapping box masks
    box_chunk_elements: int = 2**27

    def apply(self, img: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor, max_value: float = 1.0, **params: Any) -> torch.Tensor:
        return fgeo.sample(img, map_x, map_y, self.interpolation, self.border_mode, self._fill_image(max_value))

    def apply_to_mask(self, mask: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor, **params: Any) -> torch.Tensor:
        return fgeo.sample(mask, map_x, map_y, self.mask_interpolation, self.border_mode, self.fill_mask)

    def apply_to_bboxes(self, bboxes, valid, height, width, map_x=None, map_y=None, **params):
        n, nb = bboxes.shape[:2]
        if nb == 0:
            return bboxes, valid
        out = bboxes.clone()
        nonempty_all = torch.zeros_like(valid)
        per_box = n * height * width
        step = max(1, self.box_chunk_elements // max(1, per_box))
        for s in range(0, nb, step):
            masks = boxes_to_masks(bboxes[:, s : s + step], height, width).float()  # (n, k, H, W)
            k = masks.shape[1]
            warped = fgeo.sample(masks, map_x, map_y, INTER_NEAREST, BORDER_CONSTANT, 0.0) > 0.5
            boxes_k, nonempty = masks_to_boxes(warped)
            out[:, s : s + k] = torch.where(nonempty[..., None], boxes_k, bboxes[:, s : s + k])
            nonempty_all[:, s : s + k] = nonempty
        return out, valid & nonempty_all

    def apply_to_keypoints(self, keypoints, valid, height, width, map_x=None, map_y=None, **params):
        n, k = keypoints.shape[:2]
        if k == 0:
            return keypoints, valid
        dev = keypoints.device
        h, w = height, width
        cx = torch.round(keypoints[..., 0]).long()
        cy = torch.round(keypoints[..., 1]).long()
        drawn = valid & (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        # label image: value = keypoint index + 1, the largest index wins (= drawn last)
        labels = torch.zeros(n, h * w, dtype=torch.long, device=dev)
        idx = torch.arange(1, k + 1, device=dev).expand(n, k)
        for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            px, py = cx + dx, cy + dy
            ok = drawn & (px >= 0) & (px < w) & (py >= 0) & (py < h)
            flat = (py.clamp(0, h - 1) * w + px.clamp(0, w - 1))
            labels.scatter_reduce_(1, flat, torch.where(ok, idx, torch.zeros_like(idx)), reduce="amax")
        warped = fgeo.sample(labels.view(n, 1, h, w).float(), map_x, map_y, INTER_NEAREST, BORDER_CONSTANT, 0.0)
        warped = warped.view(n, h * w).round().long()
        # first (row-major) pixel of every label
        pix = torch.arange(h * w, device=dev).expand(n, h * w)
        first = torch.full((n, k + 1), h * w, dtype=torch.long, device=dev)
        first.scatter_reduce_(1, warped, pix, reduce="amin")
        first = first[:, 1:]
        found = first < h * w
        new_x = (first % w).to(keypoints.dtype)
        new_y = torch.div(first, w, rounding_mode="floor").to(keypoints.dtype)
        out = keypoints.clone()
        out[..., 0] = torch.where(found, new_x, keypoints[..., 0])
        out[..., 1] = torch.where(found, new_y, keypoints[..., 1])
        return out, valid & found


class ElasticTransform(_RemapBase):
    """albumentations ``ElasticTransform``: random displacement fields (gaussian noise normalized by its max abs, or
    uniform in [-1, 1]) smoothed with a Gaussian of ``sigma`` (BORDER_REPLICATE; kernel 17x17 if ``approximate`` else
    cv2's automatic size) and scaled by ``alpha`` pixels."""

    def __init__(
        self,
        alpha: float = 1,
        sigma: float = 50,
        interpolation: int | str = INTER_LINEAR,
        approximate: bool = False,
        same_dxdy: bool = False,
        mask_interpolation: int | str = INTER_NEAREST,
        noise_distribution: Literal["gaussian", "uniform"] = "gaussian",
        border_mode: int | str = BORDER_CONSTANT,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        super().__init__(interpolation, mask_interpolation, border_mode, fill, fill_mask, p)
        if alpha < 0 or sigma < 1:
            raise ValueError("alpha must be >= 0 and sigma >= 1")
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.approximate = approximate
        self.same_dxdy = same_dxdy
        self.noise_distribution = noise_distribution

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, h, w = batch.batch_size, batch.height, batch.width
        nf = 1 if self.same_dxdy else 2
        if self.noise_distribution == "gaussian":
            fields = rng.randn((n, nf, h, w))
            max_abs = fields.abs().amax(dim=(1, 2, 3), keepdim=True)
            fields = torch.where(max_abs > 1e-6, fields / max_abs.clamp(min=1e-12), fields)
        else:
            fields = rng.uniform(-1.0, 1.0, (n, nf, h, w))
        ksize = 17 if self.approximate else 0
        fields = fgeo.gaussian_blur_fields(fields.view(n * nf, h, w), self.sigma, ksize).view(n, nf, h, w) * self.alpha
        dx = fields[:, 0]
        dy = fields[:, 0] if self.same_dxdy else fields[:, 1]
        gx, gy = fgeo.identity_grid(n, h, w, batch.device)
        return {"map_x": gx + dx, "map_y": gy + dy}


class GridDistortion(_RemapBase):
    """albumentations ``GridDistortion``: the image is split into ``num_steps`` cells per axis whose sizes are
    multiplied by 1 + U(distort_limit); with ``normalized`` the steps are rescaled so the distortion stays within
    the image."""

    def __init__(
        self,
        num_steps: int = 5,
        distort_limit: float | Sequence[float] = (-0.3, 0.3),
        interpolation: int | str = INTER_LINEAR,
        normalized: bool = True,
        mask_interpolation: int | str = INTER_NEAREST,
        border_mode: int | str = BORDER_CONSTANT,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        super().__init__(interpolation, mask_interpolation, border_mode, fill, fill_mask, p)
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        self.num_steps = int(num_steps)
        self.distort_limit = check_bounds(symmetric_range(distort_limit, "distort_limit"), -1, 1, "distort_limit")
        self.normalized = normalized

    def _axis_map(self, steps: torch.Tensor, size: int) -> torch.Tensor:
        """albumentations ``generate_grid`` for one axis, batched: steps (n, num_steps+1) -> coords (n, size)."""
        n = steps.shape[0]
        step = size // self.num_steps
        out = torch.zeros(n, size, dtype=torch.float32, device=steps.device)
        prev = torch.zeros(n, dtype=torch.float32, device=steps.device)
        for idx in range(self.num_steps + 1):
            start = idx * step
            end = min(start + step, size)
            cur = prev + step * steps[:, idx]
            length = end - start
            if length > 0:
                t = torch.linspace(0, 1, length, device=steps.device) if length > 1 else torch.zeros(1, device=steps.device)
                out[:, start:end] = prev[:, None] + (cur - prev)[:, None] * t[None]
            prev = cur
        return out

    def _normalize(self, steps: torch.Tensor, size: int) -> torch.Tensor:
        step = size // self.num_steps
        last = min(size, (self.num_steps + 1) * step) - self.num_steps * step
        steps = steps.clone()
        steps[:, -1] = steps[:, -1] * (last / step)
        total = size / math.floor(size / self.num_steps)
        return steps * (total / steps.sum(dim=1, keepdim=True))

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, h, w = batch.batch_size, batch.height, batch.width
        sx = 1 + rng.uniform(*self.distort_limit, (n, self.num_steps + 1))
        sy = 1 + rng.uniform(*self.distort_limit, (n, self.num_steps + 1))
        if self.normalized:
            sx, sy = self._normalize(sx, w), self._normalize(sy, h)
        xx = self._axis_map(sx, w)
        yy = self._axis_map(sy, h)
        return {"map_x": xx[:, None, :].expand(n, h, w), "map_y": yy[:, :, None].expand(n, h, w)}


class OpticalDistortion(_RemapBase):
    """albumentations ``OpticalDistortion``. mode="camera": cv2.initUndistortRectifyMap with camera matrix
    [[W, 0, W/2], [0, H, H/2]], distortion coefficients (k, k, 0, 0, 0) and cv2's default new camera matrix (same
    focal lengths, principal point ((W-1)/2, (H-1)/2)); mode="fisheye": radial r * (1 + k * r_norm^2)."""

    def __init__(
        self,
        distort_limit: float | Sequence[float] = (-0.05, 0.05),
        interpolation: int | str = INTER_LINEAR,
        mask_interpolation: int | str = INTER_NEAREST,
        mode: Literal["camera", "fisheye"] = "camera",
        border_mode: int | str = BORDER_CONSTANT,
        fill: float | Sequence[float] = 0,
        fill_mask: float | Sequence[float] = 0,
        p: float = 0.5,
    ):
        super().__init__(interpolation, mask_interpolation, border_mode, fill, fill_mask, p)
        self.distort_limit = symmetric_range(distort_limit, "distort_limit")
        if mode not in ("camera", "fisheye"):
            raise ValueError("mode must be 'camera' or 'fisheye'")
        self.mode = mode

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, h, w = batch.batch_size, batch.height, batch.width
        k = rng.uniform(*self.distort_limit, n).double().view(n, 1, 1)
        dev = batch.device
        ys = torch.arange(h, dtype=torch.float64, device=dev)
        xs = torch.arange(w, dtype=torch.float64, device=dev)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        cx, cy = w / 2, h / 2
        if self.mode == "camera":
            x = (xx - (w - 1) / 2) / w
            y = (yy - (h - 1) / 2) / h
            r2 = x * x + y * y
            radial = 1 + k * r2 + k * r2 * r2
            map_x = w * (x * radial) + cx
            map_y = h * (y * radial) + cy
        else:
            x = xx - cx
            y = yy - cy
            r = torch.sqrt(x * x + y * y)
            theta = torch.atan2(y, x)
            max_r = math.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2)
            r_dist = r * (1 + k * (r / max_r) ** 2)
            map_x = r_dist * torch.cos(theta) + cx
            map_y = r_dist * torch.sin(theta) + cy
        return {"map_x": map_x.float().expand(n, h, w), "map_y": map_y.float().expand(n, h, w)}

