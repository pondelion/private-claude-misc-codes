"""Batched resampling and transformation-matrix helpers.

Coordinate convention (same as OpenCV warps): pixel (row i, col j) is at integer coordinates (x=j, y=i); the image
spans [-0.5, W-0.5]. Matrices map **source -> destination** (like the matrices passed to ``cv2.warpAffine``) and are
inverted internally to sample the source for every output pixel.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..core.constants import (
    BORDER_CONSTANT,
    BORDER_REFLECT,
    BORDER_REFLECT_101,
    BORDER_REPLICATE,
    BORDER_WRAP,
    INTER_CUBIC,
    INTER_LINEAR,
    INTER_NEAREST,
)

_MODES = {INTER_NEAREST: "nearest", INTER_LINEAR: "bilinear", INTER_CUBIC: "bicubic"}


def _fold_coords(c: torch.Tensor, size: int, border_mode: int) -> torch.Tensor:
    """Map out-of-range source coordinates back into [0, size-1] following the border mode."""
    if size == 1:
        return torch.zeros_like(c)
    if border_mode == BORDER_REPLICATE:
        return c.clamp(0, size - 1)
    if border_mode == BORDER_REFLECT_101:  # ...cba|abcd|dcb...  (reflection about pixel centers 0 and size-1)
        period = 2 * (size - 1)
        c = torch.remainder(c, period)
        return torch.where(c > size - 1, period - c, c)
    if border_mode == BORDER_REFLECT:  # ...cba|abcd|dcba...  (reflection about the pixel edges)
        period = 2 * size
        u = torch.remainder(c + 0.5, period)
        u = torch.where(u > size, period - u, u)
        return (u - 0.5).clamp(0, size - 1)
    if border_mode == BORDER_WRAP:
        return (torch.remainder(c + 0.5, size) - 0.5).clamp(0, size - 1)
    raise ValueError(border_mode)


def fill_tensor(fill: float | Sequence[float], channels: int, device: torch.device) -> torch.Tensor:
    f = torch.as_tensor(fill, dtype=torch.float32, device=device).flatten()
    if f.numel() == 1:
        f = f.expand(channels)
    if f.numel() != channels:
        raise ValueError(f"fill has {f.numel()} values but the tensor has {channels} channels")
    return f.view(1, channels, 1, 1)


def sample(
    img: torch.Tensor,
    src_x: torch.Tensor,
    src_y: torch.Tensor,
    interpolation: int = INTER_LINEAR,
    border_mode: int = BORDER_CONSTANT,
    fill: float | Sequence[float] = 0.0,
) -> torch.Tensor:
    """Sample ``img`` (n, C, H, W) at source coordinates ``src_x``/``src_y`` (n, Ho, Wo) (cv2 ``remap`` semantics).
    Non-float tensors (masks) are sampled in float32 and cast back (rounded for integer dtypes)."""
    n, c, h, w = img.shape
    dtype = img.dtype
    x = img if dtype == torch.float32 else img.float()
    mode = _MODES[interpolation]
    if border_mode != BORDER_CONSTANT:
        src_x = _fold_coords(src_x, w, border_mode)
        src_y = _fold_coords(src_y, h, border_mode)
    gx = src_x * (2.0 / (w - 1)) - 1.0 if w > 1 else torch.zeros_like(src_x)
    gy = src_y * (2.0 / (h - 1)) - 1.0 if h > 1 else torch.zeros_like(src_y)
    grid = torch.stack([gx, gy], dim=-1)
    pad = "zeros" if border_mode == BORDER_CONSTANT else "border"
    out = F.grid_sample(x, grid, mode=mode, padding_mode=pad, align_corners=True)
    if border_mode == BORDER_CONSTANT:
        f = fill_tensor(fill, c, img.device)
        if bool((f != 0).any()):
            weight = F.grid_sample(torch.ones_like(x[:, :1]), grid, mode=mode, padding_mode="zeros", align_corners=True)
            out = out + (1.0 - weight) * f
    if dtype == torch.float32:
        return out
    if dtype.is_floating_point:
        return out.to(dtype)
    if dtype == torch.bool:
        return out > 0.5
    return out.round().to(dtype)


def coords_from_matrix(matrix: torch.Tensor, out_h: int, out_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Source coordinates of every output pixel for a forward (source -> destination) 3x3 matrix (affine or
    projective). Returns (src_x, src_y), each (n, out_h, out_w) float32."""
    inv = torch.linalg.inv(matrix.to(torch.float64))
    dev = matrix.device
    ys = torch.arange(out_h, dtype=torch.float64, device=dev)
    xs = torch.arange(out_w, dtype=torch.float64, device=dev)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    a = inv[:, :, None, None, :]  # (n, 3, 1, 1, 3)
    pts = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # (Ho, Wo, 3)
    src = (a * pts).sum(-1)  # (n, 3, Ho, Wo)
    w = src[:, 2]
    return (src[:, 0] / w).float(), (src[:, 1] / w).float()


def warp(
    img: torch.Tensor,
    matrix: torch.Tensor,
    out_hw: tuple[int, int] | None = None,
    interpolation: int = INTER_LINEAR,
    border_mode: int = BORDER_CONSTANT,
    fill: float | Sequence[float] = 0.0,
) -> torch.Tensor:
    """``cv2.warpAffine`` / ``cv2.warpPerspective`` for a batch with per-sample 3x3 forward matrices (n, 3, 3)."""
    out_h, out_w = out_hw if out_hw is not None else img.shape[-2:]
    sx, sy = coords_from_matrix(matrix, out_h, out_w)
    return sample(img, sx, sy, interpolation, border_mode, fill)


# --------------------------------------------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------------------------------------------


def affine_matrix(
    translate: torch.Tensor,
    shear_deg: torch.Tensor,
    scale: torch.Tensor,
    rotate_deg: torch.Tensor,
    center: tuple[float, float],
) -> torch.Tensor:
    """albumentations ``create_affine_transformation_matrix`` (batched).
    translate/shear_deg/scale: (n, 2) as (x, y); rotate_deg: (n,); center: (cx, cy).
    M = T(center) @ T(translate) @ Shear @ Rotate @ Scale @ T(-center)."""
    n = rotate_deg.shape[0]
    dt = torch.float64
    dev = rotate_deg.device
    rot = torch.deg2rad(torch.remainder(rotate_deg.to(dt), 360.0))
    cos, sin = torch.cos(rot), torch.sin(rot)
    eye = torch.eye(3, dtype=dt, device=dev).expand(n, 3, 3)

    def mat():
        return eye.clone()

    t_c, t_mc, tr, sh, r, s = mat(), mat(), mat(), mat(), mat(), mat()
    t_c[:, 0, 2], t_c[:, 1, 2] = center[0], center[1]
    t_mc[:, 0, 2], t_mc[:, 1, 2] = -center[0], -center[1]
    tr[:, 0, 2], tr[:, 1, 2] = translate[:, 0].to(dt), translate[:, 1].to(dt)
    sh[:, 0, 1] = torch.tan(torch.deg2rad(shear_deg[:, 0].to(dt)))
    sh[:, 1, 0] = torch.tan(torch.deg2rad(shear_deg[:, 1].to(dt)))
    r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1] = cos, sin, -sin, cos
    s[:, 0, 0], s[:, 1, 1] = scale[:, 0].to(dt), scale[:, 1].to(dt)
    return t_c @ tr @ sh @ r @ s @ t_mc


def perspective_transform_from_points(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """``cv2.getPerspectiveTransform`` (batched). src, dst: (n, 4, 2) -> (n, 3, 3) float64."""
    src = src.to(torch.float64)
    dst = dst.to(torch.float64)
    n = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    rows_u = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    rows_v = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
    a = torch.cat([rows_u, rows_v], dim=1)  # (n, 8, 8)
    b = torch.cat([u, v], dim=1)  # (n, 8)
    sol = torch.linalg.solve(a, b)
    return torch.cat([sol, torch.ones(n, 1, dtype=sol.dtype, device=sol.device)], dim=1).view(n, 3, 3)


def identity_grid(n: int, h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx.expand(n, h, w), yy.expand(n, h, w)


# --------------------------------------------------------------------------------------------------------------
# exact (index based) operations
# --------------------------------------------------------------------------------------------------------------


def rot90(img: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Per-sample ``np.rot90(img, k)`` (counter-clockwise) for square images. k: (n,) in {0,1,2,3}."""
    out = img.clone()
    for kk in range(1, 4):
        sel = k == kk
        if bool(sel.any()):
            out[sel] = torch.rot90(img[sel], kk, dims=(-2, -1))
    return out


def crop_resize_coords(
    crop: torch.Tensor, out_h: int, out_w: int, interpolation: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source coordinates for ``cv2.resize(img[y1:y2, x1:x2], (out_w, out_h))`` per sample.
    crop: (n, 4) int [x1, y1, x2, y2]. Linear/cubic use the half-pixel mapping of cv2.resize; nearest uses cv2's
    ``floor(dst * scale)`` mapping."""
    crop = crop.to(torch.float32)
    x1, y1, x2, y2 = crop.unbind(-1)
    sx = ((x2 - x1) / out_w)[:, None]
    sy = ((y2 - y1) / out_h)[:, None]
    dev = crop.device
    xs = torch.arange(out_w, dtype=torch.float32, device=dev)[None]
    ys = torch.arange(out_h, dtype=torch.float32, device=dev)[None]
    if interpolation == INTER_NEAREST:
        cx = torch.minimum(torch.floor(xs * sx), (x2 - x1 - 1)[:, None]) + x1[:, None]
        cy = torch.minimum(torch.floor(ys * sy), (y2 - y1 - 1)[:, None]) + y1[:, None]
    else:
        # half-pixel mapping, clamped to the crop like cv2.resize does at the borders
        cx = ((xs + 0.5) * sx - 0.5).clamp(min=0)
        cx = torch.minimum(cx, (x2 - x1 - 1)[:, None]) + x1[:, None]
        cy = ((ys + 0.5) * sy - 0.5).clamp(min=0)
        cy = torch.minimum(cy, (y2 - y1 - 1)[:, None]) + y1[:, None]
    n = crop.shape[0]
    return cx[:, None, :].expand(n, out_h, out_w), cy[:, :, None].expand(n, out_h, out_w)


def gaussian_blur_fields(fields: torch.Tensor, sigma: float, ksize: int) -> torch.Tensor:
    """cv2.GaussianBlur((n, H, W) float32 fields, (ksize, ksize), sigma, borderType=BORDER_REPLICATE).
    ksize=0 -> cv2's automatic size for float images: round(sigma * 4 * 2 + 1) | 1."""
    from .filters import cv2_gaussian_kernel_1d, separable_filter

    if ksize <= 0:
        ksize = int(round(sigma * 4 * 2 + 1)) | 1
    k = cv2_gaussian_kernel_1d(ksize, sigma, fields.device)
    return separable_filter(fields[:, None], k, k, border="replicate")[:, 0]


__all__ = [
    "affine_matrix",
    "coords_from_matrix",
    "crop_resize_coords",
    "fill_tensor",
    "gaussian_blur_fields",
    "identity_grid",
    "perspective_transform_from_points",
    "rot90",
    "sample",
    "warp",
]
