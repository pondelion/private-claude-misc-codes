"""Keypoint handling.

Like boxes, keypoints are carried as a padded tensor ``(B, K, F)`` plus a validity mask ``(B, K)``; invisible keypoints
are marked ``valid=False`` instead of being removed.

Internally keypoints are ``(x, y, angle, scale)`` with pixel-center coordinates (the center of pixel (row i, col j) is
(x=j, y=i), same as albumentations), angle in radians and normalized to [0, 2*pi) after every transform. Columns of
the input beyond the format (e.g. labels) are passed through unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

KeypointFormat = Literal["xy", "yx", "xya", "xys", "xyas", "xysa"]
TWO_PI = 2 * math.pi

# position of (x, y, angle, scale) in each format (None = not present)
_FORMAT_INDEX = {
    "xy": (0, 1, None, None),
    "yx": (1, 0, None, None),
    "xya": (0, 1, 2, None),
    "xys": (0, 1, None, 2),
    "xyas": (0, 1, 2, 3),
    "xysa": (0, 1, 3, 2),
}


@dataclass
class KeypointParams:
    """Same meaning as ``albumentations.KeypointParams`` (minus ``label_fields``: keep per-keypoint payloads aligned
    in the padded ``(B, K, ...)`` layout, or append them as extra columns, which are passed through).

    - format: "xy", "yx", "xya", "xys", "xyas", "xysa" (a = angle, s = scale)
    - remove_invisible: keypoints outside the image (x not in [0, W) or y not in [0, H)) after a transform, inside a
      dropout hole, or lost by a distortion are marked invalid
    - angle_in_degrees: input/output angles in degrees (radians otherwise)
    - check_each_transform: filter after every transform (otherwise only at the end)
    """

    format: KeypointFormat = "xy"
    remove_invisible: bool = True
    angle_in_degrees: bool = True
    check_each_transform: bool = True

    def __post_init__(self) -> None:
        if self.format not in _FORMAT_INDEX:
            raise ValueError(f"Unknown keypoint format {self.format!r}; use one of {sorted(_FORMAT_INDEX)}")


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle, TWO_PI)


def to_internal(kps: torch.Tensor, params: KeypointParams) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, K, F) in ``params.format`` -> internal (B, K, 4) [x, y, angle(rad), scale] and the extra columns."""
    ix, iy, ia, is_ = _FORMAT_INDEX[params.format]
    n_fmt = len(params.format)
    if kps.shape[-1] < n_fmt:
        raise ValueError(f"keypoints must have at least {n_fmt} columns for format {params.format!r}")
    k = kps.float()
    zeros = torch.zeros_like(k[..., 0])
    angle = k[..., ia] if ia is not None else zeros
    if ia is not None and params.angle_in_degrees:
        angle = torch.deg2rad(angle)
    scale = k[..., is_] if is_ is not None else zeros
    internal = torch.stack([k[..., ix], k[..., iy], wrap_angle(angle), scale], dim=-1)
    return internal, kps[..., n_fmt:]


def from_internal(internal: torch.Tensor, extra: torch.Tensor, params: KeypointParams) -> torch.Tensor:
    ix, iy, ia, is_ = _FORMAT_INDEX[params.format]
    cols: list[torch.Tensor | None] = [None] * len(params.format)
    cols[ix] = internal[..., 0]
    cols[iy] = internal[..., 1]
    if ia is not None:
        a = wrap_angle(internal[..., 2])
        cols[ia] = torch.rad2deg(a) if params.angle_in_degrees else a
    if is_ is not None:
        cols[is_] = internal[..., 3]
    out = torch.stack(cols, dim=-1)
    if extra.shape[-1] > 0:
        out = torch.cat([out, extra.to(out.dtype)], dim=-1)
    return out


def filter_keypoints(kps: torch.Tensor, valid: torch.Tensor, height: int, width: int, params: KeypointParams) -> torch.Tensor:
    """albumentations ``filter_keypoints``: visible iff 0 <= x < W and 0 <= y < H."""
    if not params.remove_invisible:
        return valid
    x, y = kps[..., 0], kps[..., 1]
    return valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)


def transform_points_by_matrix(xy: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """xy (n, K, 2), matrix (n, 3, 3) (affine or projective) -> (n, K, 2)."""
    m = matrix.to(torch.float64)
    p = xy.to(torch.float64)
    tx = m[:, None, 0, 0] * p[..., 0] + m[:, None, 0, 1] * p[..., 1] + m[:, None, 0, 2]
    ty = m[:, None, 1, 0] * p[..., 0] + m[:, None, 1, 1] * p[..., 1] + m[:, None, 1, 2]
    tw = m[:, None, 2, 0] * p[..., 0] + m[:, None, 2, 1] * p[..., 1] + m[:, None, 2, 2]
    return torch.stack([tx / tw, ty / tw], dim=-1).to(xy.dtype)
