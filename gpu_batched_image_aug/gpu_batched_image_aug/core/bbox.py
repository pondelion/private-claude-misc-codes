"""Bounding box handling.

Boxes are carried through the pipeline as a padded tensor ``(B, N, 4)`` plus a validity mask ``(B, N)``: images in a
batch may have different numbers of boxes, and instead of removing boxes (as albumentations does) a filtered box is
marked ``valid=False``. Any per-box payload (labels, ids, ...) can simply be kept aligned by the caller with the same
``(B, N, ...)`` layout.

Internally boxes are always in ``pascal_voc`` pixel coordinates ``[x_min, y_min, x_max, y_max]`` with continuous
pixel-edge coordinates (the image spans ``[0, W] x [0, H]``), which is the convention albumentations uses after
denormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

BboxFormat = Literal["pascal_voc", "albumentations", "coco", "yolo"]
_EPS = 1e-7


@dataclass
class BboxParams:
    """Same meaning as ``albumentations.BboxParams`` (minus ``label_fields``, which is not needed with the padded
    ``(B, N, ...)`` layout).

    - format: input/output format. ``pascal_voc`` [x_min, y_min, x_max, y_max] pixels, ``albumentations`` same but
      normalized to [0, 1], ``coco`` [x_min, y_min, width, height] pixels, ``yolo`` [x_c, y_c, w, h] normalized.
    - min_area / min_visibility / min_width / min_height / max_accept_ratio: filtering criteria applied (like
      albumentations with ``check_each_transform=True``) after every transform that moves pixels. Visibility is
      ``area after clipping to the image / area before clipping`` of that transform's output.
    - clip: clip input boxes to the image before processing.
    - filter_invalid_bboxes: mark input boxes with non-positive area (after optional clipping) as invalid.
    """

    format: BboxFormat = "pascal_voc"
    min_area: float = 0.0
    min_visibility: float = 0.0
    min_width: float = 0.0
    min_height: float = 0.0
    clip: bool = False
    filter_invalid_bboxes: bool = False
    max_accept_ratio: float | None = None
    check_each_transform: bool = True

    def __post_init__(self) -> None:
        if self.format not in ("pascal_voc", "albumentations", "coco", "yolo"):
            raise ValueError(f"Unknown bbox format {self.format!r}")
        if not 0.0 <= self.min_visibility <= 1.0:
            raise ValueError("min_visibility must be in [0, 1]")


def to_pascal_voc(boxes: torch.Tensor, fmt: BboxFormat, height: int, width: int) -> torch.Tensor:
    b = boxes.float()
    if fmt == "pascal_voc":
        return b.clone()
    if fmt == "albumentations":
        return b * b.new_tensor([width, height, width, height])
    if fmt == "coco":
        return torch.cat([b[..., :2], b[..., :2] + b[..., 2:4]], dim=-1)
    if fmt == "yolo":
        scale = b.new_tensor([width, height, width, height])
        xc, yc, w, h = b.unbind(-1)
        return torch.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], dim=-1) * scale
    raise ValueError(fmt)


def from_pascal_voc(boxes: torch.Tensor, fmt: BboxFormat, height: int, width: int) -> torch.Tensor:
    if fmt == "pascal_voc":
        return boxes
    if fmt == "albumentations":
        return boxes / boxes.new_tensor([width, height, width, height])
    if fmt == "coco":
        return torch.cat([boxes[..., :2], boxes[..., 2:4] - boxes[..., :2]], dim=-1)
    if fmt == "yolo":
        x1, y1, x2, y2 = boxes.unbind(-1)
        out = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
        return out / boxes.new_tensor([width, height, width, height])
    raise ValueError(fmt)


def clip_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    limit = boxes.new_tensor([width, height, width, height])
    return torch.minimum(boxes.clamp(min=0), limit)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def filter_boxes(
    boxes: torch.Tensor, valid: torch.Tensor, height: int, width: int, params: BboxParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """albumentations ``filter_bboxes``: clip to the image and invalidate boxes that fail the criteria.
    Additionally, boxes lying completely outside the image (zero width or height after clipping) are always
    invalidated (albumentations removes them inside the geometric transforms via ``validate_bboxes``).
    Returns (clipped boxes, new validity)."""
    area_before = box_area(boxes)
    clipped = clip_boxes(boxes, height, width)
    area_after = box_area(clipped)
    w = clipped[..., 2] - clipped[..., 0]
    h = clipped[..., 3] - clipped[..., 1]
    keep = (
        (area_before >= _EPS)
        & (area_after >= params.min_area - _EPS)
        & (area_after / (area_before + _EPS) >= params.min_visibility)
        & (w >= params.min_width - _EPS)
        & (h >= params.min_height - _EPS)
        & (w > 0)
        & (h > 0)
    )
    if params.max_accept_ratio is not None:
        ratio = torch.maximum(w / (h + _EPS), h / (w + _EPS))
        keep &= ratio <= params.max_accept_ratio
    return clipped, valid & keep


# ---------------------------------------------------------------------------------------------------------------
# geometric helpers
# ---------------------------------------------------------------------------------------------------------------

_ELLIPSE_ANGLES = None


def _ellipse_unit(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    global _ELLIPSE_ANGLES
    if _ELLIPSE_ANGLES is None or _ELLIPSE_ANGLES.device != device:
        _ELLIPSE_ANGLES = torch.deg2rad(torch.arange(0, 360, dtype=torch.float32, device=device))
    return torch.sin(_ELLIPSE_ANGLES), torch.cos(_ELLIPSE_ANGLES)


def transform_boxes_by_matrix(
    boxes: torch.Tensor, matrix: torch.Tensor, method: Literal["largest_box", "ellipse"] = "largest_box"
) -> torch.Tensor:
    """boxes (n, N, 4) pixel-edge coordinates, matrix (n, 3, 3) mapping source -> destination in pixel-edge
    coordinates (affine or projective). Same as albumentations ``bboxes_affine_largest_box`` /
    ``bboxes_affine_ellipse`` / ``perspective_bboxes``: transform the corner points (or 360 points of the inscribed
    ellipse) and take their axis-aligned extent."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    if method == "largest_box":
        px = torch.stack([x1, x2, x2, x1], dim=-1)
        py = torch.stack([y1, y1, y2, y2], dim=-1)
    elif method == "ellipse":
        sin_a, cos_a = _ellipse_unit(boxes.device)
        hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
        px = (x1 + hw)[..., None] + hw[..., None] * sin_a
        py = (y1 + hh)[..., None] + hh[..., None] * cos_a
    else:
        raise ValueError(f"Unknown rotate_method {method!r}")
    m = matrix.to(torch.float64)
    px, py = px.to(torch.float64), py.to(torch.float64)
    tx = m[:, None, 0:1, 0] * px + m[:, None, 0:1, 1] * py + m[:, None, 0:1, 2]
    ty = m[:, None, 1:2, 0] * px + m[:, None, 1:2, 1] * py + m[:, None, 1:2, 2]
    tw = m[:, None, 2:3, 0] * px + m[:, None, 2:3, 1] * py + m[:, None, 2:3, 2]
    tx, ty = tx / tw, ty / tw
    out = torch.stack([tx.amin(-1), ty.amin(-1), tx.amax(-1), ty.amax(-1)], dim=-1)
    return out.to(boxes.dtype)


def pixel_center_to_edge_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert a transform expressed in pixel-center coordinates (cv2 warp convention, pixel (i, j) at integer
    coords) into the equivalent transform in pixel-edge coordinates used by boxes: T(+0.5) @ M @ T(-0.5).
    For an affine built around the image center this is exactly albumentations' separate ``bbox_matrix``."""
    n = matrix.shape[0]
    t = torch.eye(3, dtype=matrix.dtype, device=matrix.device).expand(n, 3, 3).clone()
    t_inv = t.clone()
    t[:, 0, 2] = 0.5
    t[:, 1, 2] = 0.5
    t_inv[:, 0, 2] = -0.5
    t_inv[:, 1, 2] = -0.5
    return t @ matrix @ t_inv


def boxes_to_masks(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(n, N, 4) -> (n, N, H, W) bool. albumentations ``bboxes_to_mask``: round each coordinate, clamp to valid pixel
    indices and fill the inclusive index range."""
    b = torch.round(boxes)  # torch.round = round-half-to-even, same as python's round used by albumentations
    x1 = b[..., 0].clamp(0, width - 1).long()
    x2 = b[..., 2].clamp(0, width - 1).long()
    y1 = b[..., 1].clamp(0, height - 1).long()
    y2 = b[..., 3].clamp(0, height - 1).long()
    xs = torch.arange(width, device=boxes.device)
    ys = torch.arange(height, device=boxes.device)
    in_x = (xs >= x1[..., None]) & (xs <= x2[..., None])  # (n, N, W)
    in_y = (ys >= y1[..., None]) & (ys <= y2[..., None])  # (n, N, H)
    return in_y[..., :, None] & in_x[..., None, :]


def masks_to_boxes(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, N, H, W) bool -> boxes (n, N, 4) and non-empty flag (n, N). albumentations ``mask_to_bboxes``: the box is
    [min col, min row, max col, max row] of the set pixels (max is the pixel *index*, not index + 1)."""
    n, nb, h, w = masks.shape
    rows = masks.any(dim=-1)  # (n, N, H)
    cols = masks.any(dim=-2)  # (n, N, W)
    nonempty = rows.any(dim=-1)
    ar_h = torch.arange(h, device=masks.device)
    ar_w = torch.arange(w, device=masks.device)
    big = torch.iinfo(torch.long).max
    y_min = torch.where(rows, ar_h, big).amin(-1)
    y_max = torch.where(rows, ar_h, -1).amax(-1)
    x_min = torch.where(cols, ar_w, big).amin(-1)
    x_max = torch.where(cols, ar_w, -1).amax(-1)
    boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1).float()
    return torch.where(nonempty[..., None], boxes, torch.zeros_like(boxes)), nonempty


def boxes_visible_extent_after_holes(
    boxes: torch.Tensor, hole_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """albumentations ``filter_bboxes_by_holes`` core (used by CoarseDropout / GridDropout / XYMasking).

    boxes (n, N, 4), hole_mask (n, H, W) bool. Boxes are truncated to int; returns
    (remaining_area (n, N), box_area (n, N), shrunk boxes (n, N, 4)) where the shrunk box is the extent of the
    rows/columns of the box that still contain at least one visible pixel."""
    n, h, w = hole_mask.shape
    b = boxes.long()  # astype(int): truncation toward zero
    x1, y1, x2, y2 = (b[..., i] for i in range(4))
    hm = hole_mask.to(torch.int32)
    # integral image with a zero row/col in front: S[y, x] = sum(hm[:y, :x])
    integ = torch.zeros(n, h + 1, w + 1, dtype=torch.int32, device=hole_mask.device)
    integ[:, 1:, 1:] = hm.cumsum(1).cumsum(2)

    def rect_sum(ya, yb, xa, xb):
        ya, yb = ya.clamp(0, h), yb.clamp(0, h)
        xa, xb = xa.clamp(0, w), xb.clamp(0, w)
        idx = torch.arange(n, device=hole_mask.device)[:, None]
        return integ[idx, yb, xb] - integ[idx, ya, xb] - integ[idx, yb, xa] + integ[idx, ya, xa]

    box_areas = (x2 - x1) * (y2 - y1)
    covered = rect_sum(y1, y2, x1, x2)
    remaining = box_areas - covered

    # per-row hole counts inside [x1, x2) -> visible rows; per-column counts inside [y1, y2) -> visible columns
    row_cum = torch.zeros(n, h, w + 1, dtype=torch.int32, device=hole_mask.device)
    row_cum[:, :, 1:] = hm.cumsum(2)
    col_cum = torch.zeros(n, h + 1, w, dtype=torch.int32, device=hole_mask.device)
    col_cum[:, 1:, :] = hm.cumsum(1)
    idx_n = torch.arange(n, device=hole_mask.device)[:, None, None]
    ar_h = torch.arange(h, device=hole_mask.device)
    ar_w = torch.arange(w, device=hole_mask.device)
    xa, xb = x1.clamp(0, w)[..., None], x2.clamp(0, w)[..., None]
    ya, yb = y1.clamp(0, h)[..., None], y2.clamp(0, h)[..., None]
    holes_in_row = row_cum[idx_n, ar_h, xb] - row_cum[idx_n, ar_h, xa]  # (n, N, H)
    row_visible = (holes_in_row < (xb - xa)) & (ar_h >= ya) & (ar_h < yb)
    holes_in_col = col_cum[idx_n, yb, ar_w] - col_cum[idx_n, ya, ar_w]  # (n, N, W)
    col_visible = (holes_in_col < (yb - ya)) & (ar_w >= xa) & (ar_w < xb)
    big = torch.iinfo(torch.long).max
    new_y1 = torch.where(row_visible, ar_h, big).amin(-1)
    new_y2 = torch.where(row_visible, ar_h, -1).amax(-1) + 1
    new_x1 = torch.where(col_visible, ar_w, big).amin(-1)
    new_x2 = torch.where(col_visible, ar_w, -1).amax(-1) + 1
    shrunk = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1).to(boxes.dtype)
    any_visible = row_visible.any(-1)
    shrunk = torch.where(any_visible[..., None], shrunk, boxes)
    return remaining, box_areas, shrunk
