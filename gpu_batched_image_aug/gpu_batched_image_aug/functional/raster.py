"""Batched rasterization of simple primitives (polygons, disks, line segments) at pixel centers."""

from __future__ import annotations

import torch


def polygon_masks(vertices: torch.Tensor, valid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Fill polygons with the even-odd rule (like cv2.fillPoly for simple polygons), including pixels whose center lies
    on an edge. vertices (n, K, P, 2) float (x, y), valid (n, K) -> (n, K, H, W) bool."""
    n, k, p, _ = vertices.shape
    dev = vertices.device
    ys = torch.arange(height, dtype=torch.float32, device=dev).view(1, 1, height, 1)
    xs = torch.arange(width, dtype=torch.float32, device=dev).view(1, 1, 1, width)
    inside = torch.zeros(n, k, height, width, dtype=torch.bool, device=dev)
    on_edge = torch.zeros_like(inside)
    v = vertices.float()
    for i in range(p):
        x0, y0 = v[:, :, i, 0, None, None], v[:, :, i, 1, None, None]
        x1, y1 = v[:, :, (i + 1) % p, 0, None, None], v[:, :, (i + 1) % p, 1, None, None]
        crosses = (y0 > ys) != (y1 > ys)
        dy = torch.where(y1 == y0, torch.ones_like(y1), y1 - y0)
        x_int = x0 + (ys - y0) * (x1 - x0) / dy
        inside ^= crosses & (xs < x_int)
        # distance of the pixel center to the segment <= 0.5 -> on the edge (cv2 draws the boundary too)
        ex, ey = x1 - x0, y1 - y0
        l2 = (ex * ex + ey * ey).clamp(min=1e-12)
        t = (((xs - x0) * ex + (ys - y0) * ey) / l2).clamp(0, 1)
        dx, dyy = xs - (x0 + t * ex), ys - (y0 + t * ey)
        on_edge |= (dx * dx + dyy * dyy) <= 0.25
    return (inside | on_edge) & valid[:, :, None, None]


def disk_coverage_count(
    centers: torch.Tensor, radii: torch.Tensor, valid: torch.Tensor, height: int, width: int, chunk: int = 64
) -> torch.Tensor:
    """Number of filled disks (cv2.circle thickness=-1 approximated by distance <= radius) covering each pixel.
    centers (n, K, 2) (x, y), radii (n, K), valid (n, K) -> (n, H, W) float."""
    n, k = radii.shape
    dev = centers.device
    ys = torch.arange(height, dtype=torch.float32, device=dev).view(1, 1, height, 1)
    xs = torch.arange(width, dtype=torch.float32, device=dev).view(1, 1, 1, width)
    count = torch.zeros(n, height, width, device=dev)
    for s in range(0, k, chunk):
        c = centers[:, s : s + chunk].float()
        r = radii[:, s : s + chunk].float()
        v = valid[:, s : s + chunk]
        d2 = (xs - c[..., 0, None, None]) ** 2 + (ys - c[..., 1, None, None]) ** 2
        count += ((d2 <= (r * r)[..., None, None]) & v[..., None, None]).float().sum(1)
    return count


def segment_masks(starts: torch.Tensor, ends: torch.Tensor, valid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Union of 1-pixel 4-connected line segments per sample, replicating OpenCV's ``LineIterator`` with
    connectivity 4 (cv2.line / cv2.polylines with ``lineType=cv2.LINE_4``): endpoints are truncated to int, iteration
    goes left to right, and at each step the error term decides between a step along the major or the minor axis.
    starts/ends (n, K, 2) (x, y), valid (n, K) -> (n, H, W) bool. Pixels outside the image are skipped."""
    n, k, _ = starts.shape
    dev = starts.device
    p0 = starts.long()
    p1 = ends.long()
    swap = p0[..., 0] > p1[..., 0]  # left to right
    a = torch.where(swap[..., None], p1, p0)
    b = torch.where(swap[..., None], p0, p1)
    dx = b[..., 0] - a[..., 0]  # >= 0
    dy_s = b[..., 1] - a[..., 1]
    sy = torch.where(dy_s < 0, -torch.ones_like(dy_s), torch.ones_like(dy_s))
    dy = dy_s.abs()
    y_major = dy > dx
    major = torch.where(y_major, dy, dx)
    minor = torch.where(y_major, dx, dy)
    steps = major + minor  # number of moves (pixels = steps + 1)
    lmax = int(steps.max().item()) if k > 0 else 0
    x, y = a[..., 0].clone(), a[..., 1].clone()
    err = torch.zeros_like(x)
    out = torch.zeros(n, height * width, device=dev)
    for i in range(lmax + 1):
        active = valid & (i <= steps)
        ok = active & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        lin = torch.where(ok, y * width + x, torch.zeros_like(x))
        out.scatter_(1, lin, ok.float())
        move_minor = err < 0
        err = torch.where(move_minor, err + 2 * major, err - 2 * minor)
        # major-axis step: x+1 (x major) or y+sy (y major); minor-axis step: the other axis
        step_x = torch.where(y_major, move_minor, ~move_minor)
        x = x + step_x.long()
        y = y + torch.where(step_x, torch.zeros_like(sy), sy)
    return out.view(n, height, width) > 0
