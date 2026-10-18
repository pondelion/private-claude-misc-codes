"""Shadow / weather effects (image only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F

from ..core.data import AugBatch
from ..core.random import Rng
from ..core.transform import ImageOnlyTransform
from ..core.utils import check_bounds, check_nondecreasing
from ..functional import color as fcolor
from ..functional.filters import box_kernels, cv2_gaussian_kernel_1d, filter2d, separable_filter
from ..functional.raster import disk_coverage_count, polygon_masks, segment_masks

__all__ = ["RandomFog", "RandomRain", "RandomShadow", "RandomSnow"]


class RandomShadow(ImageOnlyTransform):
    """albumentations ``RandomShadow``: randint(num_shadows_limit) polygons with ``shadow_dimension`` random vertices
    inside ``shadow_roi`` (fractions x_min, y_min, x_max, y_max); pixels inside are multiplied by 1 - intensity,
    intensity ~ U(shadow_intensity_range) per shadow. Polygons are filled with the even-odd rule."""

    def __init__(
        self,
        shadow_roi: Sequence[float] = (0, 0.5, 1, 1),
        num_shadows_limit: Sequence[int] = (1, 2),
        shadow_dimension: int = 5,
        shadow_intensity_range: Sequence[float] = (0.5, 0.5),
        p: float = 0.5,
    ):
        super().__init__(p)
        x1, y1, x2, y2 = (float(v) for v in shadow_roi)
        if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1):
            raise ValueError(f"Invalid shadow_roi {shadow_roi}")
        if shadow_dimension < 3:
            raise ValueError("shadow_dimension must be >= 3")
        self.shadow_roi = (x1, y1, x2, y2)
        self.num_shadows_limit = check_nondecreasing((int(num_shadows_limit[0]), int(num_shadows_limit[1])), "num_shadows_limit")
        self.shadow_dimension = int(shadow_dimension)
        self.shadow_intensity_range = check_bounds((float(shadow_intensity_range[0]), float(shadow_intensity_range[1])), 0, 1, "shadow_intensity_range")

    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        n, h, w = batch.batch_size, batch.height, batch.width
        kmax = self.num_shadows_limit[1]
        num = rng.randint(self.num_shadows_limit[0], kmax, n)
        x_lo, x_hi = int(self.shadow_roi[0] * w), int(self.shadow_roi[2] * w)
        y_lo, y_hi = int(self.shadow_roi[1] * h), int(self.shadow_roi[3] * h)
        vx = rng.randint(x_lo, max(x_lo, x_hi - 1), (n, kmax, self.shadow_dimension))  # numpy integers: high exclusive
        vy = rng.randint(y_lo, max(y_lo, y_hi - 1), (n, kmax, self.shadow_dimension))
        valid = torch.arange(kmax, device=batch.device)[None] < num[:, None]
        masks = polygon_masks(torch.stack([vx, vy], dim=-1).float(), valid, h, w)  # (n, K, H, W)
        intensity = rng.uniform(*self.shadow_intensity_range, (n, kmax))
        factor = torch.prod(torch.where(masks, (1 - intensity)[..., None, None], torch.ones_like(masks, dtype=torch.float32)), dim=1)
        return {"factor": factor[:, None]}

    def apply(self, img, factor, **params):
        return img * factor


class RandomFog(ImageOnlyTransform):
    """albumentations ``RandomFog``: intensity ~ U(fog_coef_range); white disks (particles placed in shrinking central
    regions, radius ~ U[max_r/2, max_r)) are alpha-blended one after another with alpha = alpha_coef * intensity,
    which is computed in closed form as 1 - (1 - x) * (1 - alpha)^count; then a Gaussian blur of size
    max(3, min(H, W) // 30) (odd) is applied."""

    def __init__(self, alpha_coef: float = 0.08, fog_coef_range: Sequence[float] = (0.3, 1), p: float = 0.5):
        super().__init__(p)
        self.alpha_coef = float(alpha_coef)
        self.fog_coef_range = check_bounds(check_nondecreasing((float(fog_coef_range[0]), float(fog_coef_range[1])), "fog_coef_range"), 0, 1, "fog_coef_range")

    def get_params(self, batch, rng):
        n, h, w = batch.batch_size, batch.height, batch.width
        dev = batch.device
        intensity = rng.uniform(*self.fog_coef_range, n)
        cx, cy = int(w / 2 - 0.5), int(h / 2 - 0.5)
        centers, radii = [], []
        for i, inten in enumerate(intensity.tolist()):
            region = max(1, int(w // 3 * inten))
            cw, ch, it = w, h, 0
            pos = []
            while cw > region and ch > region and it < 10:
                count = int(cw * ch / (region * region) * inten * 10)
                if count > 0:
                    xs = rng.randint(cx - cw // 2, cx + cw // 2, count)
                    ys = rng.randint(cy - ch // 2, cy + ch // 2, count)
                    pos.append(torch.stack([xs, ys], dim=1))
                cw, ch, it = int(cw * 0.9), int(ch * 0.9), it + 1
            pts = torch.cat(pos) if pos else torch.zeros(0, 2, dtype=torch.long, device=dev)
            max_r = max(2, int(min(h, w) * 0.1 * inten))
            min_r = max(1, max_r // 2)
            r = rng.randint(min_r, max(min_r, max_r - 1), pts.shape[0])
            centers.append(pts)
            radii.append(r)
        kmax = max(1, max(c.shape[0] for c in centers))
        c_pad = torch.zeros(n, kmax, 2, device=dev)
        r_pad = torch.zeros(n, kmax, device=dev)
        valid = torch.zeros(n, kmax, dtype=torch.bool, device=dev)
        for i, (c, r) in enumerate(zip(centers, radii)):
            k = c.shape[0]
            c_pad[i, :k], r_pad[i, :k], valid[i, :k] = c.float(), r.float(), True
        count = disk_coverage_count(c_pad, r_pad, valid, h, w)
        return {"count": count[:, None], "alpha": self.alpha_coef * intensity}

    def apply(self, img, count, alpha, **params):
        out = 1.0 - (1.0 - img) * torch.pow((1.0 - alpha).view(-1, 1, 1, 1), count)
        h, w = img.shape[-2:]
        k = max(3, int(min(h, w) // 30))
        k = k + 1 if k % 2 == 0 else k
        g = cv2_gaussian_kernel_1d(k, 0.0, img.device)
        return separable_filter(out, g, g).clamp_(0, 1)


class RandomRain(ImageOnlyTransform):
    """albumentations ``RandomRain``: drops are 1-pixel (``drop_width``) lines from (x, y) to (x + slant,
    y + drop_length), slant ~ U(slant_range); drawn in ``drop_color`` (0..255), box-blurred (``blur_value``), added to
    the image and the result multiplied by ``brightness_coefficient``. Number of drops by ``rain_type``."""

    def __init__(
        self,
        slant_range: Sequence[float] = (-10, 10),
        drop_length: int | None = None,
        drop_width: int = 1,
        drop_color: Sequence[int] = (200, 200, 200),
        blur_value: int = 7,
        brightness_coefficient: float = 0.7,
        rain_type: Literal["drizzle", "heavy", "torrential", "default"] = "default",
        p: float = 0.5,
    ):
        super().__init__(p)
        self.slant_range = (float(slant_range[0]), float(slant_range[1]))
        self.drop_length = drop_length
        self.drop_width = int(drop_width)
        self.drop_color = tuple(int(c) for c in drop_color)
        self.blur_value = int(blur_value)
        self.brightness_coefficient = float(brightness_coefficient)
        self.rain_type = rain_type

    def get_params(self, batch, rng):
        n, h, w = batch.batch_size, batch.height, batch.width
        num = {"drizzle": h // 4, "heavy": h, "torrential": h * 2}.get(self.rain_type, h // 3)
        length = max(1, h // 8) if self.drop_length is None else int(self.drop_length)
        slant = rng.uniform(*self.slant_range, n)
        xs = rng.randint(0, w - 1, (n, num)).float()
        ys = rng.randint(0, max(0, h - length - 1), (n, num)).float()
        starts = torch.stack([xs, ys], dim=-1)
        ends = torch.stack([torch.trunc(xs + slant[:, None]), ys + length], dim=-1)
        valid = torch.ones(n, num, dtype=torch.bool, device=batch.device)
        lines = segment_masks(starts, ends, valid, h, w).float()[:, None]
        if self.drop_width > 1:
            lines = F.max_pool2d(lines, self.drop_width, stride=1, padding=self.drop_width // 2)[..., :h, :w]
        return {"lines": lines}

    def apply(self, img, lines, **params):
        n, c = img.shape[:2]
        color = torch.tensor(self.drop_color, dtype=img.dtype, device=img.device)[:c].view(1, -1, 1, 1) / 255.0
        layer = lines * color
        if self.blur_value > 1:
            layer = filter2d(layer, box_kernels(torch.full((n,), self.blur_value, device=img.device)))
        return ((img + layer).clamp(0, 1) * self.brightness_coefficient).clamp_(0, 1)


class RandomSnow(ImageOnlyTransform):
    """albumentations ``RandomSnow`` (method="bleach"): in HLS, lightness below snow_point / 2 + 1/3 is multiplied by
    ``brightness_coeff``; snow_point ~ U(snow_point_range). method="texture" is not supported."""

    def __init__(
        self,
        brightness_coeff: float = 2.5,
        snow_point_range: Sequence[float] = (0.1, 0.3),
        method: Literal["bleach", "texture"] = "bleach",
        p: float = 0.5,
    ):
        super().__init__(p)
        if method != "bleach":
            raise NotImplementedError("Only method='bleach' is supported")
        self.brightness_coeff = float(brightness_coeff)
        self.snow_point_range = (float(snow_point_range[0]), float(snow_point_range[1]))
        self.method = method

    def get_params(self, batch, rng):
        return {"snow_point": rng.uniform(*self.snow_point_range, batch.batch_size)}

    def apply(self, img, snow_point, **params):
        hls = fcolor.rgb_to_hls(img)
        thr = (snow_point / 2 + 1 / 3).view(-1, 1, 1)
        light = hls[:, 1]
        hls[:, 1] = torch.where(light < thr, (light * self.brightness_coeff).clamp(0, 1), light)
        return fcolor.hls_to_rgb(hls).clamp_(0, 1)

