"""Base transform classes.

Probability semantics follow albumentations, but per sample: each sample of the batch independently decides whether
a transform is applied (with probability ``p``), and every applied sample draws its own random parameters. The
transform is then executed once on the stacked subset of applied samples.

Subclasses implement:
- ``get_params(batch, rng)``: draw per-sample parameters for the (sub)batch, as tensors with leading dim ``n``
- ``apply(img, **params)``: image (n, C, H, W) float32 in [0, 1]
- ``apply_to_mask(mask, **params)`` / ``apply_to_bboxes(bboxes, valid, height, width, **params)`` /
  ``apply_to_keypoints(keypoints, valid, height, width, **params)`` for dual transforms
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from .data import AugBatch
from .keypoints import wrap_angle
from .random import Rng


class BaseTransform:
    #: True for transforms whose output spatial size differs from the input (crops / resizes). They must be applied
    #: to the whole batch (p=1, not inside OneOf/SomeOf) since a batch tensor has a single size.
    changes_shape: bool = False
    #: True if the transform moves pixels (geometric). bboxes are filtered after such transforms.
    is_geometric: bool = False

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    # ---- public convenience: transform(image=..., mask=..., bboxes=...) like albumentations ----------------------
    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if args:
            raise TypeError("Call transforms with keyword arguments: t(image=..., mask=..., bboxes=...)")
        from .composition import Compose

        return Compose([self], p=1.0)(**kwargs)

    # ---- internal execution ------------------------------------------------------------------------------------
    def run(self, batch: AugBatch, rng: Rng, active: torch.Tensor | None = None, force_apply: bool = False) -> AugBatch:
        """``active``: (B,) bool, samples this transform is allowed to touch (set by OneOf/SomeOf/parents).
        ``force_apply``: skip the own probability draw (OneOf applies the chosen child with force)."""
        bsz = batch.batch_size
        draw = rng.rand(bsz)  # always drawn so the random stream does not depend on `active`
        apply_mask = torch.ones(bsz, dtype=torch.bool, device=batch.device) if force_apply else draw < self.p
        if active is not None:
            apply_mask &= active

        if self.changes_shape:
            if not bool(apply_mask.all()):
                raise ValueError(
                    f"{type(self).__name__} changes the image size, so it must be applied to every sample of the batch: "
                    "use p=1.0 and do not put it inside OneOf/SomeOf or a composition with p<1."
                )
            return self.apply_batch(batch, self._params(batch, rng))

        idx = torch.nonzero(apply_mask).squeeze(1)
        n = idx.numel()
        if n == 0:
            return batch
        if n == bsz:
            return self.apply_batch(batch, self._params(batch, rng))
        sub = batch.select(idx)
        sub = self.apply_batch(sub, self._params(sub, rng))
        batch.update(idx, sub)
        return batch

    def _params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        params = self.get_params(batch, rng)
        params["max_value"] = batch.max_value
        return params

    # ---- to implement ----------------------------------------------------------------------------------------------
    def get_params(self, batch: AugBatch, rng: Rng) -> dict[str, Any]:
        return {}

    def apply_batch(self, batch: AugBatch, params: dict[str, Any]) -> AugBatch:
        raise NotImplementedError

    def apply(self, img: torch.Tensor, **params: Any) -> torch.Tensor:
        raise NotImplementedError

    # ---- repr ------------------------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        sig = inspect.signature(type(self).__init__)
        args = []
        for name in sig.parameters:
            if name in ("self", "args", "kwargs"):
                continue
            if hasattr(self, name):
                args.append(f"{name}={getattr(self, name)!r}")
        return f"{type(self).__name__}({', '.join(args)})"


class ImageOnlyTransform(BaseTransform):
    """Changes pixel values only. Masks and boxes pass through unchanged."""

    def apply_batch(self, batch: AugBatch, params: dict[str, Any]) -> AugBatch:
        batch.image = self.apply(batch.image, **params)
        return batch


class DualTransform(BaseTransform):
    """Transform that also affects masks and/or boxes (geometric transforms, dropouts with fill_mask, ...)."""

    def apply_batch(self, batch: AugBatch, params: dict[str, Any]) -> AugBatch:
        height, width = batch.height, batch.width
        batch.image = self.apply(batch.image, **params)
        if batch.mask is not None:
            batch.mask = self.apply_to_mask(batch.mask, **params)
        if batch.bboxes is not None:
            batch.bboxes, batch.bbox_valid = self.apply_to_bboxes(
                batch.bboxes, batch.bbox_valid, height, width, bbox_params=batch.bbox_params, **params
            )
        if batch.keypoints is not None:
            kps, batch.keypoint_valid = self.apply_to_keypoints(
                batch.keypoints, batch.keypoint_valid, height, width, keypoint_params=batch.keypoint_params, **params
            )
            kps = kps.clone()
            kps[..., 2] = wrap_angle(kps[..., 2])
            batch.keypoints = kps
        return batch

    def apply_to_mask(self, mask: torch.Tensor, **params: Any) -> torch.Tensor:
        return mask

    def apply_to_bboxes(
        self, bboxes: torch.Tensor, valid: torch.Tensor, height: int, width: int, **params: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``height``/``width``: spatial size *before* the transform."""
        return bboxes, valid

    def apply_to_keypoints(
        self, keypoints: torch.Tensor, valid: torch.Tensor, height: int, width: int, **params: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """keypoints (n, K, 4) [x, y, angle (rad), scale]; ``height``/``width``: size *before* the transform.
        Angles are wrapped to [0, 2*pi) afterwards by the caller."""
        return keypoints, valid
