"""Compositions: Compose / Sequential / OneOf / SomeOf.

Probability semantics are the albumentations ones, evaluated independently per sample:
- Compose(p): the whole pipeline is applied to a sample with probability p; children use their own p.
- Sequential(p): same as Compose, for nesting.
- OneOf(p): with probability p, exactly one child is chosen per sample with probability proportional to the
  children's p and applied with force (its own p is not drawn again).
- SomeOf(n, replace, p): with probability p, n children are chosen uniformly per sample (with/without
  replacement), applied in ascending index order, each with its own probability p.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .bbox import BboxParams, filter_boxes, from_pascal_voc, to_pascal_voc
from .keypoints import KeypointParams, filter_keypoints, from_internal, to_internal
from .data import AugBatch
from .random import Rng
from .transform import BaseTransform

Node = "BaseTransform | BaseCompose"


class BaseCompose:
    changes_shape = False

    def __init__(self, transforms: Sequence[Node], p: float):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.transforms = list(transforms)
        self.p = float(p)
        for t in self.transforms:
            if not isinstance(t, (BaseTransform, BaseCompose)):
                raise TypeError(f"Expected a transform or composition, got {type(t)}")

    def _child(self, t: Node, batch: AugBatch, rng: Rng, active: torch.Tensor | None, force: bool = False) -> AugBatch:
        batch = t.run(batch, rng, active, force_apply=force)
        params = batch.bbox_params
        if batch.bboxes is not None and params is not None and params.check_each_transform:
            batch.bboxes, batch.bbox_valid = filter_boxes(batch.bboxes, batch.bbox_valid, batch.height, batch.width, params)
        kp = batch.keypoint_params
        if batch.keypoints is not None and kp is not None and kp.check_each_transform:
            batch.keypoint_valid = filter_keypoints(batch.keypoints, batch.keypoint_valid, batch.height, batch.width, kp)
        return batch

    def _own_mask(self, batch: AugBatch, rng: Rng, active: torch.Tensor | None, force_apply: bool) -> torch.Tensor:
        draw = rng.rand(batch.batch_size)
        mask = torch.ones_like(draw, dtype=torch.bool) if force_apply else draw < self.p
        return mask if active is None else mask & active

    def run(self, batch: AugBatch, rng: Rng, active: torch.Tensor | None = None, force_apply: bool = False) -> AugBatch:
        raise NotImplementedError

    def __repr__(self) -> str:
        inner = ",\n".join("  " + repr(t).replace("\n", "\n  ") for t in self.transforms)
        return f"{type(self).__name__}([\n{inner}\n], p={self.p})"


class Sequential(BaseCompose):
    def __init__(self, transforms: Sequence[Node], p: float = 0.5):
        super().__init__(transforms, p)

    def run(self, batch: AugBatch, rng: Rng, active: torch.Tensor | None = None, force_apply: bool = False) -> AugBatch:
        mask = self._own_mask(batch, rng, active, force_apply)
        if not bool(mask.any()):
            return batch
        for t in self.transforms:
            batch = self._child(t, batch, rng, mask)
        return batch


class OneOf(BaseCompose):
    def __init__(self, transforms: Sequence[Node], p: float = 0.5):
        super().__init__(transforms, p)
        weights = [t.p for t in self.transforms]
        total = sum(weights)
        self.weights = [w / total for w in weights] if total > 0 else []

    def run(self, batch: AugBatch, rng: Rng, active: torch.Tensor | None = None, force_apply: bool = False) -> AugBatch:
        mask = self._own_mask(batch, rng, active, force_apply)
        if not self.weights:
            return batch
        choice = rng.choice(len(self.transforms), batch.batch_size, probs=self.weights)
        for i, t in enumerate(self.transforms):
            sel = mask & (choice == i)
            if bool(sel.any()):
                batch = self._child(t, batch, rng, sel, force=True)
        return batch


class SomeOf(BaseCompose):
    def __init__(self, transforms: Sequence[Node], n: int = 1, replace: bool = False, p: float = 1.0):
        super().__init__(transforms, p)
        if not replace and n > len(self.transforms):
            n = len(self.transforms)
        self.n = int(n)
        self.replace = replace

    def run(self, batch: AugBatch, rng: Rng, active: torch.Tensor | None = None, force_apply: bool = False) -> AugBatch:
        mask = self._own_mask(batch, rng, active, force_apply)
        k = len(self.transforms)
        if self.n == 0 or k == 0:
            return batch
        if self.replace:
            idx = rng.choice(k, (batch.batch_size, self.n))
        else:
            idx = rng.permutation(batch.batch_size, k)[:, : self.n]
        idx, _ = torch.sort(idx, dim=1)
        for slot in range(self.n):
            for i, t in enumerate(self.transforms):
                sel = mask & (idx[:, slot] == i)
                if bool(sel.any()):
                    batch = self._child(t, batch, rng, sel)
        return batch


class Compose(BaseCompose):
    """Entry point of a pipeline.

    Example::

        aug = Compose([...], bbox_params=BboxParams(format="pascal_voc", min_visibility=0.3), seed=0)
        out = aug(image=images, mask=masks, bboxes=boxes, bbox_valid=valid)

    Inputs (all tensors on the same device, typically CUDA):
    - image: (B, C, H, W) or (B, H, W). uint8 (0..255) or floating point with values in [0, 1].
      Output has the same dtype/shape layout (uint8 is rounded back).
    - mask: optional, (B, H, W) or (B, Cm, H, W), any dtype.
    - bboxes: optional, (B, N, 4) in ``bbox_params.format``. Requires ``bbox_params``.
    - bbox_valid: optional (B, N) bool marking real boxes in the padded tensor (default: all True).
    - keypoints: optional, (B, K, F) in ``keypoint_params.format`` (extra columns are passed through). Requires
      ``keypoint_params``.
    - keypoint_valid: optional (B, K) bool (default: all True).
    Returns a dict with the same keys; ``bbox_valid`` / ``keypoint_valid`` are always returned when boxes / keypoints
    are given. Filtered boxes keep their (clipped) coordinates and get ``bbox_valid=False``; invisible keypoints keep
    their coordinates and get ``keypoint_valid=False``.
    """

    def __init__(
        self,
        transforms: Sequence[Node],
        p: float = 1.0,
        bbox_params: BboxParams | dict | None = None,
        keypoint_params: KeypointParams | dict | None = None,
        seed: int | None = None,
    ):
        super().__init__(transforms, p)
        if isinstance(bbox_params, dict):
            bbox_params = BboxParams(**bbox_params)
        if isinstance(keypoint_params, dict):
            keypoint_params = KeypointParams(**keypoint_params)
        self.bbox_params = bbox_params
        self.keypoint_params = keypoint_params
        self.seed = seed
        self._rngs: dict[torch.device, Rng] = {}

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self._rngs = {}

    def _rng(self, device: torch.device) -> Rng:
        if device not in self._rngs:
            self._rngs[device] = Rng(device, self.seed)
        return self._rngs[device]

    def run(self, batch: AugBatch, rng: Rng, active: torch.Tensor | None = None, force_apply: bool = False) -> AugBatch:
        mask = self._own_mask(batch, rng, active, force_apply)
        if not bool(mask.any()) and not any(t.changes_shape for t in self.transforms):
            return batch
        for t in self.transforms:
            batch = self._child(t, batch, rng, None if bool(mask.all()) else mask)
        return batch

    @torch.no_grad()
    def __call__(
        self,
        *,
        image: torch.Tensor,
        mask: torch.Tensor | None = None,
        bboxes: torch.Tensor | None = None,
        bbox_valid: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        keypoint_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not torch.is_tensor(image):
            raise TypeError("image must be a torch.Tensor of shape (B, C, H, W) or (B, H, W)")
        image_ndim, image_dtype = image.ndim, image.dtype
        if image_ndim == 3:
            image = image[:, None]
        if image.ndim != 4:
            raise ValueError(f"image must be (B, C, H, W) or (B, H, W), got shape {tuple(image.shape)}")
        if image_dtype == torch.uint8:
            img = image.float() / 255.0
        elif image_dtype.is_floating_point:
            img = image.float().clone()
        else:
            raise TypeError(f"Unsupported image dtype {image_dtype} (use uint8 or float in [0, 1])")
        bsz, _, height, width = img.shape

        mask_ndim = None
        if mask is not None:
            mask_ndim = mask.ndim
            if mask.ndim == 3:
                mask = mask[:, None]
            if mask.ndim != 4 or mask.shape[0] != bsz or mask.shape[-2:] != (height, width):
                raise ValueError(f"mask must be (B, H, W) or (B, Cm, H, W) matching the image, got {tuple(mask.shape)}")
            mask = mask.clone()

        boxes = valid = None
        if bboxes is not None:
            if self.bbox_params is None:
                raise ValueError("bbox_params must be set in Compose to use bboxes")
            if bboxes.ndim != 3 or bboxes.shape[0] != bsz or bboxes.shape[-1] != 4:
                raise ValueError(f"bboxes must be (B, N, 4), got {tuple(bboxes.shape)}")
            boxes = to_pascal_voc(bboxes, self.bbox_params.format, height, width)
            if bbox_valid is None:
                valid = torch.ones(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
            else:
                if bbox_valid.shape != boxes.shape[:2]:
                    raise ValueError("bbox_valid must be (B, N)")
                valid = bbox_valid.bool().clone()
            if self.bbox_params.clip:
                boxes = torch.minimum(boxes.clamp(min=0), boxes.new_tensor([width, height, width, height]))
            if self.bbox_params.filter_invalid_bboxes:
                valid &= (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])

        kps = kp_valid = kp_extra = None
        if keypoints is not None:
            if self.keypoint_params is None:
                raise ValueError("keypoint_params must be set in Compose to use keypoints")
            if keypoints.ndim != 3 or keypoints.shape[0] != bsz:
                raise ValueError(f"keypoints must be (B, K, F), got {tuple(keypoints.shape)}")
            kps, kp_extra = to_internal(keypoints, self.keypoint_params)
            if keypoint_valid is None:
                kp_valid = torch.ones(kps.shape[:2], dtype=torch.bool, device=kps.device)
            else:
                if keypoint_valid.shape != kps.shape[:2]:
                    raise ValueError("keypoint_valid must be (B, K)")
                kp_valid = keypoint_valid.bool().clone()
            kp_valid = filter_keypoints(kps, kp_valid, height, width, self.keypoint_params)

        batch = AugBatch(
            image=img,
            mask=mask,
            bboxes=boxes,
            bbox_valid=valid,
            bbox_params=self.bbox_params,
            keypoints=kps,
            keypoint_valid=kp_valid,
            keypoint_params=self.keypoint_params,
            max_value=255.0 if image_dtype == torch.uint8 else 1.0,
        )
        batch = self.run(batch, self._rng(img.device))
        if batch.bboxes is not None:  # final filtering like albumentations' postprocess (also when nothing was applied)
            batch.bboxes, batch.bbox_valid = filter_boxes(batch.bboxes, batch.bbox_valid, batch.height, batch.width, self.bbox_params)
        if batch.keypoints is not None:
            batch.keypoint_valid = filter_keypoints(batch.keypoints, batch.keypoint_valid, batch.height, batch.width, self.keypoint_params)

        out_img = batch.image
        if image_dtype == torch.uint8:
            out_img = (out_img.clamp(0, 1) * 255.0).round().to(torch.uint8)
        else:
            out_img = out_img.to(image_dtype)
        if image_ndim == 3:
            out_img = out_img[:, 0]
        result = {"image": out_img}
        if batch.mask is not None:
            result["mask"] = batch.mask[:, 0] if mask_ndim == 3 else batch.mask
        if batch.bboxes is not None:
            result["bboxes"] = from_pascal_voc(batch.bboxes, self.bbox_params.format, batch.height, batch.width)
            result["bbox_valid"] = batch.bbox_valid
        if batch.keypoints is not None:
            result["keypoints"] = from_internal(batch.keypoints, kp_extra, self.keypoint_params).to(
                keypoints.dtype if keypoints.dtype.is_floating_point else torch.float32
            )
            result["keypoint_valid"] = batch.keypoint_valid
        return result
