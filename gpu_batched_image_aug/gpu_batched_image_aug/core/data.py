"""Internal batch container passed between transforms."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .bbox import BboxParams
from .keypoints import KeypointParams


@dataclass
class AugBatch:
    """
    - image: (B, C, H, W) float32, values in [0, 1] (albumentations' float convention, max value 1.0)
    - mask: (B, Cm, H, W) or None. Any dtype; geometric transforms resample it (nearest by default) and keep the dtype
    - bboxes: (B, N, 4) float32 pascal_voc pixel coordinates, or None
    - bbox_valid: (B, N) bool, or None
    - bbox_params: filtering settings, needed by transforms that drop boxes themselves (e.g. CoarseDropout)
    - keypoints: (B, K, 4) float32 internal [x, y, angle (rad), scale] (pixel-center coordinates), or None
    - keypoint_valid: (B, K) bool, or None
    - keypoint_params: keypoint settings (remove_invisible is needed by transforms that drop keypoints themselves)
    - max_value: value scale of the *input* images (255 for uint8 input, 1.0 for float input). Fill values given by the
      user are in this scale (like albumentations) and are divided by it internally.
    """

    image: torch.Tensor
    mask: torch.Tensor | None = None
    bboxes: torch.Tensor | None = None
    bbox_valid: torch.Tensor | None = None
    bbox_params: BboxParams | None = None
    keypoints: torch.Tensor | None = None
    keypoint_valid: torch.Tensor | None = None
    keypoint_params: KeypointParams | None = None
    max_value: float = 1.0

    @property
    def batch_size(self) -> int:
        return self.image.shape[0]

    @property
    def height(self) -> int:
        return self.image.shape[-2]

    @property
    def width(self) -> int:
        return self.image.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.image.device

    def select(self, idx: torch.Tensor) -> AugBatch:
        return replace(
            self,
            image=self.image[idx],
            mask=None if self.mask is None else self.mask[idx],
            bboxes=None if self.bboxes is None else self.bboxes[idx],
            bbox_valid=None if self.bbox_valid is None else self.bbox_valid[idx],
            keypoints=None if self.keypoints is None else self.keypoints[idx],
            keypoint_valid=None if self.keypoint_valid is None else self.keypoint_valid[idx],
        )

    def update(self, idx: torch.Tensor, sub: AugBatch) -> None:
        """Write a processed subset back (in place). The subset must keep the spatial size."""
        if sub.image.shape[1:] != self.image.shape[1:]:
            raise RuntimeError("A transform applied to a subset of the batch changed the image shape.")
        self.image[idx] = sub.image
        if self.mask is not None:
            self.mask[idx] = sub.mask
        if self.bboxes is not None:
            self.bboxes[idx] = sub.bboxes
            self.bbox_valid[idx] = sub.bbox_valid
        if self.keypoints is not None:
            self.keypoints[idx] = sub.keypoints
            self.keypoint_valid[idx] = sub.keypoint_valid
