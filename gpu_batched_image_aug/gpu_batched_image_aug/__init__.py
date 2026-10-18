"""GPU batched image augmentation with an albumentations-like API.

Every transform runs on whole (B, C, H, W) batches on the GPU with independent random parameters per sample.
See README.md for the supported transforms and the differences to albumentations.
"""

from . import functional
from .core import (
    BORDER_CONSTANT,
    BORDER_REFLECT,
    BORDER_REFLECT_101,
    BORDER_REPLICATE,
    BORDER_WRAP,
    INTER_AREA,
    INTER_CUBIC,
    INTER_LINEAR,
    INTER_NEAREST,
    AugBatch,
    BaseCompose,
    BaseTransform,
    BboxParams,
    Compose,
    DualTransform,
    ImageOnlyTransform,
    KeypointParams,
    OneOf,
    Rng,
    Sequential,
    SomeOf,
)
from .transforms import *  # noqa: F403
from .transforms import __all__ as _transform_names

__version__ = "0.1.0"

__all__ = [
    "AugBatch",
    "BaseCompose",
    "BaseTransform",
    "BboxParams",
    "Compose",
    "DualTransform",
    "ImageOnlyTransform",
    "KeypointParams",
    "OneOf",
    "Rng",
    "Sequential",
    "SomeOf",
    "functional",
    "BORDER_CONSTANT",
    "BORDER_REFLECT",
    "BORDER_REFLECT_101",
    "BORDER_REPLICATE",
    "BORDER_WRAP",
    "INTER_AREA",
    "INTER_CUBIC",
    "INTER_LINEAR",
    "INTER_NEAREST",
    *_transform_names,
]
