"""End-to-end statistical comparison: the same transform (same constructor arguments) applied N times with
albumentations and once to a batch of N copies here. Per run we measure the mean absolute change of the image; the
mean and std of that statistic over the runs must agree. This checks parameter sampling (ranges, distributions,
probabilities) of the random transforms, complementing the deterministic tests with fixed parameters."""

from __future__ import annotations

import warnings

import cv2
import numpy as np
import torch
from helpers import DEVICE, natural_image

import albumentations as A
import gpu_batched_image_aug as G

N = 300
IMG = natural_image(64, 64, seed=11)

PAIRS = [
    ("Affine", dict(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-30, 30), shear=(-10, 10), p=1)),
    ("Affine", dict(scale=(0.5, 1.5), balanced_scale=True, keep_ratio=True, translate_px=(-5, 5), p=1)),
    ("ShiftScaleRotate", dict(p=1)),
    ("Rotate", dict(limit=(-60, 60), border_mode=cv2.BORDER_REFLECT_101, p=1)),
    ("Perspective", dict(scale=(0.05, 0.15), p=1)),
    ("ElasticTransform", dict(alpha=40, sigma=6, p=1)),
    ("GridDistortion", dict(num_steps=5, distort_limit=0.4, p=1)),
    ("OpticalDistortion", dict(distort_limit=0.4, p=1)),
    ("OpticalDistortion", dict(distort_limit=0.4, mode="fisheye", p=1)),
    ("RandomRotate90", dict(p=1)),
    ("D4", dict(p=1)),
    ("HorizontalFlip", dict(p=0.5)),
    ("CoarseDropout", dict(num_holes_range=(1, 5), hole_height_range=(3, 20), hole_width_range=(4, 20), p=1)),
    ("CoarseDropout", dict(num_holes_range=(1, 3), hole_height_range=(0.05, 0.3), hole_width_range=(0.1, 0.2), p=1)),
    ("CoarseDropout", dict(num_holes_range=(2, 3), fill="random_uniform", p=1)),
    ("GridDropout", dict(ratio=0.4, unit_size_range=(8, 20), p=1)),
    ("GridDropout", dict(ratio=0.3, holes_number_xy=(3, 4), random_offset=False, shift_xy=(2, 3), p=1)),
    ("XYMasking", dict(num_masks_x=(1, 3), num_masks_y=(0, 2), mask_x_length=(3, 10), mask_y_length=(2, 8), p=1)),
    ("PixelDropout", dict(dropout_prob=0.1, p=1)),
    ("PixelDropout", dict(dropout_prob=0.1, per_channel=True, drop_value=None, p=1)),
    ("ChannelDropout", dict(channel_drop_range=(1, 2), p=1)),
    ("RandomBrightnessContrast", dict(p=1)),
    ("ColorJitter", dict(p=1)),
    ("ColorJitter", dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1)),
    ("HueSaturationValue", dict(p=1)),
    ("RGBShift", dict(p=1)),
    ("RandomGamma", dict(p=1)),
    ("ChannelShuffle", dict(p=1)),
    ("ToGray", dict(p=0.5)),
    ("Solarize", dict(threshold_range=(0.2, 0.8), p=1)),
    ("Posterize", dict(num_bits=(2, 6), p=1)),
    ("FancyPCA", dict(alpha=0.3, p=1)),
    ("RandomToneCurve", dict(scale=0.2, p=1)),
    ("RandomToneCurve", dict(scale=0.2, per_channel=True, p=1)),
    ("Emboss", dict(p=1)),
    ("Sharpen", dict(p=1)),
    ("UnsharpMask", dict(p=1)),
    ("GaussNoise", dict(std_range=(0.02, 0.1), p=1)),
    ("GaussNoise", dict(std_range=(0.05, 0.1), mean_range=(-0.05, 0.05), per_channel=False, p=1)),
    ("MultiplicativeNoise", dict(multiplier=(0.7, 1.3), per_channel=True, p=1)),
    ("MultiplicativeNoise", dict(multiplier=(0.8, 1.2), elementwise=True, p=1)),
    ("ISONoise", dict(p=1)),
    ("SaltAndPepper", dict(p=1)),
    ("ShotNoise", dict(p=1)),
    ("Blur", dict(blur_limit=(3, 9), p=1)),
    ("GaussianBlur", dict(blur_limit=(3, 9), sigma_limit=(0.5, 2.5), p=1)),
    ("GaussianBlur", dict(p=1)),
    ("MedianBlur", dict(blur_limit=(3, 7), p=1)),
    ("MotionBlur", dict(blur_limit=(3, 11), p=1)),
    ("Defocus", dict(p=1)),
    ("ZoomBlur", dict(p=1)),
    ("RandomShadow", dict(p=1)),
    ("RandomShadow", dict(num_shadows_limit=(1, 3), shadow_intensity_range=(0.2, 0.8), p=1)),
    ("RandomFog", dict(p=1)),
    ("RandomRain", dict(p=1)),
    ("RandomRain", dict(rain_type="heavy", drop_width=2, p=1)),
    ("RandomSnow", dict(p=1)),
    ("ImageCompression", dict(quality_range=(10, 60), p=1)),
    ("Downscale", dict(scale_range=(0.2, 0.6), p=1)),
    ("Downscale", dict(scale_range=(0.25, 0.5), interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=1)),
]


def _albu_stats(name: str, kwargs: dict) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = getattr(A, name)(**kwargs)
        t.set_random_seed(0)
        vals = [np.abs(t(image=IMG)["image"].astype(np.float64) - IMG).mean() for _ in range(N)]
    return np.array(vals)


def _ours_stats(name: str, kwargs: dict) -> np.ndarray:
    t = getattr(G, name)(**kwargs)
    imgs = torch.from_numpy(IMG).permute(2, 0, 1)[None].expand(N, -1, -1, -1).contiguous().to(DEVICE)
    out = G.Compose([t], seed=0)(image=imgs)["image"]
    return (out.double() - imgs.double()).abs().mean(dim=(1, 2, 3)).cpu().numpy()


def test_output_change_distributions_match_albumentations():
    failures = []
    for name, kwargs in PAIRS:
        try:
            a, o = _albu_stats(name, kwargs), _ours_stats(name, kwargs)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}{kwargs}: {type(e).__name__}: {e}")
            print(f"  FAIL {failures[-1]}")
            continue
        mean_tol = 0.08 * a.mean() + 0.4 + 3 * a.std() / np.sqrt(N)
        std_tol = 0.2 * a.std() + 0.4
        ok = abs(o.mean() - a.mean()) <= mean_tol and abs(o.std() - a.std()) <= std_tol
        line = f"{name}{kwargs}: change mean albu {a.mean():.2f} / ours {o.mean():.2f}, std albu {a.std():.2f} / ours {o.std():.2f}"
        if not ok:
            failures.append(line)
        print(("  ok   " if ok else "  FAIL ") + line)
    assert not failures, "\n".join(failures)
