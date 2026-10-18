"""Pipeline semantics: dtypes / shapes, probabilities, compositions, seeding, mask & bbox consistency, and a smoke
test of every transform."""

from __future__ import annotations

import numpy as np
import torch
from helpers import DEVICE

import gpu_batched_image_aug as G


def _batch(b=16, c=3, h=64, w=64, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = (torch.rand(b, c, h, w, generator=g) * 255).to(torch.uint8).to(DEVICE)
    mask = (torch.rand(b, h, w, generator=g) * 4).long().to(DEVICE)
    x1 = torch.rand(b, 5, generator=g) * (w * 0.6)
    y1 = torch.rand(b, 5, generator=g) * (h * 0.6)
    boxes = torch.stack([x1, y1, x1 + 6 + torch.rand(b, 5, generator=g) * 15, y1 + 6 + torch.rand(b, 5, generator=g) * 15], -1).to(DEVICE)
    return img, mask, boxes


ALL_TRANSFORMS = [
    G.HorizontalFlip(p=1), G.VerticalFlip(p=1), G.Transpose(p=1), G.RandomRotate90(p=1), G.D4(p=1),
    G.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-30, 30), shear=(-10, 10), p=1),
    G.ShiftScaleRotate(p=1), G.Rotate(limit=45, border_mode=G.BORDER_REFLECT_101, p=1), G.Perspective(p=1),
    G.ElasticTransform(alpha=30, sigma=6, p=1), G.GridDistortion(p=1), G.OpticalDistortion(distort_limit=0.3, p=1),
    G.CoarseDropout(num_holes_range=(1, 4), fill="random", fill_mask=0, p=1), G.GridDropout(p=1), G.XYMasking(num_masks_x=(1, 2), mask_x_length=(4, 10), p=1),
    G.PixelDropout(dropout_prob=0.05, mask_drop_value=0, p=1), G.ChannelDropout(p=1),
    G.RandomBrightnessContrast(p=1), G.ColorJitter(p=1), G.HueSaturationValue(p=1), G.RGBShift(p=1), G.RandomGamma(p=1),
    G.ChannelShuffle(p=1), G.ToGray(p=1), G.ToSepia(p=1), G.InvertImg(p=1), G.Solarize(p=1), G.Posterize(p=1),
    G.Equalize(p=1), G.Equalize(by_channels=False, p=1), G.FancyPCA(p=1), G.RandomToneCurve(per_channel=True, p=1), G.Emboss(p=1),
    G.Sharpen(p=1), G.Sharpen(method="gaussian", p=1), G.UnsharpMask(p=1),
    G.GaussNoise(p=1), G.GaussNoise(noise_scale_factor=0.5, per_channel=False, p=1), G.MultiplicativeNoise(per_channel=True, elementwise=True, p=1),
    G.ISONoise(p=1), G.SaltAndPepper(p=1), G.ShotNoise(p=1),
    G.Blur(p=1), G.GaussianBlur(p=1), G.MedianBlur(p=1), G.MotionBlur(p=1), G.Defocus(p=1), G.ZoomBlur(p=1),
    G.RandomShadow(p=1), G.RandomFog(p=1), G.RandomRain(p=1), G.RandomRain(drop_width=2, rain_type="heavy", p=1), G.RandomSnow(p=1),
    G.ImageCompression(quality_range=(20, 80), p=1), G.Downscale(scale_range=(0.3, 0.6), interpolation_pair={"downscale": 3, "upscale": 1}, p=1),
]  # fmt: skip


def test_every_transform_runs_with_mask_and_boxes():
    img, mask, boxes = _batch()
    for t in ALL_TRANSFORMS:
        aug = G.Compose([t], bbox_params=G.BboxParams(format="pascal_voc", min_visibility=0.1), seed=0)
        out = aug(image=img, mask=mask, bboxes=boxes)
        assert out["image"].dtype == torch.uint8 and out["image"].shape == img.shape, t
        assert out["mask"].dtype == mask.dtype and out["mask"].shape == mask.shape, t
        assert out["bboxes"].shape == boxes.shape and out["bbox_valid"].shape == boxes.shape[:2], t
        assert torch.isfinite(out["bboxes"]).all(), t
        v = out["bbox_valid"]
        b = out["bboxes"][v]
        assert bool((b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()), t
        assert bool((b >= 0).all() and (b[:, [0, 2]] <= 64).all() and (b[:, [1, 3]] <= 64).all()), t
        fimg = img.float() / 255
        outf = aug(image=fimg)["image"]
        assert outf.dtype == torch.float32 and torch.isfinite(outf).all() and outf.min() >= 0 and outf.max() <= 1, t


def test_size_changing_transforms():
    img, mask, boxes = _batch(h=80, w=100)
    for t, size in [(G.RandomCrop(48, 64), (48, 64)), (G.CenterCrop(40, 50), (40, 50)), (G.RandomResizedCrop((32, 40)), (32, 40)), (G.Resize(50, 30), (50, 30)), (G.Crop(10, 5, 60, 45), (40, 50))]:
        out = G.Compose([t], bbox_params=G.BboxParams()) (image=img, mask=mask, bboxes=boxes)
        assert out["image"].shape[-2:] == size and out["mask"].shape[-2:] == size
        b = out["bboxes"][out["bbox_valid"]]
        assert bool((b[:, [0, 2]] <= size[1]).all() and (b[:, [1, 3]] <= size[0]).all())
    try:
        G.Compose([G.OneOf([G.RandomCrop(32, 32), G.HorizontalFlip()])])(image=img)
        raise AssertionError("size changing transform inside OneOf must raise")
    except ValueError:
        pass


def test_probabilities_and_compositions():
    n = 4000
    zeros = torch.zeros(n, 1, 8, 8, device=DEVICE)
    out = G.Compose([G.InvertImg(p=0.3)], seed=1)(image=zeros)["image"]
    frac = (out[:, 0, 0, 0] > 0.5).float().mean().item()
    assert abs(frac - 0.3) < 0.03
    out = G.Compose([G.InvertImg(p=1.0)], p=0.4, seed=2)(image=zeros)["image"]
    assert abs((out[:, 0, 0, 0] > 0.5).float().mean().item() - 0.4) < 0.03
    # OneOf: one child per sample, chosen proportionally to the children's p
    marks = [G.Solarize(threshold_range=(0, 0), p=w) for w in (0.25, 0.75)]
    marks[0].apply = lambda img, **k: torch.full_like(img, 0.25)  # noqa: E731
    marks[1].apply = lambda img, **k: torch.full_like(img, 0.75)  # noqa: E731
    out = G.Compose([G.OneOf(marks, p=0.8)], seed=3)(image=zeros)["image"][:, 0, 0, 0]
    assert abs((out == 0.25).float().mean().item() - 0.2) < 0.03 and abs((out == 0.75).float().mean().item() - 0.6) < 0.03
    # SomeOf: exactly n distinct children (each with own p=1) -> applied in order
    add = [G.Solarize(p=1) for _ in range(4)]
    for i, t in enumerate(add):
        t.apply = lambda img, i=i, **k: img + 10 ** i  # noqa: E731
    out = G.Compose([G.SomeOf(add, n=2)], seed=4)(image=zeros)["image"][:, 0, 0, 0].cpu().numpy()
    digits = np.stack([(out // 10**i) % 10 for i in range(4)], 1)
    assert set(np.unique(digits.sum(1))) == {2} and digits.max() == 1


def test_seed_reproducibility_and_per_sample_independence():
    img, _, _ = _batch(b=8)
    aug = G.Compose([G.Affine(rotate=(-40, 40), p=1), G.GaussNoise(p=1)], seed=123)
    a = aug(image=img)["image"]
    aug2 = G.Compose([G.Affine(rotate=(-40, 40), p=1), G.GaussNoise(p=1)], seed=123)
    assert torch.equal(a, aug2(image=img)["image"])
    same = img[:1].expand(8, -1, -1, -1).contiguous()
    o = G.Compose([G.Affine(rotate=(-40, 40), p=1)], seed=5)(image=same)["image"]
    assert all(not torch.equal(o[0], o[i]) for i in range(1, 8)), "each sample must get its own parameters"


def test_mask_follows_image_geometry():
    b, h, w = 8, 64, 64
    labels = (torch.arange(h * w, device=DEVICE).view(h, w) % 7).expand(b, h, w).contiguous()
    img = (labels.float() / 6).unsqueeze(1).expand(b, 3, h, w).contiguous()
    for t in [G.Affine(rotate=(-30, 30), scale=(0.8, 1.2), interpolation=0, p=1), G.Perspective(interpolation=0, p=1), G.ElasticTransform(alpha=20, sigma=5, interpolation=0, p=1),
              G.GridDistortion(interpolation=0, p=1), G.D4(p=1), G.RandomResizedCrop((48, 48), interpolation=0, p=1)]:
        out = G.Compose([t], seed=0)(image=img, mask=labels)
        img_lab = (out["image"][:, 0] * 6).round().long()
        assert torch.equal(img_lab, out["mask"]), t


def test_bbox_formats_round_trip_and_filtering():
    img, _, boxes = _batch(b=4, h=60, w=80)
    for fmt in ("pascal_voc", "albumentations", "coco", "yolo"):
        from gpu_batched_image_aug.core.bbox import from_pascal_voc

        inp = from_pascal_voc(boxes, fmt, 60, 80)
        out = G.Compose([], bbox_params=G.BboxParams(format=fmt))(image=img, bboxes=inp)
        assert torch.allclose(out["bboxes"], inp, atol=1e-4)
    # shifting everything out of the image invalidates the boxes
    out = G.Compose([G.Affine(translate_px={"x": (200, 200), "y": (0, 0)}, p=1)], bbox_params=G.BboxParams())(image=img, bboxes=boxes)
    assert not bool(out["bbox_valid"].any())
    # half visible boxes: kept with min_visibility 0.3, removed with 0.7
    b = torch.tensor([[[70.0, 10, 90, 30]]], device=DEVICE)
    for mv, keep in ((0.3, True), (0.7, False)):
        out = G.Compose([], bbox_params=G.BboxParams(min_visibility=mv))(image=img[:1], bboxes=b)
        assert bool(out["bbox_valid"][0, 0]) == keep
        if keep:
            assert torch.allclose(out["bboxes"][0, 0], torch.tensor([70.0, 10, 80, 30], device=DEVICE))


def test_fill_values_follow_input_scale():
    img = torch.zeros(2, 3, 32, 32, dtype=torch.uint8, device=DEVICE)
    t = G.Affine(translate_px={"x": (16, 16), "y": (0, 0)}, fill=(255, 128, 0), p=1)
    out = G.Compose([t])(image=img)["image"]
    assert out[0, :, 5, 5].tolist() == [255, 128, 0]
    out = G.Compose([t])(image=img.float() / 255)["image"]
    t2 = G.Affine(translate_px={"x": (16, 16), "y": (0, 0)}, fill=(1.0, 0.5, 0.0), p=1)
    out = G.Compose([t2])(image=img.float() / 255)["image"]
    assert torch.allclose(out[0, :, 5, 5], torch.tensor([1.0, 0.5, 0.0], device=DEVICE))
