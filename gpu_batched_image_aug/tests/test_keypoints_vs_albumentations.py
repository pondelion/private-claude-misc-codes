"""Keypoints vs albumentations: functional comparisons with identical parameters + end-to-end pipelines."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from helpers import DEVICE, natural_image, rng_np

import albumentations as A
import albumentations.augmentations.dropout.functional as fdrop
import albumentations.augmentations.geometric.functional as fg
import gpu_batched_image_aug as G
from gpu_batched_image_aug.core.data import AugBatch
from gpu_batched_image_aug.core.keypoints import KeypointParams, from_internal, to_internal
from gpu_batched_image_aug.core.random import Rng

ATOL = 1e-3


def _kps(h, w, n=12, seed=0):
    """Internal (n, 4) [x, y, angle rad in [0, 2pi), scale], inside the image."""
    g = rng_np(seed)
    x = g.uniform(0, w - 1, n)
    y = g.uniform(0, h - 1, n)
    a = g.uniform(0, 2 * math.pi, n)
    s = g.uniform(0.5, 3.0, n)
    return np.stack([x, y, a, s], 1).astype(np.float32)


def _to_albu(k):
    """internal (n, 4) -> albumentations internal (n, 5) [x, y, z, angle, scale]."""
    return np.stack([k[:, 0], k[:, 1], np.zeros(len(k), np.float32), k[:, 2], k[:, 3]], 1).astype(np.float32)


def _from_albu(k):
    return k[:, [0, 1, 3, 4]]


def _t(k):
    return torch.from_numpy(k)[None].to(DEVICE)


def _ones(n):
    return torch.ones(1, n, dtype=torch.bool, device=DEVICE)


def _angle_close(a, b, atol=ATOL):
    d = np.abs(np.remainder(a - b + math.pi, 2 * math.pi) - math.pi)
    return bool((d < atol).all())


def _assert_kp_close(ours, ref, name):
    assert ours.shape == ref.shape, f"{name}: shape {ours.shape} vs {ref.shape}"
    assert np.allclose(ours[:, :2], ref[:, :2], atol=ATOL), f"{name} xy\n{ours[:, :2]}\n{ref[:, :2]}"
    assert _angle_close(ours[:, 2], ref[:, 2]), f"{name} angle\n{ours[:, 2]}\n{ref[:, 2]}"
    assert np.allclose(ours[:, 3], ref[:, 3], atol=ATOL, rtol=1e-4), f"{name} scale"


def _run(t, k, h, w, params, **extra):
    """apply_to_keypoints + the angle wrap DualTransform.apply_batch does."""
    out, valid = t.apply_to_keypoints(_t(k), _ones(len(k)), h, w, **params, **extra)
    out = out.clone()
    out[..., 2] = torch.remainder(out[..., 2], 2 * math.pi)
    return out[0].cpu().numpy(), valid[0].cpu().numpy()


def test_format_roundtrip():
    g = rng_np(0)
    for fmt in ("xy", "yx", "xya", "xys", "xyas", "xysa"):
        for deg in (True, False):
            p = KeypointParams(format=fmt, angle_in_degrees=deg)
            data = g.uniform(0, 50, (2, 5, len(fmt) + 2)).astype(np.float32)
            if "a" in fmt:
                data[..., fmt.index("a")] = g.uniform(0, 360 if deg else 2 * math.pi, (2, 5))
            x = torch.from_numpy(data)
            internal, extra = to_internal(x, p)
            back = from_internal(internal, extra, p)
            assert torch.allclose(back, x, atol=1e-4), fmt


def test_flips_rot90_d4_transpose_keypoints():
    s = 64
    k = _kps(s, s)
    ka = _to_albu(k)
    cases = [
        (G.HorizontalFlip(), {}, lambda x: fg.keypoints_hflip(x, s)),
        (G.VerticalFlip(), {}, lambda x: fg.keypoints_vflip(x, s)),
        (G.Transpose(), {}, fg.keypoints_transpose),
    ]
    for r in range(4):
        cases.append((G.RandomRotate90(), {"factor": torch.tensor([r], device=DEVICE)}, lambda x, r=r: fg.keypoints_rot90(x, r, (s, s))))
    for i, e in enumerate(["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]):
        cases.append((G.D4(), {"element": torch.tensor([i], device=DEVICE)}, lambda x, e=e: fg.keypoints_d4(x, e, (s, s))))
    for t, params, ref_f in cases:
        ours, valid = _run(t, k, s, s, params)
        assert valid.all()
        _assert_kp_close(ours, _from_albu(ref_f(ka.copy())), f"{t} {params}")


def test_affine_family_keypoints():
    h, w = 90, 120
    k = _kps(h, w, seed=1)
    ka = _to_albu(k)
    rng = Rng(DEVICE, seed=0)
    batch = AugBatch(image=torch.zeros(1, 3, h, w, device=DEVICE))
    transforms = [
        G.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), rotate=(-180, 180), shear={"x": (-20, 20), "y": (-10, 10)}),
        G.Affine(scale={"x": (0.5, 0.8), "y": (1.2, 1.5)}, rotate=(-45, 45)),
        G.ShiftScaleRotate(),
        G.Rotate(limit=180),
    ]
    for t in transforms:
        for _ in range(5):
            params = t.get_params(batch, rng)
            m = params["matrix"][0].double().cpu().numpy()
            sc = params.get("scale")
            scale = {"x": 1.0, "y": 1.0} if sc is None else {"x": float(sc[0, 0]), "y": float(sc[0, 1])}
            ref = fg.keypoints_affine(ka.copy(), m, (h, w), scale, cv2.BORDER_CONSTANT)
            ours, _ = _run(t, k, h, w, params)
            _assert_kp_close(ours, _from_albu(ref), type(t).__name__)


def test_perspective_keypoints():
    h, w = 90, 120
    k = _kps(h, w, seed=2)
    ka = _to_albu(k)
    rng = Rng(DEVICE, seed=0)
    batch = AugBatch(image=torch.zeros(1, 3, h, w, device=DEVICE))
    t = G.Perspective(scale=(0.05, 0.2))
    for _ in range(8):
        params = t.get_params(batch, rng)
        m = params["kp_matrix"][0].double().cpu().numpy()
        r = params["kp_resize"][0].double().cpu().numpy()
        max_w, max_h = int(round(w / r[0])), int(round(h / r[1]))
        ref = fg.perspective_keypoints(ka.copy(), (h, w), m, max_w, max_h, True)
        ours, _ = _run(t, k, h, w, params)
        _assert_kp_close(ours, _from_albu(ref), "Perspective")


def test_remap_keypoints_mask_method():
    h, w = 80, 100
    rng = Rng(DEVICE, seed=0)
    batch = AugBatch(image=torch.zeros(1, 3, h, w, device=DEVICE))
    transforms = [
        G.ElasticTransform(alpha=40, sigma=6),
        G.GridDistortion(num_steps=5, distort_limit=0.4),
        G.OpticalDistortion(distort_limit=0.5, mode="camera"),
        G.OpticalDistortion(distort_limit=0.5, mode="fisheye"),
    ]
    for seed, t in enumerate(transforms):
        # include overlapping keypoints (later ones overwrite earlier ones) and points near / outside the border
        k = _kps(h, w, n=40, seed=seed)
        k[5, :2] = k[4, :2] + 0.6
        k[6, :2] = (-0.4, 10.0)
        k[7, :2] = (w - 0.6, h - 0.6)
        for _ in range(3):
            params = t.get_params(batch, rng)
            mx = params["map_x"][0].cpu().numpy().astype(np.float32)
            my = params["map_y"][0].cpu().numpy().astype(np.float32)
            # albumentations returns only the surviving keypoints; tag them with their index in an extra column
            ka = np.concatenate([_to_albu(k), np.arange(len(k), dtype=np.float32)[:, None]], 1)
            ref = fg.remap_keypoints_via_mask(ka, mx, my, (h, w))
            ours, valid = _run(t, k, h, w, params)
            ref_idx = ref[:, 5].astype(int)
            assert np.array_equal(np.nonzero(valid)[0], ref_idx), f"{t}: survivors differ"
            _assert_kp_close(ours[ref_idx], _from_albu(ref[:, :5]), type(t).__name__)


def test_dropout_keypoints():
    h, w = 64, 80
    k = _kps(h, w, n=200, seed=3)
    rng = Rng(DEVICE, seed=0)
    batch = AugBatch(image=torch.zeros(1, 3, h, w, device=DEVICE))
    kp_params = KeypointParams(format="xy")
    transforms = [
        G.CoarseDropout(num_holes_range=(3, 8), hole_height_range=(5, 20), hole_width_range=(5, 20)),
        G.GridDropout(ratio=0.4),
        G.XYMasking(num_masks_x=(1, 3), num_masks_y=(1, 2), mask_x_length=(3, 15), mask_y_length=(3, 12)),
    ]
    for t in transforms:
        for _ in range(3):
            params = t.get_params(batch, rng)
            holes = params["holes"][0][params["hole_valid"][0]].cpu().numpy()
            ka = np.concatenate([_to_albu(k), np.arange(len(k), dtype=np.float32)[:, None]], 1)
            ref_idx = fdrop.filter_keypoints_in_holes(ka, holes)[:, 5].astype(int)
            _, valid = t.apply_to_keypoints(_t(k), _ones(len(k)), h, w, keypoint_params=kp_params, **params)
            assert np.array_equal(np.nonzero(valid[0].cpu().numpy())[0], ref_idx), type(t).__name__
            # remove_invisible=False keeps everything
            _, valid2 = t.apply_to_keypoints(
                _t(k), _ones(len(k)), h, w, keypoint_params=KeypointParams(remove_invisible=False), **params
            )
            assert bool(valid2.all())


def test_pipeline_keypoints_match_albumentations():
    """Deterministic pipelines end-to-end (format conversion, degrees, filtering, crops, resize, extra columns)."""
    h, w = 96, 128
    img = natural_image(h, w)
    g = rng_np(5)
    n = 15
    kp = np.stack(
        [g.uniform(0, w - 1, n), g.uniform(0, h - 1, n), g.uniform(0, 360, n), g.uniform(0.5, 3, n)], 1
    ).astype(np.float32)
    kp[0, :2] = (2.0, 3.0)  # removed by the crop
    pipelines = [
        lambda M: [M.HorizontalFlip(p=1), M.Crop(x_min=10, y_min=5, x_max=110, y_max=85), M.Resize(64, 48)],
        lambda M: [M.VerticalFlip(p=1), M.CenterCrop(64, 64), M.Transpose(p=1)],
    ]
    for fmt in ("xyas", "xysa", "xy", "yx"):
        cols = {"x": 0, "y": 1, "a": 2, "s": 3}
        data = kp[:, [cols[c] for c in fmt]]
        for make in pipelines:
            ref = A.Compose(make(A), keypoint_params=A.KeypointParams(format=fmt, remove_invisible=True))(image=img, keypoints=data)
            ours = G.Compose(make(G), keypoint_params=G.KeypointParams(format=fmt))(
                image=torch.from_numpy(img).permute(2, 0, 1)[None].to(DEVICE), keypoints=torch.from_numpy(data)[None].to(DEVICE)
            )
            got = ours["keypoints"][0][ours["keypoint_valid"][0]].cpu().numpy()
            exp = np.asarray(ref["keypoints"], dtype=np.float32)
            assert got.shape == exp.shape, (fmt, got.shape, exp.shape)
            for j, c in enumerate(fmt):
                if c == "a":
                    assert _angle_close(np.deg2rad(got[:, j]), np.deg2rad(exp[:, j])), fmt
                else:
                    assert np.allclose(got[:, j], exp[:, j], atol=ATOL), (fmt, c)


def test_pipeline_keypoints_batch_and_extra_columns():
    """Random batched pipeline: per-sample parameters, padded keypoints, extra columns passed through, the input
    validity respected and every valid output keypoint inside the image."""
    b, h, w, k = 8, 64, 64, 20
    g = rng_np(7)
    pts = np.stack([g.uniform(0, w - 1, (b, k)), g.uniform(0, h - 1, (b, k)), g.integers(0, 5, (b, k))], -1).astype(np.float32)
    kv = torch.from_numpy(g.uniform(size=(b, k)) > 0.2).to(DEVICE)
    comp = G.Compose(
        [
            G.HorizontalFlip(),
            G.D4(),
            G.OneOf([G.Affine(rotate=(-30, 30), scale=(0.8, 1.2)), G.Perspective(), G.ElasticTransform(alpha=20, sigma=5)], p=0.9),
            G.CoarseDropout(num_holes_range=(1, 4)),
            G.RandomResizedCrop((48, 48), scale=(0.5, 1.0)),
        ],
        keypoint_params=G.KeypointParams(format="xy"),
        seed=0,
    )
    out = comp(image=torch.rand(b, 3, h, w, device=DEVICE), keypoints=torch.from_numpy(pts).to(DEVICE), keypoint_valid=kv)
    kps, valid = out["keypoints"], out["keypoint_valid"]
    assert kps.shape == (b, k, 3) and valid.shape == (b, k)
    assert torch.equal(kps[..., 2], torch.from_numpy(pts[..., 2]).to(DEVICE)), "extra column changed"
    assert not bool((valid & ~kv).any()), "invalid input keypoint became valid"
    v = kps[valid]
    assert bool(((v[:, 0] >= 0) & (v[:, 0] < 48) & (v[:, 1] >= 0) & (v[:, 1] < 48)).all())
