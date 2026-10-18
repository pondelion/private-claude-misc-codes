"""Geometric transforms vs albumentations / OpenCV with identical parameters."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from helpers import DEVICE, interior, mae, natural_image, rng_np, to_np, to_t

import albumentations.augmentations.crops.functional as fcrops
import albumentations.augmentations.geometric.functional as fg
import gpu_batched_image_aug as G
from gpu_batched_image_aug.core.bbox import pixel_center_to_edge_matrix, transform_boxes_by_matrix
from gpu_batched_image_aug.functional import geometric as fgeo

BORDERS = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_WRAP]


def _boxes(h, w, n=6, seed=0):
    g = rng_np(seed)
    x1 = g.uniform(0, w * 0.7, n)
    y1 = g.uniform(0, h * 0.7, n)
    x2 = np.minimum(x1 + g.uniform(5, w * 0.3, n), w)
    y2 = np.minimum(y1 + g.uniform(5, h * 0.3, n), h)
    return np.stack([x1, y1, x2, y2], 1).astype(np.float32)


def _norm(b, h, w):
    return b / np.array([w, h, w, h], dtype=np.float32)


def test_flips_rot90_d4_transpose_exact():
    s = 64
    img = natural_image(s, s)
    mask = (rng_np(1).integers(0, 5, (s, s))).astype(np.uint8)
    boxes = _boxes(s, s)
    t_img, t_mask = to_t(img), torch.from_numpy(mask)[None, None].to(DEVICE)
    t_box = torch.from_numpy(boxes)[None].to(DEVICE)
    valid = torch.ones(1, len(boxes), dtype=torch.bool, device=DEVICE)

    cases = [
        (G.HorizontalFlip(), {}, fg.hflip, fg.bboxes_hflip),
        (G.VerticalFlip(), {}, fg.vflip, fg.bboxes_vflip),
        (G.Transpose(), {}, fg.transpose, fg.bboxes_transpose),
    ]
    for k in range(4):
        cases.append((G.RandomRotate90(), {"factor": torch.tensor([k], device=DEVICE)}, lambda x, k=k: fg.rot90(x, k), lambda b, k=k: fg.bboxes_rot90(b, k)))
    for i, e in enumerate(["e", "r90", "r180", "r270", "v", "hvt", "h", "t"]):
        cases.append((G.D4(), {"element": torch.tensor([i], device=DEVICE)}, lambda x, e=e: fg.d4(x, e), lambda b, e=e: fg.bboxes_d4(b, e)))

    for t, params, f_img, f_box in cases:
        out = to_np(t.apply(t_img, **params))
        assert np.array_equal(out, f_img(img)), f"{t} image"
        out_m = t.apply_to_mask(t_mask, **params)[0, 0].cpu().numpy()
        assert np.array_equal(out_m, f_img(mask)), f"{t} mask"
        ob, _ = t.apply_to_bboxes(t_box, valid, s, s, **params)
        ref = f_box(_norm(boxes, s, s)) * s
        assert np.allclose(ob[0].cpu().numpy(), ref, atol=1e-4), f"{t} bboxes"


def test_affine_matrix_and_warp_match_cv2():
    h, w = 90, 120
    img = natural_image(h, w)
    for seed, (tx, ty, shx, shy, sx, sy, rot) in enumerate(
        [(0, 0, 0, 0, 1, 1, 30), (7, -5, 10, -8, 0.8, 1.2, -17), (-12, 9, 0, 0, 1.3, 1.3, 133), (3, 3, -20, 15, 0.9, 0.7, 251)]
    ):
        ref_m = fg.create_affine_transformation_matrix({"x": tx, "y": ty}, {"x": shx, "y": shy}, {"x": sx, "y": sy}, rot, fg.center((h, w)))
        m = fgeo.affine_matrix(
            torch.tensor([[tx, ty]], device=DEVICE, dtype=torch.float32),
            torch.tensor([[shx, shy]], device=DEVICE, dtype=torch.float32),
            torch.tensor([[sx, sy]], device=DEVICE, dtype=torch.float32),
            torch.tensor([rot], device=DEVICE, dtype=torch.float32),
            (w / 2 - 0.5, h / 2 - 0.5),
        )
        assert np.allclose(m[0].cpu().numpy(), ref_m, atol=1e-4), "affine matrix"
        for border in BORDERS:
            for interp in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC):
                ref = fg.warp_affine(img, ref_m, interp, (10, 200, 30), border, (h, w))
                fill = [10 / 255, 200 / 255, 30 / 255]
                out = to_np(fgeo.warp(to_t(img), m, None, interp, border, fill))
                err = mae(interior(out), interior(ref))
                lim = 1.0 if interp != cv2.INTER_NEAREST else 2.5
                assert err < lim, f"affine seed{seed} border{border} interp{interp}: MAE {err:.3f}"


def test_affine_bboxes_match_albumentations():
    h, w = 90, 120
    boxes = _boxes(h, w, 8)
    for method in ("largest_box", "ellipse"):
        for tx, ty, shx, shy, sx, sy, rot in [(5, -3, 0, 0, 1.1, 0.9, 25), (0, 0, 12, -7, 1, 1, -60)]:
            args = ({"x": tx, "y": ty}, {"x": shx, "y": shy}, {"x": sx, "y": sy}, rot)
            m_px = fg.create_affine_transformation_matrix(*args, fg.center((h, w)))
            m_box = fg.create_affine_transformation_matrix(*args, fg.center_bbox((h, w)))
            ref = fg.bboxes_affine(_norm(boxes, h, w), m_box, method, (h, w), cv2.BORDER_CONSTANT, (h, w)) * [w, h, w, h]
            m = pixel_center_to_edge_matrix(torch.tensor(m_px, device=DEVICE)[None])
            assert np.allclose(m[0].cpu().numpy(), m_box, atol=1e-6), "edge matrix == albumentations bbox_matrix"
            out = transform_boxes_by_matrix(torch.from_numpy(boxes)[None].to(DEVICE), m, method)[0].cpu().numpy()
            # albumentations validate_bboxes drops boxes that end up completely outside the image
            keep = (out[:, 2] > 0) & (out[:, 3] > 0) & (out[:, 0] < w) & (out[:, 1] < h)
            assert keep.sum() == len(ref)
            assert np.allclose(out[keep], ref, atol=1e-2), f"{method}: {np.abs(out[keep] - ref).max()}"


def test_perspective_matches_albumentations():
    h, w = 80, 100
    img = natural_image(h, w)
    g = rng_np(3)
    for _ in range(3):
        pts = fg.generate_perspective_points((h, w), 0.08, g)
        pts = fg.order_points(pts)
        ref_m, mw, mh = fg.compute_perspective_params(pts.copy(), (h, w))
        ref = fg.perspective(img, ref_m, mw, mh, 0, cv2.BORDER_CONSTANT, True, cv2.INTER_LINEAR)
        m = fgeo.perspective_transform_from_points(
            torch.tensor(pts, device=DEVICE)[None],
            torch.tensor([[0, 0], [w, 0], [w, h], [0, h]], dtype=torch.float64, device=DEVICE)[None],
        )
        s = torch.diag(torch.tensor([w / mw, h / mh, 1.0], dtype=torch.float64, device=DEVICE))[None]
        out = to_np(fgeo.warp(to_t(img), s @ m, None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0.0))
        assert mae(interior(out), interior(ref)) < 1.0
        # boxes: albumentations applies the same matrix to box coordinates
        boxes = _boxes(h, w, 5)
        ref_b = fg.perspective_bboxes(_norm(boxes, h, w), (h, w), ref_m, mw, mh, True) * [w, h, w, h]
        out_b = transform_boxes_by_matrix(torch.from_numpy(boxes)[None].to(DEVICE), (s @ m), "largest_box")[0].cpu().numpy()
        assert np.allclose(out_b, ref_b, atol=1e-2)


def test_perspective_transform_params_order():
    """Perspective.get_params reproduces albumentations' point ordering and size computation."""
    pts = torch.tensor([[[5.0, 3.0], [95.0, 7.0], [92.0, 78.0], [2.0, 74.0]]], dtype=torch.float64)
    shuffled = pts[:, [2, 0, 3, 1]]
    from gpu_batched_image_aug.transforms.geometric import _order_points

    assert torch.equal(_order_points(shuffled), pts)
    assert np.allclose(fg.order_points(shuffled[0].numpy()), pts[0].numpy())


def test_remap_matches_cv2():
    h, w = 70, 90
    img = natural_image(h, w)
    g = rng_np(5)
    dx = cv2.GaussianBlur(g.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 6)
    dy = cv2.GaussianBlur(g.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 6)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xx + dx, yy + dy
    for border in BORDERS:
        for interp in (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC):
            ref = cv2.remap(img, mx, my, interp, borderMode=border, borderValue=(0, 0, 0))
            out = to_np(fgeo.sample(to_t(img), torch.from_numpy(mx)[None].to(DEVICE), torch.from_numpy(my)[None].to(DEVICE), interp, border, 0.0))
            err = mae(interior(out), interior(ref))
            assert err < (2.5 if interp == cv2.INTER_NEAREST else 1.2), f"remap border{border} interp{interp}: {err:.3f}"


def test_elastic_field_blur_matches_cv2():
    g = rng_np(7)
    for sigma, ksize in [(50, 0), (10, 0), (6, 17)]:
        f = g.standard_normal((60, 80)).astype(np.float32)
        ref = cv2.GaussianBlur(f, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
        out = fgeo.gaussian_blur_fields(torch.from_numpy(f)[None].to(DEVICE), sigma, ksize)[0].cpu().numpy()
        assert np.allclose(out, ref, atol=1e-5), f"sigma={sigma}: {np.abs(out - ref).max()}"


def test_grid_distortion_maps_match_albumentations():
    for h, w, steps in [(64, 80, 5), (77, 91, 4), (50, 50, 7)]:
        g = rng_np(steps)
        sx = [1 + g.uniform(-0.3, 0.3) for _ in range(steps + 1)]
        sy = [1 + g.uniform(-0.3, 0.3) for _ in range(steps + 1)]
        norm = fg.normalize_grid_distortion_steps((h, w), steps, list(sx), list(sy))
        ref_x, ref_y = fg.generate_grid((h, w), norm["steps_x"], norm["steps_y"], steps)
        t = G.GridDistortion(num_steps=steps)
        tx = t._normalize(torch.tensor([sx], device=DEVICE, dtype=torch.float32), w)
        ty = t._normalize(torch.tensor([sy], device=DEVICE, dtype=torch.float32), h)
        out_x = t._axis_map(tx, w)[0].cpu().numpy()
        out_y = t._axis_map(ty, h)[0].cpu().numpy()
        assert np.allclose(out_x, ref_x[0], atol=1e-3) and np.allclose(out_y, ref_y[:, 0], atol=1e-3)


def test_optical_distortion_maps_match_albumentations():
    h, w = 60, 84
    batch = G.AugBatch(image=torch.zeros(1, 3, h, w, device=DEVICE))
    for mode, fn in (("camera", fg.get_camera_matrix_distortion_maps), ("fisheye", fg.get_fisheye_distortion_maps)):
        for k in (-0.05, 0.03, 0.3):
            t = G.OpticalDistortion(distort_limit=(k, k), mode=mode)
            p = t.get_params(batch, G.Rng(DEVICE, 0))
            rx, ry = fn((h, w), k)
            assert np.allclose(p["map_x"][0].cpu().numpy(), rx, atol=2e-3), f"{mode} k={k}"
            assert np.allclose(p["map_y"][0].cpu().numpy(), ry, atol=2e-3), f"{mode} k={k}"


def test_remap_bboxes_match_albumentations():
    h, w = 64, 80
    g = rng_np(9)
    dx = cv2.GaussianBlur(g.normal(0, 30, (h, w)).astype(np.float32), (0, 0), 8)
    dy = cv2.GaussianBlur(g.normal(0, 30, (h, w)).astype(np.float32), (0, 0), 8)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xx + dx, yy + dy
    boxes = _boxes(h, w, 6, seed=2)
    ref = fg.remap_bboxes(boxes.copy(), mx, my, (h, w))
    t = G.ElasticTransform()
    valid = torch.ones(1, len(boxes), dtype=torch.bool, device=DEVICE)
    out, ok = t.apply_to_bboxes(
        torch.from_numpy(boxes)[None].to(DEVICE), valid, h, w, map_x=torch.from_numpy(mx)[None].to(DEVICE), map_y=torch.from_numpy(my)[None].to(DEVICE)
    )
    out = out[0].cpu().numpy()
    nonempty = ok[0].cpu().numpy()
    assert np.allclose(out[nonempty], ref[nonempty], atol=1.01), np.abs(out - ref).max()


def test_resize_matches_cv2():
    img = natural_image(90, 120)
    for interp, (oh, ow) in [(cv2.INTER_LINEAR, (45, 60)), (cv2.INTER_LINEAR, (130, 70)), (cv2.INTER_NEAREST, (37, 53)), (cv2.INTER_CUBIC, (64, 64)), (cv2.INTER_AREA, (45, 60))]:
        ref = cv2.resize(img, (ow, oh), interpolation=interp)
        out = to_np(G.Resize(oh, ow, interpolation=interp).apply(to_t(img)))
        assert mae(out, ref) < 0.6, f"resize interp{interp} -> {(oh, ow)}: {mae(out, ref):.3f}"


def test_random_resized_crop_given_crop_matches_albumentations():
    img = natural_image(90, 120)
    for crop, interp in [((10, 5, 90, 70), cv2.INTER_LINEAR), ((0, 0, 120, 90), cv2.INTER_LINEAR), ((33, 20, 61, 77), cv2.INTER_NEAREST)]:
        ref = cv2.resize(fcrops.crop(img, *crop), (64, 48), interpolation=interp)
        t = G.RandomResizedCrop((48, 64), interpolation=interp)
        out = to_np(t.apply(to_t(img), crop=torch.tensor([crop], device=DEVICE)))
        assert mae(out, ref) < 0.6, f"{crop}: {mae(out, ref):.3f}"
        boxes = _boxes(90, 120, 4)
        cropped = fcrops.crop_bboxes_by_coords(_norm(boxes, 90, 120), crop, (90, 120))
        ref_b = cropped * [64, 48, 64, 48]
        out_b, _ = t.apply_to_bboxes(torch.from_numpy(boxes)[None].to(DEVICE), None, 90, 120, crop=torch.tensor([crop], device=DEVICE))
        assert np.allclose(out_b[0].cpu().numpy(), ref_b, atol=1e-3)
