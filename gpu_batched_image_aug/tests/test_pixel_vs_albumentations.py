"""Pixel-level transforms vs albumentations with identical parameters."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from helpers import DEVICE, interior, mae, max_abs, natural_image, psnr, rng_np, to_np, to_t

import albumentations as A
import albumentations.augmentations.blur.functional as fb
import albumentations.augmentations.dropout.functional as fd
import albumentations.augmentations.pixel.functional as fp
import gpu_batched_image_aug as G
from gpu_batched_image_aug.functional import color as fcolor
from gpu_batched_image_aug.functional.filters import albu_gaussian_kernels_1d, box_kernels, filter2d, median_filter, separable_filter
from gpu_batched_image_aug.functional.jpeg import jpeg
from gpu_batched_image_aug.transforms.blur import motion_kernels

IMG = natural_image(72, 96)
T_IMG = to_t(IMG)


def _t(x):
    return torch.tensor([x], device=DEVICE, dtype=torch.float32)


# ------------------------------------------------------------------------------------------------- color conversions
def test_color_space_round_trips_match_cv2():
    f = IMG.astype(np.float32) / 255
    for code, fn in [(cv2.COLOR_RGB2HSV, fcolor.rgb_to_hsv), (cv2.COLOR_RGB2HLS, fcolor.rgb_to_hls)]:
        ref = cv2.cvtColor(f, code)
        out = fn(T_IMG)[0].permute(1, 2, 0).cpu().numpy()
        hue_err = np.abs(((out[..., 0] - ref[..., 0]) + 180) % 360 - 180)
        assert hue_err.max() < 0.05 and np.abs(out[..., 1:] - ref[..., 1:]).max() < 1e-4
    hsv = fcolor.rgb_to_hsv(T_IMG)
    assert torch.allclose(fcolor.hsv_to_rgb(hsv), T_IMG, atol=1e-5)
    hls = fcolor.rgb_to_hls(T_IMG)
    assert torch.allclose(fcolor.hls_to_rgb(hls), T_IMG, atol=1e-5)
    ref = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2HLS), cv2.COLOR_HLS2RGB)
    assert np.abs(fcolor.hls_to_rgb(hls)[0].permute(1, 2, 0).cpu().numpy() - ref).max() < 1e-4


# ------------------------------------------------------------------------------------------------- color transforms
def test_brightness_contrast():
    for alpha, beta, by_max in [(1.15, 0.08, True), (0.85, -0.1, True), (1.1, 0.1, False)]:
        mean = IMG.mean() if not by_max else None
        ref = fp.multiply_add(IMG, alpha, beta * 255 if by_max else beta * mean)
        out = to_np(G.RandomBrightnessContrast(brightness_by_max=by_max).apply(T_IMG.clone(), alpha=_t(alpha), beta=_t(beta)))
        assert max_abs(out, ref) <= 1


def test_color_jitter_ops():
    t = G.ColorJitter()
    for i, (fn, f) in enumerate(
        [(fp.adjust_brightness_torchvision, 1.3), (fp.adjust_contrast_torchvision, 0.7), (fp.adjust_saturation_torchvision, 1.6), (fp.adjust_hue_torchvision, 0.12)]
    ):
        ref = fn(IMG.astype(np.float32) / 255, f)
        ref = np.clip(ref, 0, 1)
        out = t._op(T_IMG.clone(), i, _t(f))[0].permute(1, 2, 0).cpu().numpy()
        assert np.abs(out - ref).max() < 2e-3, f"op {i}: {np.abs(out - ref).max()}"


def test_hue_saturation_value():
    for hs, ss, vs in [(20, 0, 0), (-35, 25, -15), (7, -30, 40)]:
        ref = fp.shift_hsv(IMG, hs, ss, vs)
        out = to_np(G.HueSaturationValue().apply(T_IMG.clone(), hue=_t(hs), sat=_t(ss), val=_t(vs)))
        assert mae(out, ref) < 1.0, f"{(hs, ss, vs)}: {mae(out, ref):.3f}"


def test_gamma_solarize_posterize_invert_sepia_gray():
    assert max_abs(to_np(G.RandomGamma().apply(T_IMG.clone(), gamma=_t(1.3))), fp.gamma_transform(IMG, 1.3)) <= 1
    assert max_abs(to_np(G.Solarize().apply(T_IMG.clone(), threshold=_t(0.4))), fp.solarize(IMG, 0.4)) <= 1
    bits = torch.tensor([[3, 3, 3]], device=DEVICE)
    assert np.array_equal(to_np(G.Posterize().apply(T_IMG.clone(), bits=bits)), fp.posterize(IMG, 3))
    bits = torch.tensor([[2, 5, 7]], device=DEVICE)
    assert np.array_equal(to_np(G.Posterize().apply(T_IMG.clone(), bits=bits)), fp.posterize(IMG, [2, 5, 7]))
    assert np.array_equal(to_np(G.InvertImg().apply(T_IMG.clone())), fp.invert(IMG))
    sep = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    assert max_abs(to_np(G.ToSepia().apply(T_IMG.clone())), fp.linear_transformation_rgb(IMG, sep)) <= 1
    for method in ("weighted_average", "desaturation", "average", "max", "from_lab"):
        ref = fp.to_gray(IMG, 3, method)
        out = to_np(G.ToGray(method=method).apply(T_IMG.clone()))
        assert max_abs(out, ref) <= 1, f"{method}: {max_abs(out, ref)}"


def test_rgb_shift_and_channel_ops():
    shift = torch.tensor([[0.1, -0.05, 0.2]], device=DEVICE)
    ref = np.clip(IMG.astype(np.float32) + np.array([0.1, -0.05, 0.2]) * 255, 0, 255)
    assert max_abs(to_np(G.RGBShift().apply(T_IMG.clone(), shift=shift)), ref) <= 1
    perm = torch.tensor([[2, 0, 1]], device=DEVICE)
    assert np.array_equal(to_np(G.ChannelShuffle().apply(T_IMG.clone(), perm=perm)), fp.channel_shuffle(IMG, [2, 0, 1]))


def test_equalize():
    for by_channels in (True, False):
        ref = fp.equalize(IMG, by_channels=by_channels)
        out = to_np(G.Equalize(by_channels=by_channels).apply(T_IMG.clone()))
        assert max_abs(out, ref) == 0, f"by_channels={by_channels}: {max_abs(out, ref)}"


def test_fancy_pca_and_tone_curve():
    a = np.array([0.05, -0.08, 0.1], dtype=np.float32)
    ref = fp.fancy_pca(IMG.astype(np.float32) / 255, a)
    out = G.FancyPCA().apply(T_IMG.clone(), alpha_vector=torch.from_numpy(a)[None].to(DEVICE))[0].permute(1, 2, 0).cpu().numpy()
    assert np.abs(out - ref).max() < 1e-4
    for lo, hi in [(0.2, 0.8), (np.array([0.1, 0.3, 0.25]), np.array([0.7, 0.9, 0.75]))]:
        ref = fp.move_tone_curve(IMG, lo, hi)
        k = 1 if np.isscalar(lo) else 3
        out = to_np(
            G.RandomToneCurve().apply(
                T_IMG.clone(), low_y=torch.tensor(np.atleast_1d(lo), device=DEVICE, dtype=torch.float32).view(1, k), high_y=torch.tensor(np.atleast_1d(hi), device=DEVICE, dtype=torch.float32).view(1, k)
            )
        )
        assert np.array_equal(out, ref)


def test_emboss_sharpen_unsharp():
    alpha, strength = 0.4, 0.5
    em = A.Emboss._Emboss__generate_emboss_matrix(alpha, strength)
    batch = G.AugBatch(image=T_IMG)

    class FixedRng(G.Rng):
        def __init__(self, values):
            super().__init__(DEVICE, 0)
            self.values = list(values)

        def uniform(self, low, high, size):
            return torch.full((size,) if isinstance(size, int) else size, self.values.pop(0), device=DEVICE)

    k = G.Emboss().get_params(batch, FixedRng([alpha, strength]))["kernel"][0].cpu().numpy()
    assert np.allclose(k, em, atol=1e-6)
    ref = fp.convolve(IMG, em)
    assert max_abs(to_np(G.Emboss().apply(T_IMG.clone(), kernel=torch.from_numpy(em)[None].float().to(DEVICE))), ref) <= 1
    sm = A.Sharpen._Sharpen__generate_sharpening_matrix(0.3, 0.7)
    k = G.Sharpen().get_params(batch, FixedRng([0.3, 0.7]))["kernel"][0].cpu().numpy()
    assert np.allclose(k, sm, atol=1e-6)
    ref = fp.sharpen_gaussian(IMG.astype(np.float32) / 255, 0.4, 5, 1.0)
    out = G.Sharpen(method="gaussian").apply(T_IMG.clone(), alpha=_t(0.4))[0].permute(1, 2, 0).cpu().numpy()
    assert np.abs(out - np.clip(ref, 0, 1)).max() < 1e-4
    for ks, sg, al in [(5, 0.0, 0.4), (7, 1.5, 0.3)]:
        ref = fp.unsharp_mask(IMG.astype(np.float32) / 255, ks, sg, al, 10)
        out = G.UnsharpMask().apply(T_IMG.clone(), ksize=torch.tensor([ks], device=DEVICE), sigma=_t(sg), alpha=_t(al))
        diff = np.abs(out[0].permute(1, 2, 0).cpu().numpy() - ref)
        # pixels exactly at the hard threshold |residual| * 255 > threshold may flip with float rounding; the flip then
        # spreads through the blurred soft mask to its kernel neighbourhood
        assert diff.mean() < 1e-5 and (diff > 1e-4).mean() < 0.005 and diff.max() < 0.01


# ------------------------------------------------------------------------------------------------- noise
def test_multiplicative_and_salt_pepper_and_gauss_noise():
    mult = torch.tensor([0.8, 1.1, 1.25], device=DEVICE).view(1, 3, 1, 1)
    ref = fp.multiply(IMG, np.array([0.8, 1.1, 1.25], dtype=np.float32))
    assert max_abs(to_np(G.MultiplicativeNoise().apply(T_IMG.clone(), mult=mult)), ref) <= 1
    # SaltAndPepper: exact counts
    t = G.SaltAndPepper(amount=(0.05, 0.05), salt_vs_pepper=(0.5, 0.5))
    p = t.get_params(G.AugBatch(image=T_IMG.expand(4, -1, -1, -1).clone()), G.Rng(DEVICE, 0))
    n_px = int(72 * 96 * 0.05)
    assert (p["salt"] | p["pepper"]).flatten(1).sum(1).tolist() == [n_px] * 4
    assert p["salt"].flatten(1).sum(1).tolist() == [int(n_px * 0.5)] * 4
    # GaussNoise: std of the added noise matches std_range
    t = G.GaussNoise(std_range=(0.05, 0.05), per_channel=True)
    gray = torch.full((8, 3, 64, 64), 0.5, device=DEVICE)
    out = t.apply(gray.clone(), **t.get_params(G.AugBatch(image=gray), G.Rng(DEVICE, 1)))
    assert abs((out - 0.5).std().item() - 0.05) < 2e-3


def test_iso_noise_statistics_close_to_albumentations():
    f = IMG.astype(np.float32) / 255
    g = np.random.default_rng(0)
    refs = np.stack([fp.iso_noise(f.copy(), 0.03, 0.4, g) for _ in range(16)])
    t = G.ISONoise(color_shift=(0.03, 0.03), intensity=(0.4, 0.4))
    batch = G.AugBatch(image=T_IMG.expand(16, -1, -1, -1).clone())
    out = t.apply(batch.image.clone(), **t.get_params(batch, G.Rng(DEVICE, 0))).permute(0, 2, 3, 1).cpu().numpy()
    d_ref, d_out = refs - f, out - f
    assert abs(d_ref.mean() - d_out.mean()) < 2e-3 and abs(d_ref.std() - d_out.std()) < 0.1 * d_ref.std() + 1e-3


def test_shot_noise_statistics_close_to_albumentations():
    f = IMG.astype(np.float32) / 255
    g = np.random.default_rng(0)
    refs = np.stack([fp.shot_noise(f.copy(), 0.2, g) for _ in range(16)])
    t = G.ShotNoise(scale_range=(0.2, 0.2))
    batch = G.AugBatch(image=T_IMG.expand(16, -1, -1, -1).clone())
    out = t.apply(batch.image.clone(), **t.get_params(batch, G.Rng(DEVICE, 0))).permute(0, 2, 3, 1).cpu().numpy()
    assert abs(refs.mean() - out.mean()) < 3e-3 and abs(refs.std() - out.std()) < 3e-3


# ------------------------------------------------------------------------------------------------- blur
def test_box_median_gaussian_blur():
    for k in (3, 5, 9):
        ks = torch.tensor([k], device=DEVICE)
        assert max_abs(to_np(filter2d(T_IMG, box_kernels(ks))), fb.box_blur(IMG, k)) <= 1
        assert np.array_equal(to_np(median_filter(T_IMG, ks)), fb.median_blur(IMG, k))
    for sigma, ksize in [(1.2, 0), (2.5, 7), (0.7, 4)]:
        kern = fb.create_gaussian_kernel_1d(sigma, ksize)
        ref = fp.separable_convolve(IMG, kern)
        mine = albu_gaussian_kernels_1d(_t(sigma), torch.tensor([ksize], device=DEVICE))
        assert np.allclose(mine[0, (mine.shape[1] - len(kern)) // 2 :][: len(kern)].cpu().numpy(), kern, atol=1e-6)
        assert max_abs(to_np(separable_filter(T_IMG, mine, mine)), ref) <= 1


def test_mixed_kernel_sizes_in_one_batch():
    ks = torch.tensor([3, 7, 5, 9], device=DEVICE)
    imgs = T_IMG.expand(4, -1, -1, -1).contiguous()
    out = filter2d(imgs, box_kernels(ks))
    for i, k in enumerate(ks.tolist()):
        assert max_abs(to_np(out[i]), fb.box_blur(IMG, k)) <= 1
    med = median_filter(imgs, ks)
    for i, k in enumerate(ks.tolist()):
        assert np.array_equal(to_np(med[i]), fb.median_blur(IMG, k))


def test_motion_kernels_match_albumentations():
    class FixedRandom:
        def __init__(self, vals):
            self.vals = list(vals)

        def uniform(self, a, b):
            return self.vals.pop(0)

    cases = [(7, 30.0, 0.0, False, (0, 0)), (9, 123.0, 0.6, True, (0.4, -0.7)), (5, 300.0, -1.0, True, (-1.0, 1.0)), (3, 45.0, 0.3, False, (0, 0))]
    for k, ang, d, shifted, (sx, sy) in cases:
        ref = fb.create_motion_kernel(k, ang, d, shifted, FixedRandom([sx, sy]))
        ref = ref / ref.sum()
        mine = motion_kernels(torch.tensor([k, 11], device=DEVICE), torch.tensor([ang, 0.0], device=DEVICE), torch.tensor([d, 0.0], device=DEVICE), torch.tensor([sx if shifted else 0.0, 0.0], device=DEVICE), torch.tensor([sy if shifted else 0.0, 0.0], device=DEVICE))
        c = (11 - k) // 2
        assert np.allclose(mine[0, c : c + k, c : c + k].cpu().numpy(), ref, atol=1e-6), (k, ang, d)


def test_defocus_and_zoom_blur():
    batch = G.AugBatch(image=T_IMG)

    class FixedRng(G.Rng):
        def __init__(self, r, s):
            super().__init__(DEVICE, 0)
            self.r, self.s = r, s

        def randint(self, low, high, size):
            return torch.full((size,), self.r, device=DEVICE, dtype=torch.long)

        def uniform(self, low, high, size):
            return torch.full((size,), self.s, device=DEVICE)

    for r, s in [(3, 0.3), (10, 0.2)]:
        ref = fb.defocus(IMG, r, s)
        p = G.Defocus().get_params(batch, FixedRng(r, s))
        assert max_abs(to_np(G.Defocus().apply(T_IMG.clone(), **p)), ref) <= 1
    zf = np.arange(1.0, 1.2, 0.03)
    ref = fb.zoom_blur(IMG.astype(np.float32) / 255, zf)
    out = G.ZoomBlur().apply(T_IMG.clone(), step=torch.tensor([0.03], device=DEVICE, dtype=torch.float64), count=torch.tensor([len(zf)], device=DEVICE))
    assert mae(to_np(out), np.clip(ref * 255, 0, 255).round().astype(np.uint8)) < 0.6


# ------------------------------------------------------------------------------------------------- dropout
def test_coarse_dropout_fill_and_bboxes():
    holes = np.array([[10, 5, 30, 25], [50, 40, 70, 60], [20, 15, 45, 35]])
    ref = fd.cutout(IMG, holes, (10, 20, 30), np.random.default_rng(0))
    t = G.CoarseDropout(fill=(10, 20, 30))
    from gpu_batched_image_aug.transforms.dropout import _holes_to_index_map

    th = torch.tensor(holes, device=DEVICE)[None]
    index_map = _holes_to_index_map(th, torch.ones(1, 3, dtype=torch.bool, device=DEVICE), 72, 96)
    out = to_np(t.apply(T_IMG.clone(), index_map=index_map, max_value=255.0))
    assert np.array_equal(out, ref)
    boxes = np.array([[12, 8, 28, 22], [5, 5, 40, 30], [60, 10, 90, 50], [48, 38, 72, 62]], dtype=np.float32)
    for min_vis in (0.0, 0.3):
        ref_b = fd.filter_bboxes_by_holes(boxes, holes, (72, 96), 0, min_vis)
        out_b, ok = t.apply_to_bboxes(
            torch.from_numpy(boxes)[None].to(DEVICE), torch.ones(1, 4, dtype=torch.bool, device=DEVICE), 72, 96, index_map=index_map, bbox_params=G.BboxParams(min_visibility=min_vis)
        )
        got = out_b[0][ok[0]].cpu().numpy()
        assert got.shape == ref_b.shape and np.allclose(got, ref_b), (min_vis, got, ref_b)


def test_grid_dropout_structure():
    t = G.GridDropout(ratio=0.5, holes_number_xy=(4, 3), random_offset=True, p=1)
    batch = G.AugBatch(image=torch.ones(3, 3, 60, 80, device=DEVICE))
    holes, valid = t.make_holes(batch, G.Rng(DEVICE, 0))
    assert holes.shape[1] == 12 and bool(valid.all())
    hh = holes[..., 3] - holes[..., 1]
    hw = holes[..., 2] - holes[..., 0]
    assert int(hh.min()) == 10 and int(hw.min()) == 10  # cells 20x20 -> holes 10x10
    out = t.apply(batch.image.clone(), **t.get_params(batch, G.Rng(DEVICE, 0)), max_value=1.0)
    frac = (out == 0).float().mean().item()
    assert abs(frac - 12 * 100 / 4800) < 1e-6


# ------------------------------------------------------------------------------------------------- weather
def test_shadow_matches_albumentations():
    g = np.random.default_rng(3)
    verts = [np.stack([g.integers(0, 96, 5), g.integers(36, 72, 5)], 1) for _ in range(2)]
    inten = np.array([0.4, 0.6])
    ref = fp.add_shadow(IMG, verts, inten)
    from gpu_batched_image_aug.functional.raster import polygon_masks

    v = torch.tensor(np.stack(verts), device=DEVICE, dtype=torch.float32)[None]
    masks = polygon_masks(v, torch.ones(1, 2, dtype=torch.bool, device=DEVICE), 72, 96)
    ref_masks = []
    for vv in verts:
        m = np.zeros((72, 96), np.uint8)
        cv2.fillPoly(m, [vv.astype(np.int32)], 1)
        ref_masks.append(m.astype(bool))
    for i in range(2):
        a, b = masks[0, i].cpu().numpy(), ref_masks[i]
        iou = (a & b).sum() / max(1, (a | b).sum())
        assert iou > 0.95, f"polygon {i} IoU {iou:.3f}"
    factor = torch.prod(torch.where(masks, (1 - torch.tensor(inten, device=DEVICE, dtype=torch.float32))[None, :, None, None], torch.ones_like(masks, dtype=torch.float32)), dim=1)
    out = to_np(G.RandomShadow().apply(T_IMG.clone(), factor=factor[:, None]))
    assert mae(out, ref) < 1.5


def test_fog_rain_snow_match_albumentations():
    # fog: same particles / radii
    g = np.random.default_rng(1)
    pos = [(int(x), int(y)) for x, y in zip(g.integers(0, 96, 150), g.integers(0, 72, 150))]
    rad = [int(r) for r in g.integers(3, 7, 150)]
    ref = fp.add_fog(IMG, 0.6, 0.08, pos, rad)
    from gpu_batched_image_aug.functional.raster import disk_coverage_count

    cnt = disk_coverage_count(torch.tensor(pos, device=DEVICE, dtype=torch.float32)[None], torch.tensor(rad, device=DEVICE, dtype=torch.float32)[None], torch.ones(1, 150, dtype=torch.bool, device=DEVICE), 72, 96)
    out = to_np(G.RandomFog().apply(T_IMG.clone(), count=cnt[:, None], alpha=_t(0.08 * 0.6)))
    assert mae(out, ref) < 2.5, f"fog {mae(out, ref):.3f}"
    # rain: same drops
    drops = np.stack([g.integers(0, 96, 24), g.integers(0, 72 - 9, 24)], 1).astype(np.int32)
    ref = fp.add_rain(IMG, 6.0, 9, 1, (200, 200, 200), 3, 0.8, drops)
    from gpu_batched_image_aug.functional.raster import segment_masks

    st = torch.tensor(drops, device=DEVICE, dtype=torch.float32)[None]
    en = st + torch.tensor([6.0, 9.0], device=DEVICE)
    lines = segment_masks(st, en, torch.ones(1, 24, dtype=torch.bool, device=DEVICE), 72, 96).float()[:, None]
    t = G.RandomRain(drop_width=1, blur_value=3, brightness_coefficient=0.8)
    out = to_np(t.apply(T_IMG.clone(), lines=lines))
    assert mae(out, ref) < 1.5, f"rain {mae(out, ref):.3f}"
    # snow (bleach)
    ref = fp.add_snow_bleach(IMG, 0.25, 2.5)
    out = to_np(G.RandomSnow().apply(T_IMG.clone(), snow_point=_t(0.25)))
    assert mae(out, ref) < 1.0, f"snow {mae(out, ref):.3f}"


# ------------------------------------------------------------------------------------------------- compression
def test_jpeg_close_to_cv2():
    img = natural_image(96, 128, seed=4)
    for q in (10, 30, 50, 75, 95):
        ref = fp.image_compression(img, q, ".jpg")  # albumentations path (RGB fed to cv2 as BGR)
        out = to_np(jpeg(to_t(img), torch.tensor([q], device=DEVICE)))
        p_ref, p_out = psnr(ref, img), psnr(out, img)
        assert psnr(out, ref) > 32, f"q={q}: PSNR(ours, cv2) {psnr(out, ref):.1f}"
        assert abs(p_ref - p_out) < 1.5, f"q={q}: distortion PSNR cv2 {p_ref:.1f} vs ours {p_out:.1f}"


def test_downscale_matches_albumentations():
    for scale, pair in [(0.5, (cv2.INTER_NEAREST, cv2.INTER_NEAREST)), (0.33, (cv2.INTER_LINEAR, cv2.INTER_LINEAR)), (0.25, (cv2.INTER_AREA, cv2.INTER_LINEAR)), (0.4, (cv2.INTER_NEAREST, cv2.INTER_CUBIC))]:
        ref = fp.downscale(IMG, scale, pair[0], pair[1])
        t = G.Downscale(scale_range=(scale, scale), interpolation_pair={"downscale": pair[0], "upscale": pair[1]})
        p = t.get_params(G.AugBatch(image=T_IMG), G.Rng(DEVICE, 0))
        out = to_np(t.apply(T_IMG.clone(), **p))
        assert mae(interior(out, 3), interior(ref, 3)) < 1.0, f"{scale} {pair}: {mae(out, ref):.3f}"
