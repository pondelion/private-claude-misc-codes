"""Throughput: albumentations in DataLoader workers (CPU, per image, n_workers processes) vs this library (one call per
batch on the GPU, after a DataLoader that only loads).

    python benchmarks/bench_vs_albumentations.py [--batch 64] [--height 256] [--width 256] [--workers 1 4 8 16]
    python benchmarks/bench_vs_albumentations.py --per-transform --batch 256     # each transform alone (p=1)
    python benchmarks/bench_vs_albumentations.py --sweep 1 16 64 256 1024        # pipeline vs batch size (GPU only)
    python benchmarks/bench_vs_albumentations.py --sweep 1 16 64 --device cpu [--threads 1]  # this library on CPU tensors

Measured, for the same pipeline:
- albu   : DataLoader(num_workers=n) whose __getitem__ runs albumentations -> collated uint8 batch (img/s)
- gpu    : DataLoader(num_workers=n) without augmentation -> .cuda() -> gpu_batched_image_aug (img/s, end to end)
- gpu-only: the GPU augmentation alone on a resident batch (ms/batch) = GPU time taken away from training
--per-transform: albumentations ms/img on one core, the ideal 16-worker throughput (16x one core) and this library's
throughput on a batch of ``--batch`` images (use a batch large enough to saturate the GPU, e.g. 256 at 256x256).

Run on an otherwise idle GPU/CPU, otherwise the numbers are not comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gpu_batched_image_aug as G  # noqa: E402

cv2.setNumThreads(1)  # like DataLoader workers in practice; parallelism comes from the workers


def pipeline(M, p: float = 0.5):
    """Typical training pipeline; ``M`` is ``albumentations`` or ``gpu_batched_image_aug``."""
    return [
        M.HorizontalFlip(p=0.5),
        M.Affine(scale=(0.8, 1.2), rotate=(-20, 20), translate_percent=(-0.1, 0.1), p=0.7),
        M.OneOf([M.GaussianBlur(p=1), M.MotionBlur(p=1), M.MedianBlur(p=1)], p=0.3),
        M.ColorJitter(p=p),
        M.GaussNoise(std_range=(0.02, 0.08), p=p),
        M.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.2), hole_width_range=(0.05, 0.2), p=p),
        M.ImageCompression(quality_range=(40, 90), p=0.3),
    ]


def single_transforms(M):
    """(name, transform with p=1) for the per-transform comparison (mostly the ones that are heavy on the CPU)."""
    return [
        ("HorizontalFlip", M.HorizontalFlip(p=1)),
        ("ElasticTransform", M.ElasticTransform(alpha=30, sigma=5, p=1)),
        ("ElasticTransform(default)", M.ElasticTransform(p=1)),
        ("GridDistortion", M.GridDistortion(p=1)),
        ("OpticalDistortion", M.OpticalDistortion(distort_limit=0.3, p=1)),
        ("Perspective", M.Perspective(p=1)),
        ("Affine(cubic)", M.Affine(rotate=(-30, 30), scale=(0.8, 1.2), interpolation=cv2.INTER_CUBIC, p=1)),
        ("MedianBlur(3-7)", M.MedianBlur(blur_limit=(3, 7), p=1)),
        ("GaussianBlur(3-15)", M.GaussianBlur(blur_limit=(3, 15), p=1)),
        ("MotionBlur(3-21)", M.MotionBlur(blur_limit=(3, 21), p=1)),
        ("Defocus", M.Defocus(p=1)),
        ("ZoomBlur", M.ZoomBlur(p=1)),
        ("GaussNoise", M.GaussNoise(p=1)),
        ("ISONoise", M.ISONoise(p=1)),
        ("ShotNoise", M.ShotNoise(p=1)),
        ("ImageCompression", M.ImageCompression(p=1)),
        ("RandomFog", M.RandomFog(p=1)),
        ("RandomRain", M.RandomRain(p=1)),
        ("RandomSnow", M.RandomSnow(p=1)),
        ("RandomShadow", M.RandomShadow(p=1)),
        ("ColorJitter", M.ColorJitter(p=1)),
        ("HueSaturationValue", M.HueSaturationValue(p=1)),
        ("FancyPCA", M.FancyPCA(p=1)),
        ("CoarseDropout", M.CoarseDropout(p=1)),
    ]


class _Images(torch.utils.data.Dataset):
    """Synthetic uint8 images from a pre-generated pool (shared with forked workers, so loading is ~free)."""

    def __init__(self, pool: np.ndarray, length: int, aug=None):
        self.pool, self.length, self.aug = pool, length, aug

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> torch.Tensor:
        img = self.pool[i % len(self.pool)]
        if self.aug is not None:
            img = self.aug(image=img)["image"]
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


def _loader(ds, batch: int, workers: int):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None
    )


def _throughput(loader, n_batches: int, gpu_aug=None) -> float:
    """img/s over n_batches after 2 warm-up batches (batches are moved to the GPU in both modes)."""
    it = iter(loader)
    done, t0 = 0, None
    for b, x in enumerate(it):
        x = x.cuda(non_blocking=True)
        if gpu_aug is not None:
            x = gpu_aug(image=x)["image"]
        if b == 1:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        elif b > 1:
            done += x.shape[0]
        if b == n_batches + 1:
            break
    torch.cuda.synchronize()
    return done / (time.perf_counter() - t0)


def _aug_ms(aug, x: torch.Tensor, iters: int) -> float:
    """Wall time of one call on a resident batch (CUDA or CPU tensor)."""
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    for _ in range(3):
        aug(image=x)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        aug(image=x)
    sync()
    return (time.perf_counter() - t0) / iters * 1000


def _albu_ms_per_img(aug, pool: np.ndarray, n: int) -> float:
    for i in range(min(8, n)):
        aug(image=pool[i % len(pool)])
    t0 = time.perf_counter()
    for i in range(n):
        aug(image=pool[i % len(pool)])
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--batches", type=int, default=30, help="measured batches per DataLoader run")
    ap.add_argument("--iters", type=int, default=20, help="iterations for the GPU-only timing")
    ap.add_argument("--per-transform", action="store_true")
    ap.add_argument("--sweep", type=int, nargs="+", help="batch sizes: time the pipeline alone on a resident batch")
    ap.add_argument("--device", default="cuda", help="device for --sweep (cuda or cpu)")
    ap.add_argument("--threads", type=int, help="torch CPU threads (for --sweep --device cpu)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    warnings.simplefilter("ignore")

    rng = np.random.default_rng(0)
    pool = rng.integers(0, 256, (256, args.height, args.width, 3), dtype=np.uint8)
    print(f"{args.height}x{args.width}x3 uint8, CPU cores {os.cpu_count()}, torch threads {torch.get_num_threads()}, GPU {torch.cuda.get_device_name()}")

    def batch_of(b: int, device: str = "cuda") -> torch.Tensor:
        return torch.from_numpy(pool[np.arange(b) % len(pool)]).permute(0, 3, 1, 2).contiguous().to(device)

    if args.sweep:
        gpu = G.Compose(pipeline(G), seed=0)
        print(f"\npipeline on {args.device}\n{'batch':>6}{'ms/batch':>12}{'img/s':>10}")
        for b in args.sweep:
            ms = _aug_ms(gpu, batch_of(b, args.device), max(2, min(args.iters, 4096 // b)))
            print(f"{b:>6}{ms:>12.2f}{b / ms * 1000:>10.0f}", flush=True)
            torch.cuda.empty_cache()  # cached blocks of the previous size can push the next one over the VRAM size
        return

    x_gpu = batch_of(args.batch)
    if args.per_transform:
        # albu16 = ideal 16-worker throughput (16 x one core; measured DataLoader scaling is ~0.9 of that)
        print(f"\nGPU batch {args.batch}\n{'transform':<27}{'albu ms/img':>12}{'albu16 img/s':>14}{'gpu img/s':>11}{'gpu/albu16':>12}")
        for (name, ta), (_, tg) in zip(single_transforms(A), single_transforms(G)):
            a = _albu_ms_per_img(A.Compose([ta]), pool, 32)
            g = _aug_ms(G.Compose([tg], seed=0), x_gpu, 5)
            a16, gi = 16 * 1000 / a, args.batch / g * 1000
            print(f"{name:<27}{a:>12.3f}{a16:>14.0f}{gi:>11.0f}{gi / a16:>11.1f}x", flush=True)
        return

    albu = A.Compose(pipeline(A))
    gpu = G.Compose(pipeline(G), seed=0)
    length = args.batch * (args.batches + 2)
    print(f"batch {args.batch}")
    print(f"\nalbumentations, 1 core: {_albu_ms_per_img(albu, pool, 128):.3f} ms/img")
    print(f"GPU aug only (resident batch): {_aug_ms(gpu, x_gpu, args.iters):.2f} ms/batch")
    print(f"\n{'workers':>8}{'albu img/s':>14}{'load only img/s':>18}{'load+gpu aug img/s':>21}")
    for w in args.workers:
        r_albu = _throughput(_loader(_Images(pool, length, albu), args.batch, w), args.batches)
        r_load = _throughput(_loader(_Images(pool, length), args.batch, w), args.batches)
        r_gpu = _throughput(_loader(_Images(pool, length), args.batch, w), args.batches, gpu)
        print(f"{w:>8}{r_albu:>14.0f}{r_load:>18.0f}{r_gpu:>21.0f}")


if __name__ == "__main__":
    main()
