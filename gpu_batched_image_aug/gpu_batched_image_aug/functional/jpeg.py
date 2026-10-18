"""Batched JPEG compression simulation (baseline JPEG as produced by libjpeg / cv2.imencode + imdecode).

Encoder: uint8 rounding -> JFIF YCbCr -> 4:2:0 chroma subsampling (2x2 average) -> 8x8 DCT -> quantization with the
IJG standard tables scaled by quality -> rounding. Decoder: dequantization -> IDCT -> "fancy" (triangular) chroma
upsampling -> RGB -> uint8 rounding. Entropy coding is lossless and therefore skipped. The result is close to, but
not bit-identical with, libjpeg (which uses integer DCT approximations and fixed-point color conversion).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LUMA_Q = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]  # fmt: skip
_CHROMA_Q = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]  # fmt: skip


def quant_tables(quality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """IJG ``jpeg_quality_scaling`` + ``jpeg_add_quant_table`` (force_baseline): (n,) -> (n, 8, 8) luma, chroma."""
    q = quality.clamp(1, 100).double()
    scale = torch.where(q < 50, 5000.0 / q, 200.0 - 2.0 * q)
    dev = quality.device

    def table(base):
        b = torch.tensor(base, dtype=torch.float64, device=dev).view(1, 8, 8)
        return torch.floor((b * scale.view(-1, 1, 1) + 50) / 100).clamp(1, 255).float()

    return table(_LUMA_Q), table(_CHROMA_Q)


def _dct_matrix(device: torch.device) -> torch.Tensor:
    k = torch.arange(8, dtype=torch.float64, device=device)
    m = torch.cos((2 * k[None, :] + 1) * k[:, None] * math.pi / 16) * 0.5
    m[0] = m[0] / math.sqrt(2)
    return m.float()  # orthonormal DCT-II


def _blocks(x: torch.Tensor) -> torch.Tensor:
    """(n, H, W) with H, W multiples of 8 -> (n, H/8, W/8, 8, 8)"""
    n, h, w = x.shape
    return x.view(n, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)


def _unblocks(b: torch.Tensor) -> torch.Tensor:
    n, hb, wb = b.shape[:3]
    return b.permute(0, 1, 3, 2, 4).reshape(n, hb * 8, wb * 8)


def _pad_to(x: torch.Tensor, mult: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = F.pad(x[:, None], (0, pw, 0, ph), mode="replicate")[:, 0]
    return x


def _code_plane(plane: torch.Tensor, table: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
    """plane (n, H, W) in 0..255 -> lossy reconstruction (n, H, W)."""
    h, w = plane.shape[-2:]
    x = _pad_to(plane, 8) - 128.0
    b = _blocks(x)
    coef = dct @ b @ dct.T
    t = table[:, None, None]
    coef = torch.round(coef / t) * t
    rec = dct.T @ coef @ dct
    return _unblocks(rec)[:, :h, :w] + 128.0


def jpeg(img: torch.Tensor, quality: torch.Tensor, bgr_input: bool = True) -> torch.Tensor:
    """img (n, C, H, W) float [0, 1] (C = 1 or 3), quality (n,) in [1, 100] -> (n, C, H, W) float [0, 1].
    ``bgr_input``: interpret channels as BGR for the color conversion, which is what albumentations effectively does
    (it passes RGB arrays to cv2.imencode, which expects BGR)."""
    n, c, h, w = img.shape
    dct = _dct_matrix(img.device)
    q_luma, q_chroma = quant_tables(quality)
    x = (img.clamp(0, 1) * 255.0).round()
    if c == 1:
        return (_code_plane(x[:, 0], q_luma, dct).round().clamp(0, 255) / 255.0)[:, None]
    if c != 3:
        raise ValueError("JPEG simulation supports 1 or 3 channels")
    if bgr_input:
        b, g, r = x[:, 0], x[:, 1], x[:, 2]
    else:
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168735892 * r - 0.331264108 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418687589 * g - 0.081312411 * b + 128.0

    y_rec = _code_plane(y, q_luma, dct)
    # 4:2:0: average 2x2 (edge replicated), code at half resolution, triangular upsampling back
    cb_s = F.avg_pool2d(_pad_to(cb, 2)[:, None], 2)[:, 0]
    cr_s = F.avg_pool2d(_pad_to(cr, 2)[:, None], 2)[:, 0]
    cb_rec = _code_plane(cb_s, q_chroma, dct)
    cr_rec = _code_plane(cr_s, q_chroma, dct)
    hh, ww = cb_rec.shape[-2] * 2, cb_rec.shape[-1] * 2
    up = F.interpolate(torch.stack([cb_rec, cr_rec], dim=1), size=(hh, ww), mode="bilinear", align_corners=False)
    cb_up, cr_up = up[:, 0, :h, :w] - 128.0, up[:, 1, :h, :w] - 128.0

    r2 = y_rec + 1.402 * cr_up
    g2 = y_rec - 0.344136286 * cb_up - 0.714136286 * cr_up
    b2 = y_rec + 1.772 * cb_up
    out = torch.stack([b2, g2, r2] if bgr_input else [r2, g2, b2], dim=1)
    return out.round().clamp(0, 255) / 255.0
