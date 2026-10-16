"""
VisionHOPE - 分類損失 (ラベル平滑化 / Mixup・CutMix 用 Soft-Target CE)
================================================================

VisionHOPE 自体に新規の損失関数は無く、timm 標準の画像分類レシピをそのまま使う。
公式コード gh/VisionHOPE/train/classification.py:
    loss_fn = SoftTargetCrossEntropy() if (mixup or cutmix) else (
        LabelSmoothingCrossEntropy(smoothing) if smoothing else CrossEntropyLoss())
本ファイルは timm への依存を増やさずに、同じ数式をそのまま再実装したもの。

============================================================
Shape Convention
============================================================
B : バッチサイズ
C : クラス数
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ラベル平滑化付き Cross-Entropy (timm.loss.LabelSmoothingCrossEntropy と同一の式)。

    ========================================
    式
    ========================================
    logprobs = log_softmax(logits)
    nll_loss = -logprobs[y]                                  (正解クラスの負対数確率)
    smooth_loss = -mean(logprobs)                              (全クラス平均)
    loss = (1-smoothing)*nll_loss + smoothing*smooth_loss
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        # logits: (B, C), target: (B,) 整数クラスID
        logprobs = F.log_softmax(logits, dim=-1)                       # (B, C)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)  # (B,)
        smooth_loss = -logprobs.mean(dim=-1)                            # (B,)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss  # (B,)
        return loss.mean()                                                # scalar


class SoftTargetCrossEntropy(nn.Module):
    """ソフトラベル (Mixup/CutMix後の混合ラベル) に対するCross-Entropy。

    ========================================
    式
    ========================================
    loss = mean_b( sum_c( -target[b,c] * log_softmax(logits[b,c]) ) )
    """

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), target: (B, C) ソフトラベル (各行の合計は1)
        loss = torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1)  # (B,)
        return loss.mean()                                                    # scalar


def mixup_cutmix(
    images: torch.Tensor,     # (B, 3, H, W)
    targets: torch.LongTensor,  # (B,)
    num_classes: int,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    prob: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mixup または CutMix を適用し、ソフトラベルを返す (公式 train/data.py の簡略版)。

    バッチ内の画像をシャッフルした相手と混合する。確率 ``prob`` でどちらかの手法を
    (半々で) 適用し、残りはそのまま (onehotラベル) を返す。

    ========================================
    Shape
    ========================================
    入力: images (B,3,H,W), targets (B,)
    出力: mixed_images (B,3,H,W), soft_targets (B, num_classes)
    """
    import random

    B = images.shape[0]
    onehot = F.one_hot(targets, num_classes).float()  # (B, C)
    if random.random() > prob:
        return images, onehot

    perm = torch.randperm(B, device=images.device)
    use_cutmix = random.random() < 0.5
    if use_cutmix:
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        H, W = images.shape[-2:]
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
        lam_actual = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
    else:
        lam_actual = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        mixed = lam_actual * images + (1.0 - lam_actual) * images[perm]

    soft_targets = lam_actual * onehot + (1.0 - lam_actual) * onehot[perm]  # (B, C)
    return mixed, soft_targets


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    print("=== loss.py Example Usage ===\n")
    torch.manual_seed(0)

    B, C = 8, 10
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, C, (B,))

    ce = nn.CrossEntropyLoss()
    ls = LabelSmoothingCrossEntropy(smoothing=0.1)
    print(f"標準CE loss:        {ce(logits, targets).item():.4f}")
    print(f"LabelSmoothing loss: {ls(logits, targets).item():.4f}")

    images = torch.randn(B, 3, 32, 32)
    mixed_images, soft_targets = mixup_cutmix(images, targets, num_classes=C)
    print(f"\nmixup/cutmix後の images shape: {tuple(mixed_images.shape)}")
    print(f"soft_targets shape: {tuple(soft_targets.shape)}, 各行合計={soft_targets.sum(-1)[:3].tolist()}")

    stce = SoftTargetCrossEntropy()
    loss = stce(logits, soft_targets)
    loss.backward()
    print(f"\nSoftTargetCE loss: {loss.item():.4f}  入力への勾配ノルム: {logits.grad.norm().item():.4f}")


if __name__ == "__main__":
    example_usage()
