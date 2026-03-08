"""
DINOv3 Training Example - 簡略化学習サンプル
=============================================

DINOv3 の Teacher-Student SSL 学習ループを簡略化した実行可能なサンプルです。
画像ファイルパスリストを指定して実際の画像で学習検証ができます。

実際の DINOv3 構成:
  - ViT-7B (6.7B params), 256 GPU (H100), batch=4096, 1M iter
  - ここでは ViT-Tiny (数Mパラメータ) でデモ

学習フェーズ:
  Phase 1: DINO + iBOT + KoLeo (1M iter)
  Phase 2: + Gram Anchoring (精緻化)

使用方法:
  python training_example.py --image_dir /path/to/images
"""

import argparse
import glob
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ============================================================
# 簡略化モデルコンポーネント
# ============================================================

class SimplePatchEmbed(nn.Module):
    """パッチ埋め込み"""

    def __init__(self, patch_size: int = 16, embed_dim: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            patches: (B, P, D), P = (H/patch_size) * (W/patch_size)
        """
        x = self.proj(x)             # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, P, D)
        return x


class SimpleSwiGLUFFN(nn.Module):
    """SwiGLU FFN (簡略版)"""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        swiglu_hidden = int(hidden_dim * 2 / 3)
        swiglu_hidden = swiglu_hidden + (-swiglu_hidden % 8)  # 8の倍数に
        self.w1 = nn.Linear(dim, swiglu_hidden)
        self.w2 = nn.Linear(dim, swiglu_hidden)
        self.w3 = nn.Linear(swiglu_hidden, dim)

    def forward(self, x):
        # x: (B, N, D) → (B, N, D)
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SimpleTransformerBlock(nn.Module):
    """Transformer Block (簡略版, RoPE なし)"""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 3.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SimpleSwiGLUFFN(dim, int(dim * ffn_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
        Returns:
            out: (B, N, D)
        """
        # Attention path
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)  # (B, N, D)
        x = x + h

        # FFN path
        h = self.norm2(x)
        h = self.ffn(h)  # (B, N, D)
        x = x + h

        return x


class SimpleViT(nn.Module):
    """
    簡略化 Vision Transformer (DINOv3 学習デモ用)

    ViT-Tiny 相当:
      embed_dim=192, depth=6, num_heads=3, patch_size=16
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        n_storage_tokens: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_storage_tokens = n_storage_tokens

        self.patch_embed = SimplePatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.storage_tokens = nn.Parameter(
            torch.randn(1, n_storage_tokens, embed_dim) * 0.02
        )
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) - 入力画像
            masks: (B, P) optional - マスク (True=マスク)

        Returns:
            dict:
                x_norm_clstoken: (B, D)
                x_storage_tokens: (B, R, D)
                x_norm_patchtokens: (B, P, D)
        """
        B = x.shape[0]

        # 1. パッチ埋め込み
        patches = self.patch_embed(x)  # (B, P, D)
        P = patches.shape[1]

        # 2. マスク適用
        if masks is not None:
            mask_expanded = masks.unsqueeze(-1).bool()  # (B, P, 1)
            patches = torch.where(
                mask_expanded,
                self.mask_token.expand(B, P, -1),
                patches,
            )

        # 3. CLS + Storage + Patches
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        storage = self.storage_tokens.expand(B, -1, -1)  # (B, R, D)
        tokens = torch.cat([cls, storage, patches], dim=1)  # (B, 1+R+P, D)

        # 4. Transformer Blocks
        for block in self.blocks:
            tokens = block(tokens)  # (B, 1+R+P, D)

        # 5. 正規化
        tokens = self.norm(tokens)
        R = self.n_storage_tokens

        return {
            "x_norm_clstoken": tokens[:, 0],           # (B, D)
            "x_storage_tokens": tokens[:, 1:1+R],      # (B, R, D)
            "x_norm_patchtokens": tokens[:, 1+R:],     # (B, P, D)
        }


class SimpleHead(nn.Module):
    """DINO/iBOT Head (簡略版)"""

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int, n_prototypes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, n_prototypes, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


# ============================================================
# 損失関数
# ============================================================

def sinkhorn_knopp(logits: torch.Tensor, temp: float, n_iter: int = 3) -> torch.Tensor:
    """
    Sinkhorn-Knopp 正規化

    Args:
        logits: (B, K) - ロジット
        temp: 温度
        n_iter: 反復回数

    Returns:
        Q: (B, K) - 正規化確率分布
    """
    B, K = logits.shape
    Q = torch.exp(logits.float() / temp).T  # (K, B)
    Q /= Q.sum() + 1e-8

    for _ in range(n_iter):
        Q /= (Q.sum(dim=1, keepdim=True) + 1e-8) * K
        Q /= (Q.sum(dim=0, keepdim=True) + 1e-8) * B

    return Q.T  # (B, K)


def dino_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    student_temp: float = 0.1,
) -> torch.Tensor:
    """
    DINO CLS token 交差エントロピー

    Args:
        student_logits: (n_s, B, K)
        teacher_probs: (n_t, B, K)

    Returns:
        loss: scalar
    """
    n_s, B, K = student_logits.shape
    n_t = teacher_probs.shape[0]

    s_log = F.log_softmax(student_logits.float() / student_temp, dim=-1)
    loss = -torch.einsum("sbk,tbk->st", s_log, teacher_probs.float())

    # 対角除外 (同一クロップペア)
    min_st = min(n_s, n_t)
    loss[:min_st, :min_st].fill_diagonal_(0)
    count = B * (n_s * n_t - min_st)

    return loss.sum() / max(count, 1)


def ibot_loss(
    student_masked: torch.Tensor,
    teacher_masked: torch.Tensor,
    student_temp: float = 0.1,
) -> torch.Tensor:
    """
    iBOT マスクパッチ損失

    Args:
        student_masked: (n_masked, K)
        teacher_masked: (n_masked, K)

    Returns:
        loss: scalar
    """
    s_log = F.log_softmax(student_masked.float() / student_temp, dim=-1)
    loss = -torch.sum(teacher_masked.float() * s_log, dim=-1)
    return loss.mean()


def koleo_loss(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KoLeo 均一性損失

    Args:
        features: (B, D) - CLS tokens

    Returns:
        loss: scalar
    """
    x = F.normalize(features, dim=-1, p=2)
    dots = torch.mm(x, x.T)
    dots.fill_diagonal_(-1)
    nn_idx = dots.argmax(dim=1)
    nn_feats = x[nn_idx]
    dist = torch.norm(x - nn_feats, p=2, dim=-1)
    return -torch.log(dist + eps).mean()


def gram_loss(
    student_feats: torch.Tensor,
    teacher_feats: torch.Tensor,
) -> torch.Tensor:
    """
    Gram Anchoring 損失

    Args:
        student_feats: (B, P, D)
        teacher_feats: (B, P, D)

    Returns:
        loss: scalar
    """
    s = F.normalize(student_feats.float(), dim=-1)
    t = F.normalize(teacher_feats.float(), dim=-1)

    G_s = torch.bmm(s, s.transpose(1, 2))  # (B, P, P)
    G_t = torch.bmm(t, t.transpose(1, 2))  # (B, P, P)

    G_s = torch.clamp(G_s, min=0)
    G_t = torch.clamp(G_t, min=0)

    return F.mse_loss(G_s, G_t)


# ============================================================
# ダミーデータセット
# ============================================================

class DINOv3SSLDataset(Dataset):
    """
    DINOv3 学習用データセット

    画像ファイルパスのリストを受け取り、各サンプルについて
    DINOv3 の Multi-Crop 拡張を適用:
      - Global Crops: 2枚 (RandomResizedCrop → global_crop_size)
      - Local Crops: n枚 (RandomResizedCrop → local_crop_size)

    使用例:
        # ディレクトリから画像パスリストを作成
        import glob
        image_paths = glob.glob("/path/to/images/**/*.jpg", recursive=True)

        dataset = DINOv3SSLDataset(
            image_paths=image_paths,
            global_crop_size=64,
            local_crop_size=32,
        )
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_paths: List[str],
        global_crop_size: int = 64,
        local_crop_size: int = 32,
        n_global_crops: int = 2,
        n_local_crops: int = 4,
        global_crop_scale: Tuple[float, float] = (0.32, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.32),
    ):
        """
        Args:
            image_paths: 画像ファイルパスのリスト (.jpg, .png 等)
            global_crop_size: グローバルクロップのリサイズ先
            local_crop_size: ローカルクロップのリサイズ先
            n_global_crops: グローバルクロップ数 (デフォルト: 2)
            n_local_crops: ローカルクロップ数 (デフォルト: 4)
            global_crop_scale: グローバルクロップのスケール範囲
            local_crop_scale: ローカルクロップのスケール範囲
        """
        self.image_paths = image_paths
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        # --- Global Crop 拡張 ---
        # View 1: 強いぼかし
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, scale=global_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=self._blur_kernel(global_crop_size), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        # View 2: 弱いぼかし + ソラリゼーション
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size, scale=global_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=self._blur_kernel(global_crop_size), sigma=(0.1, 2.0))],
                p=0.1,
            ),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        # --- Local Crop 拡張 ---
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size, scale=local_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=self._blur_kernel(local_crop_size), sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    @staticmethod
    def _blur_kernel(crop_size: int) -> int:
        """GaussianBlur のカーネルサイズ (奇数にする)"""
        k = crop_size // 10
        return k if k % 2 == 1 else k + 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx: サンプルインデックス

        Returns:
            dict:
                global_crops: (n_global, 3, global_crop_size, global_crop_size)
                local_crops: (n_local, 3, local_crop_size, local_crop_size)
        """
        # 画像読み込み (PIL Image)
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Global Crops (2枚: 異なる拡張)
        global_crops = [self.global_transform_1(img)]
        for _ in range(self.n_global_crops - 1):
            global_crops.append(self.global_transform_2(img))

        # Local Crops (n枚)
        local_crops = [self.local_transform(img) for _ in range(self.n_local_crops)]

        return {
            "global_crops": torch.stack(global_crops),  # (n_global, 3, H_g, W_g)
            "local_crops": torch.stack(local_crops),     # (n_local, 3, H_l, W_l)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Multi-Crop バッチの collate

    各クロップタイプを (n_crops*B, 3, H, W) に変換

    Returns:
        dict:
            collated_global_crops: (n_global*B, 3, H_g, W_g)
            collated_local_crops: (n_local*B, 3, H_l, W_l)
    """
    global_crops = torch.cat([b["global_crops"] for b in batch], dim=0)
    # (n_global*B, 3, 64, 64)
    local_crops = torch.cat([b["local_crops"] for b in batch], dim=0)
    # (n_local*B, 3, 32, 32)

    return {
        "collated_global_crops": global_crops,
        "collated_local_crops": local_crops,
    }


# ============================================================
# マスク生成
# ============================================================

def generate_masks(
    batch_size: int,
    n_global_crops: int,
    patch_grid: int,
    mask_ratio_range: Tuple[float, float] = (0.1, 0.5),
    mask_probability: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    iBOT 用ランダムマスク生成

    Args:
        batch_size: バッチサイズ
        n_global_crops: グローバルクロップ数
        patch_grid: パッチグリッドサイズ (例: 8 for 64/8)
        mask_ratio_range: マスク率の範囲
        mask_probability: マスク適用確率

    Returns:
        masks: (n_global*B, P) - バイナリマスク
        masks_weight: (n_global*B, P) - パッチ重み
    """
    P = patch_grid * patch_grid
    total = n_global_crops * batch_size
    masks = torch.zeros(total, P)

    for i in range(total):
        if torch.rand(1).item() < mask_probability:
            ratio = torch.empty(1).uniform_(*mask_ratio_range).item()
            n_mask = int(P * ratio)
            indices = torch.randperm(P)[:n_mask]
            masks[i, indices] = 1

    # 重み: 各画像のマスク数の逆数 (正規化用)
    mask_counts = masks.sum(dim=1, keepdim=True).clamp(min=1)
    masks_weight = masks / mask_counts

    return masks, masks_weight


# ============================================================
# メイン学習ループ
# ============================================================

def train_one_epoch(
    student_backbone: SimpleViT,
    teacher_backbone: SimpleViT,
    student_dino_head: SimpleHead,
    teacher_dino_head: SimpleHead,
    student_ibot_head: SimpleHead,
    teacher_ibot_head: SimpleHead,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    epoch: int,
    device: torch.device,
    # Hyperparams
    n_global_crops: int = 2,
    n_local_crops: int = 4,
    patch_size: int = 8,
    ema_momentum: float = 0.999,
    teacher_temp: float = 0.07,
    student_temp: float = 0.1,
    dino_loss_weight: float = 1.0,
    ibot_loss_weight: float = 1.0,
    koleo_loss_weight: float = 0.1,
    gram_loss_weight: float = 0.0,   # Phase 1 では 0, Phase 2 では 2.0
    use_gram: bool = False,
    gram_teacher_backbone: Optional[SimpleViT] = None,
) -> Dict[str, float]:
    """
    1エポック分の DINOv3 学習

    Args:
        student_backbone: Student ViT
        teacher_backbone: Teacher ViT (EMA)
        student_dino_head: Student DINO Head
        teacher_dino_head: Teacher DINO Head
        student_ibot_head: Student iBOT Head
        teacher_ibot_head: Teacher iBOT Head
        optimizer: AdamW optimizer
        dataloader: DataLoader
        epoch: エポック番号
        device: torch.device
        ...各ハイパーパラメータ...

    Returns:
        epoch_losses: 各損失の平均値
    """
    student_backbone.train()
    teacher_backbone.eval()

    epoch_losses = {
        "dino_global": 0.0,
        "dino_local": 0.0,
        "ibot": 0.0,
        "koleo": 0.0,
        "gram": 0.0,
        "total": 0.0,
    }
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        global_crops = batch["collated_global_crops"].to(device)
        # (n_global*B, 3, 64, 64)
        local_crops = batch["collated_local_crops"].to(device)
        # (n_local*B, 3, 32, 32)

        B = global_crops.shape[0] // n_global_crops
        global_patch_grid = global_crops.shape[2] // patch_size  # 64/8=8

        # === マスク生成 ===
        masks, masks_weight = generate_masks(
            B, n_global_crops, global_patch_grid
        )
        masks = masks.to(device)            # (n_global*B, P)
        masks_weight = masks_weight.to(device)

        # ============================================================
        # Step 1: Teacher Forward (勾配なし)
        # ============================================================
        with torch.no_grad():
            teacher_out = teacher_backbone(global_crops)
            # cls: (n_global*B, D), patches: (n_global*B, P, D)

            t_cls = teacher_out["x_norm_clstoken"]           # (n_global*B, D)
            t_patches = teacher_out["x_norm_patchtokens"]     # (n_global*B, P, D)

            # DINO Head
            t_cls_logits = teacher_dino_head(t_cls)           # (n_global*B, K_dino)
            t_cls_probs = sinkhorn_knopp(t_cls_logits, teacher_temp)
            t_cls_probs = t_cls_probs.view(n_global_crops, B, -1)
            # (n_global, B, K_dino)

            # iBOT Head (マスクパッチのみ)
            mask_bool = masks.bool()                           # (n_global*B, P)
            t_masked_patches = t_patches[mask_bool]            # (n_masked, D)
            if t_masked_patches.shape[0] > 0:
                t_masked_logits = teacher_ibot_head(t_masked_patches)  # (n_masked, K_ibot)
                t_masked_probs = sinkhorn_knopp(t_masked_logits, teacher_temp)
            else:
                t_masked_probs = torch.zeros(0, teacher_ibot_head.last_layer.out_features).to(device)

        # ============================================================
        # Step 2: Student Forward
        # ============================================================
        # --- Global Crops (マスクあり) ---
        s_global_out = student_backbone(global_crops, masks=masks)
        s_global_cls = s_global_out["x_norm_clstoken"]           # (n_global*B, D)
        s_global_patches = s_global_out["x_norm_patchtokens"]    # (n_global*B, P, D)

        # DINO Head on global CLS
        s_global_cls_logits = student_dino_head(s_global_cls)    # (n_global*B, K_dino)
        s_global_cls_logits = s_global_cls_logits.view(n_global_crops, B, -1)
        # (n_global, B, K_dino)

        # iBOT Head on masked patches
        s_masked_patches = s_global_patches[mask_bool]           # (n_masked, D)
        if s_masked_patches.shape[0] > 0:
            s_masked_logits = student_ibot_head(s_masked_patches)  # (n_masked, K_ibot)
        else:
            s_masked_logits = torch.zeros(0, student_ibot_head.last_layer.out_features).to(device)

        # --- Local Crops (マスクなし) ---
        s_local_out = student_backbone(local_crops)
        s_local_cls = s_local_out["x_norm_clstoken"]             # (n_local*B, D)

        s_local_cls_logits = student_dino_head(s_local_cls)      # (n_local*B, K_dino)
        s_local_cls_logits = s_local_cls_logits.view(n_local_crops, B, -1)
        # (n_local, B, K_dino)

        # ============================================================
        # Step 3: 損失計算
        # ============================================================

        # --- DINO Global Loss ---
        l_dino_global = dino_loss(
            s_global_cls_logits, t_cls_probs, student_temp
        ) * dino_loss_weight
        # scalar

        # --- DINO Local Loss ---
        l_dino_local = dino_loss(
            s_local_cls_logits, t_cls_probs, student_temp
        ) * dino_loss_weight
        # scalar

        # --- iBOT Loss ---
        if s_masked_logits.shape[0] > 0:
            l_ibot = ibot_loss(
                s_masked_logits, t_masked_probs, student_temp
            ) * ibot_loss_weight
        else:
            l_ibot = torch.tensor(0.0, device=device)
        # scalar

        # --- KoLeo Loss ---
        l_koleo = koleo_loss(s_global_cls[:B]) * koleo_loss_weight
        # scalar (最初の global crop の CLS のみ使用)

        # --- Gram Loss (Phase 2) ---
        l_gram = torch.tensor(0.0, device=device)
        if use_gram and gram_teacher_backbone is not None:
            with torch.no_grad():
                gram_out = gram_teacher_backbone(global_crops)
                gram_teacher_patches = gram_out["x_norm_patchtokens"]
                # (n_global*B, P, D)
            l_gram = gram_loss(
                s_global_patches, gram_teacher_patches
            ) * gram_loss_weight
        # scalar

        # --- 合計損失 ---
        total_loss = l_dino_global + l_dino_local + l_ibot + l_koleo + l_gram

        # ============================================================
        # Step 4: 逆伝播 + 最適化
        # ============================================================
        optimizer.zero_grad()
        total_loss.backward()
        # 勾配クリッピング (DINOv3: max_norm=3.0)
        torch.nn.utils.clip_grad_norm_(
            list(student_backbone.parameters())
            + list(student_dino_head.parameters())
            + list(student_ibot_head.parameters()),
            max_norm=3.0,
        )
        optimizer.step()

        # ============================================================
        # Step 5: EMA Teacher 更新
        # ============================================================
        with torch.no_grad():
            m = ema_momentum
            for t_p, s_p in zip(teacher_backbone.parameters(), student_backbone.parameters()):
                t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)
            for t_p, s_p in zip(teacher_dino_head.parameters(), student_dino_head.parameters()):
                t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)
            for t_p, s_p in zip(teacher_ibot_head.parameters(), student_ibot_head.parameters()):
                t_p.data.mul_(m).add_(s_p.data, alpha=1 - m)

        # ログ記録
        epoch_losses["dino_global"] += l_dino_global.item()
        epoch_losses["dino_local"] += l_dino_local.item()
        epoch_losses["ibot"] += l_ibot.item()
        epoch_losses["koleo"] += l_koleo.item()
        epoch_losses["gram"] += l_gram.item()
        epoch_losses["total"] += total_loss.item()
        n_batches += 1

        if batch_idx % 10 == 0:
            print(
                f"  Batch {batch_idx}/{len(dataloader)} | "
                f"DINO_G: {l_dino_global.item():.4f} | "
                f"DINO_L: {l_dino_local.item():.4f} | "
                f"iBOT: {l_ibot.item():.4f} | "
                f"KoLeo: {l_koleo.item():.4f} | "
                f"Gram: {l_gram.item():.4f} | "
                f"Total: {total_loss.item():.4f}"
            )

    # 平均損失
    for key in epoch_losses:
        epoch_losses[key] /= max(n_batches, 1)

    return epoch_losses


# ============================================================
# メイン関数
# ============================================================

def collect_image_paths(image_dir: str) -> List[str]:
    """
    ディレクトリから画像ファイルパスを再帰的に収集

    Args:
        image_dir: 画像ディレクトリのパス

    Returns:
        image_paths: ソート済み画像パスリスト
    """
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
        paths.extend(glob.glob(os.path.join(image_dir, "**", ext.upper()), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(
            f"No images found in {image_dir}. "
            f"Supported extensions: {extensions}"
        )
    return paths


def main():
    """
    DINOv3 学習デモ

    Phase 1: DINO + iBOT + KoLeo
    Phase 2: + Gram Anchoring

    使用方法:
      python training_example.py --image_dir /path/to/images
      python training_example.py --image_dir /path/to/images --batch_size 16 --num_epochs_phase1 10
    """
    parser = argparse.ArgumentParser(description="DINOv3 Training Example")
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="画像ディレクトリのパス (再帰的に .jpg/.png 等を検索)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs_phase1", type=int, default=3)
    parser.add_argument("--num_epochs_phase2", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("DINOv3 Training Example (Simplified)")
    print("=" * 60)

    # === 画像パス収集 ===
    image_paths = collect_image_paths(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    # === 設定 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # モデル設定 (デモ用に小さくする)
    config = {
        "img_size_global": 64,
        "img_size_local": 32,
        "patch_size": 8,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 3,
        "n_storage_tokens": 2,
        # Head 設定
        "dino_n_prototypes": 1024,   # 実際: 256K
        "ibot_n_prototypes": 512,    # 実際: 96K
        "head_hidden_dim": 384,
        "head_bottleneck_dim": 64,
        # 学習設定
        "n_global_crops": 2,
        "n_local_crops": 4,
        "batch_size": args.batch_size,
        "num_epochs_phase1": args.num_epochs_phase1,
        "num_epochs_phase2": args.num_epochs_phase2,
        "lr": args.lr,
        "weight_decay": 0.04,
        "ema_momentum": 0.996,
        "teacher_temp": 0.07,
        "student_temp": 0.1,
        # 損失重み
        "dino_loss_weight": 1.0,
        "ibot_loss_weight": 1.0,
        "koleo_loss_weight": 0.1,
        "gram_loss_weight": 2.0,
    }

    # === モデル構築 ===
    print("\nBuilding models...")

    # Student
    student_backbone = SimpleViT(
        img_size=config["img_size_global"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        n_storage_tokens=config["n_storage_tokens"],
    ).to(device)

    student_dino_head = SimpleHead(
        config["embed_dim"], config["head_hidden_dim"],
        config["head_bottleneck_dim"], config["dino_n_prototypes"],
    ).to(device)

    student_ibot_head = SimpleHead(
        config["embed_dim"], config["head_hidden_dim"],
        config["head_bottleneck_dim"], config["ibot_n_prototypes"],
    ).to(device)

    # Teacher (Student のコピー, 勾配なし)
    teacher_backbone = SimpleViT(
        img_size=config["img_size_global"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        n_storage_tokens=config["n_storage_tokens"],
    ).to(device)

    teacher_dino_head = SimpleHead(
        config["embed_dim"], config["head_hidden_dim"],
        config["head_bottleneck_dim"], config["dino_n_prototypes"],
    ).to(device)

    teacher_ibot_head = SimpleHead(
        config["embed_dim"], config["head_hidden_dim"],
        config["head_bottleneck_dim"], config["ibot_n_prototypes"],
    ).to(device)

    # Teacher を Student で初期化
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    teacher_dino_head.load_state_dict(student_dino_head.state_dict())
    teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())

    # Teacher の勾配無効化
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_dino_head.parameters():
        p.requires_grad = False
    for p in teacher_ibot_head.parameters():
        p.requires_grad = False

    # パラメータ数表示
    n_params_student = sum(
        p.numel() for p in student_backbone.parameters()
    ) + sum(
        p.numel() for p in student_dino_head.parameters()
    ) + sum(
        p.numel() for p in student_ibot_head.parameters()
    )
    print(f"Student parameters: {n_params_student:,}")

    # === Optimizer ===
    # DINOv3: AdamW, 定数 LR (コサインスケジュールなし)
    optimizer = torch.optim.AdamW(
        list(student_backbone.parameters())
        + list(student_dino_head.parameters())
        + list(student_ibot_head.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    # === データセット ===
    dataset = DINOv3SSLDataset(
        image_paths=image_paths,
        global_crop_size=config["img_size_global"],
        local_crop_size=config["img_size_local"],
        n_global_crops=config["n_global_crops"],
        n_local_crops=config["n_local_crops"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")

    # === Phase 1: DINO + iBOT + KoLeo ===
    print("\n" + "=" * 60)
    print("Phase 1: Pre-training (DINO + iBOT + KoLeo)")
    print("=" * 60)

    for epoch in range(config["num_epochs_phase1"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs_phase1']}")
        losses = train_one_epoch(
            student_backbone=student_backbone,
            teacher_backbone=teacher_backbone,
            student_dino_head=student_dino_head,
            teacher_dino_head=teacher_dino_head,
            student_ibot_head=student_ibot_head,
            teacher_ibot_head=teacher_ibot_head,
            optimizer=optimizer,
            dataloader=dataloader,
            epoch=epoch,
            device=device,
            n_global_crops=config["n_global_crops"],
            n_local_crops=config["n_local_crops"],
            patch_size=config["patch_size"],
            ema_momentum=config["ema_momentum"],
            teacher_temp=config["teacher_temp"],
            student_temp=config["student_temp"],
            dino_loss_weight=config["dino_loss_weight"],
            ibot_loss_weight=config["ibot_loss_weight"],
            koleo_loss_weight=config["koleo_loss_weight"],
            gram_loss_weight=0.0,  # Phase 1: Gram なし
            use_gram=False,
        )
        print(f"  Epoch {epoch+1} avg: {losses}")

    # === Phase 2: + Gram Anchoring ===
    print("\n" + "=" * 60)
    print("Phase 2: Refinement with Gram Anchoring")
    print("=" * 60)

    # Gram Teacher = 現在の Teacher のスナップショット
    gram_teacher_backbone = SimpleViT(
        img_size=config["img_size_global"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        n_storage_tokens=config["n_storage_tokens"],
    ).to(device)
    gram_teacher_backbone.load_state_dict(teacher_backbone.state_dict())
    for p in gram_teacher_backbone.parameters():
        p.requires_grad = False

    for epoch in range(config["num_epochs_phase2"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs_phase2']}")
        losses = train_one_epoch(
            student_backbone=student_backbone,
            teacher_backbone=teacher_backbone,
            student_dino_head=student_dino_head,
            teacher_dino_head=teacher_dino_head,
            student_ibot_head=student_ibot_head,
            teacher_ibot_head=teacher_ibot_head,
            optimizer=optimizer,
            dataloader=dataloader,
            epoch=epoch + config["num_epochs_phase1"],
            device=device,
            n_global_crops=config["n_global_crops"],
            n_local_crops=config["n_local_crops"],
            patch_size=config["patch_size"],
            ema_momentum=config["ema_momentum"],
            teacher_temp=config["teacher_temp"],
            student_temp=config["student_temp"],
            dino_loss_weight=config["dino_loss_weight"],
            ibot_loss_weight=config["ibot_loss_weight"],
            koleo_loss_weight=config["koleo_loss_weight"],
            gram_loss_weight=config["gram_loss_weight"],  # Phase 2: Gram あり
            use_gram=True,
            gram_teacher_backbone=gram_teacher_backbone,
        )
        print(f"  Epoch {epoch+1} avg: {losses}")

    # === 推論デモ ===
    print("\n" + "=" * 60)
    print("Inference Demo")
    print("=" * 60)

    student_backbone.eval()
    with torch.no_grad():
        # テスト画像 (データセットの先頭画像を使用)
        test_img = Image.open(image_paths[0]).convert("RGB")
        test_transform = transforms.Compose([
            transforms.Resize((config["img_size_global"], config["img_size_global"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DINOv3SSLDataset.IMAGENET_MEAN,
                std=DINOv3SSLDataset.IMAGENET_STD,
            ),
        ])
        test_image = test_transform(test_img).unsqueeze(0).to(device)  # (1, 3, 64, 64)
        out = student_backbone(test_image)

        cls_token = out["x_norm_clstoken"]        # (1, 192)
        patch_tokens = out["x_norm_patchtokens"]  # (1, 64, 192)

        print(f"\nTest image: {image_paths[0]}")
        print(f"Input image shape: {test_image.shape}")
        print(f"CLS token shape: {cls_token.shape}")
        print(f"Patch tokens shape: {patch_tokens.shape}")
        print(f"  → {patch_tokens.shape[1]} patches of {patch_tokens.shape[2]}-dim features")

        # 密特徴の品質チェック
        cls_norm = F.normalize(cls_token, dim=-1)
        patch_norm = F.normalize(patch_tokens, dim=-1)
        cls_patch_sim = torch.einsum("bd,bpd->bp", cls_norm, patch_norm)
        print(f"CLS-Patch cosine similarity: mean={cls_patch_sim.mean():.4f}, "
              f"std={cls_patch_sim.std():.4f}")

        # Gram 行列の構造
        gram = torch.bmm(patch_norm, patch_norm.transpose(1, 2))
        print(f"Gram matrix shape: {gram.shape}")
        print(f"Gram matrix stats: mean={gram.mean():.4f}, "
              f"diagonal_mean={gram.diagonal(dim1=1, dim2=2).mean():.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
