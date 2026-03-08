"""
LightGlue Training Example - 簡略化学習サンプル
===============================================

HomographicDatasetを使ったLightGlueの学習例
2段階学習（Stage 1: 対応関係、Stage 2: 確信度）を実装

CPU環境でも動作するように調整済み

【公式実装との違い】
- 特徴抽出器を簡易化 (SuperPointの代わりに軽量CNN)
- バッチサイズを縮小
- キーポイント数を削減

【参考】
- 論文: LightGlue: Local Feature Matching at Light Speed (ICCV 2023)
- 公式実装: https://github.com/cvg/LightGlue

Shape Convention:
    B: バッチサイズ
    M: Image A のキーポイント数
    N: Image B のキーポイント数
    D: 特徴量次元 (feature_dim, default 128)
    C: 隠れ次元 (hidden_dim, default 128)
    H_img, W_img: 画像の高さ・幅
    H_feat, W_feat: 特徴マップの高さ・幅 (H_img/4, W_img/4)
    L: Transformerレイヤー数 (default 4)
    H: Attention ヘッド数 (default 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import random
from PIL import Image


# ============================================================
# Homographic Dataset (ALIKED実装から流用)
# ============================================================

class HomographicDataset(Dataset):
    """
    ランダムなHomography変換を適用した画像ペアを生成

    LightGlueの事前学習データセット:
        - Oxford-Paris 1M (170K画像)
        - ランダムなHomography変換
        - 強い光学的拡張
    """

    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        max_keypoints: int = 256,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.max_keypoints = max_keypoints

        # 画像ファイルを収集
        self.image_files = list(self.image_dir.glob('*.jpg'))
        self.image_files += list(self.image_dir.glob('*.png'))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(self.image_files)} images")

    def __len__(self) -> int:
        return len(self.image_files)

    def _random_homography(self) -> torch.Tensor:
        """
        ランダムなHomography行列を生成

        変換の構成:
            1. 回転 (±15度)
            2. スケール (0.8~1.2)
            3. 平行移動 (±20px)
            4. 透視変換 (小さなパースペクティブ)

        出力:
            H: (3, 3) Homography行列
        """
        # 基本パラメータ
        angle = random.uniform(-15, 15) * np.pi / 180
        scale = random.uniform(0.8, 1.2)
        tx = random.uniform(-20, 20)
        ty = random.uniform(-20, 20)

        # 回転・スケール行列
        cos_a = np.cos(angle) * scale
        sin_a = np.sin(angle) * scale

        H = torch.tensor([
            [cos_a, -sin_a, tx],
            [sin_a, cos_a, ty],
            [random.uniform(-1e-4, 1e-4), random.uniform(-1e-4, 1e-4), 1.0]
        ], dtype=torch.float32)
        # H: (3, 3)

        return H

    def _apply_homography(
        self,
        image: torch.Tensor,
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        画像にHomographyを適用

        入力:
            image: (3, H_img, W_img) 入力画像
            H: (3, 3) Homography行列

        処理:
            1. H^{-1} を計算 (逆ワーピング用)
            2. グリッドを生成し逆変換
            3. grid_sample でワーピング

        出力:
            warped_image: (3, H_img, W_img) 変換後の画像
            valid_mask: (H_img, W_img) 有効なピクセルのマスク (bool)
        """
        # image: (3, H_img, W_img)
        _, H_img, W_img = image.shape

        # 逆行列を計算（ワーピング用）
        # H: (3, 3) → inverse → (3, 3)
        H_inv = torch.inverse(H)
        # H_inv: (3, 3)

        # グリッドを生成
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H_img, dtype=torch.float32),
            torch.arange(W_img, dtype=torch.float32),
            indexing='ij'
        )
        # y_grid: (H_img, W_img), x_grid: (H_img, W_img)

        # Homogeneous coordinates
        ones = torch.ones_like(x_grid)
        # ones: (H_img, W_img)
        grid_homo = torch.stack([x_grid, y_grid, ones], dim=-1)
        # grid_homo: (H_img, W_img, 3)

        # 逆変換を適用
        # grid_homo: (H_img, W_img, 3) @ H_inv^T: (3, 3) → (H_img, W_img, 3)
        grid_warped = grid_homo @ H_inv.T
        # grid_warped: (H_img, W_img, 3)

        # 正規化 (ホモジニアス → ユークリッド)
        # [..., :2]: (H_img, W_img, 2) / [..., 2:3]: (H_img, W_img, 1) → (H_img, W_img, 2)
        grid_warped = grid_warped[..., :2] / grid_warped[..., 2:3]
        # grid_warped: (H_img, W_img, 2)

        # grid_sampleの形式に変換 (-1, 1)
        grid_warped[..., 0] = 2 * grid_warped[..., 0] / (W_img - 1) - 1
        grid_warped[..., 1] = 2 * grid_warped[..., 1] / (H_img - 1) - 1
        # grid_warped: (H_img, W_img, 2) ∈ [-1, 1]

        # ワーピング
        image_batch = image.unsqueeze(0)
        # image_batch: (1, 3, H_img, W_img)
        grid_batch = grid_warped.unsqueeze(0)
        # grid_batch: (1, H_img, W_img, 2)

        warped = F.grid_sample(
            image_batch, grid_batch,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        # warped: (1, 3, H_img, W_img)
        warped = warped.squeeze(0)
        # warped: (3, H_img, W_img)

        # 有効マスク
        # grid_warped: (H_img, W_img, 2)
        valid_mask = (
            (grid_warped[..., 0] >= -1) & (grid_warped[..., 0] <= 1) &
            (grid_warped[..., 1] >= -1) & (grid_warped[..., 1] <= 1)
        )
        # valid_mask: (H_img, W_img) bool

        return warped, valid_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        画像ペアとHomographyを返す

        出力:
            dict with:
                - image_a: (3, H_img, W_img) 元画像
                - image_b: (3, H_img, W_img) 変換後画像
                - H_ab: (3, 3) Homography行列 (A→B)
                - valid_mask: (H_img, W_img) 有効マスク (bool)
        """
        # 画像を読み込み
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.image_size)

        # Tensorに変換
        # numpy: (H_img, W_img, 3) → permute → (3, H_img, W_img)
        image_a = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        # image_a: (3, H_img, W_img) ∈ [0, 1]

        # ランダムHomographyを生成
        H_ab = self._random_homography()
        # H_ab: (3, 3)

        # 画像を変換
        # image_a: (3, H_img, W_img), H_ab: (3, 3)
        # → image_b: (3, H_img, W_img), valid_mask: (H_img, W_img)
        image_b, valid_mask = self._apply_homography(image_a, H_ab)

        return {
            'image_a': image_a,      # (3, H_img, W_img)
            'image_b': image_b,      # (3, H_img, W_img)
            'H_ab': H_ab,            # (3, 3)
            'valid_mask': valid_mask, # (H_img, W_img)
        }


# ============================================================
# 簡易特徴抽出器 (SuperPointの代替)
# ============================================================

class SimpleFeatureExtractor(nn.Module):
    """
    簡易特徴抽出器

    実際のLightGlueは以下の特徴抽出器をサポート:
        - SuperPoint (default)
        - DISK
        - ALIKED
        - SIFT (古典的手法)

    ここでは学習目的の簡易版を実装
    """

    def __init__(
        self,
        feature_dim: int = 128,
        max_keypoints: int = 256,
    ):
        """
        Args:
            feature_dim: 特徴量次元 (D)
            max_keypoints: 最大キーポイント数 (N)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.max_keypoints = max_keypoints

        # 簡易CNN: 4x downsampling
        # (B, 3, H_img, W_img) → (B, 128, H_img/4, W_img/4)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),       # (B, 3, H, W) → (B, 64, H, W)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # (B, 64, H, W) → (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),     # (B, 64, H/2, W/2) → (B, 128, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), # (B, 128, H/2, W/2) → (B, 128, H/4, W/4)
            nn.ReLU(),
        )

        # スコアヘッド (キーポイント検出用)
        # (B, 128, H_feat, W_feat) → (B, 1, H_feat, W_feat)
        self.score_head = nn.Conv2d(128, 1, 1)

        # 特徴ヘッド
        # (B, 128, H_feat, W_feat) → (B, D, H_feat, W_feat)
        self.desc_head = nn.Conv2d(128, feature_dim, 1)

    def forward(
        self,
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        キーポイントと特徴量を抽出

        入力:
            image: (B, 3, H_img, W_img)

        処理:
            1. CNN で特徴マップ抽出: (B, 128, H_feat, W_feat)
            2. スコアマップ計算: (B, H_feat, W_feat)
            3. Top-k でキーポイント選択: (B, N, 2)
            4. 対応する特徴量を取得: (B, N, D)

        出力:
            dict with:
                - keypoints: (B, N, 2) キーポイント座標 (元画像スケール)
                - descriptors: (B, N, D) 特徴量
                - scores: (B, N) スコア
        """
        # image: (B, 3, H_img, W_img)
        B, _, H, W = image.shape

        # ========================================
        # Step 1: エンコード
        # ========================================
        # image: (B, 3, H_img, W_img) → encoder → (B, 128, H_feat, W_feat)
        # where H_feat = H_img/4, W_feat = W_img/4
        features = self.encoder(image)
        # features: (B, 128, H_feat, W_feat)

        # ========================================
        # Step 2: スコアマップ
        # ========================================
        # features: (B, 128, H_feat, W_feat) → Conv2d(128, 1, 1) → (B, 1, H_feat, W_feat)
        # → sigmoid → (B, 1, H_feat, W_feat) → squeeze(1) → (B, H_feat, W_feat)
        score_map = torch.sigmoid(self.score_head(features))
        # score_map: (B, 1, H_feat, W_feat)
        score_map = score_map.squeeze(1)
        # score_map: (B, H_feat, W_feat) ∈ [0, 1]

        # ========================================
        # Step 3: 特徴マップ
        # ========================================
        # features: (B, 128, H_feat, W_feat) → Conv2d(128, D, 1) → (B, D, H_feat, W_feat)
        # → L2 normalize → (B, D, H_feat, W_feat)
        desc_map = self.desc_head(features)
        # desc_map: (B, D, H_feat, W_feat)
        desc_map = F.normalize(desc_map, dim=1)
        # desc_map: (B, D, H_feat, W_feat) - L2正規化済み

        # ========================================
        # Step 4: キーポイントを抽出 (top-k)
        # ========================================
        H_feat, W_feat = score_map.shape[-2:]
        # H_feat = H_img/4, W_feat = W_img/4

        # score_map: (B, H_feat, W_feat) → view → (B, H_feat*W_feat)
        score_flat = score_map.view(B, -1)
        # score_flat: (B, H_feat*W_feat)

        # Top-kを選択
        k = min(self.max_keypoints, H_feat * W_feat)
        # topk: (B, H_feat*W_feat) → topk(k) → values: (B, k), indices: (B, k)
        topk_scores, topk_indices = score_flat.topk(k, dim=-1)
        # topk_scores: (B, N) where N = k
        # topk_indices: (B, N) - フラットインデックス

        # インデックスを2D座標に変換
        # topk_indices // W_feat: 行インデックス (y)
        # topk_indices % W_feat: 列インデックス (x)
        topk_y = (topk_indices // W_feat).float()
        # topk_y: (B, N)
        topk_x = (topk_indices % W_feat).float()
        # topk_x: (B, N)

        # 元の画像サイズにスケール
        # 特徴マップ座標 → 画像座標: x * (W_img / W_feat), y * (H_img / H_feat)
        keypoints = torch.stack([
            topk_x * (W / W_feat),
            topk_y * (H / H_feat),
        ], dim=-1)
        # keypoints: (B, N, 2) - [x, y] in image coordinates

        # ========================================
        # Step 5: 対応する特徴量を取得
        # ========================================
        # desc_map: (B, D, H_feat, W_feat) → view → (B, D, H_feat*W_feat)
        desc_flat = desc_map.view(B, self.feature_dim, -1)
        # desc_flat: (B, D, H_feat*W_feat)

        # → permute → (B, H_feat*W_feat, D)
        desc_flat = desc_flat.permute(0, 2, 1)
        # desc_flat: (B, H_feat*W_feat, D)

        # topk_indices: (B, N) → unsqueeze(-1) → (B, N, 1) → expand → (B, N, D)
        # gather: (B, H_feat*W_feat, D) → select indices → (B, N, D)
        descriptors = torch.gather(
            desc_flat, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.feature_dim)
        )
        # descriptors: (B, N, D)

        return {
            'keypoints': keypoints,    # (B, N, 2)
            'descriptors': descriptors, # (B, N, D)
            'scores': topk_scores,     # (B, N)
        }


# ============================================================
# 簡易LightGlue (学習用)
# ============================================================

class SimpleLightGlue(nn.Module):
    """
    簡易LightGlue (学習用)

    主要コンポーネント:
        1. 入力投影: Linear(D, C)
        2. 位置エンコーディング: MLP(2 → C)
        3. Self/Cross Attention: MultiheadAttention
        4. Match Assignment (Double Softmax)
        5. Deep Supervision
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        """
        Args:
            feature_dim: 入力特徴量次元 (D)
            hidden_dim: 隠れ次元 (C)
            n_layers: Transformerレイヤー数 (L)
            n_heads: Attention ヘッド数 (H)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 入力投影: Linear(D, C)
        # (B, N, D) → (B, N, C)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # 位置エンコーディング (Learnable Fourier)
        # (B, N, 2) → (B, N, C)
        self.pos_encoding = nn.Sequential(
            nn.Linear(2, 32),      # (B, N, 2) → (B, N, 32)
            nn.ReLU(),
            nn.Linear(32, hidden_dim),  # (B, N, 32) → (B, N, C)
        )

        # Transformerレイヤー
        # Self-Attention: (B, N, C) → (B, N, C)
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        # Cross-Attention: (B, M, C), (B, N, C) → (B, M, C)
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        # FFN: (B, N, C) → (B, N, 2C) → (B, N, C)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),  # (B, N, C) → (B, N, 2C)
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),  # (B, N, 2C) → (B, N, C)
            )
            for _ in range(n_layers)
        ])
        # LayerNorm: (B, N, C) → (B, N, C)
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers * 3)  # self, cross, ffn
        ])

        # Match Assignment: Linear(C, 1)
        # (B, N, C) → (B, N, 1) matchability logit
        self.matchability = nn.Linear(hidden_dim, 1)

        # 確信度分類器 (Stage 2)
        # (B, N, C) → (B, N, 1) confidence
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # (B, N, C) → (B, N, C)
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),           # (B, N, C) → (B, N, 1)
        )

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        マッチングを実行

        入力:
            desc0: (B, M, D) Image A の特徴量
            desc1: (B, N, D) Image B の特徴量
            kpts0: (B, M, 2) Image A のキーポイント [x, y]
            kpts1: (B, N, 2) Image B のキーポイント [x, y]

        出力:
            dict with:
                - scores_per_layer: List[(B, M+1, N+1)] 各レイヤーのlog assignment (長さ L)
                - final_scores: (B, M+1, N+1) 最終レイヤーのスコア
                - confidences: Tuple[(B, M), (B, N)] 確信度
                - final_features: Tuple[(B, M, C), (B, N, C)] 最終特徴
        """
        # desc0: (B, M, D), desc1: (B, N, D)
        # kpts0: (B, M, 2), kpts1: (B, N, 2)
        B, M, D = desc0.shape
        _, N, _ = desc1.shape

        # ========================================
        # Step 1: 入力投影
        # ========================================
        # input_proj: Linear(D, C)
        # desc0: (B, M, D) → (B, M, C)
        x0 = self.input_proj(desc0)
        # x0: (B, M, C)

        # desc1: (B, N, D) → (B, N, C)
        x1 = self.input_proj(desc1)
        # x1: (B, N, C)

        # ========================================
        # Step 2: 位置エンコーディング
        # ========================================
        # キーポイントを正規化 (0~1)
        # kpts0: (B, M, 2) / 256.0 → (B, M, 2)
        kpts0_norm = kpts0 / 256.0  # 簡略化
        # kpts0_norm: (B, M, 2) ∈ [0, 1]
        kpts1_norm = kpts1 / 256.0
        # kpts1_norm: (B, N, 2) ∈ [0, 1]

        # pos_encoding: MLP(2 → 32 → C)
        # kpts0_norm: (B, M, 2) → (B, M, C)
        pos0 = self.pos_encoding(kpts0_norm)
        # pos0: (B, M, C)
        pos1 = self.pos_encoding(kpts1_norm)
        # pos1: (B, N, C)

        # ========================================
        # Step 3: Deep Supervision: 各レイヤーでスコアを記録
        # ========================================
        scores_per_layer = []  # List[(B, M+1, N+1)]

        # ========================================
        # Step 4: Transformerレイヤー
        # ========================================
        for layer_idx in range(self.n_layers):
            # --- Self-Attention (with position) ---
            # Query/Key に位置情報を加算、Valueはそのまま
            # x0: (B, M, C) + pos0: (B, M, C) → (B, M, C)
            q0 = x0 + pos0
            # q0: (B, M, C)
            k0 = x0 + pos0
            # k0: (B, M, C)
            q1 = x1 + pos1
            # q1: (B, N, C)
            k1 = x1 + pos1
            # k1: (B, N, C)

            norm_idx = layer_idx * 3

            # Self-Attention for image 0
            # MultiheadAttention(q0, k0, x0): (B, M, C) → (B, M, C)
            x0_attn, _ = self.self_attns[layer_idx](q0, k0, x0)
            # x0_attn: (B, M, C)
            # Residual + LayerNorm
            x0 = self.norms[norm_idx](x0 + x0_attn)
            # x0: (B, M, C)

            # Self-Attention for image 1
            # MultiheadAttention(q1, k1, x1): (B, N, C) → (B, N, C)
            x1_attn, _ = self.self_attns[layer_idx](q1, k1, x1)
            # x1_attn: (B, N, C)
            x1 = self.norms[norm_idx](x1 + x1_attn)
            # x1: (B, N, C)

            # --- Cross-Attention (bidirectional) ---
            # Image 0 attends to Image 1
            # MultiheadAttention(x0, x1, x1): (B, M, C), (B, N, C) → (B, M, C)
            x0_cross, _ = self.cross_attns[layer_idx](x0, x1, x1)
            # x0_cross: (B, M, C)

            # Image 1 attends to Image 0
            # MultiheadAttention(x1, x0, x0): (B, N, C), (B, M, C) → (B, N, C)
            x1_cross, _ = self.cross_attns[layer_idx](x1, x0, x0)
            # x1_cross: (B, N, C)

            # Residual + LayerNorm
            x0 = self.norms[norm_idx + 1](x0 + x0_cross)
            # x0: (B, M, C)
            x1 = self.norms[norm_idx + 1](x1 + x1_cross)
            # x1: (B, N, C)

            # --- FFN ---
            # ffn: (B, N, C) → (B, N, 2C) → (B, N, C)
            # Residual + LayerNorm
            x0 = self.norms[norm_idx + 2](x0 + self.ffns[layer_idx](x0))
            # x0: (B, M, C)
            x1 = self.norms[norm_idx + 2](x1 + self.ffns[layer_idx](x1))
            # x1: (B, N, C)

            # --- このレイヤーでのスコアを計算 (Deep Supervision) ---
            # x0: (B, M, C), x1: (B, N, C) → scores: (B, M+1, N+1)
            scores = self._compute_assignment(x0, x1)
            # scores: (B, M+1, N+1)
            scores_per_layer.append(scores)

        # ========================================
        # Step 5: 確信度を計算
        # ========================================
        # confidence: MLP(C → C → 1)
        # x0: (B, M, C) → (B, M, 1) → sigmoid → (B, M, 1) → squeeze → (B, M)
        conf0 = torch.sigmoid(self.confidence(x0).squeeze(-1))
        # conf0: (B, M) ∈ [0, 1]

        # x1: (B, N, C) → (B, N, 1) → sigmoid → (B, N, 1) → squeeze → (B, N)
        conf1 = torch.sigmoid(self.confidence(x1).squeeze(-1))
        # conf1: (B, N) ∈ [0, 1]

        return {
            'scores_per_layer': scores_per_layer,  # List[(B, M+1, N+1)] 長さ L
            'final_scores': scores_per_layer[-1],   # (B, M+1, N+1)
            'confidences': (conf0, conf1),          # ((B, M), (B, N))
            'final_features': (x0, x1),             # ((B, M, C), (B, N, C))
        }

    def _compute_assignment(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Double Softmax + Matchability でassignment matrixを計算

        入力:
            x0: (B, M, C) Image A の特徴記述子埋め込み
            x1: (B, N, C) Image B の特徴記述子埋め込み

        処理:
            1. 類似度行列: einsum → (B, M, N)
            2. Double Softmax: log_softmax(row) + log_softmax(col) → (B, M, N)
            3. Matchability: sigmoid(linear) → (B, M, 1), (B, N, 1)
            4. 結合: log_p + log_sigma → (B, M+1, N+1)

        出力:
            log_assignment: (B, M+1, N+1)
        """
        # x0: (B, M, C), x1: (B, N, C)
        B, M, C = x0.shape
        _, N, _ = x1.shape

        # ========================================
        # Step 1: 類似度行列
        # ========================================
        # einsum: (B, M, C) × (B, N, C) → (B, M, N)
        sim = torch.einsum('bmd,bnd->bmn', x0, x1)
        # sim: (B, M, N)

        # スケーリング: sim / sqrt(C)
        sim = sim / (C ** 0.5)
        # sim: (B, M, N)

        # ========================================
        # Step 2: Double Softmax (log domain)
        # ========================================
        # Row-wise: sim: (B, M, N) → dim=-1 → (B, M, N)
        log_row = F.log_softmax(sim, dim=-1)
        # log_row: (B, M, N)

        # Col-wise: sim: (B, M, N) → dim=-2 → (B, M, N)
        log_col = F.log_softmax(sim, dim=-2)
        # log_col: (B, M, N)

        # Combined: (B, M, N) + (B, M, N) → (B, M, N)
        log_p = log_row + log_col
        # log_p: (B, M, N) = log(softmax_row × softmax_col)

        # ========================================
        # Step 3: Matchability
        # ========================================
        # matchability: Linear(C, 1)
        # x0: (B, M, C) → (B, M, 1) → sigmoid → (B, M, 1)
        sigma0 = torch.sigmoid(self.matchability(x0))
        # sigma0: (B, M, 1) ∈ [0, 1]

        # x1: (B, N, C) → (B, N, 1) → sigmoid → (B, N, 1)
        sigma1 = torch.sigmoid(self.matchability(x1))
        # sigma1: (B, N, 1) ∈ [0, 1]

        # ========================================
        # Step 4: Log assignment matrix (dustbin付き)
        # ========================================
        # (B, M+1, N+1)
        log_assignment = torch.full(
            (B, M + 1, N + 1), fill_value=-10.0,
            device=x0.device, dtype=x0.dtype
        )
        # log_assignment: (B, M+1, N+1) 初期値 -10.0

        # P_ij = sigma_i * sigma_j * softmax_i * softmax_j
        # log(sigma0): (B, M, 1)
        log_sigma0 = torch.log(sigma0 + 1e-8)
        # log_sigma0: (B, M, 1)

        # log(sigma1): (B, N, 1)
        log_sigma1 = torch.log(sigma1 + 1e-8)
        # log_sigma1: (B, N, 1)

        # log_sigma0 + log_sigma1^T: (B, M, 1) + (B, 1, N) → broadcast → (B, M, N)
        log_sigma = log_sigma0 + log_sigma1.transpose(-1, -2)
        # log_sigma: (B, M, N) = log(σ_A^i × σ_B^j)

        # マッチング部分: log_p + log_sigma: (B, M, N) + (B, M, N) → (B, M, N)
        log_assignment[:, :M, :N] = log_p + log_sigma
        # log_assignment[:, :M, :N]: (B, M, N)

        # Dustbin (unmatchable)
        # A点iがunmatchable: log(1 - σ_A^i)
        # sigma0: (B, M, 1) → squeeze(-1) → (B, M) → 1-σ → log → (B, M)
        log_assignment[:, :M, -1] = torch.log(1 - sigma0.squeeze(-1) + 1e-8)
        # log_assignment[:, :M, -1]: (B, M)

        # B点jがunmatchable: log(1 - σ_B^j)
        # sigma1: (B, N, 1) → squeeze(-1) → (B, N) → 1-σ → log → (B, N)
        log_assignment[:, -1, :N] = torch.log(1 - sigma1.squeeze(-1) + 1e-8)
        # log_assignment[:, -1, :N]: (B, N)

        log_assignment[:, -1, -1] = 0  # dummy
        # log_assignment: (B, M+1, N+1)

        return log_assignment
        # return: (B, M+1, N+1)


# ============================================================
# Ground Truth 計算
# ============================================================

def compute_ground_truth_matches(
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    H: torch.Tensor,
    match_threshold: float = 3.0,
    unmatch_threshold: float = 5.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Homographyから Ground Truth マッチを計算

    入力:
        kpts0: (M, 2) Image A のキーポイント [x, y]
        kpts1: (N, 2) Image B のキーポイント [x, y]
        H: (3, 3) Homography行列 (A→B)
        match_threshold: マッチ判定閾値 (default: 3px), スカラー
        unmatch_threshold: unmatchable判定閾値 (default: 5px), スカラー

    処理:
        1. kpts0 → homogeneous → H変換 → 正規化 → kpts0_warped: (M, 2)
        2. 距離行列計算: (M, N)
        3. MNNでマッチ抽出
        4. unmatchable判定

    出力:
        matches: List[(i, j)] マッチペア
        unmatchable_A: List[i] Aのunmatchable点
        unmatchable_B: List[j] Bのunmatchable点
    """
    # kpts0: (M, 2), kpts1: (N, 2), H: (3, 3)
    M, N = kpts0.shape[0], kpts1.shape[0]

    # ========================================
    # Step 1: キーポイントを変換
    # ========================================
    ones = torch.ones(M, 1, device=kpts0.device)
    # ones: (M, 1)

    # kpts0: (M, 2) + ones: (M, 1) → cat → (M, 3)
    kpts0_homo = torch.cat([kpts0, ones], dim=1)
    # kpts0_homo: (M, 3)

    # H変換: (M, 3) @ (3, 3)^T → (M, 3)
    kpts0_warped_homo = kpts0_homo @ H.T
    # kpts0_warped_homo: (M, 3)

    # 正規化: (M, 2) / (M, 1) → (M, 2)
    kpts0_warped = kpts0_warped_homo[:, :2] / kpts0_warped_homo[:, 2:3]
    # kpts0_warped: (M, 2)

    # ========================================
    # Step 2: 距離行列
    # ========================================
    # kpts0_warped: (M, 2) → unsqueeze(1) → (M, 1, 2)
    # kpts1: (N, 2) → unsqueeze(0) → (1, N, 2)
    # diff: (M, 1, 2) - (1, N, 2) → (M, N, 2)
    diff = kpts0_warped.unsqueeze(1) - kpts1.unsqueeze(0)
    # diff: (M, N, 2)

    # norm: (M, N, 2) → (M, N)
    dist = torch.norm(diff, dim=-1)
    # dist: (M, N)

    # ========================================
    # Step 3: マッチを抽出 (Mutual Nearest Neighbor)
    # ========================================
    matches = []

    # dist: (M, N) → dim=1で最小 → (M,), (M,)
    min_dist_a, argmin_a = dist.min(dim=1)
    # min_dist_a: (M,), argmin_a: (M,)

    # dist: (M, N) → dim=0で最小 → (N,), (N,)
    min_dist_b, argmin_b = dist.min(dim=0)
    # min_dist_b: (N,), argmin_b: (N,)

    for i in range(M):
        j = argmin_a[i].item()
        if argmin_b[j].item() == i and min_dist_a[i] < match_threshold:
            matches.append((i, j))

    # ========================================
    # Step 4: Unmatchable
    # ========================================
    matched_A = set(m[0] for m in matches)
    matched_B = set(m[1] for m in matches)

    # dist[i]: (N,) → min → スカラー > unmatch_threshold → unmatchable
    unmatchable_A = [i for i in range(M) if i not in matched_A and dist[i].min() > unmatch_threshold]
    # dist[:, j]: (M,) → min → スカラー > unmatch_threshold → unmatchable
    unmatchable_B = [j for j in range(N) if j not in matched_B and dist[:, j].min() > unmatch_threshold]

    return matches, unmatchable_A, unmatchable_B


# ============================================================
# 損失計算
# ============================================================

def compute_correspondence_loss(
    scores: torch.Tensor,
    matches: List[Tuple[int, int]],
    unmatchable_A: List[int],
    unmatchable_B: List[int],
) -> torch.Tensor:
    """
    対応関係損失を計算

    入力:
        scores: (M+1, N+1) log assignment matrix (single sample)
        matches: List[(i, j)] ground truth マッチ
        unmatchable_A: List[i]
        unmatchable_B: List[j]

    損失 = L_positive + L_negative

    L_positive: -mean(log P[matches]) → スカラー
    L_negative: -mean(log P[unmatchable]) → スカラー

    出力:
        loss: スカラー
    """
    # scores: (M+1, N+1)
    device = scores.device

    # ========================================
    # Positive loss
    # ========================================
    if len(matches) > 0:
        # matches → (K_match, 2)
        match_indices = torch.tensor(matches, dtype=torch.long, device=device)
        # match_indices: (K_match, 2)

        # scores[i, j] → (K_match,)
        match_scores = scores[match_indices[:, 0], match_indices[:, 1]]
        # match_scores: (K_match,)

        # -mean → スカラー
        loss_positive = -match_scores.mean()
        # loss_positive: スカラー
    else:
        loss_positive = torch.tensor(0.0, device=device)

    # ========================================
    # Negative loss
    # ========================================
    M, N = scores.shape[0] - 1, scores.shape[1] - 1

    if len(unmatchable_A) > 0:
        # unmatchable_A → (K_unA,)
        idx_A = torch.tensor(unmatchable_A, dtype=torch.long, device=device)
        # scores[idx_A, -1]: dustbin列 → (K_unA,) → -mean → スカラー
        loss_neg_A = -scores[idx_A, -1].mean()
    else:
        loss_neg_A = torch.tensor(0.0, device=device)

    if len(unmatchable_B) > 0:
        # unmatchable_B → (K_unB,)
        idx_B = torch.tensor(unmatchable_B, dtype=torch.long, device=device)
        # scores[-1, idx_B]: dustbin行 → (K_unB,) → -mean → スカラー
        loss_neg_B = -scores[-1, idx_B].mean()
    else:
        loss_neg_B = torch.tensor(0.0, device=device)

    # (スカラー + スカラー) / 2 → スカラー
    loss_negative = (loss_neg_A + loss_neg_B) / 2

    return loss_positive + loss_negative
    # return: スカラー


def compute_deep_supervision_loss(
    scores_per_layer: List[torch.Tensor],
    matches: List[Tuple[int, int]],
    unmatchable_A: List[int],
    unmatchable_B: List[int],
) -> torch.Tensor:
    """
    Deep Supervision損失を計算

    入力:
        scores_per_layer: List[(B, M+1, N+1)] 各レイヤーのlog assignment (長さ L)
        matches: List[(i, j)] ground truth マッチ
        unmatchable_A: List[i]
        unmatchable_B: List[j]

    処理:
        全レイヤーで損失を計算し平均
        ※バッチの最初の要素のみ使用 (簡略化)

    出力:
        loss: スカラー
    """
    total_loss = 0
    for scores in scores_per_layer:
        # scores: (B, M+1, N+1) → [0]: (M+1, N+1) バッチの最初の要素
        layer_loss = compute_correspondence_loss(
            scores[0], matches, unmatchable_A, unmatchable_B
        )
        # layer_loss: スカラー
        total_loss = total_loss + layer_loss

    # スカラー / L → スカラー
    return total_loss / len(scores_per_layer)
    # return: スカラー


# ============================================================
# 学習ループ
# ============================================================

def train_epoch_stage1(
    feature_extractor: nn.Module,
    matcher: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """
    Stage 1: 対応関係予測の学習 (1エポック)

    Deep Supervisionを使用:
        - 全レイヤーで損失を計算
        - 早期レイヤーでも意味のある予測を学習

    処理フロー (各バッチ):
        1. 画像ペアを取得: image_a: (B, 3, H, W), image_b: (B, 3, H, W)
        2. 特徴抽出: keypoints: (B, N, 2), descriptors: (B, N, D)
        3. マッチング: scores_per_layer: List[(B, M+1, N+1)]
        4. GT計算: matches, unmatchable_A, unmatchable_B
        5. Deep Supervision Loss: スカラー
        6. Backward + Step
    """
    feature_extractor.train()
    matcher.train()

    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # image_a: (B, 3, H_img, W_img), image_b: (B, 3, H_img, W_img)
        image_a = batch['image_a'].to(device)
        image_b = batch['image_b'].to(device)
        # H_ab: (B, 3, 3)
        H_ab = batch['H_ab'].to(device)

        # 特徴抽出
        # image_a: (B, 3, H, W) → feats_a:
        #   keypoints: (B, N, 2), descriptors: (B, N, D), scores: (B, N)
        feats_a = feature_extractor(image_a)
        feats_b = feature_extractor(image_b)

        # マッチング
        # desc: (B, N, D), kpts: (B, N, 2) → outputs:
        #   scores_per_layer: List[(B, M+1, N+1)] 長さ L
        outputs = matcher(
            feats_a['descriptors'],
            feats_b['descriptors'],
            feats_a['keypoints'],
            feats_b['keypoints'],
        )

        # Ground Truth計算 (バッチの最初の要素のみ)
        # kpts: (B, N, 2) → [0]: (N, 2), H_ab: (B, 3, 3) → [0]: (3, 3)
        matches, unmatchable_A, unmatchable_B = compute_ground_truth_matches(
            feats_a['keypoints'][0],
            feats_b['keypoints'][0],
            H_ab[0],
        )

        # Deep Supervision Loss
        # scores_per_layer: List[(B, M+1, N+1)] → スカラー
        loss = compute_deep_supervision_loss(
            outputs['scores_per_layer'],
            matches, unmatchable_A, unmatchable_B
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch [{batch_idx + 1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} (Avg: {avg_loss:.4f})")

    return {'total_loss': total_loss / num_batches}


def train_epoch_stage2(
    feature_extractor: nn.Module,
    matcher: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """
    Stage 2: 確信度分類器の学習 (1エポック)

    重要:
        - 特徴抽出器とマッチャーは固定
        - 確信度分類器のみ学習

    処理フロー (各バッチ):
        1. 特徴抽出 (no_grad): keypoints: (B, N, 2), descriptors: (B, N, D)
        2. マッチング: scores_per_layer, confidences
        3. 各レイヤーのマッチと最終レイヤーを比較 → labels: (M,)
        4. BCE(confidence, labels) → スカラー
        5. Backward + Step (確信度分類器のみ)
    """
    feature_extractor.eval()  # 固定
    matcher.train()  # 確信度分類器のみ学習

    # 確信度分類器以外の勾配を無効化
    for name, param in matcher.named_parameters():
        if 'confidence' not in name:
            param.requires_grad = False

    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # image_a: (B, 3, H_img, W_img), image_b: (B, 3, H_img, W_img)
        image_a = batch['image_a'].to(device)
        image_b = batch['image_b'].to(device)

        with torch.no_grad():
            # 特徴抽出 (固定)
            # → keypoints: (B, N, 2), descriptors: (B, N, D)
            feats_a = feature_extractor(image_a)
            feats_b = feature_extractor(image_b)

        # マッチング
        # → scores_per_layer: List[(B, M+1, N+1)], confidences: ((B, M), (B, N))
        outputs = matcher(
            feats_a['descriptors'],
            feats_b['descriptors'],
            feats_a['keypoints'],
            feats_b['keypoints'],
        )

        # 最終レイヤーのマッチ結果
        # final_scores: (B, M+1, N+1) → [0, :-1, :-1]: (M, N) → argmax(dim=-1) → (M,)
        final_scores = outputs['final_scores']
        # final_scores: (B, M+1, N+1)
        final_matches = final_scores[0, :-1, :-1].argmax(dim=-1)
        # final_matches: (M,) - 各A点の最終レイヤーでのマッチ先

        # 各レイヤーのマッチ結果と比較
        conf_loss = 0
        n_layers = len(outputs['scores_per_layer']) - 1  # 最終レイヤーは除く

        for layer_idx in range(n_layers):
            layer_scores = outputs['scores_per_layer'][layer_idx]
            # layer_scores: (B, M+1, N+1)

            # [0, :-1, :-1]: (M, N) → argmax(dim=-1) → (M,)
            layer_matches = layer_scores[0, :-1, :-1].argmax(dim=-1)
            # layer_matches: (M,) - レイヤーℓでの各A点のマッチ先

            # ラベル: 最終レイヤーと同じマッチならconfident
            # layer_matches: (M,) == final_matches: (M,) → (M,) bool → float → (M,)
            labels = (layer_matches == final_matches).float()
            # labels: (M,) ∈ {0.0, 1.0}

            # BCE loss (確信度)
            # conf0: (B, M), conf1: (B, N)
            conf0, conf1 = outputs['confidences']
            # conf0[0]: (M,), labels: (M,) → スカラー
            loss = F.binary_cross_entropy(conf0[0], labels)
            # loss: スカラー
            conf_loss = conf_loss + loss

        # 全レイヤー平均: スカラー / (L-1) → スカラー
        conf_loss = conf_loss / n_layers

        # Backward
        optimizer.zero_grad()
        conf_loss.backward()
        optimizer.step()

        total_loss += conf_loss.item()

        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch [{batch_idx + 1}/{num_batches}] "
                  f"Conf Loss: {conf_loss.item():.4f} (Avg: {avg_loss:.4f})")

    # 勾配を再有効化
    for param in matcher.parameters():
        param.requires_grad = True

    return {'conf_loss': total_loss / num_batches}


def validate_epoch(
    feature_extractor: nn.Module,
    matcher: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    検証 (1エポック)

    処理フロー (各バッチ):
        1. 特徴抽出 (no_grad): (B, N, 2), (B, N, D)
        2. マッチング: scores_per_layer: List[(B, M+1, N+1)]
        3. GT計算
        4. Loss計算: スカラー
        5. 精度計算 (簡易): argmax予測 vs GT

    出力:
        dict with 'val_loss': float, 'match_accuracy': float
    """
    feature_extractor.eval()
    matcher.eval()

    total_loss = 0.0
    total_matches = 0
    correct_matches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # image_a: (B, 3, H, W), image_b: (B, 3, H, W)
            image_a = batch['image_a'].to(device)
            image_b = batch['image_b'].to(device)
            # H_ab: (B, 3, 3)
            H_ab = batch['H_ab'].to(device)

            # 特徴抽出
            # → keypoints: (B, N, 2), descriptors: (B, N, D)
            feats_a = feature_extractor(image_a)
            feats_b = feature_extractor(image_b)

            # マッチング
            # → scores_per_layer: List[(B, M+1, N+1)]
            outputs = matcher(
                feats_a['descriptors'],
                feats_b['descriptors'],
                feats_a['keypoints'],
                feats_b['keypoints'],
            )

            # Ground Truth
            gt_matches, unmatchable_A, unmatchable_B = compute_ground_truth_matches(
                feats_a['keypoints'][0],  # (N, 2)
                feats_b['keypoints'][0],  # (N, 2)
                H_ab[0],                  # (3, 3)
            )

            # Loss
            # scores_per_layer: List[(B, M+1, N+1)] → スカラー
            loss = compute_deep_supervision_loss(
                outputs['scores_per_layer'],
                gt_matches, unmatchable_A, unmatchable_B
            )
            total_loss += loss.item()

            # 精度計算 (簡易)
            # final_scores: (B, M+1, N+1) → [0, :-1, :-1]: (M, N) → argmax → (M,)
            final_scores = outputs['final_scores']
            pred_matches = final_scores[0, :-1, :-1].argmax(dim=-1)
            # pred_matches: (M,) - 各A点の予測マッチ先

            gt_match_dict = {i: j for i, j in gt_matches}
            for i, j_pred in enumerate(pred_matches.tolist()):
                if i in gt_match_dict:
                    total_matches += 1
                    if gt_match_dict[i] == j_pred:
                        correct_matches += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_matches / max(1, total_matches)

    return {
        'val_loss': avg_loss,
        'match_accuracy': accuracy,
    }


# ============================================================
# メイン学習関数
# ============================================================

def train_lightglue(
    train_image_dir: str,
    val_image_dir: Optional[str] = None,
    num_epochs_stage1: int = 10,
    num_epochs_stage2: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: str = 'cpu',
    save_dir: str = './checkpoints',
):
    """
    LightGlue学習のメイン関数

    2段階学習:
        Stage 1: 対応関係予測 (Deep Supervision)
        Stage 2: 確信度分類器

    Args:
        train_image_dir: 学習用画像ディレクトリ
        val_image_dir: 検証用画像ディレクトリ
        num_epochs_stage1: Stage 1のエポック数
        num_epochs_stage2: Stage 2のエポック数
        batch_size: バッチサイズ (B)
        learning_rate: 学習率
        device: 'cpu' or 'cuda'
        save_dir: チェックポイント保存先
    """
    print("=" * 60)
    print("LightGlue Training Example")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Stage 1 Epochs: {num_epochs_stage1}")
    print(f"Stage 2 Epochs: {num_epochs_stage2}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print()

    # 保存ディレクトリ
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # データセット
    print("Creating datasets...")
    train_dataset = HomographicDataset(image_dir=train_image_dir)

    if val_image_dir is not None:
        val_dataset = HomographicDataset(image_dir=val_image_dir)
    else:
        total_size = len(train_dataset)
        val_size = int(total_size * 0.1)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # モデル
    print("Creating models...")
    feature_extractor = SimpleFeatureExtractor(
        feature_dim=128,      # D = 128
        max_keypoints=256,    # N = 256
    ).to(device)

    matcher = SimpleLightGlue(
        feature_dim=128,      # D = 128
        hidden_dim=128,       # C = 128
        n_layers=4,           # L = 4
        n_heads=4,            # H = 4
    ).to(device)

    total_params = sum(p.numel() for p in feature_extractor.parameters())
    total_params += sum(p.numel() for p in matcher.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print()

    # ========================================
    # Stage 1: 対応関係予測
    # ========================================
    print("=" * 60)
    print("STAGE 1: Correspondence Learning (Deep Supervision)")
    print("=" * 60)

    optimizer_stage1 = optim.Adam(
        list(feature_extractor.parameters()) + list(matcher.parameters()),
        lr=learning_rate,
    )
    scheduler_stage1 = optim.lr_scheduler.StepLR(
        optimizer_stage1, step_size=5, gamma=0.5
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs_stage1):
        epoch_start = time.time()

        print(f"\nStage 1 - Epoch [{epoch + 1}/{num_epochs_stage1}]")
        print("-" * 60)

        # 学習
        train_losses = train_epoch_stage1(
            feature_extractor, matcher, train_loader,
            optimizer_stage1, device, epoch
        )

        # 検証
        val_losses = validate_epoch(
            feature_extractor, matcher, val_loader, device
        )

        scheduler_stage1.step()

        epoch_time = time.time() - epoch_start

        print(f"\n  Train Loss: {train_losses['total_loss']:.4f}")
        print(f"  Val Loss: {val_losses['val_loss']:.4f}")
        print(f"  Match Accuracy: {val_losses['match_accuracy']:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer_stage1.param_groups[0]['lr']:.6f}")

        # ベストモデル保存
        if val_losses['val_loss'] < best_val_loss:
            best_val_loss = val_losses['val_loss']
            torch.save({
                'epoch': epoch,
                'feature_extractor': feature_extractor.state_dict(),
                'matcher': matcher.state_dict(),
                'val_loss': best_val_loss,
            }, save_path / 'best_stage1.pth')
            print(f"  -> Best Stage 1 model saved!")

    # ========================================
    # Stage 2: 確信度分類器
    # ========================================
    print("\n" + "=" * 60)
    print("STAGE 2: Confidence Classifier Learning")
    print("=" * 60)

    # 確信度分類器のパラメータのみ
    conf_params = [p for n, p in matcher.named_parameters() if 'confidence' in n]
    optimizer_stage2 = optim.Adam(conf_params, lr=learning_rate * 0.1)

    for epoch in range(num_epochs_stage2):
        epoch_start = time.time()

        print(f"\nStage 2 - Epoch [{epoch + 1}/{num_epochs_stage2}]")
        print("-" * 60)

        # 学習 (確信度分類器のみ)
        train_losses = train_epoch_stage2(
            feature_extractor, matcher, train_loader,
            optimizer_stage2, device, epoch
        )

        epoch_time = time.time() - epoch_start

        print(f"\n  Conf Loss: {train_losses['conf_loss']:.4f}")
        print(f"  Time: {epoch_time:.2f}s")

    # 最終モデル保存
    torch.save({
        'feature_extractor': feature_extractor.state_dict(),
        'matcher': matcher.state_dict(),
    }, save_path / 'final_model.pth')

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best Stage 1 Val Loss: {best_val_loss:.4f}")
    print("=" * 60)


# ============================================================
# エントリーポイント
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        train_image_dir = sys.argv[1]
        train_lightglue(
            train_image_dir=train_image_dir,
            num_epochs_stage1=10,
            num_epochs_stage2=3,
            batch_size=2,
            learning_rate=1e-4,
            device='cpu',
            save_dir='./checkpoints_lightglue',
        )
    else:
        print("Usage: python training_example.py <train_image_dir>")
        print()
        print("Example:")
        print("  python training_example.py /path/to/oxford_images")
        print()
        print("Note:")
        print("  train_image_dir should contain .jpg or .png images")
        print("  (e.g., Oxford-Paris dataset)")
        print()
        print("=" * 60)
        print("Training Process Overview")
        print("=" * 60)
        print("""
LightGlue uses a 2-stage training process:

STAGE 1: Correspondence Learning
--------------------------------
  - Train both feature extractor and matcher
  - Use Deep Supervision (loss at all layers)
  - This enables faster convergence vs SuperGlue
  - Duration: ~2 GPU-days (official)

STAGE 2: Confidence Classifier
------------------------------
  - Freeze feature extractor and matcher weights
  - Only train confidence classifier
  - Learn when to stop early (adaptive inference)
  - Duration: ~0.5 GPU-days (official)

Key Differences from SuperGlue:
  1. Deep Supervision -> 3.5x faster training
  2. Rotary PE (relative) vs Absolute PE
  3. Double Softmax vs Sinkhorn (100x faster)
  4. Adaptive depth/width for faster inference
""")
