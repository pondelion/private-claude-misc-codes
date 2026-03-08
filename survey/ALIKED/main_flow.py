"""
ALIKED Main Flow - 簡略化疑似コード
=====================================

ALIKED: A Lighter Keypoint and Descriptor Extraction Network
via Deformable Transformation

主要特徴:
- SDDH (Sparse Deformable Descriptor Head) - 変形可能な記述子抽出
- DKD (Differentiable Keypoint Detection) - 微分可能なキーポイント検出
- 超軽量: 0.19M (Tiny) ~ 0.98M (Normal-32) パラメータ
- リアルタイム性能: 125 FPS (Tiny) ~ 75 FPS (Normal-32)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class ALIKED(nn.Module):
    """
    ALIKED: A Lighter Keypoint and Descriptor Extraction Network

    モデルバリアント:
    - aliked-t16: Tiny (c1=8, c2=16, c3=32, c4=64, dim=64, M=16)
    - aliked-n16: Normal 16-stride (c1=16, c2=32, c3=64, c4=128, dim=128, M=16)
    - aliked-n32: Normal 32-stride (c1=16, c2=32, c3=64, c4=128, dim=128, M=32)
    - aliked-n16rot: Normal with rotation augmentation

    処理フロー:
    1. Feature Encoding: Multi-scale特徴抽出
    2. Feature Aggregation: Multi-scale特徴統合
    3. Keypoint Detection: DKDでsub-pixelキーポイント検出
    4. Descriptor Extraction: SDDHで変形可能記述子抽出
    """

    def __init__(
        self,
        c1: int = 16,      # Block1 channels
        c2: int = 32,      # Block2 channels
        c3: int = 64,      # Block3 channels
        c4: int = 128,     # Block4 channels
        dim: int = 128,    # Descriptor dimension
        M: int = 16,       # Number of deformable sample positions
        K: int = 3         # Kernel size for SDDH
    ):
        super().__init__()

        self.dim = dim
        self.M = M

        # ========================================
        # 1. Feature Encoding (4ブロック)
        # ========================================
        from blocks import ConvBlock, ResBlock

        # Block1: Low-level feature extraction (stride=1)
        self.block1 = ConvBlock(3, c1, stride=1)

        # Block2: (stride=2, total stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block2 = ConvBlock(c1, c2, stride=1)

        # Block3: Deformable Conv (stride=4, total stride=8)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.block3 = ResBlock(c2, c3, use_dcn=True)

        # Block4: Deformable Conv (stride=4, total stride=32)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.block4 = ResBlock(c3, c4, use_dcn=True)

        # ========================================
        # 2. Feature Aggregation
        # ========================================
        # Multi-scale特徴を統合
        self.ublock4 = UpsampleBlock(c4, dim // 4, scale=32)
        self.ublock3 = UpsampleBlock(c3, dim // 4, scale=8)
        self.ublock2 = UpsampleBlock(c2, dim // 4, scale=2)
        self.ublock1 = nn.Sequential(
            nn.Conv2d(c1, dim // 4, kernel_size=1),
            nn.SELU()
        )

        # Concatenation後の特徴: dim channels

        # ========================================
        # 3. Score Map Head (SMH)
        # ========================================
        self.score_head = nn.Sequential(
            nn.Conv2d(dim, 8, kernel_size=1),
            nn.SELU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.SELU(),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # ========================================
        # 4. Differentiable Keypoint Detection (DKD)
        # ========================================
        from soft_detect import DKD
        self.dkd = DKD(
            radius=2,           # NMSカーネル半径
            top_k=5000,         # 最大キーポイント数
            scores_th=0.2,      # スコア閾値
            n_limit=20000       # 制限
        )

        # ========================================
        # 5. Sparse Deformable Descriptor Head (SDDH)
        # ========================================
        from blocks import SDDH
        self.sddh = SDDH(
            in_dim=dim,
            out_dim=dim,
            M=M,                # デフォーマブルサンプル位置数
            K=K                 # パッチサイズ
        )

    def forward(
        self,
        images: torch.Tensor,
        top_k: int = 5000,
        scores_th: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        ALIKEDフォワードパス

        入力:
            images: (B, 3, H, W) - RGB画像
            top_k: int - 最大キーポイント数
            scores_th: float - キーポイントスコア閾値

        出力:
            {
                'keypoints': (B, N, 2) - キーポイント座標 [x, y]
                'descriptors': (B, N, dim) - 記述子
                'scores': (B, N) - キーポイントスコア
                'score_map': (B, 1, H, W) - スコアマップ
            }
        """

        B, _, H, W = images.shape

        # ========================================
        # Step 1: Feature Encoding
        # ========================================

        # Block1: stride=1
        x1 = self.block1(images)
        # x1: (B, c1, H, W)

        # Block2: stride=2
        x2 = self.pool2(x1)
        x2 = self.block2(x2)
        # x2: (B, c2, H/2, W/2)

        # Block3: stride=8 (with Deformable Conv)
        x3 = self.pool3(x2)
        x3 = self.block3(x3)
        # x3: (B, c3, H/8, W/8)

        # Block4: stride=32 (with Deformable Conv)
        x4 = self.pool4(x3)
        x4 = self.block4(x4)
        # x4: (B, c4, H/32, W/32)

        # ========================================
        # Step 2: Feature Aggregation
        # ========================================

        # 全てをH×Wにアップサンプリング
        f1 = self.ublock1(x1)           # (B, dim/4, H, W)
        f2 = self.ublock2(x2)           # (B, dim/4, H, W)
        f3 = self.ublock3(x3)           # (B, dim/4, H, W)
        f4 = self.ublock4(x4)           # (B, dim/4, H, W)

        # Concatenate
        features = torch.cat([f1, f2, f3, f4], dim=1)
        # features: (B, dim, H, W)

        # ========================================
        # Step 3: Score Map Estimation
        # ========================================

        score_map = self.score_head(features)
        # score_map: (B, 1, H, W)

        # ========================================
        # Step 4: Differentiable Keypoint Detection (DKD)
        # ========================================

        keypoints, scores = self.dkd(
            score_map,
            top_k=top_k,
            scores_th=scores_th
        )
        # keypoints: (B, N, 2) - sub-pixel座標 [x, y]
        # scores: (B, N) - キーポイントスコア

        # ========================================
        # Step 5: Sparse Deformable Descriptor Head (SDDH)
        # ========================================

        descriptors = self.sddh(features, keypoints)
        # descriptors: (B, N, dim)

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'scores': scores,
            'score_map': score_map
        }

    def extract_dense_map(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        密な特徴マップとスコアマップを抽出 (訓練用)

        入力:
            images: (B, 3, H, W)

        出力:
            features: (B, dim, H, W)
            score_map: (B, 1, H, W)
        """

        B, _, H, W = images.shape

        # Feature Encoding
        x1 = self.block1(images)
        x2 = self.block2(self.pool2(x1))
        x3 = self.block3(self.pool3(x2))
        x4 = self.block4(self.pool4(x3))

        # Feature Aggregation
        f1 = self.ublock1(x1)
        f2 = self.ublock2(x2)
        f3 = self.ublock3(x3)
        f4 = self.ublock4(x4)

        features = torch.cat([f1, f2, f3, f4], dim=1)

        # Score Map
        score_map = self.score_head(features)

        return features, score_map

class UpsampleBlock(nn.Module):
    """アップサンプリングブロック"""

    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.scale = scale

    def forward(self, x):
        # 公式実装に合わせて、先にチャネル削減してからupsample
        x = self.conv(x)
        x = nn.functional.selu(x)

        if self.scale > 1:
            x = nn.functional.interpolate(
                x,
                scale_factor=self.scale,
                mode='bilinear',
                align_corners=False
            )

        return x

# ============================================
# 使用例
# ============================================

def example_aliked_usage():
    """ALIKED使用例（CPU 12GBでも動作するように調整）"""

    # モデル作成 (Normal-16)
    model = ALIKED(
        c1=16, c2=32, c3=64, c4=128,
        dim=128,
        M=16,
        K=3
    )
    model.eval()  # 評価モード

    # ダミー入力（32で割り切れるサイズ: 256x320）
    # バッチサイズも 2 → 1 に削減
    images = torch.randn(1, 3, 256, 320)

    # フォワードパス（勾配計算を無効化してメモリ節約）
    with torch.no_grad():
        # top_k も削減: 1000 → 300
        outputs = model(images, top_k=300, scores_th=0.2)

    print(f"Keypoints: {outputs['keypoints'].shape}")      # (1, N, 2)
    print(f"Descriptors: {outputs['descriptors'].shape}")  # (1, N, 128)
    print(f"Scores: {outputs['scores'].shape}")            # (1, N)
    print(f"Score map: {outputs['score_map'].shape}")      # (1, 1, 256, 320)

    # 画像マッチング例（同じく小さいサイズで）
    img_a = torch.randn(1, 3, 256, 320)
    img_b = torch.randn(1, 3, 256, 320)

    # キーポイント・記述子抽出
    with torch.no_grad():
        out_a = model(img_a, top_k=300, scores_th=0.2)
        out_b = model(img_b, top_k=300, scores_th=0.2)

    # Mutual Nearest Neighbor (mNN) マッチング
    desc_a = out_a['descriptors'][0]  # (N_a, 128)
    desc_b = out_b['descriptors'][0]  # (N_b, 128)

    # Cosine similarity
    sim_matrix = torch.matmul(desc_a, desc_b.T)  # (N_a, N_b)

    # mNN matching
    nn_ab = torch.argmax(sim_matrix, dim=1)      # (N_a,)
    nn_ba = torch.argmax(sim_matrix, dim=0)      # (N_b,)

    # Mutual check
    ids_a = torch.arange(desc_a.shape[0])
    mutual_mask = (nn_ba[nn_ab] == ids_a)

    matches_a = ids_a[mutual_mask]
    matches_b = nn_ab[mutual_mask]

    print(f"Number of matches: {matches_a.shape[0]}")

if __name__ == "__main__":
    example_aliked_usage()
