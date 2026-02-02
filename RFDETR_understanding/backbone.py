"""
RF-DETR バックボーン - 簡略化疑似コード
======================================

DINOv2 + Windowed Attention + Multi-scale Projector
RF-DETRの特徴的なバックボーン実装
"""

import torch
import torch.nn as nn
from typing import List, Dict


class DINOv2WithWindowedAttention(nn.Module):
    """
    DINOv2 + Windowed Attention

    RF-DETRの特徴:
    - DINOv2をバックボーンとして使用
    - Windowed Attentionで計算量削減
    - 特定レイヤーのみグローバルアテンション
    - Multi-scaleで特徴抽出
    """

    def __init__(
        self,
        model_size: str = 'base',         # 'nano', 'small', 'medium', 'base', 'large'
        patch_size: int = 14,             # パッチサイズ
        extract_layers: List[int] = [2, 5, 8, 11],  # 特徴抽出レイヤー
        num_windows: int = 2,             # ウィンドウ数 (2 or 4)
        hidden_dim: int = 768             # 特徴次元
    ):
        super().__init__()

        self.patch_size = patch_size
        self.extract_layers = extract_layers
        self.num_windows = num_windows

        # ========================================
        # DINOv2 ベースモデル (timm使用を想定)
        # ========================================
        # 実際には timm.create_model('vit_base_patch14_dinov2') 等を使用
        self.dinov2 = self._create_dinov2_model(model_size, patch_size)

        # ========================================
        # Windowed Attention の設定
        # ========================================
        # 特定のブロックのみウィンドウアテンションを適用
        # 特徴抽出レイヤー以外にウィンドウ化を適用
        total_blocks = self._get_num_blocks(model_size)
        self.window_block_indexes = self._compute_window_blocks(
            total_blocks, extract_layers
        )
        # 例: total_blocks=12, extract_layers=[2,5,8,11]
        #     window_blocks = [0,1,3,4,6,7,9,10] (2,5,8,11以外)


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        DINOv2 + Windowed Attention フォワードパス

        入力:
            x: (B, 3, H, W) - RGB画像
                B: バッチサイズ
                3: RGB
                H, W: 画像サイズ (例: 640x640)

        出力:
            features: List[Tensor] 長さ len(extract_layers)
                各要素: (B, hidden_dim, H_patch, W_patch)
                例: [(B, 768, 46, 46)] for 640x640, patch_size=14
        """

        B, C, H, W = x.shape

        # ========================================
        # Step 1: パッチ埋め込み
        # ========================================

        # 画像をパッチに分割して埋め込み
        # (B, 3, H, W) -> (B, N_patches, hidden_dim)
        x = self.dinov2.patch_embed(x)
        # x: (B, H_patch * W_patch, hidden_dim)
        # H_patch = H // patch_size, W_patch = W // patch_size

        H_patch, W_patch = H // self.patch_size, W // self.patch_size
        N_patches = H_patch * W_patch

        # CLSトークンと位置埋め込みを追加 (DINOv2標準)
        x = x + self.dinov2.pos_embed
        # x: (B, N_patches, hidden_dim)


        # ========================================
        # Step 2: Transformer ブロック (Windowed Attention)
        # ========================================

        features = []

        for block_idx, block in enumerate(self.dinov2.blocks):
            # ウィンドウ化が必要か判定
            use_windowed = block_idx in self.window_block_indexes

            if use_windowed:
                # Windowed Attention適用
                x = self._apply_windowed_attention(
                    x, block, H_patch, W_patch
                )
            else:
                # 通常のグローバルアテンション
                x = block(x)

            # 特徴抽出レイヤーの場合、保存
            if block_idx in self.extract_layers:
                # (B, N_patches, hidden_dim) -> (B, hidden_dim, H_patch, W_patch)
                feat = x.transpose(1, 2).reshape(B, -1, H_patch, W_patch)
                features.append(feat)

        return features


    def _apply_windowed_attention(
        self,
        x: torch.Tensor,
        block: nn.Module,
        H_patch: int,
        W_patch: int
    ) -> torch.Tensor:
        """
        Windowed Attention の適用

        RF-DETRの重要ポイント:
        - 画像を num_windows x num_windows のウィンドウに分割
        - 各ウィンドウ内でのみアテンション計算
        - 計算量を O(N^2) から O(N^2 / num_windows^2) に削減

        入力:
            x: (B, N_patches, hidden_dim)
            block: Transformerブロック
            H_patch, W_patch: パッチグリッドサイズ

        出力:
            x: (B, N_patches, hidden_dim) - Windowed Attention適用後
        """

        B, N, C = x.shape

        # ========================================
        # ウィンドウ分割
        # ========================================

        # (B, N_patches, C) -> (B, H_patch, W_patch, C)
        x_2d = x.reshape(B, H_patch, W_patch, C)

        # ウィンドウサイズ計算
        # 例: H_patch=46, num_windows=2 -> window_h=23
        window_h = H_patch // self.num_windows
        window_w = W_patch // self.num_windows

        # ウィンドウに分割
        # (B, H_patch, W_patch, C)
        # -> (B, num_windows, window_h, num_windows, window_w, C)
        # -> (B, num_windows, num_windows, window_h, window_w, C)
        x_windows = x_2d.reshape(
            B,
            self.num_windows, window_h,
            self.num_windows, window_w,
            C
        )
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5)
        # x_windows: (B, num_windows, num_windows, window_h, window_w, C)

        # ウィンドウをバッチ次元にフラット化
        # (B, num_windows^2, window_h*window_w, C)
        num_windows_total = self.num_windows * self.num_windows
        x_windows = x_windows.reshape(
            B, num_windows_total, window_h * window_w, C
        )

        # バッチとウィンドウをマージ
        # (B * num_windows^2, window_h*window_w, C)
        x_windows = x_windows.reshape(
            B * num_windows_total, window_h * window_w, C
        )


        # ========================================
        # 各ウィンドウ内でアテンション
        # ========================================

        # Transformerブロック適用 (各ウィンドウ独立)
        x_windows = block(x_windows)
        # x_windows: (B * num_windows^2, window_h*window_w, C)


        # ========================================
        # ウィンドウを元に戻す
        # ========================================

        # (B * num_windows^2, window_h*window_w, C)
        # -> (B, num_windows^2, window_h*window_w, C)
        x_windows = x_windows.reshape(
            B, num_windows_total, window_h * window_w, C
        )

        # -> (B, num_windows, num_windows, window_h, window_w, C)
        x_windows = x_windows.reshape(
            B, self.num_windows, self.num_windows,
            window_h, window_w, C
        )

        # -> (B, H_patch, W_patch, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5)
        x_2d = x_windows.reshape(B, H_patch, W_patch, C)

        # -> (B, N_patches, C)
        x = x_2d.reshape(B, N, C)

        return x


    def _compute_window_blocks(
        self,
        total_blocks: int,
        extract_layers: List[int]
    ) -> List[int]:
        """
        ウィンドウアテンションを適用するブロックを計算

        特徴抽出レイヤー以外にウィンドウ化を適用
        """

        window_blocks = []
        for i in range(total_blocks):
            if i not in extract_layers:
                window_blocks.append(i)

        return window_blocks


    def _get_num_blocks(self, model_size: str) -> int:
        """モデルサイズからブロック数を取得"""

        blocks_map = {
            'nano': 12,
            'small': 12,
            'medium': 12,
            'base': 12,
            'large': 24
        }
        return blocks_map.get(model_size, 12)


    def _create_dinov2_model(self, model_size: str, patch_size: int):
        """DINOv2モデルの作成 (実際にはtimm使用)"""

        # 実際の実装では timm.create_model を使用
        # ここではダミーを返す
        class DummyDINOv2(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, 768, kernel_size=patch_size, stride=patch_size)
                self.pos_embed = nn.Parameter(torch.zeros(1, 1, 768))
                self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])

        return DummyDINOv2()


class MultiScaleProjector(nn.Module):
    """
    Multi-scale Feature Pyramid Network

    RF-DETRの特徴:
    - DINOv2の単一スケール特徴からマルチスケール特徴を生成
    - P3 (2x upsampling), P4 (1x), P5 (0.5x downsampling)
    - C2fブロック (YOLOv8風) で特徴融合
    """

    def __init__(
        self,
        in_dims: List[int] = [768, 768, 768, 768],  # 各レイヤーの入力次元
        out_dim: int = 256,                          # 統一出力次元
        scales: List[str] = ['P3', 'P4', 'P5']       # 生成スケール
    ):
        super().__init__()

        self.scales = scales
        self.out_dim = out_dim

        # ========================================
        # 各スケールの生成モジュール
        # ========================================

        # P4: ベーススケール (1x)
        # 最後のDINOv2レイヤーの特徴を使用
        self.p4_proj = nn.Sequential(
            nn.Conv2d(in_dims[-1], out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim)
        )

        # P3: 高解像度スケール (2x upsampling)
        # ConvTranspose2dでアップサンプリング
        if 'P3' in scales:
            self.p3_upsample = nn.ConvTranspose2d(
                out_dim, out_dim,
                kernel_size=2, stride=2
            )
            self.p3_fusion = C2fBlock(
                in_channels=out_dim + in_dims[2],  # P4からのアップサンプル + layer 8の特徴
                out_channels=out_dim,
                num_blocks=3
            )

        # P5: 低解像度スケール (0.5x downsampling)
        # Conv + Stride=2でダウンサンプリング
        if 'P5' in scales:
            self.p5_downsample = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
            self.p5_fusion = C2fBlock(
                in_channels=out_dim,
                out_channels=out_dim,
                num_blocks=3
            )


    def forward(
        self,
        backbone_features: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-scale特徴ピラミッド生成

        入力:
            backbone_features: List[Tensor] 長さ4
                [
                    (B, 768, H/14, W/14),  # layer 2
                    (B, 768, H/14, W/14),  # layer 5
                    (B, 768, H/14, W/14),  # layer 8
                    (B, 768, H/14, W/14)   # layer 11
                ]

        出力:
            pyramid: Dict {
                'P3': (B, 256, H/8, W/8),   # 高解像度
                'P4': (B, 256, H/16, W/16), # 中解像度
                'P5': (B, 256, H/32, W/32)  # 低解像度
            }
        """

        feat_layer2 = backbone_features[0]   # (B, 768, H/14, W/14)
        feat_layer5 = backbone_features[1]   # (B, 768, H/14, W/14)
        feat_layer8 = backbone_features[2]   # (B, 768, H/14, W/14)
        feat_layer11 = backbone_features[3]  # (B, 768, H/14, W/14)

        pyramid = {}

        # ========================================
        # P4: ベーススケール (H/16, W/16相当)
        # ========================================

        # 最終レイヤーの特徴を投影
        p4 = self.p4_proj(feat_layer11)
        # p4: (B, 256, H/14, W/14)

        # H/16に調整 (ダウンサンプリング)
        p4 = nn.functional.avg_pool2d(p4, kernel_size=2, stride=1, padding=0)
        # p4: (B, 256, H/16, W/16) 相当
        pyramid['P4'] = p4


        # ========================================
        # P3: 高解像度スケール (H/8, W/8相当)
        # ========================================

        if 'P3' in self.scales:
            # P4をアップサンプリング
            p3_up = self.p3_upsample(p4)
            # p3_up: (B, 256, H/8, W/8)

            # layer 8の特徴をリサイズして結合
            feat8_resized = nn.functional.interpolate(
                feat_layer8,
                size=p3_up.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # 特徴融合
            p3_concat = torch.cat([p3_up, feat8_resized], dim=1)
            # p3_concat: (B, 256+768, H/8, W/8)

            p3 = self.p3_fusion(p3_concat)
            # p3: (B, 256, H/8, W/8)
            pyramid['P3'] = p3


        # ========================================
        # P5: 低解像度スケール (H/32, W/32相当)
        # ========================================

        if 'P5' in self.scales:
            # P4をダウンサンプリング
            p5 = self.p5_downsample(p4)
            # p5: (B, 256, H/32, W/32)

            p5 = self.p5_fusion(p5)
            # p5: (B, 256, H/32, W/32)
            pyramid['P5'] = p5


        return pyramid


class C2fBlock(nn.Module):
    """
    C2f Block (YOLOv8風のCSP Bottleneck)

    RF-DETRの特徴融合に使用
    軽量で効率的な特徴融合
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        shortcut: bool = True
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Bottleneckブロック
        self.blocks = nn.ModuleList([
            Bottleneck(out_channels, out_channels, shortcut)
            for _ in range(num_blocks)
        ])

        # 最終畳み込み
        self.conv_final = nn.Conv2d(
            out_channels * (2 + num_blocks),
            out_channels,
            kernel_size=1
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        C2fブロックのフォワードパス

        入力:
            x: (B, in_channels, H, W)

        出力:
            out: (B, out_channels, H, W)
        """

        # 初期分岐
        x1 = self.conv1(x)  # (B, out_channels, H, W)
        x2 = self.conv2(x)  # (B, out_channels, H, W)

        # Bottleneckブロックを順次適用
        features = [x1, x2]
        out = x2

        for block in self.blocks:
            out = block(out)
            features.append(out)

        # 全特徴を結合
        out = torch.cat(features, dim=1)
        # out: (B, out_channels * (2 + num_blocks), H, W)

        # 最終畳み込み
        out = self.conv_final(out)
        # out: (B, out_channels, H, W)

        return out


class Bottleneck(nn.Module):
    """基本的なBottleneckブロック"""

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = shortcut and in_channels == out_channels


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)

        if self.shortcut:
            out = out + identity

        return out


# ============================================
# 使用例
# ============================================

def example_backbone():
    """バックボーンの使用例"""

    # DINOv2 + Windowed Attention
    backbone = DINOv2WithWindowedAttention(
        model_size='base',
        patch_size=14,
        extract_layers=[2, 5, 8, 11],
        num_windows=2
    )

    # ダミー入力
    images = torch.randn(2, 3, 640, 640)

    # フォワードパス
    features = backbone(images)

    print("DINOv2 Features:")
    for i, feat in enumerate(features):
        print(f"  Layer {i}: {feat.shape}")


def example_projector():
    """Projectorの使用例"""

    # Multi-scale Projector
    projector = MultiScaleProjector(
        in_dims=[768, 768, 768, 768],
        out_dim=256,
        scales=['P3', 'P4', 'P5']
    )

    # ダミー特徴
    backbone_features = [
        torch.randn(2, 768, 46, 46),  # layer 2
        torch.randn(2, 768, 46, 46),  # layer 5
        torch.randn(2, 768, 46, 46),  # layer 8
        torch.randn(2, 768, 46, 46)   # layer 11
    ]

    # フォワードパス
    pyramid = projector(backbone_features)

    print("\nMulti-scale Pyramid:")
    for scale, feat in pyramid.items():
        print(f"  {scale}: {feat.shape}")


if __name__ == "__main__":
    example_backbone()
    example_projector()
