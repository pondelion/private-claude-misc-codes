"""
SAM3 Segmentation Head - 簡略化疑似コード
=======================================

オブジェクトクエリからピクセル単位のマスクを生成するセグメンテーションヘッド
"""

import torch
import torch.nn as nn
from typing import List, Dict


class SegmentationHead(nn.Module):
    """
    Universal Segmentation Head

    Decoderのクエリとバックボーンの特徴を使ってマスクを生成
    MaskFormerスタイルのアーキテクチャ
    """

    def __init__(
        self,
        d_model: int = 256,
        num_upsampling_stages: int = 3,
        mask_dim: int = 256
    ):
        super().__init__()

        self.d_model = d_model
        self.mask_dim = mask_dim

        # ========================================
        # Pixel Decoder (特徴マップのアップサンプリング)
        # ========================================

        # Multi-scale特徴を統合してアップサンプリング
        self.pixel_decoder = PixelDecoder(
            in_channels=d_model,
            mask_dim=mask_dim,
            num_stages=num_upsampling_stages
        )

        # ========================================
        # Mask Predictor (クエリ x ピクセル特徴)
        # ========================================

        # クエリをマスク埋め込み空間に投影
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, mask_dim)
        )


    def forward(
        self,
        queries: torch.Tensor,
        vision_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        マスク予測のフォワードパス

        入力:
            queries: (B, N_q, 256) - Decoderからのオブジェクトクエリ
                B: バッチサイズ
                N_q: クエリ数 (通常200)
                256: 特徴次元

            vision_features: List[Tensor] - Multi-scale画像特徴
                [
                    (B, 256, H/4, W/4),   # スケール 4.0x
                    (B, 256, H/8, W/8),   # スケール 2.0x
                    (B, 256, H/16, W/16), # スケール 1.0x
                    (B, 256, H/32, W/32)  # スケール 0.5x
                ]

        出力:
            pred_masks: (B, N_q, H, W) - 予測マスク
                B: バッチサイズ
                N_q: クエリ数
                H, W: 元の画像サイズ (例: 1008x1008)
        """

        B, N_q, C = queries.shape

        # ========================================
        # Step 1: Pixel Decoder (高解像度特徴の生成)
        # ========================================

        # Multi-scale特徴をアップサンプリングして統合
        pixel_embeddings = self.pixel_decoder(vision_features)
        # pixel_embeddings: (B, mask_dim, H, W)
        #   B: バッチサイズ
        #   mask_dim: ピクセル埋め込み次元 (通常256)
        #   H, W: アップサンプリング後の解像度 (元画像サイズ)


        # ========================================
        # Step 2: クエリをマスク埋め込み空間に投影
        # ========================================

        mask_queries = self.mask_embed(queries)
        # mask_queries: (B, N_q, mask_dim)
        #   B: バッチサイズ
        #   N_q: クエリ数
        #   mask_dim: マスク埋め込み次元


        # ========================================
        # Step 3: マスク予測 (Einstein Sum)
        # ========================================

        # クエリとピクセル埋め込みのドット積でマスクを生成
        # (B, N_q, mask_dim) x (B, mask_dim, H, W) -> (B, N_q, H, W)
        pred_masks = torch.einsum('bqc,bchw->bqhw', mask_queries, pixel_embeddings)
        # pred_masks: (B, N_q, H, W)
        #   各クエリに対して空間的なマスクを生成
        #   値は logits (sigmoid前)

        return pred_masks


class PixelDecoder(nn.Module):
    """
    Pixel Decoder (FPN風のアップサンプリングモジュール)

    Multi-scale特徴を統合して高解像度のピクセル埋め込みを生成
    """

    def __init__(
        self,
        in_channels: int = 256,
        mask_dim: int = 256,
        num_stages: int = 3
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mask_dim = mask_dim
        self.num_stages = num_stages

        # 各スケールの特徴を処理する畳み込み
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, mask_dim, kernel_size=1)
            for _ in range(4)  # 4スケール
        ])

        # アップサンプリング後の特徴を精製
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1)
            )
            for _ in range(num_stages)
        ])

        # 最終的な出力投影
        self.final_conv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1)
        )


    def forward(self, vision_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Multi-scale特徴をアップサンプリング

        入力:
            vision_features: List[Tensor] 長さ4
                [
                    (B, 256, H/4, W/4),   # スケール 0 (最高解像度)
                    (B, 256, H/8, W/8),   # スケール 1
                    (B, 256, H/16, W/16), # スケール 2
                    (B, 256, H/32, W/32)  # スケール 3 (最低解像度)
                ]

        出力:
            pixel_embeddings: (B, mask_dim, H, W)
                H, W: 元画像サイズ (スケール0よりさらにアップサンプリング)
        """

        # ========================================
        # Step 1: 各スケールの特徴を統一次元に投影
        # ========================================

        lateral_features = []
        for idx, feat in enumerate(vision_features):
            # feat: (B, 256, H_i, W_i)
            lateral = self.lateral_convs[idx](feat)
            # lateral: (B, mask_dim, H_i, W_i)
            lateral_features.append(lateral)

        # lateral_features: List[(B, mask_dim, H_i, W_i)] 長さ4


        # ========================================
        # Step 2: Top-Down経路 (FPN風の統合)
        # ========================================

        # 最も低解像度から開始
        current = lateral_features[-1]  # (B, mask_dim, H/32, W/32)

        # 高解像度に向かって逐次的にアップサンプリング&統合
        for idx in range(len(lateral_features) - 2, -1, -1):
            # idx: 2 -> 1 -> 0 (低解像度から高解像度へ)

            # 現在の特徴を2倍にアップサンプリング
            upsampled = nn.functional.interpolate(
                current,
                scale_factor=2,
                mode='nearest'
            )
            # upsampled: (B, mask_dim, H_new, W_new)

            # 同じスケールのlateral特徴と加算
            current = upsampled + lateral_features[idx]
            # current: (B, mask_dim, H_i, W_i)

        # current: (B, mask_dim, H/4, W/4) - 最高解像度のFPN特徴


        # ========================================
        # Step 3: さらにアップサンプリング (元画像サイズへ)
        # ========================================

        for stage_idx, conv in enumerate(self.output_convs):
            # 2倍アップサンプリング
            current = nn.functional.interpolate(
                current,
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            )
            # current: (B, mask_dim, H_new, W_new)

            # 畳み込みで精製
            current = conv(current)
            # current: (B, mask_dim, H_new, W_new)

        # num_stages=3の場合、H/4 -> H/2 -> H -> 2H と3回アップサンプリング


        # ========================================
        # Step 4: 最終的な投影
        # ========================================

        pixel_embeddings = self.final_conv(current)
        # pixel_embeddings: (B, mask_dim, H, W)

        return pixel_embeddings


# ============================================
# 補足: マスクのポストプロセス
# ============================================

class MaskPostProcessor:
    """
    マスク予測のポストプロセス処理
    """

    @staticmethod
    def threshold_masks(
        pred_masks: torch.Tensor,
        threshold: float = 0.0
    ) -> torch.Tensor:
        """
        マスクロジットを二値マスクに変換

        入力:
            pred_masks: (B, N_q, H, W) - マスクロジット
            threshold: 閾値

        出力:
            binary_masks: (B, N_q, H, W) - 二値マスク
        """
        # Sigmoidで確率に変換
        prob_masks = torch.sigmoid(pred_masks)

        # 閾値処理
        binary_masks = (prob_masks > threshold).float()

        return binary_masks


    @staticmethod
    def filter_by_score(
        pred_masks: torch.Tensor,
        pred_scores: torch.Tensor,
        score_threshold: float = 0.5,
        max_detections: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        スコアでマスクをフィルタリング

        入力:
            pred_masks: (B, N_q, H, W) - 予測マスク
            pred_scores: (B, N_q) - 信頼度スコア
            score_threshold: スコア閾値
            max_detections: 最大検出数

        出力:
            filtered_results: Dict {
                'masks': List[Tensor] - バッチごとのフィルタ後マスク
                'scores': List[Tensor] - バッチごとのスコア
                'indices': List[Tensor] - 選択されたインデックス
            }
        """
        B, N_q, H, W = pred_masks.shape

        filtered_masks = []
        filtered_scores = []
        filtered_indices = []

        for b in range(B):
            # スコアでフィルタ
            valid_mask = pred_scores[b] > score_threshold
            valid_scores = pred_scores[b][valid_mask]
            valid_indices = torch.nonzero(valid_mask).squeeze(1)

            # スコアでソート
            if len(valid_scores) > 0:
                sorted_indices = torch.argsort(valid_scores, descending=True)
                sorted_indices = sorted_indices[:max_detections]

                selected_indices = valid_indices[sorted_indices]
                selected_scores = valid_scores[sorted_indices]
                selected_masks = pred_masks[b][selected_indices]

                filtered_indices.append(selected_indices)
                filtered_scores.append(selected_scores)
                filtered_masks.append(selected_masks)
            else:
                # 検出なし
                filtered_indices.append(torch.tensor([], dtype=torch.long))
                filtered_scores.append(torch.tensor([], dtype=torch.float32))
                filtered_masks.append(torch.zeros(0, H, W))

        return {
            'masks': filtered_masks,
            'scores': filtered_scores,
            'indices': filtered_indices
        }


    @staticmethod
    def resize_masks(
        masks: torch.Tensor,
        target_size: tuple
    ) -> torch.Tensor:
        """
        マスクを指定サイズにリサイズ

        入力:
            masks: (B, N_q, H, W) or (N_q, H, W)
            target_size: (H_new, W_new)

        出力:
            resized_masks: (B, N_q, H_new, W_new) or (N_q, H_new, W_new)
        """
        return nn.functional.interpolate(
            masks,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )


# ============================================
# 使用例
# ============================================

def example_segmentation_usage():
    """Segmentation Headの使用例"""

    # 初期化
    seg_head = SegmentationHead(
        d_model=256,
        num_upsampling_stages=3,
        mask_dim=256
    )

    # ダミー入力
    B = 2  # バッチサイズ
    N_q = 200  # クエリ数

    queries = torch.randn(B, N_q, 256)

    # Multi-scale特徴 (例: 1008x1008の画像)
    vision_features = [
        torch.randn(B, 256, 252, 252),  # H/4, W/4
        torch.randn(B, 256, 126, 126),  # H/8, W/8
        torch.randn(B, 256, 63, 63),    # H/16, W/16
        torch.randn(B, 256, 31, 31)     # H/32, W/32
    ]

    # フォワードパス
    pred_masks = seg_head(queries, vision_features)
    print("Predicted masks shape:", pred_masks.shape)
    # (2, 200, 1008, 1008) - 2バッチ, 200オブジェクト, 1008x1008解像度

    # ========================================
    # ポストプロセス
    # ========================================

    # ダミースコア
    pred_scores = torch.rand(B, N_q)

    # スコアでフィルタリング
    processor = MaskPostProcessor()
    filtered = processor.filter_by_score(
        pred_masks,
        pred_scores,
        score_threshold=0.5,
        max_detections=100
    )

    # バッチ0の結果
    print(f"Batch 0: {len(filtered['masks'][0])} objects detected")
    print(f"  Masks shape: {filtered['masks'][0].shape}")
    print(f"  Scores: {filtered['scores'][0][:5]}")  # 上位5個

    # 二値化
    binary_masks = processor.threshold_masks(pred_masks, threshold=0.0)
    print("Binary masks shape:", binary_masks.shape)


if __name__ == "__main__":
    example_segmentation_usage()
