"""
RF-DETR メインフロー - 簡略化疑似コード
=====================================

RF-DETRの物体検出とインスタンスセグメンテーションの全体処理フロー
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class RFDETR(nn.Module):
    """
    RF-DETR: Real-time Detection Transformer

    DINOv2バックボーン + 2ステージTransformerによる高速物体検出
    オプションでインスタンスセグメンテーションもサポート
    """

    def __init__(
        self,
        num_classes: int = 80,          # クラス数 (COCO: 80)
        num_queries: int = 300,          # クエリ数 (検出: 300, セグメンテーション: 200)
        hidden_dim: int = 256,           # 特徴次元
        num_decoder_layers: int = 3,     # Decoderレイヤー数
        num_groups: int = 13,            # Group DETRのグループ数
        with_segmentation: bool = False  # セグメンテーション有効化
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.with_segmentation = with_segmentation

        # ========================================
        # 1. バックボーン: DINOv2 + Windowed Attention
        # ========================================
        self.backbone = DINOv2WithWindowedAttention(
            model_size='base',           # 'nano', 'small', 'medium', 'base', 'large'
            patch_size=14,               # パッチサイズ
            extract_layers=[2, 5, 8, 11],  # 特徴抽出レイヤー
            num_windows=2,               # Windowedアテンションのウィンドウ数
            hidden_dim=768               # DINOv2の内部次元
        )

        # ========================================
        # 2. Multi-Scale Projector: 特徴ピラミッド生成
        # ========================================
        self.projector = MultiScaleProjector(
            in_dims=[768, 768, 768, 768],  # 各レイヤーの入力次元
            out_dim=hidden_dim,            # 出力次元 (256)
            scales=['P3', 'P4', 'P5']      # 生成するスケール
        )
        # P3: 2x upsampling (高解像度)
        # P4: 1x (元解像度)
        # P5: 0.5x downsampling (低解像度)

        # ========================================
        # 3. 位置エンコーディング
        # ========================================
        self.position_encoding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2,
            temperature=10000,
            normalize=True
        )

        # ========================================
        # 4. Transformer Encoder (2ステージの Stage 1)
        # ========================================
        # Note: RF-DETRではEncoderは軽量またはスキップ可能
        # 2ステージ設計でエンコーダ出力からプロポーザル生成
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=1,  # 軽量
            num_heads=8
        )

        # ========================================
        # 5. Transformer Decoder (2ステージの Stage 2)
        # ========================================
        self.decoder = TransformerDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=8,
            num_groups=num_groups,
            use_deformable_attention=True
        )

        # ========================================
        # 6. 検出ヘッド
        # ========================================
        # クラス分類ヘッド
        self.class_embed = nn.Linear(hidden_dim, num_classes)

        # ボックス回帰ヘッド
        self.bbox_embed = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=4,  # (cx, cy, w, h)
            num_layers=3
        )

        # ========================================
        # 7. セグメンテーションヘッド (オプション)
        # ========================================
        if with_segmentation:
            self.segmentation_head = SegmentationHead(
                in_dim=hidden_dim,
                num_blocks=4,
                bottleneck_ratio=1,
                downsample_ratio=4
            )

        # ========================================
        # 8. クエリ埋め込み (学習可能)
        # ========================================
        # Group DETR用: グループごとに異なる埋め込み
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.reference_points = nn.Embedding(num_queries, 4)  # (cx, cy, w, h)


    def forward(
        self,
        images: torch.Tensor,                    # (B, 3, H, W)
        targets: Optional[List[Dict]] = None     # 学習時のGT
    ) -> Dict[str, torch.Tensor]:
        """
        RF-DETR メイン推論フロー

        入力:
            images: (B, 3, H, W) - RGB画像
                B: バッチサイズ
                3: RGB チャンネル
                H, W: 画像サイズ (例: 640x640)

        出力:
            Dict {
                'pred_logits': (B, num_queries, num_classes) - クラス分類スコア
                'pred_boxes': (B, num_queries, 4) - 予測ボックス (cx,cy,w,h) 正規化座標
                'pred_masks': (B, num_queries, H, W) - 予測マスク (セグメンテーション有効時)
                'aux_outputs': List[Dict] - 中間レイヤー出力 (Deep Supervision用)
            }
        """

        B, C, H, W = images.shape

        # ========================================
        # STAGE 1: バックボーン特徴抽出
        # ========================================

        # 1.1 DINOv2で multi-scale特徴抽出
        backbone_features = self.backbone(images)
        # backbone_features: List[Tensor] 長さ4
        #   [
        #     (B, 768, H/14, W/14),  # layer 2
        #     (B, 768, H/14, W/14),  # layer 5
        #     (B, 768, H/14, W/14),  # layer 8
        #     (B, 768, H/14, W/14)   # layer 11
        #   ]
        # 軸の意味: B=バッチ, 768=DINOv2特徴次元, H/14, W/14=パッチグリッド

        # 1.2 Multi-scale Projectorで特徴ピラミッド生成
        pyramid_features = self.projector(backbone_features)
        # pyramid_features: Dict {
        #   'P3': (B, 256, H/8, W/8),   # 高解像度 (2xアップサンプリング)
        #   'P4': (B, 256, H/16, W/16), # 中解像度
        #   'P5': (B, 256, H/32, W/32)  # 低解像度 (0.5xダウンサンプリング)
        # }

        # リストに変換 (Multi-scale Deformable Attention用)
        multi_scale_features = [
            pyramid_features['P3'],
            pyramid_features['P4'],
            pyramid_features['P5']
        ]

        # 1.3 位置エンコーディング生成
        pos_encodings = []
        for feat in multi_scale_features:
            pos_enc = self.position_encoding(feat)
            # pos_enc: (B, 256, H_i, W_i)
            pos_encodings.append(pos_enc)


        # ========================================
        # STAGE 2: Transformer Encoder (プロポーザル生成)
        # ========================================

        # 2.1 特徴を平坦化
        src_flatten = []
        pos_flatten = []
        spatial_shapes = []

        for level, (feat, pos) in enumerate(zip(multi_scale_features, pos_encodings)):
            B, C, H_i, W_i = feat.shape
            spatial_shapes.append((H_i, W_i))

            # Flatten: (B, C, H, W) -> (B, H*W, C) -> (H*W, B, C)
            feat_flat = feat.flatten(2).permute(2, 0, 1)
            pos_flat = pos.flatten(2).permute(2, 0, 1)

            src_flatten.append(feat_flat)
            pos_flatten.append(pos_flat)

        # 全スケールを連結
        src_flatten = torch.cat(src_flatten, dim=0)  # (HW_total, B, 256)
        pos_flatten = torch.cat(pos_flatten, dim=0)  # (HW_total, B, 256)
        # HW_total = H_P3*W_P3 + H_P4*W_P4 + H_P5*W_P5

        # 2.2 Encoderでプロポーザル生成
        memory = self.encoder(
            src=src_flatten,
            pos=pos_flatten
        )
        # memory: (HW_total, B, 256) - エンコード済み特徴

        # 2.3 エンコーダ出力からプロポーザル生成
        # 各位置を潜在的なオブジェクト候補として扱う
        output_proposals = self._gen_encoder_output_proposals(
            memory, spatial_shapes
        )
        # output_proposals: (B, HW_total, 4) - 各位置の参照ボックス

        # Top-Kプロポーザルを選択
        topk_proposals, topk_indices = self._select_topk_proposals(
            memory, output_proposals, k=self.num_queries
        )
        # topk_proposals: (B, num_queries, 4)
        # topk_indices: (B, num_queries)


        # ========================================
        # STAGE 3: Transformer Decoder (クエリ精製)
        # ========================================

        # 3.1 クエリ初期化
        # プロポーザルから選ばれた特徴をクエリとして使用
        query_features = self._gather_features(memory, topk_indices)
        # query_features: (num_queries, B, 256)

        # 学習可能な埋め込みを加算
        query_embed = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)
        # query_embed: (num_queries, B, 256)
        query_features = query_features + query_embed

        # 参照点の初期化
        reference_points_init = topk_proposals  # (B, num_queries, 4)
        reference_points_init = reference_points_init.sigmoid()  # [0, 1]に正規化

        # 3.2 Decoderで精製
        decoder_outputs = self.decoder(
            queries=query_features,                # (num_queries, B, 256)
            memory=memory,                         # (HW_total, B, 256)
            reference_points=reference_points_init,  # (B, num_queries, 4)
            spatial_shapes=spatial_shapes,
            pos_encodings=pos_flatten
        )
        # decoder_outputs: Dict {
        #   'hs': (num_layers, num_queries, B, 256) - 各レイヤーの隠れ状態
        #   'reference_points': (num_layers, B, num_queries, 4) - 精製された参照点
        # }


        # ========================================
        # STAGE 4: 検出ヘッド (クラス分類 + ボックス回帰)
        # ========================================

        # 最終レイヤーの出力
        final_hidden_states = decoder_outputs['hs'][-1]  # (num_queries, B, 256)
        final_hidden_states = final_hidden_states.permute(1, 0, 2)  # (B, num_queries, 256)

        # 4.1 クラス分類
        pred_logits = self.class_embed(final_hidden_states)
        # pred_logits: (B, num_queries, num_classes)

        # 4.2 ボックス回帰
        pred_boxes = self.bbox_embed(final_hidden_states)
        # pred_boxes: (B, num_queries, 4) - (cx, cy, w, h) 正規化座標
        pred_boxes = pred_boxes.sigmoid()  # [0, 1]に正規化


        # ========================================
        # STAGE 5: セグメンテーションヘッド (オプション)
        # ========================================

        pred_masks = None
        if self.with_segmentation:
            # 早期レイヤーの高解像度特徴を使用
            spatial_features = backbone_features[0]  # (B, 768, H/14, W/14)

            # セグメンテーションヘッドでマスク生成
            pred_masks = self.segmentation_head(
                query_features=final_hidden_states,  # (B, num_queries, 256)
                spatial_features=spatial_features     # (B, 768, H/14, W/14)
            )
            # pred_masks: (B, num_queries, H/4, W/4) - マスクロジット


        # ========================================
        # STAGE 6: 補助出力 (Deep Supervision用)
        # ========================================

        aux_outputs = []
        for layer_idx in range(len(decoder_outputs['hs']) - 1):
            layer_hs = decoder_outputs['hs'][layer_idx].permute(1, 0, 2)
            # (B, num_queries, 256)

            aux_logits = self.class_embed(layer_hs)
            aux_boxes = self.bbox_embed(layer_hs).sigmoid()

            aux_outputs.append({
                'pred_logits': aux_logits,  # (B, num_queries, num_classes)
                'pred_boxes': aux_boxes     # (B, num_queries, 4)
            })


        # ========================================
        # 出力の整理
        # ========================================

        outputs = {
            'pred_logits': pred_logits,      # (B, num_queries, num_classes)
            'pred_boxes': pred_boxes,        # (B, num_queries, 4)
            'aux_outputs': aux_outputs       # List[Dict] - 中間出力
        }

        if pred_masks is not None:
            outputs['pred_masks'] = pred_masks  # (B, num_queries, H/4, W/4)

        return outputs


    def _gen_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        エンコーダ出力から初期プロポーザルを生成

        各空間位置を中心とした参照ボックスを生成

        入力:
            memory: (HW_total, B, 256)
            spatial_shapes: List[(H_i, W_i)]

        出力:
            proposals: (B, HW_total, 4) - 各位置の参照ボックス
        """

        HW_total, B, C = memory.shape

        proposals = []
        offset = 0

        for H, W in spatial_shapes:
            # グリッド座標生成
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, H),
                torch.linspace(0, 1, W),
                indexing='ij'
            )
            # grid_x, grid_y: (H, W)

            # 参照ボックス: 各グリッド位置を中心とした小さなボックス
            scale = 0.05  # 初期ボックスサイズ
            boxes = torch.stack([
                grid_x.flatten(),           # cx
                grid_y.flatten(),           # cy
                torch.ones(H*W) * scale,    # w
                torch.ones(H*W) * scale     # h
            ], dim=1)  # (H*W, 4)

            proposals.append(boxes)
            offset += H * W

        # 全スケールを連結
        proposals = torch.cat(proposals, dim=0)  # (HW_total, 4)
        proposals = proposals.unsqueeze(0).expand(B, -1, -1)  # (B, HW_total, 4)

        return proposals


    def _select_topk_proposals(
        self,
        memory: torch.Tensor,
        proposals: torch.Tensor,
        k: int = 300
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Top-Kプロポーザルを選択

        オブジェクトネススコアに基づいて上位K個を選択

        入力:
            memory: (HW_total, B, 256)
            proposals: (B, HW_total, 4)
            k: 選択数

        出力:
            topk_proposals: (B, k, 4)
            topk_indices: (B, k)
        """

        # オブジェクトネススコアを計算 (簡略化: ノルムを使用)
        memory_permuted = memory.permute(1, 0, 2)  # (B, HW_total, 256)
        objectness_scores = memory_permuted.norm(dim=2)  # (B, HW_total)

        # Top-K選択
        topk_scores, topk_indices = torch.topk(objectness_scores, k, dim=1)
        # topk_indices: (B, k)

        # プロポーザルを選択
        topk_proposals = torch.gather(
            proposals,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        )
        # topk_proposals: (B, k, 4)

        return topk_proposals, topk_indices


    def _gather_features(
        self,
        memory: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        インデックスに基づいて特徴を収集

        入力:
            memory: (HW_total, B, 256)
            indices: (B, k)

        出力:
            gathered: (k, B, 256)
        """

        memory_permuted = memory.permute(1, 0, 2)  # (B, HW_total, 256)

        gathered = torch.gather(
            memory_permuted,
            1,
            indices.unsqueeze(-1).expand(-1, -1, memory.shape[2])
        )
        # gathered: (B, k, 256)

        gathered = gathered.permute(1, 0, 2)  # (k, B, 256)

        return gathered


# ============================================
# 使用例
# ============================================

def example_detection():
    """RF-DETRの物体検出使用例"""

    # モデル初期化 (検出のみ)
    model = RFDETR(
        num_classes=80,
        num_queries=300,
        hidden_dim=256,
        num_decoder_layers=3,
        with_segmentation=False
    )
    model.eval()

    # ダミー入力
    images = torch.randn(2, 3, 640, 640)  # バッチサイズ2

    # 推論
    with torch.no_grad():
        outputs = model(images)

    # 結果取得
    pred_logits = outputs['pred_logits']  # (2, 300, 80)
    pred_boxes = outputs['pred_boxes']    # (2, 300, 4)

    # クラススコア (sigmoid + 閾値フィルタ)
    pred_scores = pred_logits.sigmoid()   # (2, 300, 80)
    max_scores, pred_classes = pred_scores.max(dim=2)  # (2, 300)

    # 信頼度でフィルタ (例: 0.5以上)
    for b in range(2):
        valid_mask = max_scores[b] > 0.5
        valid_boxes = pred_boxes[b][valid_mask]      # (N_valid, 4)
        valid_classes = pred_classes[b][valid_mask]  # (N_valid,)
        valid_scores = max_scores[b][valid_mask]     # (N_valid,)

        print(f"画像{b}: {valid_mask.sum()}個のオブジェクトを検出")


def example_segmentation():
    """RF-DETRのインスタンスセグメンテーション使用例"""

    # モデル初期化 (セグメンテーション有効)
    model = RFDETR(
        num_classes=80,
        num_queries=200,  # セグメンテーション用は200
        hidden_dim=256,
        num_decoder_layers=4,
        with_segmentation=True
    )
    model.eval()

    # ダミー入力
    images = torch.randn(1, 3, 432, 432)  # セグメンテーション用解像度

    # 推論
    with torch.no_grad():
        outputs = model(images)

    # 結果取得
    pred_logits = outputs['pred_logits']  # (1, 200, 80)
    pred_boxes = outputs['pred_boxes']    # (1, 200, 4)
    pred_masks = outputs['pred_masks']    # (1, 200, 108, 108) - H/4, W/4

    print(f"検出: {pred_logits.shape}")
    print(f"ボックス: {pred_boxes.shape}")
    print(f"マスク: {pred_masks.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("RF-DETR 物体検出デモ")
    print("=" * 60)
    example_detection()

    print("\n" + "=" * 60)
    print("RF-DETR インスタンスセグメンテーションデモ")
    print("=" * 60)
    example_segmentation()
