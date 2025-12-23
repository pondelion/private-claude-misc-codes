"""
SAM3 Transformer Decoder - 簡略化疑似コード
==========================================

オブジェクトクエリを精製してボックス予測を行うTransformer Decoder
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with Object Queries

    学習可能なクエリをメモリ(エンコーダ出力)とクロスアテンションで精製し、
    各オブジェクトのボックス座標を反復的に改善
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_queries: int = 200,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_queries = num_queries

        # Decoderレイヤー
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # ボックス予測ヘッド (各レイヤーで共有可能)
        self.bbox_embed = nn.ModuleList([
            MLP(d_model, d_model, 4, 3)  # 出力4次元 (cx, cy, w, h)
            for _ in range(num_layers)
        ])

        # クラス埋め込み (スコアリング用、オプション)
        self.class_embed = nn.ModuleList([
            nn.Linear(d_model, 1)  # バイナリスコア
            for _ in range(num_layers)
        ])

        # Reference point用のMLP
        self.ref_point_head = MLP(d_model, d_model, 2, 2)  # (x, y)中心座標


    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decoderのフォワードパス

        入力:
            queries: (N_q, B, 256) - オブジェクトクエリ
                N_q: クエリ数 (通常200)
                B: バッチサイズ
                256: 特徴次元

            memory: (HW, B, 256) - Encoderからのメモリ
                HW: 空間位置数 (multi-scaleのflatten)
                B: バッチサイズ
                256: 特徴次元

            text_features: (L_text, B, 256) - テキスト特徴 (optional)
                L_text: テキストトークン長
                B: バッチサイズ
                256: 特徴次元

        出力:
            Dict {
                'hs': (num_layers, B, N_q, 256) - 各レイヤーのクエリ表現
                'pred_boxes': (num_layers, B, N_q, 4) - 各レイヤーの予測ボックス
                'pred_logits': (num_layers, B, N_q, 1) - 各レイヤーの信頼度
            }
        """

        N_q, B, C = queries.shape
        HW, _, _ = memory.shape

        # 出力を保存するリスト
        intermediate_queries = []
        intermediate_boxes = []
        intermediate_logits = []

        # 現在のクエリ状態
        query_states = queries  # (N_q, B, 256)

        # Reference pointの初期化 (クエリから2D座標を生成)
        reference_points = self.ref_point_head(queries)  # (N_q, B, 2)
        reference_points = reference_points.sigmoid()  # [0, 1]に正規化
        # reference_points: (N_q, B, 2) - 各クエリの参照座標 (x, y)

        # ========================================
        # 各Decoderレイヤーを順次処理
        # ========================================

        for layer_idx, layer in enumerate(self.layers):
            # ----------------------------------------
            # Decoderレイヤー実行
            # ----------------------------------------
            query_states = layer(
                queries=query_states,           # (N_q, B, 256)
                memory=memory,                  # (HW, B, 256)
                text_features=text_features,    # (L_text, B, 256)
                reference_points=reference_points  # (N_q, B, 2)
            )
            # query_states: (N_q, B, 256) - 更新されたクエリ

            # ----------------------------------------
            # ボックス予測
            # ----------------------------------------
            # 現在のクエリからボックスをデコード
            bbox_delta = self.bbox_embed[layer_idx](query_states)
            # bbox_delta: (N_q, B, 4) - デルタ値 (オフセット)

            # Reference pointからの相対位置として解釈
            # bbox_delta: [delta_cx, delta_cy, log(w), log(h)]
            pred_boxes = self._apply_deltas(reference_points, bbox_delta)
            # pred_boxes: (N_q, B, 4) - (cx, cy, w, h) 正規化座標 [0, 1]

            # ----------------------------------------
            # クラスロジット (信頼度) 予測
            # ----------------------------------------
            pred_logits = self.class_embed[layer_idx](query_states)
            # pred_logits: (N_q, B, 1)

            # ----------------------------------------
            # 中間出力を保存
            # ----------------------------------------
            intermediate_queries.append(query_states)  # (N_q, B, 256)
            intermediate_boxes.append(pred_boxes)      # (N_q, B, 4)
            intermediate_logits.append(pred_logits)    # (N_q, B, 1)

            # ----------------------------------------
            # Reference pointの更新 (iterative refinement)
            # ----------------------------------------
            # 予測されたボックス中心を次のreference pointとして使用
            reference_points = pred_boxes[..., :2].detach()  # (N_q, B, 2)

        # ========================================
        # 出力の整理
        # ========================================

        # リストをテンソルにスタック
        intermediate_queries = torch.stack(intermediate_queries, dim=0)
        # (num_layers, N_q, B, 256) -> (num_layers, B, N_q, 256)
        intermediate_queries = intermediate_queries.permute(0, 2, 1, 3)

        intermediate_boxes = torch.stack(intermediate_boxes, dim=0)
        # (num_layers, N_q, B, 4) -> (num_layers, B, N_q, 4)
        intermediate_boxes = intermediate_boxes.permute(0, 2, 1, 3)

        intermediate_logits = torch.stack(intermediate_logits, dim=0)
        # (num_layers, N_q, B, 1) -> (num_layers, B, N_q, 1)
        intermediate_logits = intermediate_logits.permute(0, 2, 1, 3)

        return {
            'hs': intermediate_queries,          # (num_layers, B, N_q, 256)
            'pred_boxes': intermediate_boxes,    # (num_layers, B, N_q, 4)
            'pred_logits': intermediate_logits   # (num_layers, B, N_q, 1)
        }


    def _apply_deltas(
        self,
        reference_points: torch.Tensor,
        bbox_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Reference pointにデルタを適用してボックスを生成

        入力:
            reference_points: (N_q, B, 2) - (cx, cy)
            bbox_delta: (N_q, B, 4) - [delta_cx, delta_cy, log(w), log(h)]

        出力:
            boxes: (N_q, B, 4) - (cx, cy, w, h) 正規化座標
        """
        # デルタを分解
        delta_cxy = bbox_delta[..., :2]  # (N_q, B, 2)
        log_wh = bbox_delta[..., 2:]     # (N_q, B, 2)

        # 中心座標の更新
        cxy = reference_points + delta_cxy  # (N_q, B, 2)
        cxy = cxy.sigmoid()  # [0, 1]にクリップ

        # 幅・高さの計算
        wh = log_wh.sigmoid()  # (N_q, B, 2) [0, 1]

        # ボックスを結合
        boxes = torch.cat([cxy, wh], dim=-1)  # (N_q, B, 4)
        return boxes


class TransformerDecoderLayer(nn.Module):
    """
    単一のTransformer Decoderレイヤー

    Self-Attention -> Cross-Attention (text) -> Cross-Attention (image) -> FFN
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        # Self-Attention (クエリ間の相互作用)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # Cross-Attention to Text
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # Cross-Attention to Image Memory
        self.memory_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=False
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)


    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoderレイヤーのフォワードパス

        入力:
            queries: (N_q, B, 256) - オブジェクトクエリ
            memory: (HW, B, 256) - エンコーダメモリ
            text_features: (L_text, B, 256) - テキスト特徴
            reference_points: (N_q, B, 2) - 参照座標

        出力:
            queries: (N_q, B, 256) - 更新されたクエリ
        """

        # ========================================
        # 1. Self-Attention (クエリ間の相互作用)
        # ========================================
        q2, _ = self.self_attn(
            query=queries,   # (N_q, B, 256)
            key=queries,     # (N_q, B, 256)
            value=queries    # (N_q, B, 256)
        )
        # q2: (N_q, B, 256)

        queries = queries + q2
        queries = self.norm1(queries)  # (N_q, B, 256)


        # ========================================
        # 2. Cross-Attention to Text (テキストプロンプトへの注意)
        # ========================================
        if text_features is not None:
            q2, _ = self.text_cross_attn(
                query=queries,        # (N_q, B, 256) - 各クエリが
                key=text_features,    # (L_text, B, 256) - テキストに注目
                value=text_features   # (L_text, B, 256)
            )
            # q2: (N_q, B, 256)

            queries = queries + q2
            queries = self.norm2(queries)  # (N_q, B, 256)


        # ========================================
        # 3. Cross-Attention to Image Memory (画像特徴への注意)
        # ========================================

        # 位置バイアスの計算 (reference pointsを使用)
        if reference_points is not None:
            # Box-to-point relative position bias (BoxRPB)
            # 実装の詳細は省略 (RoPEやデフォルマブルアテンションを使用可能)
            position_bias = self._compute_position_bias(reference_points, memory)
        else:
            position_bias = None

        q2, _ = self.memory_cross_attn(
            query=queries,     # (N_q, B, 256) - 各クエリが
            key=memory,        # (HW, B, 256) - 画像メモリに注目
            value=memory,      # (HW, B, 256)
            # attn_mask=position_bias  # 位置バイアス (optional)
        )
        # q2: (N_q, B, 256)

        queries = queries + q2
        queries = self.norm3(queries)  # (N_q, B, 256)


        # ========================================
        # 4. Feed-Forward Network
        # ========================================
        q2 = self.ffn(queries)  # (N_q, B, 256)

        queries = queries + q2
        queries = self.norm4(queries)  # (N_q, B, 256)

        return queries


    def _compute_position_bias(self, reference_points, memory):
        """
        位置バイアスの計算 (BoxRPB等)

        実際にはRoPEやデフォルマブルアテンションを使用
        ここでは詳細を省略
        """
        # 省略: 実装の詳細はSAM3の実コードを参照
        return None


# ============================================
# 補助モジュール
# ============================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (シンプルな全結合ネットワーク)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DotProductScoring(nn.Module):
    """
    ドット積ベースのスコアリング

    クエリとテキスト特徴の類似度を計算して信頼度スコアを生成
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # スコアリング用のプロジェクション
        self.query_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)

        # 温度パラメータ
        self.temperature = nn.Parameter(torch.tensor(1.0))


    def forward(
        self,
        queries: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        スコアを計算

        入力:
            queries: (B, N_q, 256) - オブジェクトクエリ
                B: バッチサイズ
                N_q: クエリ数
                256: 特徴次元

            text_features: (L_text, B, 256) - テキスト特徴
                L_text: テキスト長
                B: バッチサイズ
                256: 特徴次元

        出力:
            scores: (B, N_q) - 各クエリのスコア
        """

        # クエリのプロジェクション
        queries_proj = self.query_proj(queries)  # (B, N_q, 256)
        queries_proj = nn.functional.normalize(queries_proj, dim=-1)

        # テキスト特徴のプロジェクション
        text_features = text_features.permute(1, 0, 2)  # (B, L_text, 256)
        text_proj = self.text_proj(text_features)  # (B, L_text, 256)
        text_proj = nn.functional.normalize(text_proj, dim=-1)

        # テキストをプーリング (平均)
        text_pooled = text_proj.mean(dim=1)  # (B, 256)

        # ドット積でスコア計算
        scores = torch.einsum('bqd,bd->bq', queries_proj, text_pooled)
        # scores: (B, N_q)

        # 温度スケーリング
        scores = scores / self.temperature

        return scores


# ============================================
# 使用例
# ============================================

def example_decoder_usage():
    """Decoderの使用例"""

    # 初期化
    decoder = TransformerDecoder(
        d_model=256,
        num_layers=6,
        num_heads=8,
        num_queries=200
    )

    # ダミー入力
    B = 2  # バッチサイズ
    N_q = 200  # クエリ数
    HW = 5000  # 空間位置数
    L_text = 30  # テキスト長

    queries = torch.randn(N_q, B, 256)
    memory = torch.randn(HW, B, 256)
    text_features = torch.randn(L_text, B, 256)

    # フォワードパス
    outputs = decoder(
        queries=queries,
        memory=memory,
        text_features=text_features
    )

    # 出力
    print("Hidden states shape:", outputs['hs'].shape)
    # (6, 2, 200, 256) - 6レイヤー, 2バッチ, 200クエリ, 256次元

    print("Predicted boxes shape:", outputs['pred_boxes'].shape)
    # (6, 2, 200, 4) - 6レイヤー, 2バッチ, 200ボックス, 4座標

    print("Predicted logits shape:", outputs['pred_logits'].shape)
    # (6, 2, 200, 1) - 6レイヤー, 2バッチ, 200クエリ, 1スコア

    # 最終レイヤーの出力を使用
    final_boxes = outputs['pred_boxes'][-1]  # (2, 200, 4)
    final_queries = outputs['hs'][-1]  # (2, 200, 256)

    # スコアリング
    scoring = DotProductScoring(d_model=256)
    scores = scoring(final_queries, text_features)  # (2, 200)

    print("Final scores shape:", scores.shape)


if __name__ == "__main__":
    example_decoder_usage()
