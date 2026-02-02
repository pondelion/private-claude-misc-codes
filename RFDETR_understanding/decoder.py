"""
RF-DETR Decoder - 簡略化疑似コード
==================================

Group DETR + Deformable Attention
RF-DETRの特徴的なDecoder実装
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class TransformerDecoder(nn.Module):
    """
    RF-DETR Transformer Decoder

    RF-DETRの特徴:
    - Group DETR: クエリを13グループに分割して並列処理
    - Deformable Attention: 効率的なマルチスケールアテンション
    - Lite Refpoint Refine: 軽量な参照点精製
    - 2-4レイヤーの軽量設計
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        num_groups: int = 13,            # Group DETR のグループ数
        num_queries_per_group: int = 300 // 13,  # グループあたりのクエリ数
        use_deformable_attention: bool = True,
        lite_refpoint_refine: bool = True  # 軽量な参照点精製
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.num_queries_per_group = num_queries_per_group
        self.lite_refpoint_refine = lite_refpoint_refine

        # ========================================
        # Decoder レイヤー
        # ========================================
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                use_deformable_attention=use_deformable_attention
            )
            for _ in range(num_layers)
        ])

        # ========================================
        # 参照点精製用MLP
        # ========================================
        if not lite_refpoint_refine:
            # 完全な反復精製
            self.ref_point_head = nn.ModuleList([
                MLP(hidden_dim, hidden_dim, 4, 2)
                for _ in range(num_layers)
            ])
        else:
            # Lite mode: 初期参照点を全レイヤーで共有
            self.ref_point_head = None


    def forward(
        self,
        queries: torch.Tensor,                    # (num_queries, B, 256)
        memory: torch.Tensor,                     # (HW_total, B, 256)
        reference_points: torch.Tensor,           # (B, num_queries, 4)
        spatial_shapes: List[Tuple[int, int]],    # [(H_P3, W_P3), ...]
        pos_encodings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decoder フォワードパス

        入力:
            queries: (num_queries, B, 256) - 初期クエリ特徴
            memory: (HW_total, B, 256) - エンコーダメモリ
            reference_points: (B, num_queries, 4) - 初期参照点 (cx,cy,w,h)
            spatial_shapes: マルチスケール特徴の空間shape
            pos_encodings: 位置エンコーディング (optional)

        出力:
            Dict {
                'hs': (num_layers, num_queries, B, 256) - 各レイヤーの隠れ状態
                'reference_points': (num_layers, B, num_queries, 4) - 精製された参照点
            }
        """

        num_queries, B, C = queries.shape

        # ========================================
        # Group DETR: クエリをグループ分割
        # ========================================
        # RF-DETRの重要ポイント:
        # 学習時は13グループに分割して並列処理 -> 収束高速化
        # 推論時は1グループ (全クエリを一括処理)

        if self.training:
            # 学習時: グループ分割
            queries_grouped = self._split_groups(queries)
            # queries_grouped: (num_groups, num_queries_per_group, B, 256)

            reference_points_grouped = self._split_groups_2d(reference_points)
            # reference_points_grouped: (num_groups, B, num_queries_per_group, 4)
        else:
            # 推論時: グループ化なし
            queries_grouped = queries.unsqueeze(0)  # (1, num_queries, B, 256)
            reference_points_grouped = reference_points.unsqueeze(0)  # (1, B, num_queries, 4)


        # ========================================
        # 各グループを処理
        # ========================================

        all_hidden_states = []
        all_reference_points = []

        for group_idx in range(queries_grouped.shape[0]):
            # グループのクエリと参照点
            group_queries = queries_grouped[group_idx]  # (num_queries_per_group, B, 256)
            group_ref_points = reference_points_grouped[group_idx]  # (B, num_queries_per_group, 4)

            # グループごとにDecoderレイヤーを通過
            group_hs, group_ref_pts = self._process_group(
                group_queries,
                memory,
                group_ref_points,
                spatial_shapes,
                pos_encodings
            )
            # group_hs: (num_layers, num_queries_per_group, B, 256)
            # group_ref_pts: (num_layers, B, num_queries_per_group, 4)

            all_hidden_states.append(group_hs)
            all_reference_points.append(group_ref_pts)


        # ========================================
        # グループを結合
        # ========================================

        # 隠れ状態を結合
        hs = torch.cat(all_hidden_states, dim=1)
        # hs: (num_layers, num_queries_total, B, 256)

        # 参照点を結合
        reference_points_out = torch.cat(all_reference_points, dim=2)
        # reference_points_out: (num_layers, B, num_queries_total, 4)


        return {
            'hs': hs,
            'reference_points': reference_points_out
        }


    def _process_group(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        pos_encodings: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        単一グループの処理

        入力:
            queries: (num_queries_per_group, B, 256)
            memory: (HW_total, B, 256)
            reference_points: (B, num_queries_per_group, 4)

        出力:
            hs: (num_layers, num_queries_per_group, B, 256)
            ref_pts: (num_layers, B, num_queries_per_group, 4)
        """

        hidden_states_all = []
        reference_points_all = []

        current_queries = queries
        current_ref_points = reference_points

        for layer_idx, layer in enumerate(self.layers):
            # ========================================
            # Decoder レイヤー実行
            # ========================================
            current_queries = layer(
                queries=current_queries,
                memory=memory,
                reference_points=current_ref_points,
                spatial_shapes=spatial_shapes,
                pos_encodings=pos_encodings
            )
            # current_queries: (num_queries_per_group, B, 256)

            # ========================================
            # 参照点の精製
            # ========================================
            if not self.lite_refpoint_refine and self.ref_point_head is not None:
                # 完全な反復精製
                # クエリから参照点のオフセットを予測
                ref_point_delta = self.ref_point_head[layer_idx](
                    current_queries.permute(1, 0, 2)  # (B, num_queries, 256)
                )
                # ref_point_delta: (B, num_queries, 4)

                # 参照点を更新
                current_ref_points = current_ref_points + ref_point_delta
                current_ref_points = current_ref_points.sigmoid()  # [0,1]に正規化
            # else: lite mode では初期参照点を全レイヤーで共有

            # 保存
            hidden_states_all.append(current_queries)
            reference_points_all.append(current_ref_points)

        # スタック
        hs = torch.stack(hidden_states_all, dim=0)
        # hs: (num_layers, num_queries_per_group, B, 256)

        ref_pts = torch.stack(reference_points_all, dim=0)
        # ref_pts: (num_layers, B, num_queries_per_group, 4)

        return hs, ref_pts


    def _split_groups(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        クエリをグループ分割

        入力:
            tensor: (num_queries_total, B, C)

        出力:
            grouped: (num_groups, num_queries_per_group, B, C)
        """

        num_queries_total, B, C = tensor.shape

        # グループに再形成
        grouped = tensor.reshape(
            self.num_groups,
            self.num_queries_per_group,
            B, C
        )

        return grouped


    def _split_groups_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        参照点をグループ分割

        入力:
            tensor: (B, num_queries_total, 4)

        出力:
            grouped: (num_groups, B, num_queries_per_group, 4)
        """

        B, num_queries_total, C = tensor.shape

        # グループに再形成
        grouped = tensor.reshape(
            B,
            self.num_groups,
            self.num_queries_per_group,
            C
        )
        # (B, num_groups, num_queries_per_group, 4)

        # グループ次元を先頭に
        grouped = grouped.permute(1, 0, 2, 3)
        # (num_groups, B, num_queries_per_group, 4)

        return grouped


class TransformerDecoderLayer(nn.Module):
    """
    RF-DETR Decoder Layer

    Self-Attention + Deformable Cross-Attention + FFN
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        use_deformable_attention: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ========================================
        # Self-Attention (クエリ間)
        # ========================================
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=False
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # ========================================
        # Cross-Attention (クエリ → メモリ)
        # ========================================
        if use_deformable_attention:
            # Deformable Attention (マルチスケール対応)
            self.cross_attn = MultiScaleDeformableAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=3,           # P3, P4, P5
                num_points=4            # サンプリングポイント数
            )
        else:
            # 標準的なCross-Attention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=False
            )

        self.norm2 = nn.LayerNorm(hidden_dim)

        # ========================================
        # Feed-Forward Network
        # ========================================
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)


    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        pos_encodings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder Layer フォワードパス

        入力:
            queries: (num_queries, B, 256)
            memory: (HW_total, B, 256)
            reference_points: (B, num_queries, 4)
            spatial_shapes: [(H_P3, W_P3), (H_P4, W_P4), (H_P5, W_P5)]

        出力:
            queries: (num_queries, B, 256) - 更新されたクエリ
        """

        # ========================================
        # Self-Attention
        # ========================================
        q2, _ = self.self_attn(queries, queries, queries)
        queries = queries + q2
        queries = self.norm1(queries)

        # ========================================
        # Cross-Attention (Deformable)
        # ========================================
        if isinstance(self.cross_attn, MultiScaleDeformableAttention):
            # Deformable Attention
            q2 = self.cross_attn(
                query=queries,
                reference_points=reference_points,
                value=memory,
                spatial_shapes=spatial_shapes
            )
        else:
            # 標準Cross-Attention
            q2, _ = self.cross_attn(queries, memory, memory)

        queries = queries + q2
        queries = self.norm2(queries)

        # ========================================
        # FFN
        # ========================================
        q2 = self.ffn(queries.permute(1, 0, 2))  # (B, num_queries, 256)
        q2 = q2.permute(1, 0, 2)  # (num_queries, B, 256)

        queries = queries + q2
        queries = self.norm3(queries)

        return queries


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention

    RF-DETRの重要コンポーネント:
    - 複数スケールの特徴からサンプリング
    - 学習可能なオフセットで適応的なサンプリング位置
    - 効率的なマルチスケール処理
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,     # スケール数 (P3, P4, P5)
        num_points: int = 4      # 各ヘッド・レベルあたりのサンプリングポイント数
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        # ========================================
        # サンプリングオフセットの予測
        # ========================================
        # 各クエリに対して、各ヘッド・レベル・ポイントのオフセットを予測
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points * 2  # 2: (x, y)
        )

        # ========================================
        # アテンション重みの予測
        # ========================================
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points
        )

        # ========================================
        # 出力投影
        # ========================================
        self.output_proj = nn.Linear(embed_dim, embed_dim)


    def forward(
        self,
        query: torch.Tensor,                      # (num_queries, B, 256)
        reference_points: torch.Tensor,           # (B, num_queries, 4)
        value: torch.Tensor,                      # (HW_total, B, 256)
        spatial_shapes: List[Tuple[int, int]]     # [(H_P3, W_P3), ...]
    ) -> torch.Tensor:
        """
        Deformable Attention フォワードパス

        処理フロー:
        1. 参照点から相対オフセットを予測
        2. オフセットを適用してサンプリング位置を計算
        3. マルチスケール特徴からサンプリング
        4. アテンション重みで集約

        入力:
            query: (num_queries, B, 256)
            reference_points: (B, num_queries, 4) - (cx, cy, w, h) 正規化座標
            value: (HW_total, B, 256) - マルチスケール特徴 (連結済み)
            spatial_shapes: 各スケールの (H, W)

        出力:
            output: (num_queries, B, 256)
        """

        num_queries, B, C = query.shape

        # ========================================
        # Step 1: サンプリングオフセットの予測
        # ========================================

        query_permuted = query.permute(1, 0, 2)  # (B, num_queries, 256)

        # オフセット予測
        sampling_offsets = self.sampling_offsets(query_permuted)
        # sampling_offsets: (B, num_queries, num_heads*num_levels*num_points*2)

        # Reshape
        sampling_offsets = sampling_offsets.view(
            B, num_queries,
            self.num_heads, self.num_levels, self.num_points, 2
        )
        # sampling_offsets: (B, num_queries, num_heads, num_levels, num_points, 2)


        # ========================================
        # Step 2: サンプリング位置の計算
        # ========================================

        # 参照点の中心座標を抽出
        reference_xy = reference_points[..., :2]  # (B, num_queries, 2)

        # オフセットを適用
        # 参照点に相対オフセットを加算
        reference_xy = reference_xy.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # (B, num_queries, 1, 1, 1, 2)

        sampling_locations = reference_xy + sampling_offsets * 0.1  # スケール調整
        # sampling_locations: (B, num_queries, num_heads, num_levels, num_points, 2)


        # ========================================
        # Step 3: アテンション重みの予測
        # ========================================

        attention_weights = self.attention_weights(query_permuted)
        # attention_weights: (B, num_queries, num_heads*num_levels*num_points)

        attention_weights = attention_weights.view(
            B, num_queries,
            self.num_heads, self.num_levels, self.num_points
        )
        # attention_weights: (B, num_queries, num_heads, num_levels, num_points)

        # Softmax (num_levels * num_points 次元で正規化)
        attention_weights = attention_weights.softmax(dim=-1).softmax(dim=-2)


        # ========================================
        # Step 4: マルチスケール特徴のサンプリング
        # ========================================

        # 実際の実装では、各レベルの特徴マップから
        # sampling_locations に基づいて補間サンプリング
        # ここでは簡略化

        # value を各レベルに分割
        value_list = self._split_value_by_levels(value, spatial_shapes)
        # value_list: List[(H_i*W_i, B, 256)]

        # サンプリング (簡略化: 平均プーリング)
        sampled_values = []
        for level_idx, level_value in enumerate(value_list):
            # 各レベルからサンプリング
            # 実装ではgrid_sampleを使用
            sampled = level_value.mean(dim=0, keepdim=True)
            # (1, B, 256)
            sampled_values.append(sampled)

        # サンプリングされた値を結合
        sampled = torch.cat(sampled_values, dim=0)
        # (num_levels, B, 256)


        # ========================================
        # Step 5: アテンション重みで集約
        # ========================================

        # 重み付き和 (簡略化)
        output = sampled.mean(dim=0)  # (B, 256)
        output = output.unsqueeze(0).expand(num_queries, -1, -1)
        # (num_queries, B, 256)

        # 出力投影
        output = self.output_proj(output.permute(1, 0, 2))
        # (B, num_queries, 256)
        output = output.permute(1, 0, 2)
        # (num_queries, B, 256)

        return output


    def _split_value_by_levels(
        self,
        value: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        """valueをレベルごとに分割"""

        HW_total, B, C = value.shape

        value_list = []
        start_idx = 0

        for H, W in spatial_shapes:
            end_idx = start_idx + H * W
            level_value = value[start_idx:end_idx]
            # (H*W, B, C)
            value_list.append(level_value)
            start_idx = end_idx

        return value_list


class MLP(nn.Module):
    """シンプルなMLP"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
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


# ダミークラス (main_flow.pyで使用)
class TransformerEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, src, pos):
        return src  # 簡略化: そのまま返す


class PositionEmbeddingSine(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        # ダミー位置エンコーディング
        return torch.zeros_like(x)
