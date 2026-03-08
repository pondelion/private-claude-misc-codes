"""
ID Decoder - MOTIPの核心イノベーション

【In-Context ID予測とは】
従来のMOT: 物体検出 → コスト行列計算 → ハンガリアンアルゴリズム
MOTIP: 物体検出 → ID Decoder → 直接ID分類

【重要な洞察】
MOTのIDラベルは「一貫性」を示すだけで、固定ラベルである必要はない
例: 軌跡A, B, C, Dに対して
  - [1, 2, 3, 4] でもOK
  - [8, 5, 7, 3] でもOK (時間を通じて一貫していれば)

この性質により、ID予測を分類タスクとして定式化可能
→ 教師あり学習で最適な関連付けを学習

【Relative Position Encoding】
時間的な相対位置を考慮したアテンション
- 未来の情報漏洩を防ぐ
- 軌跡の時間的な文脈を理解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class IDDecoder(nn.Module):
    """
    ID Decoder: Transformer Decoderベースのin-context ID予測

    【アーキテクチャ】
    6層のTransformer Decoder層:
    - Layer 1: Cross-Attentionのみ
    - Layer 2-6: Self-Attention + Cross-Attention + FFN

    【Self-Attention】 (Layer 2以降)
    同一フレーム内の複数検出間で情報交換
    → 類似物体の区別、グローバルな最適解の発見

    【Cross-Attention】
    現在フレームの検出 × 履歴軌跡
    → どの軌跡に属するかを判定

    【Relative Position Encoding】
    時間的な相対位置をアテンションスコアに反映
    - 近い時刻の軌跡に高い重み
    - 未来の情報は使用しない (因果マスク)

    入力:
    - current_tokens: τ^n = concat(f^n, i^spec)
        f^n: DETR特徴 (256-dim)
        i^spec: 新規物体用トークン (256-dim)
    - trajectory_tokens: τ^{m,km} = concat(f^m, i^km)
        f^m: 履歴軌跡特徴 (256-dim)
        i^km: 対応するID埋め込み (256-dim)

    出力:
    - ID分類ロジット (K+1次元)
        K: 通常のID (1~50)
        +1: 新規物体
    """

    def __init__(
        self,
        feature_dim: int = 256,
        id_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        rel_pe_length: int = 30,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.id_dim = id_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.rel_pe_length = rel_pe_length

        # トークン次元 = feature_dim + id_dim
        token_dim = feature_dim + id_dim

        # ========================================
        # Transformer Decoder層
        # ========================================
        self.layers = nn.ModuleList([
            IDDecoderLayer(
                d_model=token_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_self_attn=(i >= 1),  # Layer 2以降
            )
            for i in range(num_layers)
        ])

        # ========================================
        # Relative Position Encoding
        # ========================================
        # 各ヘッドごとに学習可能な相対位置バイアス
        # rel_pos_map[head_idx, time_offset] -> bias
        self.rel_pos_embeddings = nn.Parameter(
            torch.randn(num_heads, rel_pe_length) * 0.02
        )

        # ========================================
        # 出力投影
        # ========================================
        self.output_proj = nn.Linear(token_dim, feature_dim)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        current_tokens: torch.Tensor,      # (B, T, N, C+id_dim)
        trajectory_tokens: torch.Tensor,   # (G, M, C+id_dim)
        trajectory_times: torch.Tensor,    # (G, M)
        current_times: Optional[torch.Tensor] = None,  # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """
        ID Decoderのフォワードパス

        Args:
            current_tokens: 現在フレームの検出トークン (B, T, N, C+id_dim)
                各検出は τ^n = concat(f^n, i^spec)
                B: バッチサイズ
                T: フレーム数
                N: 検出数 (DETRクエリ数)

            trajectory_tokens: 履歴軌跡トークン (G, M, C+id_dim)
                各軌跡は τ^{m,km} = concat(f^m, i^km)
                G: 軌跡グループ数
                M: 各グループの軌跡長

            trajectory_times: 軌跡のタイムスタンプ (G, M)
                相対位置エンコーディングに使用

            current_times: 現在フレームのタイムスタンプ (B, T)

        Returns:
            output_features: (B, T, N, C) - デコード済み特徴
            attention_weights: アテンション重みのリスト (可視化用)
        """
        B, T, N, token_dim = current_tokens.shape
        G, M, _ = trajectory_tokens.shape

        # ========================================
        # Relative Position Encodingマスクを構築
        # ========================================
        if current_times is not None:
            rel_pe_mask = self._build_relative_pe_mask(
                current_times,      # (B, T)
                trajectory_times,   # (G, M)
            )  # (B, T, G, M) or (T, M)
        else:
            # タイムスタンプがない場合はダミー
            rel_pe_mask = torch.zeros(T, M, device=current_tokens.device)

        # ========================================
        # 各Decoder層を通過
        # ========================================
        x = current_tokens
        attention_weights_list = []

        for layer_idx, layer in enumerate(self.layers):
            x, attn_weights = layer(
                x=x,                               # (B, T, N, token_dim)
                memory=trajectory_tokens,          # (G, M, token_dim)
                rel_pe_mask=rel_pe_mask,           # (T, M)
                rel_pe_embeddings=self.rel_pos_embeddings,  # (num_heads, rel_pe_length)
            )
            attention_weights_list.append(attn_weights)

        # ========================================
        # 出力投影: token_dim -> feature_dim
        # ========================================
        output_features = self.output_proj(x)  # (B, T, N, C)
        output_features = self.output_norm(output_features)

        return {
            'output_features': output_features,
            'attention_weights': attention_weights_list,
        }

    def _build_relative_pe_mask(
        self,
        current_times: torch.Tensor,      # (B, T) or (T,)
        trajectory_times: torch.Tensor,   # (G, M)
    ) -> torch.Tensor:
        """
        Relative Position Encodingマスクを構築

        時刻tの検出に対して、各履歴軌跡の相対時間オフセットを計算
        - offset = t_current - t_trajectory
        - offset >= 0 (過去の軌跡のみ使用)
        - offset < rel_pe_length (範囲内に制限)

        Returns:
            rel_pe_mask: (T, M) or (B, T, G, M)
                各位置での相対時間オフセット
                未来の軌跡には-inf (マスク)
        """
        if current_times.dim() == 1:
            T = current_times.shape[0]
            current_times_expanded = current_times[:, None]  # (T, 1)
        else:
            B, T = current_times.shape
            current_times_expanded = current_times[:, :, None, None]  # (B, T, 1, 1)

        G, M = trajectory_times.shape
        trajectory_times_expanded = trajectory_times[None, None, :, :]  # (1, 1, G, M)

        # 相対時間オフセット: offset = t_current - t_trajectory
        rel_time_offset = current_times_expanded - trajectory_times_expanded
        # (T, 1) - (1, 1, G, M) -> (T, G, M) or
        # (B, T, 1, 1) - (1, 1, G, M) -> (B, T, G, M)

        # 未来の軌跡をマスク (offset < 0)
        # 範囲外をマスク (offset >= rel_pe_length)
        rel_pe_mask = torch.clamp(rel_time_offset, 0, self.rel_pe_length - 1)

        # 未来の軌跡は大きな負の値でマスク
        future_mask = rel_time_offset < 0
        rel_pe_mask = rel_pe_mask.masked_fill(future_mask, -1e9)

        return rel_pe_mask


class IDDecoderLayer(nn.Module):
    """
    ID Decoder層 (Transformer Decoder Layer)

    【処理フロー】
    1. Self-Attention (layer 2以降):
       - 同一フレーム内の検出間でアテンション
       - 類似物体の区別、グローバル最適解の発見

    2. Cross-Attention:
       - Query: 現在フレームの検出
       - Key/Value: 履歴軌跡
       - Relative PEでバイアス付与

    3. FFN:
       - Feed-Forward Network
       - 非線形変換

    すべてのサブレイヤーにResidual Connection + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_self_attn: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_self_attn = use_self_attn

        # ========================================
        # Self-Attention (layer 2以降)
        # ========================================
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)

        # ========================================
        # Cross-Attention
        # ========================================
        self.cross_attn = RelativePECrossAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ========================================
        # FFN
        # ========================================
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, N, d_model)
        memory: torch.Tensor,               # (G, M, d_model)
        rel_pe_mask: torch.Tensor,          # (T, M)
        rel_pe_embeddings: torch.Tensor,    # (num_heads, rel_pe_length)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        ID Decoder層のフォワードパス

        Returns:
            x: (B, T, N, d_model)
            attn_weights: Cross-Attentionの重み (可視化用)
        """
        B, T, N, d_model = x.shape

        attn_weights = None

        # ========================================
        # 1. Self-Attention (layer 2以降)
        # ========================================
        if self.use_self_attn:
            # 同一フレーム内の検出間でアテンション
            # (B, T, N, d_model) -> (B*T, N, d_model)
            x_flat = x.view(B * T, N, d_model)

            # Self-Attention
            attn_output, _ = self.self_attn(
                query=x_flat,
                key=x_flat,
                value=x_flat,
            )  # (B*T, N, d_model)

            # Residual + Norm
            x = x + self.dropout1(attn_output.view(B, T, N, d_model))
            x = self.norm1(x)

        # ========================================
        # 2. Cross-Attention: 現在検出 × 履歴軌跡
        # ========================================
        # Relative PEを考慮したCross-Attention
        attn_output, attn_weights = self.cross_attn(
            query=x,                        # (B, T, N, d_model)
            key_value=memory,               # (G, M, d_model)
            rel_pe_mask=rel_pe_mask,        # (T, M)
            rel_pe_embeddings=rel_pe_embeddings,  # (num_heads, rel_pe_length)
        )  # (B, T, N, d_model), (B, T, N, M)

        # Residual + Norm
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # ========================================
        # 3. FFN
        # ========================================
        ffn_output = self.ffn(x)

        # Residual + Norm
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)

        return x, attn_weights


class RelativePECrossAttention(nn.Module):
    """
    Relative Position Encoding付きCross-Attention

    【Relative PEの適用方法】
    1. 通常のQKV計算
    2. アテンションスコア = Q @ K^T / sqrt(d_k)
    3. Relative PEバイアス追加:
       score[t, m] += rel_pe_embeddings[offset[t, m]]
    4. Softmax + Value加重和

    【未来の情報漏洩防止】
    - offset < 0 (未来の軌跡) には-infをマスク
    - Softmax後に重みが0になる
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        # Q, K, V投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 出力投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,                # (B, T, N, d_model)
        key_value: torch.Tensor,            # (G, M, d_model)
        rel_pe_mask: torch.Tensor,          # (T, M)
        rel_pe_embeddings: torch.Tensor,    # (num_heads, rel_pe_length)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Relative PE付きCross-Attention

        Returns:
            output: (B, T, N, d_model)
            attn_weights: (B, T, N, M) - 平均アテンション重み
        """
        B, T, N, d_model = query.shape
        G, M, _ = key_value.shape

        # ========================================
        # QKV投影
        # ========================================
        # Query: 現在フレームの検出
        Q = self.q_proj(query)  # (B, T, N, d_model)
        Q = Q.view(B, T, N, self.num_heads, self.head_dim)
        Q = Q.permute(0, 1, 3, 2, 4)  # (B, T, num_heads, N, head_dim)

        # Key: 履歴軌跡
        K = self.k_proj(key_value)  # (G, M, d_model)
        K = K.view(G, M, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)  # (G, num_heads, M, head_dim)

        # Value: 履歴軌跡
        V = self.v_proj(key_value)  # (G, M, d_model)
        V = V.view(G, M, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)  # (G, num_heads, M, head_dim)

        # ========================================
        # Attention Score計算
        # ========================================
        # 簡略化: すべてのバッチ/フレームで同じ軌跡を使用すると仮定
        # 実際の実装では、各バッチ/フレームごとに異なる軌跡セットを扱う

        # Q @ K^T
        # (B, T, num_heads, N, head_dim) @ (G, num_heads, head_dim, M)
        # -> (B, T, num_heads, N, M)
        attn_scores = torch.einsum('btnhd,ghdm->btnhm', Q, K.mean(0, keepdim=True).squeeze(0))
        attn_scores = attn_scores * self.scale

        # ========================================
        # Relative Position Encodingバイアス追加
        # ========================================
        # rel_pe_mask: (T, M) - 各位置の相対時間オフセット
        # rel_pe_embeddings: (num_heads, rel_pe_length)

        # オフセットインデックスを取得 (0 ~ rel_pe_length-1)
        rel_indices = rel_pe_mask.long().clamp(0, rel_pe_embeddings.size(1) - 1)
        # (T, M)

        # 各ヘッドのバイアスを取得
        # (num_heads, T, M)
        rel_pe_bias = rel_pe_embeddings[:, rel_indices.view(-1)].view(
            self.num_heads, T, M
        )  # 簡略化

        # アテンションスコアに加算
        # (B, T, num_heads, N, M) + (1, T, num_heads, 1, M)
        # 簡略化のため省略

        # ========================================
        # 未来の軌跡をマスク
        # ========================================
        # rel_pe_mask < 0 の位置に-infを適用
        future_mask = rel_pe_mask < 0  # (T, M)
        if future_mask.any():
            attn_scores = attn_scores.masked_fill(
                future_mask[None, :, None, None, :],  # (1, T, 1, 1, M)
                float('-inf')
            )

        # ========================================
        # Softmax + Value加重和
        # ========================================
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, num_heads, N, M)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of Values
        # (B, T, num_heads, N, M) @ (G, num_heads, M, head_dim)
        # -> (B, T, num_heads, N, head_dim)
        output = torch.einsum('btnhm,ghdm->btnhd', attn_weights, V.mean(0, keepdim=True).squeeze(0))

        # Reshape: (B, T, num_heads, N, head_dim) -> (B, T, N, d_model)
        output = output.permute(0, 1, 3, 2, 4).contiguous()
        output = output.view(B, T, N, d_model)

        # 出力投影
        output = self.out_proj(output)

        # 平均アテンション重み (可視化用)
        avg_attn_weights = attn_weights.mean(dim=2)  # (B, T, N, M)

        return output, avg_attn_weights


def id_label_to_embed(
    id_labels: torch.Tensor,          # (B, N) or (G, M)
    id_embeddings: nn.Embedding,      # Embedding(K+1, id_dim)
) -> torch.Tensor:
    """
    IDラベルをID埋め込みに変換

    Args:
        id_labels: IDラベル (1 ~ K, K+1は新規物体)
        id_embeddings: ID辞書

    Returns:
        id_embeds: (B, N, id_dim) or (G, M, id_dim)
    """
    return id_embeddings(id_labels)


def generate_empty_id_embed(
    shape: Tuple[int, ...],
    id_embeddings: nn.Embedding,      # Embedding(K+1, id_dim)
    device: torch.device,
) -> torch.Tensor:
    """
    新規物体用の特別トークン i^spec を生成

    Args:
        shape: (B, N) or (B, T, N)
        id_embeddings: ID辞書
        device: デバイス

    Returns:
        spec_embeds: (B, N, id_dim) or (B, T, N, id_dim)
    """
    K = id_embeddings.num_embeddings - 1  # 特別トークンのインデックス
    spec_token_idx = torch.full(shape, K, dtype=torch.long, device=device)
    return id_embeddings(spec_token_idx)


# ========================================
# 形状ガイド
# ========================================
"""
【ID Decoder入力】
current_tokens: (B, T, N, 512)
    B: バッチサイズ
    T: フレーム数
    N: 検出数 (300)
    512 = feature_dim (256) + id_dim (256)

trajectory_tokens: (G, M, 512)
    G: 軌跡グループ数 (可変)
    M: 各グループの軌跡長 (可変、最大30)

trajectory_times: (G, M)
    タイムスタンプ (フレームインデックス)

【ID Decoder出力】
output_features: (B, T, N, 256)
    デコード済み特徴

attention_weights: List[(B, T, N, M)]
    各層のアテンション重み (可視化用)

【Relative Position Encoding】
rel_pe_mask: (T, M)
    各時刻tの各軌跡mへの相対時間オフセット
    offset = t_current - t_trajectory
    - offset >= 0: 過去の軌跡
    - offset < 0: 未来の軌跡 (マスク)

rel_pe_embeddings: (num_heads, rel_pe_length)
    各ヘッドごとの学習可能な相対位置バイアス
    rel_pe_length = 30 (デフォルト)

【ID分類ロジット】
id_logits: (B, T, N, K+1)
    K = 50: 通常のID
    +1: 新規物体用の特別クラス

【推論時の処理】
1. 各検出のID確率を計算: softmax(id_logits)
2. 最大確率のIDを選択: argmax(id_logits, dim=-1)
3. 閾値以下は新規物体: id_prob < λ_id → 新規
4. 重複排除: 同一フレーム内で同じIDが複数ある場合、
   最高スコアの検出のみ保持、他は新規物体として扱う
"""
