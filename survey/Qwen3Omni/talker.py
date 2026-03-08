"""
Qwen3-Omni Talker (MoE) + MTP Module
======================================

MoE (Mixture of Experts) アーキテクチャの音声トークン生成器。
Thinker からのマルチモーダル表現を受け取り、RVQ (Residual Vector Quantization)
コードブックトークンを階層的に自己回帰生成する。

パラメータ: 3B総パラメータ / 0.3B活性パラメータ (MoE)
MTPモジュール: 80M (Dense Transformer)

主な差分 (vs Qwen2.5-Omni Talker):
    - アーキテクチャ: Dense → MoE (3B-A0.3B)
    - コードブック: 単一 → マルチコードブック (RVQ) + MTP
    - 入力: Thinkerテキスト隠れ状態のみ → マルチモーダル特徴量を直接受容
    - 生成遅延: ブロックコンテキスト待ち → 最初のトークンから即座に波形出力
    - 融合方式: Dual-Track加算融合 (codec+thinker隠れ) → 階層的コードブック予測
    - 話者: Chelsie+Ethan → Chelsie(女性)+Ethan(男性)+Aiden(男性)
    - システムプロンプト: 共有 → Thinker/Talker独立 (分離テキスト表現)

アーキテクチャ概要:

    Thinker からの入力:
        1. 高レベルテキスト表現 (履歴テキストトークン)
        2. マルチモーダル表現 (中間層からの audio/visual 埋め込み)
        3. 現在ターンのストリーミングテキスト
        ※ Qwen2.5-Omni のようなテキストのみの隠れ状態ではなく、
          マルチモーダル特徴量を直接受け取る → 音声-映像連携発話が可能

    Talker Backbone (MoE Transformer):
        入力: 集約されたコードブック特徴量 (現在フレーム全コードブックの集約)
              + Thinker マルチモーダル表現
        出力: 次フレームの第0コードブックトークン予測
        ※ Left-context-only 生成: 最初のトークンから即座に波形出力可能

    MTP Module (Dense Transformer, 80M):
        入力: Talker Backbone の隠れ状態 + 第0コードブック埋め込み
        出力: 現在フレームの残余コードブックトークン (第1~K-1)
        ※ 超軽量固定ステップ自己回帰 → 低メモリ帯域、高効率バッチ推論
        ※ 固定KVキャッシュサイズ (固定ステップ数)

    マルチコードブック自己回帰スキーム:
        フレーム t:
            Backbone: codebook_features(t) → zeroth_codebook(t+1) を予測
            MTP:      backbone_hidden(t) + zeroth_emb(t) → codebook_1(t)~codebook_{K-1}(t)
        フレーム t+1:
            全コードブック(t+1) を集約 → Backbone へ入力
            ... 繰り返し

    RVQ デコーダ:
        全コードブックトークン → 波形 (left-context-only で即座出力)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


# ============================================
# MoE (Mixture of Experts) コンポーネント
# ============================================

class ExpertFFN(nn.Module):
    """
    単一エキスパートの FFN (SwiGLU)

    MoE における個別エキスパートネットワーク。
    SwiGLU 活性化関数を使用。

    入力: (B, L, hidden_size)
    出力: (B, L, hidden_size)
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        """
        入力: x (B, L, hidden_size)
        出力: (B, L, hidden_size)

        SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Mixture of Experts レイヤー

    Top-K ルーティングでトークンごとに K 個のエキスパートを選択し、
    重み付き合成で出力を得る。

    Qwen3-Omni Talker: 3B総パラメータ, 0.3B活性 → 約10:1比率
    エキスパート数とTop-K の具体値は非公開だが、
    Qwen3ベースのMoEパターンに準拠 (例: 128 experts, top-2)

    入力: (B, L, hidden_size)
    出力: (B, L, hidden_size)
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts=16,
        top_k=2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # ルーティングゲート: hidden_size → num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # エキスパート群
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        入力: x (B, L, hidden_size)
        出力: (B, L, hidden_size)

        処理:
            1. ゲートスコア計算: (B, L, num_experts)
            2. Top-K 選択: 各トークンで K 個のエキスパートを選択
            3. 選択エキスパートの重み付き合成
        """
        B, L, D = x.shape

        # ルーティングスコア
        router_logits = self.gate(x)  # (B, L, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, L, num_experts)

        # Top-K エキスパート選択
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        # top_k_probs:   (B, L, top_k) - 選択エキスパートの重み
        # top_k_indices: (B, L, top_k) - 選択エキスパートのインデックス

        # 重みの正規化
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # エキスパートの重み付き合成 (簡略実装: ループベース)
        # ※ 実装では token-to-expert ディスパッチで効率化
        output = torch.zeros_like(x)  # (B, L, D)
        x_flat = x.reshape(-1, D)  # (B*L, D)
        top_k_indices_flat = top_k_indices.reshape(-1, self.top_k)  # (B*L, top_k)
        top_k_probs_flat = top_k_probs.reshape(-1, self.top_k)  # (B*L, top_k)

        for k in range(self.top_k):
            expert_indices = top_k_indices_flat[:, k]  # (B*L,)
            expert_weights = top_k_probs_flat[:, k]    # (B*L,)

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)  # (B*L,) bool
                if mask.any():
                    expert_input = x_flat[mask]  # (num_selected, D)
                    expert_output = self.experts[expert_id](
                        expert_input.unsqueeze(0)
                    ).squeeze(0)  # (num_selected, D)
                    weighted_output = expert_output * expert_weights[mask].unsqueeze(-1)
                    output.reshape(-1, D)[mask] += weighted_output

        return output


# ============================================
# Talker MoE Decoder Layer
# ============================================

class TalkerMoEDecoderLayer(nn.Module):
    """
    Talker の単一 MoE Decoder レイヤー

    Self-Attention + MoE FFN で構成。
    Dense FFN の代わりに MoE を使用し、計算効率を向上。

    構成:
        RMSNorm → Self-Attention → Residual
        RMSNorm → MoE FFN → Residual

    入力: (B, L, hidden_size)
    出力: (B, L, hidden_size), (K_cache, V_cache)
    """

    def __init__(
        self,
        hidden_size=1024,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=2048,
        num_experts=16,
        top_k=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # GQA (Grouped Query Attention): KVヘッド数 < Qヘッド数
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # MoE FFN
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)

    def forward(self, x, past_kv=None):
        """
        入力:
            x: (B, L, hidden_size)
            past_kv: Optional[(K_cache, V_cache)]
        出力:
            x: (B, L, hidden_size)
            (K_cache, V_cache): KVキャッシュ
        """
        # Self-Attention (GQA)
        residual = x
        x = self.input_layernorm(x)
        q = self.q_proj(x)  # (B, L, hidden_size)
        k = self.k_proj(x)  # (B, L, num_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, L, num_kv_heads * head_dim)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)

        # 簡略化: GQA の KV ヘッドをリピートして Q ヘッド数に合わせる
        B, L_q, _ = q.shape
        _, L_kv, _ = k.shape

        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k_grouped = k.view(B, L_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_grouped = v.view(B, L_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # KV ヘッドのリピート (GQA)
        repeats = self.num_heads // self.num_kv_heads
        k_expanded = k_grouped.repeat_interleave(repeats, dim=1)  # (B, num_heads, L_kv, head_dim)
        v_expanded = v_grouped.repeat_interleave(repeats, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded, is_causal=(past_kv is None)
        )
        # attn_out: (B, num_heads, L_q, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, L_q, self.hidden_size)
        x = residual + self.o_proj(attn_out)

        # MoE FFN
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.moe(x)

        return x, (k, v)


# ============================================
# Multi-Codebook Head
# ============================================

class MultiCodebookHead(nn.Module):
    """
    マルチコードブック集約・予測ヘッド

    RVQ (Residual Vector Quantization) の各コードブックに対する
    埋め込みテーブルと線形予測ヘッドを管理する。

    構成:
        - codebook_embeddings: K 個の埋め込みテーブル (各コードブック用)
        - zeroth_head: Talker Backbone 出力から第0コードブックを予測する線形層

    集約方式:
        各フレームの K 個のコードブック埋め込みを加算して集約し、
        次フレームの Backbone 入力とする。

    入力/出力:
        aggregate_codebooks: K 個のトークンID → 集約埋め込み (B, 1, hidden_size)
        predict_zeroth:      Backbone 隠れ状態 → 第0コードブック logits
    """

    def __init__(
        self,
        hidden_size,
        num_codebooks=4,
        codebook_size=1024,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # 各コードブックの埋め込みテーブル
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_size)
            for _ in range(num_codebooks)
        ])

        # 第0コードブック予測ヘッド (Backbone → zeroth codebook)
        self.zeroth_head = nn.Linear(hidden_size, codebook_size, bias=False)

    def aggregate_codebooks(self, codebook_tokens):
        """
        全コードブックトークンを集約して Backbone 入力を生成

        入力:
            codebook_tokens: (B, num_codebooks) - 各コードブックのトークンID
                codebook_tokens[:, 0] → 第0コードブック
                codebook_tokens[:, 1] → 第1コードブック
                ...

        出力:
            aggregated: (B, 1, hidden_size) - 集約埋め込み

        集約方式: 各コードブック埋め込みの加算
            aggregated = sum(emb_k(codebook_tokens[:, k]) for k in range(K))
        """
        B = codebook_tokens.shape[0]
        aggregated = torch.zeros(
            B, 1, self.codebook_embeddings[0].embedding_dim,
            device=codebook_tokens.device,
            dtype=self.codebook_embeddings[0].weight.dtype,
        )

        for k in range(self.num_codebooks):
            token_ids = codebook_tokens[:, k]  # (B,)
            emb = self.codebook_embeddings[k](token_ids)  # (B, hidden_size)
            aggregated[:, 0, :] += emb  # 加算集約

        return aggregated  # (B, 1, hidden_size)

    def predict_zeroth(self, backbone_hidden):
        """
        Backbone 隠れ状態から次フレームの第0コードブックを予測

        入力:
            backbone_hidden: (B, 1, hidden_size) - Backbone 出力
        出力:
            logits: (B, 1, codebook_size) - 第0コードブックの予測分布
        """
        return self.zeroth_head(backbone_hidden)  # (B, 1, codebook_size)


# ============================================
# MTP Module (Dense Transformer, 80M)
# ============================================

class MTPDecoderLayer(nn.Module):
    """
    MTP Module の単一 Dense Decoder レイヤー

    MoE ではなく通常の Dense FFN を使用 (80M 全体で軽量)。
    固定ステップ自己回帰のため KV キャッシュサイズは固定。

    入力: (B, L, mtp_hidden_size)
    出力: (B, L, mtp_hidden_size)
    """

    def __init__(self, hidden_size=256, num_heads=4, intermediate_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

    def forward(self, x):
        """
        入力: x (B, L, hidden_size)
        出力: x (B, L, hidden_size)

        ※ MTP は固定ステップ (num_codebooks-1 ステップ) の短い系列を処理。
          KV キャッシュは使わず、毎回フルアテンション (系列が短いため)。
        """
        # Self-Attention
        residual = x
        x = self.norm1(x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        B, L, _ = q.shape
        num_heads = self.hidden_size // self.head_dim
        q = q.view(B, L, num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.hidden_size)
        x = residual + self.o_proj(attn_out)

        # SwiGLU FFN (Dense)
        residual = x
        x = self.norm2(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        return x


class MTPModule(nn.Module):
    """
    MTP (Multi-Token Prediction) Module - 80M Dense Transformer

    Talker Backbone が予測した第0コードブックに続いて、
    現在フレームの残余コードブック (第1~K-1) を固定ステップ自己回帰で生成。

    超軽量 (80M) かつ固定ステップ数 → 低メモリ帯域で効率的バッチ推論が可能。

    処理フロー:
        Step 0: backbone_hidden + zeroth_codebook_emb → MTP入力
        Step 1: → codebook_1 予測
        Step 2: + codebook_1_emb → codebook_2 予測
        ...
        Step K-2: → codebook_{K-1} 予測

    入力:
        backbone_hidden: (B, 1, talker_hidden) - Backbone の隠れ状態
        zeroth_token:    (B,)                  - 第0コードブックトークンID
    出力:
        residual_tokens: (B, num_codebooks-1)  - 残余コードブックトークンID

    パラメータ: ~80M (固定)
    """

    def __init__(
        self,
        talker_hidden_size=1024,
        mtp_hidden_size=256,
        num_layers=4,
        num_heads=4,
        mtp_intermediate_size=1024,
        num_codebooks=4,
        codebook_size=1024,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.mtp_hidden_size = mtp_hidden_size

        # Backbone 隠れ状態 → MTP 次元への射影
        self.input_proj = nn.Linear(talker_hidden_size, mtp_hidden_size, bias=False)

        # 各コードブックの埋め込み (MTP 用, mtp_hidden_size 次元)
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, mtp_hidden_size)
            for _ in range(num_codebooks)
        ])

        # Dense Transformer レイヤー
        self.layers = nn.ModuleList([
            MTPDecoderLayer(
                hidden_size=mtp_hidden_size,
                num_heads=num_heads,
                intermediate_size=mtp_intermediate_size,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(mtp_hidden_size)

        # 各残余コードブック用の予測ヘッド
        # codebook_heads[k] は第(k+1)コードブックを予測
        self.codebook_heads = nn.ModuleList([
            nn.Linear(mtp_hidden_size, codebook_size, bias=False)
            for _ in range(num_codebooks - 1)
        ])

    def forward(self, backbone_hidden, zeroth_token):
        """
        残余コードブック (第1~K-1) の固定ステップ自己回帰生成

        入力:
            backbone_hidden: (B, 1, talker_hidden_size) - Backbone 隠れ状態
            zeroth_token:    (B,)                       - 第0コードブックトークンID

        出力:
            residual_logits: list of (B, 1, codebook_size) × (num_codebooks-1)
                各残余コードブックの logits
            residual_tokens: (B, num_codebooks-1)
                各残余コードブックのサンプリング済みトークンID

        処理:
            初期入力 = input_proj(backbone_hidden) + zeroth_emb
            Step k (k=0..K-2):
                Transformer → codebook_heads[k] → codebook_{k+1} 予測
                次ステップ入力に codebook_{k+1} の埋め込みを加算
        """
        B = backbone_hidden.shape[0]

        # Backbone 隠れ状態を MTP 次元に射影
        h = self.input_proj(backbone_hidden)  # (B, 1, mtp_hidden_size)

        # 第0コードブック埋め込みを加算
        zeroth_emb = self.codebook_embeddings[0](zeroth_token)  # (B, mtp_hidden_size)
        h = h + zeroth_emb.unsqueeze(1)  # (B, 1, mtp_hidden_size)

        residual_logits = []
        residual_tokens = []

        # 固定ステップ自己回帰: K-1 ステップ
        for k in range(self.num_codebooks - 1):
            # Dense Transformer 通過
            x = h
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)  # (B, 1, mtp_hidden_size)

            # 第(k+1)コードブック予測
            logits_k = self.codebook_heads[k](x)  # (B, 1, codebook_size)
            residual_logits.append(logits_k)

            # Greedy サンプリング (推論時)
            token_k = logits_k[:, 0, :].argmax(dim=-1)  # (B,)
            residual_tokens.append(token_k)

            # 次ステップ: 予測トークンの埋め込みを入力に加算
            if k < self.num_codebooks - 2:
                next_emb = self.codebook_embeddings[k + 1](token_k)  # (B, mtp_hidden_size)
                h = h + next_emb.unsqueeze(1)

        # 残余トークンを結合
        residual_tokens = torch.stack(residual_tokens, dim=1)  # (B, num_codebooks-1)

        return residual_logits, residual_tokens


# ============================================
# Talker MoE (完全モデル)
# ============================================

class TalkerMoE(nn.Module):
    """
    Qwen3-Omni Talker - MoE Transformer (3B-A0.3B)

    Thinker からマルチモーダル表現を受け取り、RVQ コードブックトークンを
    階層的に自己回帰生成する音声合成モジュール。

    Qwen2.5-Omni の Dense Talker から大幅にアーキテクチャ変更:
        - Dense → MoE (3B総パラメータ, 0.3B活性)
        - 単一コードブック → マルチコードブック (RVQ) + MTP
        - Thinker テキスト隠れ状態の消費 → マルチモーダル特徴量を直接受容
        - ブロックコンテキスト待ち → Left-context-only で即座波形出力

    入力:
        thinker_text_repr:    (B, L_text, thinker_dim)  - テキスト表現 (履歴+現在ターン)
        thinker_multimodal:   (B, L_mm, thinker_dim)    - マルチモーダル表現 (中間層)
        ※ Thinker/Talker は独立システムプロンプトを持てる (分離テキスト表現)

    出力:
        all_codebook_tokens: (B, T_frames, num_codebooks)
            T_frames フレーム × K コードブック のトークン行列
            → RVQ デコーダで波形に変換

    生成ループ (1フレームあたり):
        1. 前フレームの全コードブックトークンを集約 → Backbone 入力
        2. Backbone (MoE Transformer): 次フレームの第0コードブック予測
        3. MTP Module: 現フレームの残余コードブック (第1~K-1) 予測
        4. 全コードブックトークンを RVQ デコーダに渡して波形出力

    話者タイプ:
        - 'Chelsie' (女性)
        - 'Ethan'   (男性)
        - 'Aiden'   (男性)  ← Qwen3-Omni で新規追加
    """

    def __init__(
        self,
        thinker_dim=4096,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=2048,
        num_experts=16,
        top_k=2,
        num_codebooks=4,
        codebook_size=1024,
        mtp_hidden_size=256,
        mtp_num_layers=4,
        mtp_num_heads=4,
        mtp_intermediate_size=1024,
    ):
        """
        パラメータ:
            thinker_dim:          Thinker 出力次元 (4096)
            hidden_size:          Talker Backbone 隠れ次元
            num_layers:           MoE Transformer レイヤー数
            num_heads:            アテンションヘッド数
            num_kv_heads:         KV ヘッド数 (GQA)
            intermediate_size:    エキスパート FFN 中間次元
            num_experts:          MoE エキスパート数
            top_k:                Top-K ルーティング
            num_codebooks:        RVQ コードブック数 (K)
            codebook_size:        各コードブックの語彙サイズ
            mtp_hidden_size:      MTP Module 隠れ次元
            mtp_num_layers:       MTP Transformer レイヤー数
            mtp_num_heads:        MTP アテンションヘッド数
            mtp_intermediate_size: MTP FFN 中間次元
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        # ========================================
        # 特殊トークン (各コードブック共通)
        # ========================================
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        # ========================================
        # Thinker → Talker 射影
        # ========================================
        # テキスト表現射影
        self.text_proj = nn.Linear(thinker_dim, hidden_size, bias=False)
        # マルチモーダル表現射影
        self.multimodal_proj = nn.Linear(thinker_dim, hidden_size, bias=False)

        # ========================================
        # マルチコードブックヘッド (集約 + 第0予測)
        # ========================================
        self.codebook_head = MultiCodebookHead(
            hidden_size=hidden_size,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )

        # ========================================
        # MoE Transformer Backbone
        # ========================================
        self.layers = nn.ModuleList([
            TalkerMoEDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)

        # ========================================
        # MTP Module (残余コードブック生成)
        # ========================================
        self.mtp = MTPModule(
            talker_hidden_size=hidden_size,
            mtp_hidden_size=mtp_hidden_size,
            num_layers=mtp_num_layers,
            num_heads=mtp_num_heads,
            mtp_intermediate_size=mtp_intermediate_size,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )

    def _build_context(self, thinker_text_repr, thinker_multimodal):
        """
        Thinker からのコンテキストを構築

        入力:
            thinker_text_repr:  (B, L_text, thinker_dim) - テキスト表現
            thinker_multimodal: (B, L_mm, thinker_dim)   - マルチモーダル表現

        出力:
            context: (B, L_text + L_mm, hidden_size) - 結合・射影済みコンテキスト

        ※ Qwen2.5-Omni は Thinker の隠れ状態を1トークンずつ消費する
          Dual-Track 方式だったが、Qwen3-Omni では マルチモーダル特徴量を
          直接受け取ることで、音声-映像連携した自然な発話が可能になった。
        """
        text_proj = self.text_proj(thinker_text_repr)        # (B, L_text, hidden_size)
        mm_proj = self.multimodal_proj(thinker_multimodal)    # (B, L_mm, hidden_size)

        # テキスト + マルチモーダル を結合
        context = torch.cat([text_proj, mm_proj], dim=1)  # (B, L_text+L_mm, hidden_size)
        return context

    def backbone_forward(self, x, past_key_values=None):
        """
        MoE Transformer Backbone のフォワードパス

        入力:
            x: (B, L, hidden_size) - 入力 (コンテキスト or コードブック集約)
            past_key_values: KV キャッシュ

        出力:
            hidden: (B, L, hidden_size) - Backbone 出力
            new_past_key_values: 更新された KV キャッシュ
        """
        new_past = []
        hidden = x
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden, kv = layer(hidden, past_kv)
            new_past.append(kv)

        hidden = self.norm(hidden)
        return hidden, new_past

    def forward_one_frame(
        self,
        prev_codebook_tokens,
        context=None,
        past_key_values=None,
    ):
        """
        1フレーム分のフォワードパス (自己回帰の1ステップ)

        入力:
            prev_codebook_tokens: (B, num_codebooks) - 前フレームの全コードブックトークン
                ※ 初回は BOS トークンで埋める
            context: (B, L_ctx, hidden_size) - Thinker コンテキスト (prefill 時のみ)
            past_key_values: KV キャッシュ

        出力:
            frame_tokens: (B, num_codebooks) - 現フレームの全コードブックトークン
                [:, 0]   → 第0コードブック (Backbone が予測した次フレーム用, ここでは現フレーム)
                [:, 1:]  → 残余コードブック (MTP が予測)
            zeroth_logits: (B, 1, codebook_size) - 第0コードブック logits
            residual_logits: list of (B, 1, codebook_size) - 残余コードブック logits
            past_key_values: 更新された KV キャッシュ

        処理フロー:
            1. 前フレーム全コードブックを集約 → (B, 1, hidden_size)
            2. [prefill時] コンテキストと結合
            3. Backbone (MoE): → 隠れ状態
            4. zeroth_head: → 第0コードブック予測
            5. MTP: backbone_hidden + zeroth_emb → 残余コードブック予測
        """
        B = prev_codebook_tokens.shape[0]

        # Step 1: 前フレーム全コードブックを集約
        aggregated = self.codebook_head.aggregate_codebooks(prev_codebook_tokens)
        # aggregated: (B, 1, hidden_size)

        # Step 2: Prefill 時はコンテキストと結合
        if context is not None:
            backbone_input = torch.cat([context, aggregated], dim=1)
            # backbone_input: (B, L_ctx + 1, hidden_size)
        else:
            backbone_input = aggregated
            # backbone_input: (B, 1, hidden_size)

        # Step 3: MoE Backbone
        backbone_hidden, past_key_values = self.backbone_forward(
            backbone_input, past_key_values
        )
        # backbone_hidden: (B, L, hidden_size) - 最後のトークンが現在ステップ

        # 最後のトークンの隠れ状態を使用
        current_hidden = backbone_hidden[:, -1:, :]  # (B, 1, hidden_size)

        # Step 4: 第0コードブック予測
        zeroth_logits = self.codebook_head.predict_zeroth(current_hidden)
        # zeroth_logits: (B, 1, codebook_size)
        zeroth_token = zeroth_logits[:, 0, :].argmax(dim=-1)  # (B,)

        # Step 5: MTP で残余コードブック予測
        residual_logits, residual_tokens = self.mtp(current_hidden, zeroth_token)
        # residual_tokens: (B, num_codebooks - 1)

        # 全コードブックトークンを結合
        frame_tokens = torch.cat([
            zeroth_token.unsqueeze(1),  # (B, 1)
            residual_tokens,            # (B, num_codebooks - 1)
        ], dim=1)
        # frame_tokens: (B, num_codebooks)

        return frame_tokens, zeroth_logits, residual_logits, past_key_values

    def generate(
        self,
        thinker_text_repr,
        thinker_multimodal,
        speaker="Chelsie",
        max_frames=200,
    ):
        """
        音声フレーム系列の自己回帰生成

        入力:
            thinker_text_repr:  (B, L_text, thinker_dim) - テキスト表現
            thinker_multimodal: (B, L_mm, thinker_dim)   - マルチモーダル表現
            speaker: str - 話者名 ('Chelsie', 'Ethan', 'Aiden')
            max_frames: int - 最大生成フレーム数

        出力:
            all_tokens: (B, T_frames, num_codebooks) - 生成された全コードブックトークン
                → RVQ デコーダで波形に変換

        生成パラメータ:
            話者: 'Chelsie' (女性), 'Ethan' (男性), 'Aiden' (男性)
            ※ Left-context-only 生成: 最初のフレームから即座に波形出力可能
              (Qwen2.5-Omni のブロックコンテキスト待ちが不要)
        """
        B = thinker_text_repr.shape[0]
        device = thinker_text_repr.device

        # コンテキスト構築
        context = self._build_context(thinker_text_repr, thinker_multimodal)
        # context: (B, L_text + L_mm, hidden_size)

        # BOS トークンで初期化
        prev_tokens = torch.full(
            (B, self.num_codebooks), self.bos_token_id,
            device=device, dtype=torch.long,
        )
        # prev_tokens: (B, num_codebooks) - 全コードブック BOS

        all_frame_tokens = []
        past_key_values = None

        for t in range(max_frames):
            frame_tokens, _, _, past_key_values = self.forward_one_frame(
                prev_codebook_tokens=prev_tokens,
                context=context if t == 0 else None,  # prefill は初回のみ
                past_key_values=past_key_values,
            )
            # frame_tokens: (B, num_codebooks)

            all_frame_tokens.append(frame_tokens)

            # EOS チェック (第0コードブックが EOS なら停止)
            if (frame_tokens[:, 0] == self.eos_token_id).all():
                break

            prev_tokens = frame_tokens

        # 全フレームを結合
        all_tokens = torch.stack(all_frame_tokens, dim=1)
        # all_tokens: (B, T_frames, num_codebooks)

        return all_tokens


# ============================================
# 使用例
# ============================================

def example_talker_moe():
    """
    Qwen3-Omni Talker (MoE) + MTP Module の使用例

    各モジュールを縮小サイズでインスタンス化し、
    フォワードパスを実際に実行して形状・動作を確認する。
    """

    # =========================================
    # ハイパーパラメータ (縮小版)
    # =========================================
    # 実モデル: 3B-A0.3B MoE, MTP 80M
    thinker_dim = 256       # 実モデル: 4096 (Thinker 出力次元)
    hidden_size = 128       # 実モデル: ~1024+ (Talker Backbone)
    num_layers = 2          # 実モデル: ~24
    num_heads = 4
    num_kv_heads = 2        # GQA
    intermediate_size = 256
    num_experts = 4         # 実モデル: 非公開 (数十~百以上)
    top_k = 2
    num_codebooks = 4       # RVQ コードブック数
    codebook_size = 64      # 実モデル: ~1024
    mtp_hidden_size = 64    # 実モデル: ~256
    mtp_num_layers = 2      # 実モデル: ~4
    mtp_num_heads = 2
    mtp_intermediate_size = 128

    B = 1
    L_text = 10    # テキストトークン数
    L_mm = 5       # マルチモーダルトークン数

    # =========================================
    # 1. MultiCodebookHead の検証
    # =========================================
    print("=" * 60)
    print("[1] MultiCodebookHead の検証")
    print("=" * 60)

    codebook_head = MultiCodebookHead(
        hidden_size=hidden_size,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )

    # ダミーコードブックトークン
    dummy_tokens = torch.randint(0, codebook_size, (B, num_codebooks))
    # dummy_tokens: (1, 4) - 4コードブック分のトークンID

    aggregated = codebook_head.aggregate_codebooks(dummy_tokens)
    assert aggregated.shape == (B, 1, hidden_size), \
        f"集約形状エラー: 期待 ({B}, 1, {hidden_size}), 実際 {aggregated.shape}"
    print(f"  コードブックトークン入力: {dummy_tokens.shape}  (B, num_codebooks)")
    print(f"  集約埋め込み出力:         {aggregated.shape}  (B, 1, hidden_size)")

    # 第0コードブック予測
    dummy_hidden = torch.randn(B, 1, hidden_size)
    zeroth_logits = codebook_head.predict_zeroth(dummy_hidden)
    assert zeroth_logits.shape == (B, 1, codebook_size), \
        f"第0予測形状エラー: 期待 ({B}, 1, {codebook_size}), 実際 {zeroth_logits.shape}"
    print(f"  Backbone隠れ状態入力:     {dummy_hidden.shape}  (B, 1, hidden_size)")
    print(f"  第0コードブック logits:   {zeroth_logits.shape}  (B, 1, codebook_size)")
    print()

    # =========================================
    # 2. MTP Module の検証
    # =========================================
    print("=" * 60)
    print("[2] MTP Module の検証")
    print("=" * 60)

    mtp = MTPModule(
        talker_hidden_size=hidden_size,
        mtp_hidden_size=mtp_hidden_size,
        num_layers=mtp_num_layers,
        num_heads=mtp_num_heads,
        mtp_intermediate_size=mtp_intermediate_size,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )
    mtp.eval()

    backbone_hidden = torch.randn(B, 1, hidden_size)
    zeroth_token = torch.randint(0, codebook_size, (B,))

    with torch.no_grad():
        residual_logits, residual_tokens = mtp(backbone_hidden, zeroth_token)

    assert len(residual_logits) == num_codebooks - 1, \
        f"残余logits数エラー: 期待 {num_codebooks - 1}, 実際 {len(residual_logits)}"
    for k, rl in enumerate(residual_logits):
        assert rl.shape == (B, 1, codebook_size), \
            f"残余logits[{k}]形状エラー: 期待 ({B}, 1, {codebook_size}), 実際 {rl.shape}"
    assert residual_tokens.shape == (B, num_codebooks - 1), \
        f"残余トークン形状エラー: 期待 ({B}, {num_codebooks - 1}), 実際 {residual_tokens.shape}"

    print(f"  Backbone隠れ状態:    {backbone_hidden.shape}  (B, 1, talker_hidden)")
    print(f"  第0トークン入力:     {zeroth_token.shape}     (B,)")
    print(f"  残余コードブック数:  {num_codebooks - 1}")
    for k, rl in enumerate(residual_logits):
        print(f"    codebook_{k+1} logits: {rl.shape}  (B, 1, codebook_size)")
    print(f"  残余トークン出力:    {residual_tokens.shape}  (B, num_codebooks-1)")
    print(f"  残余トークン値:      {residual_tokens[0].tolist()}")
    print()

    # =========================================
    # 3. MoE Layer の検証
    # =========================================
    print("=" * 60)
    print("[3] MoE Layer の検証")
    print("=" * 60)

    moe_layer = MoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    moe_layer.eval()

    dummy_input = torch.randn(B, 3, hidden_size)
    with torch.no_grad():
        moe_output = moe_layer(dummy_input)
    assert moe_output.shape == dummy_input.shape, \
        f"MoE出力形状エラー: 期待 {dummy_input.shape}, 実際 {moe_output.shape}"
    print(f"  MoE入力:   {dummy_input.shape}  (B, L, hidden_size)")
    print(f"  MoE出力:   {moe_output.shape}  (B, L, hidden_size)")
    print(f"  エキスパート数: {num_experts}, Top-K: {top_k}")
    print(f"  活性パラメータ比: ~1/{num_experts // top_k} (top_k/num_experts)")
    print()

    # =========================================
    # 4. TalkerMoE 完全モデルの検証
    # =========================================
    print("=" * 60)
    print("[4] TalkerMoE 完全モデルの検証")
    print("=" * 60)

    talker = TalkerMoE(
        thinker_dim=thinker_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        mtp_hidden_size=mtp_hidden_size,
        mtp_num_layers=mtp_num_layers,
        mtp_num_heads=mtp_num_heads,
        mtp_intermediate_size=mtp_intermediate_size,
    )
    talker.eval()

    # パラメータ数の表示
    total_params = sum(p.numel() for p in talker.parameters())
    backbone_params = sum(p.numel() for p in talker.layers.parameters())
    mtp_params = sum(p.numel() for p in talker.mtp.parameters())
    print(f"  [パラメータ数 (縮小版)]")
    print(f"    全体:     {total_params:,}")
    print(f"    Backbone: {backbone_params:,}")
    print(f"    MTP:      {mtp_params:,}")
    print()

    # --- Thinker 出力のダミーデータ ---
    thinker_text_repr = torch.randn(B, L_text, thinker_dim)
    thinker_multimodal = torch.randn(B, L_mm, thinker_dim)

    # --- コンテキスト構築 ---
    context = talker._build_context(thinker_text_repr, thinker_multimodal)
    assert context.shape == (B, L_text + L_mm, hidden_size), \
        f"コンテキスト形状エラー: 期待 ({B}, {L_text + L_mm}, {hidden_size}), 実際 {context.shape}"
    print(f"  [コンテキスト構築]")
    print(f"    テキスト表現入力:       {thinker_text_repr.shape}  (B, L_text, thinker_dim)")
    print(f"    マルチモーダル表現入力: {thinker_multimodal.shape}  (B, L_mm, thinker_dim)")
    print(f"    結合コンテキスト:       {context.shape}  (B, L_text+L_mm, hidden_size)")
    print()

    # --- 1フレーム生成 ---
    prev_tokens = torch.full((B, num_codebooks), talker.bos_token_id, dtype=torch.long)
    with torch.no_grad():
        frame_tokens, zeroth_logits, residual_logits, past_kv = talker.forward_one_frame(
            prev_codebook_tokens=prev_tokens,
            context=context,
            past_key_values=None,
        )

    assert frame_tokens.shape == (B, num_codebooks), \
        f"フレームトークン形状エラー: 期待 ({B}, {num_codebooks}), 実際 {frame_tokens.shape}"
    assert zeroth_logits.shape == (B, 1, codebook_size), \
        f"第0logits形状エラー: 期待 ({B}, 1, {codebook_size}), 実際 {zeroth_logits.shape}"
    assert len(residual_logits) == num_codebooks - 1
    assert past_kv is not None and len(past_kv) == num_layers

    print(f"  [1フレーム生成 (prefill)]")
    print(f"    前フレームトークン:   {prev_tokens.shape}    (B, num_codebooks) = BOS")
    print(f"    生成フレームトークン: {frame_tokens.shape}   (B, num_codebooks)")
    print(f"    第0 codebook logits: {zeroth_logits.shape}  (B, 1, codebook_size)")
    print(f"    残余 codebook logits: {len(residual_logits)} 個, 各 {residual_logits[0].shape}")
    print(f"    KVキャッシュ層数:     {len(past_kv)}")
    print(f"    生成トークン値:       {frame_tokens[0].tolist()}")
    print()

    # --- 2フレーム目 (decode ステップ) ---
    with torch.no_grad():
        frame_tokens_2, _, _, past_kv_2 = talker.forward_one_frame(
            prev_codebook_tokens=frame_tokens,
            context=None,  # prefill 済みなので context は不要
            past_key_values=past_kv,
        )
    assert frame_tokens_2.shape == (B, num_codebooks)
    print(f"  [2フレーム目 (decode)]")
    print(f"    入力トークン: {frame_tokens[0].tolist()}")
    print(f"    出力トークン: {frame_tokens_2[0].tolist()}")
    print()

    # --- 自己回帰生成 ---
    max_frames = 8
    with torch.no_grad():
        all_tokens = talker.generate(
            thinker_text_repr=thinker_text_repr,
            thinker_multimodal=thinker_multimodal,
            speaker="Chelsie",
            max_frames=max_frames,
        )
    # all_tokens: (B, T_frames, num_codebooks)
    assert all_tokens.shape[0] == B
    assert all_tokens.shape[1] <= max_frames
    assert all_tokens.shape[2] == num_codebooks

    print(f"  [自己回帰生成]")
    print(f"    最大フレーム数: {max_frames}")
    print(f"    生成結果:       {all_tokens.shape}  (B, T_frames, num_codebooks)")
    print(f"    生成フレーム数: {all_tokens.shape[1]}")
    print()
    print(f"    フレームごとのトークン:")
    for t in range(all_tokens.shape[1]):
        tokens = all_tokens[0, t].tolist()
        label = "  (BOS後初回)" if t == 0 else ""
        print(f"      frame {t}: {tokens}{label}")
    print()

    # =========================================
    # 5. Qwen2.5-Omni との比較まとめ
    # =========================================
    print("=" * 60)
    print("[5] Qwen2.5-Omni Talker との比較")
    print("=" * 60)
    print()
    print(f"  {'項目':<24} {'Qwen2.5-Omni':<28} {'Qwen3-Omni'}")
    print(f"  {'-'*24} {'-'*28} {'-'*28}")
    print(f"  {'アーキテクチャ':<20} {'Dense Transformer':<28} {'MoE Transformer (3B-A0.3B)'}")
    print(f"  {'コードブック':<22} {'単一 (8295語彙)':<28} {'マルチ RVQ (K個)'}")
    print(f"  {'残余コードブック':<20} {'なし':<28} {'MTP Module (80M Dense)'}")
    print(f"  {'入力ソース':<22} {'Thinkerテキスト隠れ状態':<24} {'マルチモーダル特徴量直接'}")
    print(f"  {'融合方式':<24} {'Dual-Track加算融合':<28} {'階層的コードブック予測'}")
    print(f"  {'波形出力遅延':<22} {'ブロックコンテキスト待ち':<24} {'最初のトークンから即座'}")
    print(f"  {'システムプロンプト':<20} {'Thinkerと共有':<28} {'Thinker/Talker独立'}")
    print(f"  {'話者':<24} {'Chelsie, Ethan':<28} {'Chelsie, Ethan, Aiden'}")
    print()
    print(f"  [Qwen3-Omni の主要改善点]")
    print(f"    1. MoE化: 3B総パラメータだが0.3Bのみ活性 → 推論効率が高い")
    print(f"    2. マルチコードブック: RVQ + MTP で高品質音声再構成")
    print(f"    3. マルチモーダル入力: 音声/映像特徴量を直接受容 → 連携発話")
    print(f"    4. Left-context-only: 最初のトークンから即座に波形出力可能")
    print(f"    5. 独立プロンプト: Thinker/Talker で異なるシステムプロンプト")


if __name__ == "__main__":
    example_talker_moe()
