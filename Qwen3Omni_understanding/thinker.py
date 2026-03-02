"""
Qwen3-Omni Thinker - MoE Transformer LLM 疑似コード
=====================================================

マルチモーダル統合 LLM (Thinker = "脳")
テキスト/音声/画像/動画の特徴を統合してテキストを生成する。

Qwen3 (Yang et al., 2025a) をベースに初期化された
MoE (Mixture-of-Experts) Transformer デコーダ。

アーキテクチャ概要:
    - 30B-A3B: 総パラメータ 30B、アクティブパラメータ 3B
    - MoE 構造: 共有エキスパート + ルーテッドエキスパート (top-k ルーティング)
    - TM-RoPE (Time-aligned Multimodal RoPE) による統一的な位置エンコーディング
    - vocab_size: 151,643
    - コンテキスト長: 8192 → 32768 (S3 長コンテキスト学習後)

TM-RoPE (vs Qwen2.5-Omni の M-RoPE):
    - 3次元: temporal, height, width
    - 回転角度配分: temporal=24, height=20, width=20 (インターリーブ)
    - Qwen2.5-Omni: temporal=16, height=24, width=24
    - 音声: 共有位置ID + 絶対時間エンコーディング (80ms/ID)
    - 画像: temporal 固定 + height/width 空間座標
    - 動画: temporal 単調増加 (80ms粒度) + height/width
    - 固定2秒チャンクを廃止 (Qwen2.5-Omni からの変更)

Talker との接続:
    - Thinker の中間層から隠れ状態を抽出して Talker に渡す
      (最終層だけでなく中間層)
    - マルチモーダル特徴も直接 Talker に渡す
      (テキスト表現だけでなくマルチモーダル特徴)

主な差分 (vs Qwen2.5-Omni Thinker):
    - Dense → MoE: 7B dense → 30B-A3B MoE
    - M-RoPE → TM-RoPE: 回転角度配分の変更、絶対時間エンコーディング
    - mrope_section: [16, 24, 24] → [24, 20, 20] (temporal重視に変更)
    - 固定2秒チャンク → 連続的な時間エンコーディング
    - 最終層のみ → 中間層からも Talker へ隠れ状態を転送
    - KVキャッシュメモリ: MoE によりアクティブパラメータが少なく、
      長系列での KV キャッシュメモリ効率が向上
    - 高並行性: MoE により推論時の並行処理性能が向上
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# TM-RoPE (Time-aligned Multimodal RoPE)
# ============================================

class TMRoPE(nn.Module):
    """
    TM-RoPE: Time-aligned Multimodal Rotary Position Embedding

    Qwen2.5-Omni の M-RoPE を拡張し、絶対時間エンコーディングを追加。
    RoPE を3軸に分解して、異なるモダリティの位置情報を統一的にエンコードする。

    3軸: (temporal, height, width)
        - テキスト: 3軸同一 → 標準1D-RoPEと等価
        - 音声: 共有位置ID + 絶対時間ID (80ms/ID)
          ※ Qwen2.5-Omni は 40ms/ID だったが、AuT の 12.5Hz に合わせて 80ms/ID に変更
        - 画像: temporal 固定 + height/width 空間座標
        - 動画: temporal 単調増加 (80ms粒度) + height/width 空間座標
          ※ 固定2秒チャンクを廃止 (Qwen2.5-Omni からの変更)

    回転角度配分 (インターリーブ):
        temporal: 24 rotary angles
        height:   20 rotary angles
        width:    20 rotary angles
        合計: 64 angles → head_dim = 128

    vs Qwen2.5-Omni:
        temporal: 16, height: 24, width: 24
        → temporal の割当を増加 (16→24)、height/width を削減 (24→20)
        → 時間軸の表現力を強化し、音声・動画の時間整合性を向上
    """

    def __init__(self, dim: int, max_position: int = 32768, base: float = 10000.0):
        """
        パラメータ:
            dim: 回転埋め込みの次元 (= head_dim, 通常128)
            max_position: 最大位置数 (32768: S3 長コンテキスト学習後)
            base: 基底周波数
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力:
            x: (B, L, D) - 入力テンソル (shapeのみ使用)
            position_ids: (3, B, L) - 3軸の位置ID
                [0]: temporal - 時間位置 (絶対時間エンコーディング)
                [1]: height - 高さ位置
                [2]: width - 幅位置

        出力:
            cos: (3, B, L, head_dim) - コサイン成分
            sin: (3, B, L, head_dim) - サイン成分

        ★ M-RoPE (Qwen2.5-Omni) との違い:
            - 位置IDの計算方法:
              M-RoPE: 音声は40ms/ID、動画は固定2秒チャンク
              TM-RoPE: 音声は80ms/ID (AuT対応)、動画は連続80ms粒度
            - 角度配分:
              M-RoPE: [16, 24, 24] → temporal の情報量が少ない
              TM-RoPE: [24, 20, 20] → temporal の情報量を増加
        """
        # inv_freq を3軸分に拡張
        inv_freq_expanded = self.inv_freq[None, None, :, None].float()
        # inv_freq_expanded: (1, 1, dim//2, 1)

        inv_freq_expanded = inv_freq_expanded.expand(3, position_ids.shape[1], -1, 1)
        # inv_freq_expanded: (3, B, dim//2, 1)

        position_ids_expanded = position_ids[:, :, None, :].float()
        # position_ids_expanded: (3, B, 1, L)

        # 3軸それぞれで周波数を計算
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        # freqs: (3, B, L, dim//2)

        # cos/sin用に2倍に拡張
        emb = torch.cat([freqs, freqs], dim=-1)
        # emb: (3, B, L, dim)

        cos = emb.cos()
        sin = emb.sin()
        # cos, sin: (3, B, L, dim)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    テンソルの後半をネガティブにして前半と入れ替え (RoPEの回転操作)

    入力/出力: (..., dim)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_tmrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TM-RoPE を Query と Key に適用

    ★ コアの仕組み:
        head_dim をセクションに分割し、各セクションに異なる軸の位置を適用
        mrope_section = [m1, m2, m3] where m1+m2+m3 = head_dim//2

        Qwen3-Omni: mrope_section = [24, 20, 20] (head_dim=128)
        Qwen2.5-Omni: mrope_section = [16, 24, 24] (head_dim=128)

        チャンネル [0:m1*2]         → temporal 軸の cos/sin を適用
        チャンネル [m1*2:(m1+m2)*2] → height 軸の cos/sin を適用
        チャンネル [(m1+m2)*2:]     → width 軸の cos/sin を適用

    temporal への割当増加 (16→24) により:
        - 音声の時間整合性が向上 (80msの絶対時間を高精度に表現)
        - 動画フレームの時間的順序をより正確に区別
        - height/width は画像・動画の空間解像度に比べ20でも十分

    入力:
        q: (B, num_heads, L, head_dim) - Query
        k: (B, num_heads, L, head_dim) - Key
        cos: (3, B, L, head_dim) - 3軸のコサイン
        sin: (3, B, L, head_dim) - 3軸のサイン
        mrope_section: [m1, m2, m3] - 各軸に割り当てる回転角度数
            Qwen3-Omni デフォルト: [24, 20, 20]

    出力:
        q_rotated: (B, num_heads, L, head_dim)
        k_rotated: (B, num_heads, L, head_dim)
    """
    # セクションを2倍 (cos/sinペア)
    mrope_section_doubled = [s * 2 for s in mrope_section]
    # 例: [24, 20, 20] → [48, 40, 40]

    # cos/sin を3軸のセクションに分割
    cos_sections = torch.split(cos, mrope_section_doubled, dim=-1)
    sin_sections = torch.split(sin, mrope_section_doubled, dim=-1)
    # cos_sections: List of (3, B, L, section_dim) x 3

    # 各セクションから対応する軸を選択
    cos_combined = torch.cat([
        cos_sections[0][0:1, ...],   # temporal軸 → セクション0 (48 チャンネル)
        cos_sections[1][1:2, ...],   # height軸 → セクション1 (40 チャンネル)
        cos_sections[2][2:3, ...],   # width軸 → セクション2 (40 チャンネル)
    ], dim=-1).squeeze(0)
    # cos_combined: (B, L, head_dim)

    sin_combined = torch.cat([
        sin_sections[0][0:1, ...],
        sin_sections[1][1:2, ...],
        sin_sections[2][2:3, ...],
    ], dim=-1).squeeze(0)
    # sin_combined: (B, L, head_dim)

    # unsqueeze for num_heads dimension
    cos_combined = cos_combined.unsqueeze(1)
    sin_combined = sin_combined.unsqueeze(1)
    # cos_combined, sin_combined: (B, 1, L, head_dim)

    # 標準的なRoPE適用
    q_rotated = (q * cos_combined) + (rotate_half(q) * sin_combined)
    k_rotated = (k * cos_combined) + (rotate_half(k) * sin_combined)
    # q_rotated, k_rotated: (B, num_heads, L, head_dim)

    return q_rotated, k_rotated


# ============================================
# MoE (Mixture-of-Experts) コンポーネント
# ============================================

class ExpertMLP(nn.Module):
    """
    MoE の単一エキスパート (SwiGLU MLP)

    各エキスパートは独立した SwiGLU FFN を持つ。
    Qwen3 MoE ベースのアーキテクチャに準拠。

    入力: (*, hidden_size)
    出力: (*, hidden_size)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        """
        パラメータ:
            hidden_size: 入出力の隠れ次元
            intermediate_size: FFN中間次元 (各エキスパートの中間次元)
        """
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU: down(silu(gate(x)) * up(x))

        入力: (*, hidden_size)
        出力: (*, hidden_size)
        """
        gate = F.silu(self.gate_proj(x))
        # gate: (*, intermediate_size)
        up = self.up_proj(x)
        # up: (*, intermediate_size)
        return self.down_proj(gate * up)
        # 出力: (*, hidden_size)


class MoEGate(nn.Module):
    """
    MoE ゲートネットワーク (ルーター)

    入力トークンをどのエキスパートに割り当てるかを決定する。
    Top-k ルーティングにより、各トークンは k 個のエキスパートで処理される。

    入力: (B*L, hidden_size)
    出力:
        topk_weights: (B*L, top_k) - 選択されたエキスパートの重み (正規化済み)
        topk_indices: (B*L, top_k) - 選択されたエキスパートのインデックス
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        """
        パラメータ:
            hidden_size: 入力の隠れ次元
            num_experts: エキスパート総数 (例: 128)
            top_k: 各トークンに割り当てるエキスパート数 (例: 8)
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力: (N, hidden_size) where N = B*L (フラットなトークン列)
        出力:
            topk_weights: (N, top_k) - softmax 正規化された重み
            topk_indices: (N, top_k) - 選択されたエキスパートID

        ルーティング処理:
            1. 全エキスパートへのスコアを計算: gate(x) → (N, num_experts)
            2. Top-k 選択: 上位 k 個のエキスパートを選択
            3. 選択されたエキスパートの重みを softmax で正規化
        """
        # ゲートスコア計算
        logits = self.gate(x)
        # logits: (N, num_experts)

        # Top-k 選択
        topk_weights, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        # topk_weights: (N, top_k) - 上位 k 個のスコア
        # topk_indices: (N, top_k) - 上位 k 個のインデックス

        # Softmax 正規化 (選択されたエキスパート間で)
        topk_weights = F.softmax(topk_weights, dim=-1)
        # topk_weights: (N, top_k) - 正規化された重み (合計=1)

        return topk_weights, topk_indices


class MoELayer(nn.Module):
    """
    MoE (Mixture-of-Experts) レイヤー

    共有エキスパート + ルーテッドエキスパートで構成。
    Qwen3 MoE アーキテクチャに準拠。

    構造:
        - 共有エキスパート (shared_expert): 全トークンに常に適用される FFN
        - ルーテッドエキスパート (experts): ゲートにより選択的に適用される FFN 群
        - ゲートネットワーク (gate): Top-k ルーティングを実行

    Qwen2.5-Omni (Dense) との違い:
        - Dense: 単一の SwiGLU MLP (全トークンに同じ FFN)
        - MoE: 共有エキスパート + Top-k 選択のルーテッドエキスパート
        - メリット:
          * 総パラメータ数は大きいが、推論時のアクティブパラメータは少ない
          * KV キャッシュメモリ効率が向上 (attention 部分は同じサイズ)
          * 高並行性: 異なるエキスパートを並列処理可能

    入力: (B, L, hidden_size)
    出力: (B, L, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        shared_intermediate_size: int,
        num_experts: int = 128,
        top_k: int = 8,
    ):
        """
        パラメータ:
            hidden_size: 隠れ次元
            intermediate_size: 各ルーテッドエキスパートの中間次元
            shared_intermediate_size: 共有エキスパートの中間次元
            num_experts: ルーテッドエキスパートの総数 (例: 128)
            top_k: 各トークンに割り当てるエキスパート数 (例: 8)

        ★ パラメータ数の内訳 (30B-A3B の場合):
            - 共有エキスパート: hidden_size × shared_intermediate_size × 3 (gate/up/down)
            - ルーテッドエキスパート: num_experts × hidden_size × intermediate_size × 3
            - アクティブ: 共有 + top_k 個のルーテッド
              → 総パラメータ ~30B のうちアクティブ ~3B
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 共有エキスパート: 全トークンに常に適用
        self.shared_expert = ExpertMLP(hidden_size, shared_intermediate_size)

        # ルーテッドエキスパート: ゲートにより選択的に適用
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

        # ゲートネットワーク
        self.gate = MoEGate(hidden_size, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: (B, L, hidden_size)
        出力: (B, L, hidden_size)

        処理フロー:
            1. 共有エキスパートの出力を計算 (全トークン)
            2. ゲートで Top-k エキスパートを選択
            3. 選択されたエキスパートの出力を重み付き加算
            4. 共有エキスパート出力 + ルーテッドエキスパート出力

        ★ 実装上の注意:
            実際の実装では Expert Parallelism を使用し、
            各エキスパートを異なる GPU に配置して並列処理する。
            ここでは簡略化のため逐次処理。
        """
        B, L, D = x.shape

        # --- Step 1: 共有エキスパート ---
        shared_output = self.shared_expert(x)
        # shared_output: (B, L, hidden_size)

        # --- Step 2: ゲートによるルーティング ---
        x_flat = x.reshape(B * L, D)
        # x_flat: (N, hidden_size) where N = B*L

        topk_weights, topk_indices = self.gate(x_flat)
        # topk_weights: (N, top_k) - 各トークンの top_k エキスパート重み
        # topk_indices: (N, top_k) - 各トークンの top_k エキスパートID

        # --- Step 3: ルーテッドエキスパートの出力計算 ---
        routed_output = torch.zeros_like(x_flat)
        # routed_output: (N, hidden_size)

        for i in range(self.top_k):
            expert_indices = topk_indices[:, i]   # (N,) - i番目のエキスパートID
            expert_weights = topk_weights[:, i]   # (N,) - i番目の重み

            for expert_id in range(self.num_experts):
                # このエキスパートに割り当てられたトークンを取得
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    # expert_input: (num_assigned, hidden_size)

                    expert_output = self.experts[expert_id](expert_input)
                    # expert_output: (num_assigned, hidden_size)

                    # 重み付き加算
                    routed_output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output

        routed_output = routed_output.reshape(B, L, D)
        # routed_output: (B, L, hidden_size)

        # --- Step 4: 共有 + ルーテッド ---
        output = shared_output + routed_output
        # output: (B, L, hidden_size)

        return output


# ============================================
# Thinker Decoder Layer (MoE)
# ============================================

class ThinkerMoEDecoderLayer(nn.Module):
    """
    Thinker MoE LLM の単一 Transformer Decoder レイヤー

    構成: RMSNorm → Self-Attention (TM-RoPE, GQA) → RMSNorm → MoE

    vs Qwen2.5-Omni の ThinkerDecoderLayer:
        - FFN: SwiGLU MLP (Dense) → MoE (共有 + ルーテッドエキスパート)
        - RoPE: M-RoPE → TM-RoPE (角度配分変更)
        - 他は同構造: RMSNorm, GQA, KVキャッシュ

    アテンション部分のパラメータは Dense と同等サイズ
    → KVキャッシュのメモリは同じだが、FFN部分は MoE により
      アクティブパラメータが少ないため、全体として効率的
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        intermediate_size: int = 1024,
        shared_intermediate_size: int = 2048,
        num_experts: int = 128,
        top_k: int = 8,
    ):
        """
        パラメータ:
            hidden_size: 隠れ次元
            num_heads: Query ヘッド数
            num_kv_heads: Key/Value ヘッド数 (GQA)
            intermediate_size: 各ルーテッドエキスパートの中間次元
            shared_intermediate_size: 共有エキスパートの中間次元
            num_experts: ルーテッドエキスパート数
            top_k: 各トークンに割り当てるエキスパート数
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads

        # Self-Attention (GQA)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # ★ MoE FFN (Qwen2.5-Omni の SwiGLU MLP を置換)
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            shared_intermediate_size=shared_intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

        # RMSNorm
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mrope_section: List[int],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        入力:
            hidden_states: (B, L, hidden_size) - 入力隠れ状態
            cos: (3, B, L, head_dim) - TM-RoPE コサイン
            sin: (3, B, L, head_dim) - TM-RoPE サイン
            mrope_section: [m1, m2, m3] - 各軸のチャンネル割当
                Qwen3-Omni: [24, 20, 20]
            attention_mask: (B, 1, L, L) - 因果マスク (optional)
            past_key_value: KVキャッシュ (optional)

        出力:
            hidden_states: (B, L, hidden_size) - 出力隠れ状態
            past_key_value: 更新されたKVキャッシュ
        """
        B, L, D = hidden_states.shape
        residual = hidden_states

        # ========================================
        # 1. RMSNorm + Self-Attention (GQA + TM-RoPE)
        # ========================================
        hidden_states = self.input_layernorm(hidden_states)
        # hidden_states: (B, L, hidden_size)

        # Q, K, V の計算
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # q: (B, L, num_heads * head_dim)
        # k, v: (B, L, num_kv_heads * head_dim)

        # ヘッド分割
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, L, head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # k, v: (B, num_kv_heads, L, head_dim)

        # ★ TM-RoPE 適用 (M-RoPE から変更: mrope_section が異なる)
        q, k = apply_tmrope(q, k, cos, sin, mrope_section)
        # q: (B, num_heads, L, head_dim) - 位置情報が埋め込まれた Query
        # k: (B, num_kv_heads, L, head_dim) - 位置情報が埋め込まれた Key

        # GQA: K/V をグループ数分繰り返す
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            # k, v: (B, num_heads, L, head_dim)

        # KV キャッシュの結合 (推論時)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_past_key_value = (k, v)

        # Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # attn_weights: (B, num_heads, L, L_kv)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (B, num_heads, L, head_dim)

        # ヘッド結合
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        # attn_output: (B, L, hidden_size)

        attn_output = self.o_proj(attn_output)
        # attn_output: (B, L, hidden_size)

        hidden_states = residual + attn_output
        # hidden_states: (B, L, hidden_size)

        # ========================================
        # 2. RMSNorm + MoE FFN
        # ========================================
        # ★ Dense MLP → MoE に変更 (Qwen2.5-Omni からの主要変更点)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states: (B, L, hidden_size)

        hidden_states = self.moe(hidden_states)
        # hidden_states: (B, L, hidden_size)
        # MoE 内部:
        #   共有エキスパート: 全トークンに適用
        #   ルーテッドエキスパート: top_k 個のみアクティブ
        #   → アクティブパラメータ ≪ 総パラメータ

        hidden_states = residual + hidden_states
        # hidden_states: (B, L, hidden_size)

        return hidden_states, new_past_key_value


# ============================================
# Thinker MoE Transformer (完全モデル)
# ============================================

class ThinkerMoETransformer(nn.Module):
    """
    Qwen3-Omni Thinker: MoE Transformer LLM

    マルチモーダル入力を統合してテキストを生成する LLM。
    Qwen3 (Yang et al., 2025a) をベースに初期化。

    構成:
        - Audio Tower: AuT (Audio Transformer, ~650M)
        - Visual: ViT ベース Vision Encoder
        - Text Model: MoE Transformer Decoder (30B-A3B)
        - LM Head: テキスト生成用の線形層
        - TM-RoPE: 時間整合マルチモーダル回転位置エンコーディング

    vs Qwen2.5-Omni ThinkerForConditionalGeneration:
        - 7B Dense → 30B-A3B MoE
        - M-RoPE [16,24,24] → TM-RoPE [24,20,20]
        - Whisper → AuT (audio_encoder.py 参照)
        - 最終層のみ → 中間層からも Talker に隠れ状態を転送
        - 固定2秒チャンク → 連続的な時間エンコーディング

    Talker との接続:
        - middle_layer_index で指定した中間層の隠れ状態を Talker に渡す
        - Talker は最終層の隠れ状態ではなく、中間層の表現を受け取る
          → より低レベルな特徴が音声生成に有用
        - マルチモーダル特徴も直接 Talker に渡す
          (テキスト表現だけでなくエンコーダ出力も)

    パラメータ数:
        - 総パラメータ: ~30B
        - アクティブパラメータ: ~3B (推論時に実際に計算されるパラメータ)
        - KV キャッシュ: アテンション部分のみ → Dense と同等サイズ
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_layers: int = 28,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        intermediate_size: int = 1024,
        shared_intermediate_size: int = 2048,
        num_experts: int = 128,
        top_k: int = 8,
        vocab_size: int = 151643,
        mrope_section: Optional[List[int]] = None,
        middle_layer_index: Optional[int] = None,
    ):
        """
        パラメータ:
            hidden_size: 隠れ次元
            num_layers: Transformer デコーダ層数
            num_heads: Query ヘッド数
            num_kv_heads: Key/Value ヘッド数 (GQA)
            intermediate_size: 各ルーテッドエキスパートの中間次元
            shared_intermediate_size: 共有エキスパートの中間次元
            num_experts: ルーテッドエキスパート数
            top_k: 各トークンに割り当てるエキスパート数
            vocab_size: 語彙サイズ (151,643)
            mrope_section: TM-RoPE の各軸チャンネル割当
                デフォルト: [24, 20, 20] (Qwen3-Omni)
                Qwen2.5-Omni: [16, 24, 24]
            middle_layer_index: Talker に渡す中間層のインデックス
                None の場合: num_layers // 2 (中央の層)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # TM-RoPE セクション: temporal=24, height=20, width=20
        if mrope_section is None:
            mrope_section = [24, 20, 20]
        self.mrope_section = mrope_section

        # Talker に渡す中間層のインデックス
        if middle_layer_index is None:
            middle_layer_index = num_layers // 2
        self.middle_layer_index = middle_layer_index

        # ========================================
        # モデルコンポーネント
        # ========================================

        # テキスト埋め込み
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # TM-RoPE
        self.rotary_emb = TMRoPE(
            dim=hidden_size // num_heads,  # head_dim
        )

        # MoE Transformer Decoder レイヤー
        self.layers = nn.ModuleList([
            ThinkerMoEDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                shared_intermediate_size=shared_intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
            )
            for _ in range(num_layers)
        ])

        # 最終 LayerNorm
        self.norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        TM-RoPE の位置ID計算

        入力:
            input_ids: (B, L) - トークンID
            image_grid_thw: (num_images, 3) - 画像の [T, H, W]
            video_grid_thw: (num_videos, 3) - 動画の [T, H, W]
            audio_feature_lengths: (num_audios,) - 音声トークン数

        出力:
            position_ids: (3, B, L) - 3軸の位置ID
                [0]: temporal - 時間位置 (絶対時間エンコーディング)
                [1]: height - 高さ位置
                [2]: width - 幅位置

        ★ 位置ID割当ルール (Qwen2.5-Omni との差分):

        テキスト:
            temporal = height = width = 同一の連番
            → 標準1D-RoPEと等価 (変更なし)

        音声:
            temporal = height = width = 共有位置ID + 絶対時間ID
            1 ID = 80ms (AuT の 12.5Hz に対応)
            ※ Qwen2.5-Omni: 40ms/ID (Whisper 25Hz)
            ※ 固定2秒チャンクを廃止 → 連続的な位置IDを使用

        画像:
            temporal = 固定値 (画像開始位置)
            height = 空間高さ座標 [0, 1, ..., H-1]
            width = 空間幅座標 [0, 1, ..., W-1]
            (変更なし)

        動画:
            temporal = 単調増加の時間ID (80ms粒度)
            height = 空間高さ座標
            width = 空間幅座標
            ※ Qwen2.5-Omni: 固定2秒チャンクで視覚+聴覚をインターリーブ
            ※ Qwen3-Omni: 固定チャンクなし、80ms粒度で連続

        クロスモダリティ:
            次のモダリティの開始位置 = 前のモダリティの最大位置ID + 1
        """
        B, L = input_ids.shape
        position_ids = torch.zeros(3, B, L, dtype=torch.long)

        # 簡略化: テキストのみの場合
        # 実際の実装では特殊トークンを解析して各モダリティの位置を計算
        for b in range(B):
            text_pos = 0
            for i in range(L):
                # テキストトークン: 3軸同一
                position_ids[0, b, i] = text_pos  # temporal
                position_ids[1, b, i] = text_pos  # height
                position_ids[2, b, i] = text_pos  # width
                text_pos += 1

        return position_ids

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Thinker MoE のフォワードパス

        入力:
            input_ids: (B, L) - トークンID (inputs_embeds と排他)
            inputs_embeds: (B, L, hidden_size) - 入力埋め込み (マルチモーダル統合済み)
            position_ids: (3, B, L) - TM-RoPE 位置ID
            attention_mask: (B, L) - アテンションマスク
            past_key_values: KVキャッシュリスト (推論時)
            labels: (B, L) - 教師ラベル (学習時)

        出力:
            Dict {
                'logits': (B, L, vocab_size) - ボキャブラリ全体の確率
                'hidden_states': (B, L, hidden_size) - 最終隠れ状態
                'middle_hidden_states': (B, L, hidden_size) - 中間層の隠れ状態 (Talker 用)
                'loss': scalar - Cross-Entropy Loss (学習時)
                'past_key_values': KVキャッシュ (推論時)
            }

        ★ Qwen2.5-Omni との重要な違い:
            - middle_hidden_states を返す: Talker は最終層ではなく
              中間層の隠れ状態を受け取って音声生成を行う
            - MoE により各トークンのアクティブパラメータは ~3B
              (総パラメータ 30B のうち)
        """
        B = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        L = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]

        # ========================================
        # Step 1: 入力埋め込み
        # ========================================
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # inputs_embeds: (B, L, hidden_size)

        # ========================================
        # Step 2: TM-RoPE 計算
        # ========================================
        if position_ids is None:
            position_ids = self.get_rope_index(input_ids)
        # position_ids: (3, B, L)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        # cos, sin: (3, B, L, head_dim)

        # ========================================
        # Step 3: MoE Transformer Decoder
        # ========================================
        hidden_states = inputs_embeds
        new_past_key_values = []
        middle_hidden_states = None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, new_past_kv = layer(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                mrope_section=self.mrope_section,
                attention_mask=attention_mask,
                past_key_value=past_kv,
            )
            # hidden_states: (B, L, hidden_size)
            new_past_key_values.append(new_past_kv)

            # ★ 中間層の隠れ状態を保存 (Talker に渡すため)
            if i == self.middle_layer_index:
                middle_hidden_states = hidden_states.clone()
                # middle_hidden_states: (B, L, hidden_size)
                # Talker はこの中間表現を使って音声を生成する
                # 最終層よりも中間層の方がマルチモーダルな情報を保持している

        # ========================================
        # Step 4: 最終 LayerNorm
        # ========================================
        hidden_states = self.norm(hidden_states)
        # hidden_states: (B, L, hidden_size)

        # ========================================
        # Step 5: LM Head
        # ========================================
        logits = self.lm_head(hidden_states)
        # logits: (B, L, vocab_size)

        # ========================================
        # Step 6: Loss 計算 (学習時)
        # ========================================
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # shift_logits: (B, L-1, vocab_size)
            # shift_labels: (B, L-1)

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "middle_hidden_states": middle_hidden_states,
            "loss": loss,
            "past_key_values": new_past_key_values,
        }


# ============================================
# 使用例
# ============================================

def example_tmrope():
    """
    TM-RoPE の使用例

    TMRoPE と apply_tmrope を実際にインスタンス化・実行して形状を確認する。
    Qwen2.5-Omni の M-RoPE との差分も比較する。
    """

    head_dim = 128
    mrope_section_qwen3 = [24, 20, 20]  # ★ Qwen3-Omni の TM-RoPE
    mrope_section_qwen25 = [16, 24, 24]  # Qwen2.5-Omni の M-RoPE (参考)
    B, L = 1, 20

    # --- TM-RoPE インスタンス化 ---
    rotary_emb = TMRoPE(dim=head_dim)

    # --- テキストのみの位置ID: 3軸同一 → 標準1D-RoPEと等価 ---
    position_ids_text = torch.arange(L).unsqueeze(0).expand(3, B, L)
    # position_ids_text: (3, 1, 20) - 3軸すべて [0, 1, 2, ..., 19]

    x_dummy = torch.randn(B, L, 2048)  # shape 参照用
    cos, sin = rotary_emb(x_dummy, position_ids_text)
    assert cos.shape == (3, B, L, head_dim), f"Expected (3, {B}, {L}, {head_dim}), got {cos.shape}"
    assert sin.shape == (3, B, L, head_dim), f"Expected (3, {B}, {L}, {head_dim}), got {sin.shape}"

    # --- Q, K に TM-RoPE 適用 ---
    num_heads = 16
    q = torch.randn(B, num_heads, L, head_dim)
    k = torch.randn(B, num_heads, L, head_dim)

    q_rot, k_rot = apply_tmrope(q, k, cos, sin, mrope_section_qwen3)
    assert q_rot.shape == (B, num_heads, L, head_dim), \
        f"Expected ({B}, {num_heads}, {L}, {head_dim}), got {q_rot.shape}"
    assert k_rot.shape == (B, num_heads, L, head_dim), \
        f"Expected ({B}, {num_heads}, {L}, {head_dim}), got {k_rot.shape}"

    # --- 音声の位置ID例: 共有位置ID + 絶対時間 (80ms/ID) ---
    # テキスト5トークン + 音声10トークン + テキスト5トークン = 20トークン
    L_audio = 20
    pos_ids_audio = torch.zeros(3, B, L_audio, dtype=torch.long)
    # テキスト部分 (0-4): 3軸同一
    for i in range(5):
        pos_ids_audio[:, :, i] = i
    # 音声部分 (5-14): 3軸同一、80ms/ID
    # 10トークン × 80ms = 800ms の音声
    for i in range(10):
        pos_ids_audio[:, :, 5 + i] = 5 + i  # 共有位置ID
    # テキスト部分 (15-19): 前のmax+1から連番
    for i in range(5):
        pos_ids_audio[:, :, 15 + i] = 15 + i

    cos_a, sin_a = rotary_emb(torch.randn(B, L_audio, 2048), pos_ids_audio)
    assert cos_a.shape == (3, B, L_audio, head_dim)

    # --- 動画の位置ID例: temporal 単調増加 (80ms粒度) + height/width ---
    # テキスト3トークン + 動画フレーム2枚 (各2×2=4トークン) + テキスト2トークン = 13トークン
    L_video = 13
    pos_ids_video = torch.zeros(3, B, L_video, dtype=torch.long)
    # テキスト部分 (0-2): 3軸同一
    for i in range(3):
        pos_ids_video[:, :, i] = i
    # 動画フレーム1 (3-6): temporal=3 (固定)、h/w は空間座標
    pos_ids_video[0, :, 3:7] = 3                                    # temporal: 0ms
    pos_ids_video[1, :, 3:7] = torch.tensor([0, 0, 1, 1])           # height
    pos_ids_video[2, :, 3:7] = torch.tensor([0, 1, 0, 1])           # width
    # 動画フレーム2 (7-10): temporal=4 (80ms後)、h/w は空間座標
    pos_ids_video[0, :, 7:11] = 4                                   # temporal: 80ms
    pos_ids_video[1, :, 7:11] = torch.tensor([0, 0, 1, 1])          # height
    pos_ids_video[2, :, 7:11] = torch.tensor([0, 1, 0, 1])          # width
    # テキスト部分 (11-12): 前のmax+1から連番
    for i in range(2):
        pos_ids_video[:, :, 11 + i] = 5 + i

    cos_v, sin_v = rotary_emb(torch.randn(B, L_video, 2048), pos_ids_video)
    assert cos_v.shape == (3, B, L_video, head_dim)

    print("[TM-RoPE 使用例]")
    print(f"  head_dim={head_dim}")
    print(f"  Qwen3-Omni mrope_section={mrope_section_qwen3} (temporal重視)")
    print(f"  Qwen2.5-Omni mrope_section={mrope_section_qwen25} (空間重視)")
    print()
    print(f"  テキストのみ: position_ids {position_ids_text.shape} → cos/sin {cos.shape}")
    print(f"  Q回転後: {q_rot.shape}, K回転後: {k_rot.shape}")
    print()
    print(f"  音声入力 (L={L_audio}):")
    print(f"    temporal: {pos_ids_audio[0, 0].tolist()}")
    print(f"    height:   {pos_ids_audio[1, 0].tolist()}")
    print(f"    width:    {pos_ids_audio[2, 0].tolist()}")
    print(f"    → 3軸同一、1ID = 80ms (Qwen2.5-Omni: 40ms)")
    print()
    print(f"  動画入力 (L={L_video}):")
    print(f"    temporal: {pos_ids_video[0, 0].tolist()}")
    print(f"    height:   {pos_ids_video[1, 0].tolist()}")
    print(f"    width:    {pos_ids_video[2, 0].tolist()}")
    print(f"    → temporal 単調増加 (80ms粒度)、固定2秒チャンク廃止")
    print()
    print("  [vs Qwen2.5-Omni M-RoPE]")
    print("    角度配分: [16,24,24] → [24,20,20] (temporal +50%)")
    print("    音声ID: 40ms/ID → 80ms/ID (AuT 12.5Hz 対応)")
    print("    動画: 固定2秒チャンク → 連続80ms粒度")


def example_moe_layer():
    """
    MoE レイヤーの使用例

    MoELayer (共有エキスパート + ルーテッドエキスパート) を
    縮小サイズでインスタンス化し、フォワードパスを実行して形状を確認する。
    """

    # --- 縮小版 MoE レイヤー ---
    moe = MoELayer(
        hidden_size=256,
        intermediate_size=128,           # 各ルーテッドエキスパートの中間次元
        shared_intermediate_size=256,    # 共有エキスパートの中間次元
        num_experts=8,                   # 実モデルは128 等
        top_k=2,                         # 実モデルは8 等
    )
    moe.eval()

    B, L = 1, 10
    x = torch.randn(B, L, 256)

    with torch.no_grad():
        output = moe(x)

    assert output.shape == (B, L, 256), f"Expected ({B}, {L}, 256), got {output.shape}"

    # --- ルーティングの確認 ---
    x_flat = x.reshape(B * L, 256)
    with torch.no_grad():
        topk_weights, topk_indices = moe.gate(x_flat)

    assert topk_weights.shape == (B * L, 2), f"Expected ({B * L}, 2), got {topk_weights.shape}"
    assert topk_indices.shape == (B * L, 2), f"Expected ({B * L}, 2), got {topk_indices.shape}"

    # 重みの合計が1であることを確認 (softmax 正規化)
    weight_sums = topk_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
        f"エキスパート重みの合計が1ではない: {weight_sums}"

    # --- パラメータ数の比較 ---
    moe_params = sum(p.numel() for p in moe.parameters())
    dense_params = 256 * 256 * 3  # 仮の Dense FFN (gate/up/down)

    print("[MoE レイヤー使用例]")
    print(f"  設定: hidden={256}, experts={8}, top_k={2}")
    print(f"  入力: {x.shape}")
    print(f"  出力: {output.shape}")
    print()
    print(f"  ルーティング:")
    print(f"    topk_weights: {topk_weights.shape} (重み合計 ≈ 1.0)")
    print(f"    topk_indices: {topk_indices.shape}")
    print(f"    選択されたエキスパートID (先頭5トークン):")
    for t in range(min(5, B * L)):
        print(f"      トークン{t}: experts={topk_indices[t].tolist()}, "
              f"weights={topk_weights[t].tolist()}")
    print()
    print(f"  パラメータ数:")
    print(f"    MoE: {moe_params:,} (共有 + {8}個のルーテッド)")
    print(f"    Dense (参考): {dense_params:,}")
    print(f"    アクティブ: 共有 + top_{2}個のルーテッド")
    print(f"    → 総パラメータ > Dense だがアクティブパラメータ < Dense")


def example_thinker_forward():
    """
    Thinker MoE のフォワードパスの使用例

    ThinkerMoETransformer を縮小サイズでインスタンス化し、
    学習時・推論時のフォワードパスを実行して形状を確認する。
    中間層の隠れ状態 (Talker 用) の抽出も確認する。
    """

    # --- 縮小版 Thinker MoE ---
    num_layers = 4
    middle_layer_index = 2  # 4層中の第2層 (0-indexed) から Talker へ
    hidden_size = 256
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads  # = 64
    vocab_size = 1000

    thinker = ThinkerMoETransformer(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=128,           # 各ルーテッドエキスパートの中間次元
        shared_intermediate_size=256,    # 共有エキスパートの中間次元
        num_experts=4,                   # 実モデルは128 等
        top_k=2,                         # 実モデルは8 等
        vocab_size=vocab_size,
        mrope_section=[16, 16, 16],      # head_dim=64 に合わせて簡略化
        middle_layer_index=middle_layer_index,
    )
    thinker.eval()

    B, L = 1, 30

    # --- 入力: トークンID ---
    input_ids = torch.randint(0, vocab_size, (B, L))

    # --- TM-RoPE 位置ID (テキストのみ: 3軸同一) ---
    position_ids = torch.arange(L).unsqueeze(0).expand(3, B, L)
    # position_ids: (3, 1, 30)

    # --- フォワードパス (学習時: labels 付き) ---
    labels = input_ids.clone()
    labels[:, :10] = -100  # 最初の10トークンはシステムプロンプト等 → 無視

    with torch.no_grad():
        outputs = thinker(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
        )

    logits = outputs["logits"]
    hidden_states = outputs["hidden_states"]
    middle_hidden = outputs["middle_hidden_states"]
    loss = outputs["loss"]

    assert logits.shape == (B, L, vocab_size), \
        f"Expected ({B}, {L}, {vocab_size}), got {logits.shape}"
    assert hidden_states.shape == (B, L, hidden_size), \
        f"Expected ({B}, {L}, {hidden_size}), got {hidden_states.shape}"
    assert middle_hidden.shape == (B, L, hidden_size), \
        f"Expected ({B}, {L}, {hidden_size}), got {middle_hidden.shape}"
    assert loss is not None, "学習時の loss が None"

    # --- 中間層と最終層の隠れ状態が異なることを確認 ---
    assert not torch.allclose(middle_hidden, hidden_states, atol=1e-3), \
        "中間層と最終層の隠れ状態が同一 → middle_layer_index の設定に問題あり"

    # --- フォワードパス (推論時: inputs_embeds) ---
    inputs_embeds = torch.randn(B, L, hidden_size)
    with torch.no_grad():
        outputs_emb = thinker(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
    assert outputs_emb["logits"].shape == (B, L, vocab_size)
    assert outputs_emb["middle_hidden_states"].shape == (B, L, hidden_size)

    # --- KVキャッシュ付き推論 (自己回帰) ---
    # Step 1: Prefill
    with torch.no_grad():
        outputs_prefill = thinker(
            input_ids=input_ids[:, :20],
            position_ids=position_ids[:, :, :20],
        )
    past_kv = outputs_prefill["past_key_values"]
    assert len(past_kv) == num_layers, f"Expected {num_layers} layers, got {len(past_kv)}"
    # KVキャッシュ: (B, num_heads, L_kv, head_dim)
    # GQA 展開後: num_heads (num_kv_heads ではない)
    assert past_kv[0][0].shape == (B, num_heads, 20, head_dim), \
        f"Expected ({B}, {num_heads}, 20, {head_dim}), got {past_kv[0][0].shape}"

    # Step 2: Decode (1トークンずつ)
    next_ids = input_ids[:, 20:21]
    next_pos = position_ids[:, :, 20:21]
    with torch.no_grad():
        outputs_decode = thinker(
            input_ids=next_ids,
            position_ids=next_pos,
            past_key_values=past_kv,
        )
    assert outputs_decode["logits"].shape == (B, 1, vocab_size), \
        f"Expected ({B}, 1, {vocab_size}), got {outputs_decode['logits'].shape}"
    # KVキャッシュの K 長が 21 に増加
    assert outputs_decode["past_key_values"][0][0].shape == (B, num_heads, 21, head_dim), \
        f"Expected ({B}, {num_heads}, 21, {head_dim}), got {outputs_decode['past_key_values'][0][0].shape}"

    print("[Thinker MoE フォワードパス例]")
    print(f"  モデル: hidden={hidden_size}, layers={num_layers}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"  MoE: experts={4}, top_k={2}, "
          f"shared_intermediate={256}, routed_intermediate={128}")
    print(f"  TM-RoPE: mrope_section=[16,16,16] (縮小版)")
    print(f"  Middle layer index: {middle_layer_index} (Talker 用)")
    print(f"  入力: input_ids {input_ids.shape}, position_ids {position_ids.shape}")
    print()
    print(f"  学習時 (labels付き):")
    print(f"    logits:               {logits.shape}  (B, L, vocab_size)")
    print(f"    hidden_states:        {hidden_states.shape}  (B, L, hidden_size)")
    print(f"    middle_hidden_states: {middle_hidden.shape}  (B, L, hidden_size) ★Talker用")
    print(f"    loss:                 {loss.item():.4f}")
    print()
    print(f"  推論時 (KVキャッシュ):")
    print(f"    Prefill: input (1,20) → KV cache K={past_kv[0][0].shape}")
    print(f"    Decode:  input (1,1)  → KV cache K={outputs_decode['past_key_values'][0][0].shape}")
    print(f"    出力logits: {outputs_decode['logits'].shape}")
    print()
    print("  [vs Qwen2.5-Omni Thinker]")
    print("    FFN: Dense SwiGLU → MoE (共有 + ルーテッドエキスパート)")
    print("    RoPE: M-RoPE [16,24,24] → TM-RoPE [24,20,20]")
    print("    パラメータ: 7B Dense → 30B-A3B MoE")
    print("    Talker接続: 最終層のみ → 中間層 (middle_hidden_states)")
    print("    コンテキスト: 32768 (変更なし、S3 長コンテキスト学習)")


if __name__ == "__main__":
    example_tmrope()
    print()
    print("=" * 60)
    print()
    example_moe_layer()
    print()
    print("=" * 60)
    print()
    example_thinker_forward()
