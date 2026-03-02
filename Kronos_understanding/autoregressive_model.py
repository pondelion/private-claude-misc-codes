"""
Kronos Understanding - Autoregressive Model (簡略化疑似コード)

階層的自己回帰Transformer。トークン化されたK線系列から
次のタイムステップのトークンを Coarse-to-Fine で予測する。

対応する公式実装:
  - model/kronos.py: Kronos クラス
  - model/module.py: 各コンポーネント

論文参照: Section 3 "Hierarchical Autoregressive Modeling"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Kronos メインモデル
# ============================================================

class Kronos(nn.Module):
    """
    Kronos: 階層的自己回帰Transformer for K線予測

    アーキテクチャ:
        1. HierarchicalEmbedding: s1/s2トークンを結合埋め込み
        2. TemporalEmbedding: 時間特徴の埋め込み
        3. Causal Transformer: N層の因果的Self-Attention
        4. DualHead: s1予測ヘッド
        5. DependencyAwareLayer + DualHead: s2条件付き予測

    予測の分解 (Chain Rule):
        p(b_t | b_{<t}) = p(s1_t | b_{<t}) * p(s2_t | b_{<t}, s1_t)

    モデルファミリー:
        Kronos_small: 8層, d=512,  8head,  24.7M params
        Kronos_base:  12層, d=832, 16head, 102.3M params
        Kronos_large: 18層, d=1664, 32head, 499.2M params
    """

    def __init__(
        self,
        s1_bits=10,         # Coarseサブトークン ビット数 → 語彙 2^10 = 1024
        s2_bits=10,         # Fineサブトークン ビット数 → 語彙 2^10 = 1024
        n_layers=8,         # Transformerレイヤー数
        d_model=512,        # モデル次元
        n_heads=8,          # Attention ヘッド数
        ff_dim=1024,        # FFN中間次元
        ffn_dropout_p=0.25,
        attn_dropout_p=0.1,
        resid_dropout_p=0.25,
        token_dropout_p=0.1,
        learn_te=True,      # 学習可能なTemporalEmbedding
    ):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.s1_vocab_size = 2 ** s1_bits   # 1024
        self.d_model = d_model

        # --- 入力埋め込み ---
        self.embedding = HierarchicalEmbedding(s1_bits, s2_bits, d_model)
        self.time_emb = TemporalEmbedding(d_model, learn_te)
        self.token_drop = nn.Dropout(token_dropout_p)

        # --- Causal Transformer ---
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # --- 予測ヘッド ---
        self.dep_layer = DependencyAwareLayer(d_model)
        self.head = DualHead(s1_bits, s2_bits, d_model)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        入力:
            s1_ids: (B, T) - Coarseトークン ID ∈ [0, 1023]
            s2_ids: (B, T) - Fineトークン ID ∈ [0, 1023]
            stamp:  (B, T, 5) - 時間特徴 [minute, hour, weekday, day, month]
            padding_mask: (B, T) - パディングマスク (True=パディング)
            use_teacher_forcing: Trueなら s1_targets を使用
            s1_targets: (B, T) - Teacher Forcing用の正解s1

        出力:
            s1_logits: (B, T, 1024) - Coarseトークンのロジット
            s2_logits: (B, T, 1024) - Fineトークンのロジット (s1に条件付き)

        処理フロー:
            [s1_ids, s2_ids] (B,T) × 2
                ↓ HierarchicalEmbedding
            x (B, T, d_model) = fusion(cat(emb_s1, emb_s2))
                ↓ + TemporalEmbedding
            x (B, T, d_model)
                ↓ Token Dropout
            x (B, T, d_model)
                ↓ N × TransformerBlock (Causal)
            h (B, T, d_model)
                ↓ RMSNorm
            h (B, T, d_model)
                ↓ DualHead.forward (s1予測)
            s1_logits (B, T, 1024)
                ↓ サンプリング → ŝ1
            ŝ1 (B, T)
                ↓ DependencyAwareLayer (Cross-Attention: q=emb(ŝ1), kv=h)
            h_update (B, T, d_model)
                ↓ DualHead.cond_forward (s2予測)
            s2_logits (B, T, 1024)
        """
        # === Step 1: 埋め込み ===
        x = self.embedding([s1_ids, s2_ids])
        # x: (B, T, d_model)

        # === Step 2: 時間埋め込みの加算 ===
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            # time_embedding: (B, T, d_model)
            x = x + time_embedding

        x = self.token_drop(x)

        # === Step 3: Causal Transformer ===
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        # x (= h): (B, T, d_model)

        # === Step 4: s1予測 (Coarse Subtoken) ===
        s1_logits = self.head(x)
        # s1_logits: (B, T, 1024)

        # === Step 5: s1のサンプリング ===
        if use_teacher_forcing:
            # 学習初期: 正解s1を使用
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            # 通常学習/推論: サンプリングしたs1を使用
            # ※ exposure bias軽減のため、学習時もサンプリング
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            # s1_probs: (B, T, 1024)
            sample_s1 = torch.multinomial(
                s1_probs.view(-1, self.s1_vocab_size), 1
            ).view(s1_ids.shape)
            # sample_s1: (B, T)
            sibling_embed = self.embedding.emb_s1(sample_s1)
        # sibling_embed: (B, T, d_model)

        # === Step 6: DependencyAwareLayer + s2予測 ===
        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        # x2 (= h_update): (B, T, d_model)

        s2_logits = self.head.cond_forward(x2)
        # s2_logits: (B, T, 1024)

        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        s1のみ予測 (推論時ステップ1で使用)

        入力: s1_ids (B, T), s2_ids (B, T), stamp (B, T, 5)
        出力:
            s1_logits: (B, T, 1024)
            context: (B, T, d_model) - Transformer出力 (s2予測用に保持)
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x  # contextも返す

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """
        s2予測 (推論時ステップ2で使用)

        入力:
            context: (B, T, d_model) - decode_s1の出力
            s1_ids: (B, T) or (B, 1) - サンプリング済みs1
        出力:
            s2_logits: (B, T, 1024) or (B, 1, 1024)
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


# ============================================================
# HierarchicalEmbedding
# ============================================================

class HierarchicalEmbedding(nn.Module):
    """
    s1/s2サブトークンの階層的埋め込み

    処理:
        1. s1_ids → Embedding(1024, d_model) → s1_emb × √d_model
        2. s2_ids → Embedding(1024, d_model) → s2_emb × √d_model
        3. concat([s1_emb, s2_emb]) → Linear(2*d_model, d_model) → fused

    入力: [s1_ids (B,T), s2_ids (B,T)] or composite_ids (B,T)
    出力: (B, T, d_model)

    ※ √d_model によるスケーリングはTransformer系モデルの標準的手法
    """

    def __init__(self, s1_bits=10, s2_bits=10, d_model=512):
        super().__init__()
        self.s2_bits = s2_bits
        self.d_model = d_model

        vocab_s1 = 2 ** s1_bits  # 1024
        vocab_s2 = 2 ** s2_bits  # 1024

        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, token_ids):
        """
        入力: [s1_ids, s2_ids] 各 (B, T)
        出力: (B, T, d_model)
        """
        if isinstance(token_ids, (tuple, list)):
            s1_ids, s2_ids = token_ids
        else:
            # 複合IDの場合はビットシフトで分割
            s1_ids = token_ids >> self.s2_bits
            s2_ids = token_ids & ((1 << self.s2_bits) - 1)

        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        # s1_emb: (B, T, d_model)

        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        # s2_emb: (B, T, d_model)

        # Concatenate + Linear fusion
        fused = self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))
        # cat: (B, T, 2*d_model) → fused: (B, T, d_model)

        return fused


# ============================================================
# DependencyAwareLayer
# ============================================================

class DependencyAwareLayer(nn.Module):
    """
    s2予測をs1に条件付けるためのCross-Attentionモジュール

    論文 Eq. 7:
        h_t^update = CrossAttn(q = e_c(ŝ1_t), k = h_t, v = h_t)
        p(s2_t | b_{<t}, s1_t) = softmax(W_f * h_t^update)

    処理:
        query = emb_s1(ŝ1)     ... サンプリングされたs1の埋め込み
        key   = h              ... Transformerの出力
        value = h
        → Cross-Attention + Residual + RMSNorm

    入力:
        hidden_states: (B, T, d_model) - Transformer出力
        sibling_embed: (B, T, d_model) - s1の埋め込み
    出力: (B, T, d_model) - s1情報で更新された表現
    """

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        """
        入力:
            hidden_states: (B, T, d_model) - Transformer出力 (key/value)
            sibling_embed: (B, T, d_model) - s1埋め込み (query)
        出力: (B, T, d_model) - 更新された表現
        """
        attn_out = self.cross_attn(
            query=sibling_embed,        # s1情報
            key=hidden_states,          # コンテキスト
            value=hidden_states,        # コンテキスト
            key_padding_mask=key_padding_mask
        )
        # attn_out: (B, T, d_model)

        # Residual + Norm
        return self.norm(hidden_states + attn_out)
        # output: (B, T, d_model)


# ============================================================
# DualHead
# ============================================================

class DualHead(nn.Module):
    """
    s1/s2それぞれの予測ヘッド

    forward():      h → s1_logits (Coarse予測)
    cond_forward():  h_update → s2_logits (Fine予測, s1条件付き)

    損失計算:
        CE_s1 = CrossEntropy(s1_logits, s1_targets)
        CE_s2 = CrossEntropy(s2_logits, s2_targets)
        L = (CE_s1 + CE_s2) / 2
    """

    def __init__(self, s1_bits=10, s2_bits=10, d_model=512):
        super().__init__()
        self.vocab_s1 = 2 ** s1_bits  # 1024
        self.vocab_s2 = 2 ** s2_bits  # 1024
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)  # (d_model → 1024)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)  # (d_model → 1024)

    def forward(self, x):
        """s1予測: (B, T, d_model) → (B, T, 1024)"""
        return self.proj_s1(x)

    def cond_forward(self, x2):
        """s2予測 (s1条件付き): (B, T, d_model) → (B, T, 1024)"""
        return self.proj_s2(x2)

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        """
        学習損失の計算

        入力:
            s1_logits: (B, T, 1024)
            s2_logits: (B, T, 1024)
            s1_targets: (B, T) - 正解s1 ID
            s2_targets: (B, T) - 正解s2 ID

        出力: (total_loss, ce_s1, ce_s2)
        """
        if padding_mask is not None:
            valid = (padding_mask == 0)
            s1_logits = s1_logits[valid]
            s2_logits = s2_logits[valid]
            s1_targets = s1_targets[valid]
            s2_targets = s2_targets[valid]

        ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
        ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        return (ce_s1 + ce_s2) / 2, ce_s1, ce_s2


# ============================================================
# TemporalEmbedding
# ============================================================

class TemporalEmbedding(nn.Module):
    """
    時間特徴の埋め込み

    5種類の時間特徴を独立にEmbeddingし、全て加算:
        - minute: 0-59 (60エントリ)
        - hour:   0-23 (24エントリ)
        - weekday: 0-6 (7エントリ)
        - day:    1-31 (32エントリ)
        - month:  1-12 (13エントリ)

    金融市場の周期性を捕捉:
        - 日中パターン (寄り付き/昼休み/引け)
        - 曜日効果 (月曜効果等)
        - 月末リバランス
        - 季節性 (年末ラリー等)

    入力: stamp (B, T, 5) - [minute, hour, weekday, day, month]
    出力: (B, T, d_model) - 全特徴の埋め込みの和
    """

    def __init__(self, d_model, learn_pe=True):
        super().__init__()
        # 学習可能 or 固定 (Sinusoidal) 埋め込み
        Embed = nn.Embedding if learn_pe else FixedEmbedding

        self.minute_embed  = Embed(60, d_model)   # 分: 0-59
        self.hour_embed    = Embed(24, d_model)   # 時: 0-23
        self.weekday_embed = Embed(7, d_model)    # 曜日: 0-6 (月=0)
        self.day_embed     = Embed(32, d_model)   # 日: 1-31
        self.month_embed   = Embed(13, d_model)   # 月: 1-12

    def forward(self, x):
        """
        入力: x (B, T, 5) - [minute, hour, weekday, day, month]
        出力: (B, T, d_model)
        """
        x = x.long()

        minute_x  = self.minute_embed(x[:, :, 0])   # (B, T, d_model)
        hour_x    = self.hour_embed(x[:, :, 1])      # (B, T, d_model)
        weekday_x = self.weekday_embed(x[:, :, 2])   # (B, T, d_model)
        day_x     = self.day_embed(x[:, :, 3])       # (B, T, d_model)
        month_x   = self.month_embed(x[:, :, 4])     # (B, T, d_model)

        # 全て加算 (連結ではない)
        return minute_x + hour_x + weekday_x + day_x + month_x


# ============================================================
# Transformer コンポーネント
# ============================================================

class TransformerBlock(nn.Module):
    """
    Pre-LN Causal Transformer Block

    構造:
        x → RMSNorm → CausalSelfAttn(RoPE) → Residual
          → RMSNorm → SwiGLU FFN → Residual

    入力/出力: (B, T, d_model)

    ※ Causal mask により未来の情報は参照不可
    """

    def __init__(self, d_model, n_heads, ff_dim, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None):
        """
        入力: x (B, T, d_model)
        出力: x (B, T, d_model)
        """
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x, key_padding_mask=key_padding_mask)

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    位置情報を回転行列として注入:
        q' = q * cos(θ) + rotate(q) * sin(θ)
        k' = k * cos(θ) + rotate(k) * sin(θ)

    特徴:
        - 相対位置情報を自然にエンコード
        - 系列長に対して外挿性が高い
        - 追加パラメータ不要 (固定)

    入力: q, k 各 (B, n_heads, T, head_dim)
    出力: q', k' 各 (B, n_heads, T, head_dim)
    """

    def __init__(self, dim):
        super().__init__()
        # inv_freq: [1/10000^(0/dim), 1/10000^(2/dim), ..., 1/10000^((dim-2)/dim)]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k):
        """
        入力: q, k 各 (B, n_heads, T, head_dim)
        出力: q_rotated, k_rotated 各 (B, n_heads, T, head_dim)
        """
        seq_len = q.shape[-2]
        t = torch.arange(seq_len, device=q.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        cos = emb.cos()[None, None, :, :]  # (1, 1, T, head_dim)
        sin = emb.sin()[None, None, :, :]

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q_out = q * cos + rotate_half(q) * sin
        k_out = k * cos + rotate_half(k) * sin
        return q_out, k_out


class MultiHeadAttentionWithRoPE(nn.Module):
    """
    Causal Multi-Head Self-Attention with RoPE

    処理:
        1. x → Q, K, V (各 d_model → n_heads × head_dim)
        2. RoPE適用: Q', K' = RoPE(Q, K)
        3. Scaled Dot-Product Attention (causal mask付き)
        4. 出力射影

    入力: x (B, T, d_model)
    出力: (B, T, d_model)
    """

    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None):
        B, T, _ = x.shape

        # 射影 + マルチヘッド分割
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, n_heads, T, head_dim)

        # RoPE適用
        q, k = self.rotary(q, k)

        # Causal Scaled Dot-Product Attention
        # is_causal=True: 自動的に下三角マスクを適用
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True
        )
        # attn_output: (B, n_heads, T, head_dim)

        # ヘッド結合 + 出力射影
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.out_proj(attn_output))
        # output: (B, T, d_model)


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    """
    Cross-Attention with RoPE (DependencyAwareLayer用)

    Self-Attentionとの違い:
        - query, key, valueが異なるソースから来る
        - query = s1の埋め込み
        - key = value = Transformerの隠れ状態

    入力: query (B, T, d), key (B, T, d), value (B, T, d)
    出力: (B, T, d_model)
    """

    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, query, key, value, key_padding_mask=None):
        B, q_len, _ = query.shape
        _, kv_len, _ = key.shape

        q = self.q_proj(query).view(B, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        # 訓練時のみcausal (推論時はcausal不要のケースも)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=self.training
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.resid_dropout(self.out_proj(attn_output))


# ============================================================
# 共通コンポーネント
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU_FFN(nn.Module):
    """SwiGLU Feed-Forward Network: FFN(x) = W2 * (SiLU(W1*x) ⊙ W3*x)"""

    def __init__(self, d_model, ff_dim, dropout_p=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class FixedEmbedding(nn.Module):
    """Sinusoidal位置埋め込み (非学習)"""

    def __init__(self, c_in, d_model):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


# ============================================================
# 使用例
# ============================================================

if __name__ == "__main__":
    # Kronos_small の構成
    model = Kronos(
        s1_bits=10, s2_bits=10,
        n_layers=8, d_model=512, n_heads=8, ff_dim=1024,
        ffn_dropout_p=0.25, attn_dropout_p=0.1, resid_dropout_p=0.25,
        token_dropout_p=0.1, learn_te=True,
    )

    B, T = 4, 128
    s1_ids = torch.randint(0, 1024, (B, T))  # (4, 128)
    s2_ids = torch.randint(0, 1024, (B, T))  # (4, 128)
    stamp = torch.stack([
        torch.randint(0, 60, (B, T)),   # minute
        torch.randint(0, 24, (B, T)),   # hour
        torch.randint(0, 7, (B, T)),    # weekday
        torch.randint(1, 32, (B, T)),   # day
        torch.randint(1, 13, (B, T)),   # month
    ], dim=-1).float()                   # (4, 128, 5)

    # === 学習時: Forward ===
    s1_logits, s2_logits = model(s1_ids[:, :-1], s2_ids[:, :-1], stamp[:, :-1])
    # s1_logits: (4, 127, 1024)
    # s2_logits: (4, 127, 1024)

    loss, ce_s1, ce_s2 = model.head.compute_loss(
        s1_logits, s2_logits,
        s1_ids[:, 1:], s2_ids[:, 1:]
    )
    print(f"Loss: {loss:.4f} (CE_s1={ce_s1:.4f}, CE_s2={ce_s2:.4f})")

    # === 推論時: 2段階デコード ===
    with torch.no_grad():
        # Step 1: s1予測
        s1_logits, context = model.decode_s1(s1_ids, s2_ids, stamp)
        s1_pred = s1_logits[:, -1, :].argmax(dim=-1)  # (4,) - 最後のステップ

        # Step 2: s2予測 (s1条件付き)
        s2_logits = model.decode_s2(context, s1_pred.unsqueeze(1))
        s2_pred = s2_logits[:, -1, :].argmax(dim=-1)  # (4,)

        print(f"Predicted s1: {s1_pred}")
        print(f"Predicted s2: {s2_pred}")
