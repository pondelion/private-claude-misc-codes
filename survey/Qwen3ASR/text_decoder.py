"""
Qwen3-ASR - Qwen3 LM Text Decoder 詳細
========================================

このファイルはQwen3-ASRのText Decoder (Qwen3 LM) の
詳細な処理フローを理解するための疑似コードです。

論文: https://arxiv.org/abs/2601.21337
公式実装: qwen_asr/core/transformers_backend/modeling_qwen3_asr.py

処理フロー:
1. Token Embedding (vocab_size=151936 → D_hidden)
2. Audio特徴をmasked scatterで統合
3. MRoPE Position IDs生成 (3軸)
4. 32層 Transformer Decoder (Causal Attention + SwiGLU MLP)
5. LM Head → logits

============================================================
Shape Convention
============================================================
B:            バッチサイズ
T_text:       テキストトークン数 (プロンプト + <audio>プレースホルダー)
T_audio:      Audio Encoder出力フレーム数 (12.5Hz)
T_combined:   統合後のトークン数 (= T_text, <audio>が展開済み)
D_hidden:     隠れ次元 (4096 for 1.7B / 1536 for 0.6B)
D_inter:      SwiGLU MLP中間次元 (22016 for 1.7B / 8960 for 0.6B)
N_heads:      アテンションヘッド数 (32 for 1.7B / 12 for 0.6B)
N_kv_heads:   KVヘッド数 (GQA用, = N_heads in 1.7B)
D_head:       ヘッド次元 (128)
V:            語彙サイズ (151936)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================
# 1. 統合モデル (Audio + Text)
# ============================================================

class Qwen3ASRThinkerForConditionalGeneration(nn.Module):
    """
    Audio + Text 統合モデル (generate() のメインロジック)

    ========================================
    構成
    ========================================
    - audio_tower: AuT Audio Encoder
        → audio_encoder.py 参照
    - model: Qwen3ASRThinkerTextModel (下記)
    - lm_head: Linear(D_hidden → V)
    """

    def __init__(self, config):
        super().__init__()
        # Audio Encoder (audio_encoder.py 参照)
        self.audio_tower = None  # Qwen3ASRAudioEncoder(config.audio_config)

        # Text Decoder
        self.model = Qwen3ASRThinkerTextModel(config.text_config)

        # LM Head (重み共有なし)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,   # 4096
            config.text_config.vocab_size,    # 151936
            bias=False,
        )

        # 特殊トークンID
        self.audio_token_id = 152064       # <|audio|>

    def forward(
        self,
        input_ids: torch.Tensor,               # (B, T_text)
        attention_mask: torch.Tensor,           # (B, T_text)
        input_features: torch.Tensor,           # (B, T_mel_max, D_mel)
        feature_attention_mask: torch.Tensor,   # (B, T_mel_max)
        labels: Optional[torch.Tensor] = None,  # (B, T_text) 学習時
        **kwargs,
    ):
        """
        Forward Pass

        ========================================
        Shape
        ========================================
        入力:
            input_ids:              (B, T_text) int64
            attention_mask:         (B, T_text) int64
            input_features:         (B, T_mel_max, D_mel=128) float
            feature_attention_mask: (B, T_mel_max) int64
            labels:                 (B, T_text) int64 (学習時のみ, -100でマスク)

        出力:
            logits: (B, T_combined, V=151936)
            loss:   scalar (学習時のみ)
        """

        # ========================================
        # 1. Audio Encoding (バッチ内各サンプルを個別処理)
        # ========================================
        # AuT Encoderはバッチ処理せず、1サンプルずつ処理
        # (音声長がサンプルごとに異なるため)
        audio_features_list = []
        for i in range(input_features.shape[0]):
            mask_i = feature_attention_mask[i]              # (T_mel_max,)
            valid_feat = input_features[i][mask_i.bool()]   # (T_mel_valid, D_mel)

            # Audio Encoder forward
            # (1, D_mel, T_mel_valid) → (1, T_audio, D_aut_out=3584)
            audio_feat = self.audio_tower(
                valid_feat.unsqueeze(0).transpose(1, 2)
            )
            audio_features_list.append(audio_feat.squeeze(0))
            # audio_feat: (T_audio_i, 3584)

        # ========================================
        # 2. Token Embedding
        # ========================================
        inputs_embeds = self.model.embed_tokens(input_ids)
        # inputs_embeds: (B, T_text, D_hidden=4096)

        # ========================================
        # 3. Audio特徴のMasked Scatter
        # ========================================
        # <|audio|> トークン位置にAudio Encoder出力を配置
        #
        # 処理:
        #   input_ids:     [SYS, ..., <aud>, <aud>, ..., <aud>, USER, ..., ASST]
        #   inputs_embeds: [emb, ..., aud_0, aud_1, ..., aud_N, emb,  ..., emb ]
        #
        # <audio>トークン数 = Audio Encoder出力フレーム数 (Processor側で調整済み)
        for i in range(input_ids.shape[0]):
            audio_mask = (input_ids[i] == self.audio_token_id)  # (T_text,)
            audio_positions = audio_mask.nonzero(as_tuple=True)[0]  # (T_audio_i,)

            audio_feat = audio_features_list[i]  # (T_audio_i, D_aut_out=3584)

            # D_aut_out (3584) を D_hidden (4096) 空間にマッピング
            # ※実際の実装ではembed_tokensのforward内でscatter
            inputs_embeds[i, audio_positions] = audio_feat
            # 注: 次元不一致 (3584 vs 4096) は内部のprojection layerで解決

        # ========================================
        # 4. MRoPE Position IDs生成
        # ========================================
        position_ids = self._compute_mrope_position_ids(
            input_ids, audio_features_list
        )
        # position_ids: (3, B, T_combined)
        # [0]: Temporal位置, [1]: Height位置, [2]: Width位置

        # ========================================
        # 5. Qwen3 LM Forward
        # ========================================
        hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # hidden_states: (B, T_combined, D_hidden=4096)

        # ========================================
        # 6. LM Head
        # ========================================
        logits = self.lm_head(hidden_states)
        # logits: (B, T_combined, V=151936)

        # ========================================
        # 7. Loss計算 (学習時)
        # ========================================
        loss = None
        if labels is not None:
            # Next-token prediction loss
            # Shift: logits[:-1] vs labels[1:]
            shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
            shift_labels = labels[..., 1:].contiguous()        # (B, T-1)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # (B*(T-1), V)
                shift_labels.view(-1),                          # (B*(T-1),)
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    def _compute_mrope_position_ids(self, input_ids, audio_features_list):
        """
        MRoPE (Multi-axis RoPE) Position IDs の計算

        ========================================
        3軸の位置エンコーディング
        ========================================
        Qwen3-Omniから継承された3D位置エンコーディング:

        1. Temporal (時間軸): 24周波数次元
           - テキスト: 連番でインクリメント
           - 音声: フレームごとにインクリメント

        2. Height (高さ軸): 20周波数次元
           - テキスト: Temporal と同じ値
           - 音声: 0 固定

        3. Width (幅軸): 20周波数次元
           - テキスト: Temporal と同じ値
           - 音声: 0 固定

        RoPEの次元配分:
            D_head = 128
            → Temporal: 24次元 × 2 (sin/cos) = 48
            → Height:   20次元 × 2 = 40
            → Width:    20次元 × 2 = 40
            → 合計: 128

        インターリーブ配置:
            [T0, H0, W0, T1, H1, W1, ..., T23, H19, W19, ...]

        出力: (3, B, T_combined) int64
        """
        pass


# ============================================================
# 2. Text Model (Transformer Decoder)
# ============================================================

class Qwen3ASRThinkerTextModel(nn.Module):
    """
    Qwen3 LM Text Model

    ========================================
    構成
    ========================================
    - embed_tokens: Embedding(V=151936, D_hidden=4096)
    - layers: 32 × TransformerDecoderLayer
    - norm: RMSNorm(D_hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,     # 151936
            config.hidden_size,    # 4096
        )

        self.layers = nn.ModuleList([
            Qwen3ASRThinkerTextDecoderLayer(config)
            for _ in range(config.num_hidden_layers)  # 32
        ])

        self.norm = Qwen3ASRTextRMSNorm(
            config.hidden_size,    # 4096
            eps=config.rms_norm_eps,  # 1e-6
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,         # (B, T, D_hidden=4096)
        attention_mask: torch.Tensor,        # (B, T)
        position_ids: torch.Tensor,          # (3, B, T) MRoPE用
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            inputs_embeds:  (B, T, D_hidden=4096)
            attention_mask: (B, T) int64
            position_ids:   (3, B, T) int64 - MRoPE 3軸

        出力:
            hidden_states: (B, T, D_hidden=4096)
        """
        hidden_states = inputs_embeds  # (B, T, 4096)

        # Causal attention mask構築
        # (B, 1, T, T) - 上三角がTrue (未来のトークンをマスク)
        causal_mask = self._make_causal_mask(attention_mask)

        # 32層のTransformer Decoder
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
        # hidden_states: (B, T, 4096)

        # 最終RMSNorm
        hidden_states = self.norm(hidden_states)
        # hidden_states: (B, T, 4096)

        return hidden_states

    def _make_causal_mask(self, attention_mask):
        """
        Causal attention mask の構築

        出力: (B, 1, T, T) float
            - 未来のトークン位置が -inf
            - パディング位置が -inf
        """
        pass


# ============================================================
# 3. Decoder Layer
# ============================================================

class Qwen3ASRThinkerTextDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer

    ========================================
    構成 (Pre-LayerNorm + Residual)
    ========================================
    1. input_layernorm → Self-Attention → Residual
    2. post_attention_layernorm → SwiGLU MLP → Residual

    ========================================
    Shape
    ========================================
    入力/出力: (B, T, D_hidden=4096)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size  # 4096

        # Pre-Attention LayerNorm
        self.input_layernorm = Qwen3ASRTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Causal Self-Attention
        self.self_attn = Qwen3ASRTextAttention(config)

        # Pre-MLP LayerNorm
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # SwiGLU MLP
        self.mlp = Qwen3ASRTextMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,      # (B, T, 4096)
        attention_mask: torch.Tensor,      # (B, 1, T, T)
        position_ids: torch.Tensor,        # (3, B, T)
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            hidden_states:  (B, T, D_hidden=4096)
            attention_mask: (B, 1, T, T) float (causal mask)
            position_ids:   (3, B, T) int64

        出力:
            hidden_states: (B, T, D_hidden=4096)
        """
        # ========================================
        # 1. Self-Attention Block
        # ========================================
        residual = hidden_states                          # (B, T, 4096)
        hidden_states = self.input_layernorm(hidden_states)  # (B, T, 4096)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )                                                 # (B, T, 4096)
        hidden_states = residual + hidden_states          # (B, T, 4096)

        # ========================================
        # 2. MLP Block
        # ========================================
        residual = hidden_states                          # (B, T, 4096)
        hidden_states = self.post_attention_layernorm(hidden_states)  # (B, T, 4096)
        hidden_states = self.mlp(hidden_states)           # (B, T, 4096)
        hidden_states = residual + hidden_states          # (B, T, 4096)

        return hidden_states


# ============================================================
# 4. Self-Attention (RoPE + QKV Norm)
# ============================================================

class Qwen3ASRTextAttention(nn.Module):
    """
    Causal Self-Attention with RoPE and QK-Norm

    ========================================
    特徴
    ========================================
    - Causal (自己回帰) attention
    - RoPE (Rotary Position Embedding) 適用
    - Query/Key正規化 (RMSNorm on head dim)
    - GQA (Grouped Query Attention) 対応
      - 1.7B: N_heads=32, N_kv_heads=32 (MHA)
      - 0.6B: N_heads=12, N_kv_heads=4 (GQA, ratio=3)

    ========================================
    Shape
    ========================================
    入力:  (B, T, D_hidden=4096)
    出力:  (B, T, D_hidden=4096)
    Q:     (B, N_heads, T, D_head) = (B, 32, T, 128)
    K:     (B, N_kv_heads, T, D_head) = (B, 32, T, 128)
    V:     (B, N_kv_heads, T, D_head) = (B, 32, T, 128)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size          # 4096
        self.num_heads = config.num_attention_heads     # 32
        self.num_kv_heads = config.num_key_value_heads  # 32 (1.7B) / 4 (0.6B)
        self.head_dim = config.head_dim                # 128
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 1 (1.7B) / 3 (0.6B)

        self.scaling = self.head_dim ** -0.5  # 1/sqrt(128) ≈ 0.0884

        # QKV射影
        self.q_proj = nn.Linear(
            self.hidden_size,                           # 4096
            self.num_heads * self.head_dim,              # 32 × 128 = 4096
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,                           # 4096
            self.num_kv_heads * self.head_dim,           # 32 × 128 = 4096 (1.7B)
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,                           # 4096
            self.num_kv_heads * self.head_dim,           # 32 × 128 = 4096 (1.7B)
            bias=False,
        )

        # 出力射影
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,  # 4096
            self.hidden_size,                # 4096
            bias=False,
        )

        # QK正規化 (RMSNorm on head dimension)
        self.q_norm = Qwen3ASRTextRMSNorm(self.head_dim)  # 128
        self.k_norm = Qwen3ASRTextRMSNorm(self.head_dim)  # 128

        # RoPE パラメータ
        self.rope_theta = config.rope_theta  # 5,000,000.0

    def forward(
        self,
        hidden_states: torch.Tensor,      # (B, T, 4096)
        attention_mask: torch.Tensor,      # (B, 1, T, T)
        position_ids: torch.Tensor,        # (3, B, T)
    ) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            hidden_states:  (B, T, D_hidden=4096)
            attention_mask: (B, 1, T, T) float
            position_ids:   (3, B, T) int64

        中間:
            q: (B, N_heads=32, T, D_head=128)
            k: (B, N_kv_heads=32, T, D_head=128)
            v: (B, N_kv_heads=32, T, D_head=128)
            attn_weights: (B, N_heads, T, T)

        出力:
            output: (B, T, D_hidden=4096)
        """
        B, T, _ = hidden_states.shape

        # ========================================
        # 1. QKV射影
        # ========================================
        q = self.q_proj(hidden_states)  # (B, T, N_heads * D_head = 4096)
        k = self.k_proj(hidden_states)  # (B, T, N_kv_heads * D_head)
        v = self.v_proj(hidden_states)  # (B, T, N_kv_heads * D_head)

        # Reshape to multi-head
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, 32, T, 128)

        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # k: (B, 32, T, 128)  [1.7B] or (B, 4, T, 128) [0.6B]

        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v: (B, 32, T, 128)  [1.7B] or (B, 4, T, 128) [0.6B]

        # ========================================
        # 2. QK正規化 (RMSNorm)
        # ========================================
        # 各ヘッドの128次元に対してRMSNorm
        q = self.q_norm(q)  # (B, N_heads, T, 128)
        k = self.k_norm(k)  # (B, N_kv_heads, T, 128)

        # ========================================
        # 3. RoPE (Rotary Position Embedding) 適用
        # ========================================
        # MRoPE: 3軸の位置エンコーディングを適用
        #
        # D_head = 128 を3軸に分割:
        #   Temporal: 24 pairs → 48次元
        #   Height:   20 pairs → 40次元
        #   Width:    20 pairs → 40次元
        #   合計: 128次元
        #
        # 各軸のRoPE:
        #   q[..., dim_start:dim_end] に回転行列を適用
        #   k[..., dim_start:dim_end] に同じ回転行列を適用
        q, k = apply_mrope(q, k, position_ids, self.rope_theta)
        # q: (B, 32, T, 128) - 位置情報エンコード済み
        # k: (B, 32, T, 128) - 位置情報エンコード済み

        # ========================================
        # 4. GQA: KVヘッドの繰り返し (0.6Bモデルのみ)
        # ========================================
        if self.num_kv_groups > 1:
            # (B, N_kv=4, T, 128) → (B, N_heads=12, T, 128)
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, T, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, T, self.head_dim)

        # ========================================
        # 5. Attention Score計算
        # ========================================
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # attn_weights: (B, 32, T, T)

        # Causal mask + Padding mask適用
        attn_weights = attn_weights + attention_mask
        # attention_mask: (B, 1, T, T) - 未来/パディング位置が -inf

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(q.dtype)
        # attn_weights: (B, 32, T, T)

        # ========================================
        # 6. Attention × V
        # ========================================
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (B, 32, T, 128)

        # ========================================
        # 7. Multi-head結合 + 出力射影
        # ========================================
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.hidden_size)
        # attn_output: (B, T, 4096)

        output = self.o_proj(attn_output)
        # output: (B, T, 4096)

        return output


# ============================================================
# 5. RoPE (Rotary Position Embedding)
# ============================================================

def apply_mrope(
    q: torch.Tensor,           # (B, N_heads, T, D_head=128)
    k: torch.Tensor,           # (B, N_kv_heads, T, D_head=128)
    position_ids: torch.Tensor,  # (3, B, T)
    theta: float = 5_000_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-axis RoPE の適用

    ========================================
    RoPEの数式
    ========================================
    回転行列R(θ):
        [cos(mθ)  -sin(mθ)] [x_2i  ]
        [sin(mθ)   cos(mθ)] [x_2i+1]

    θ_i = theta^(-2i/d)  (i = 0, 1, ..., d/2-1)

    ========================================
    MRoPE 3軸の次元配分
    ========================================
    D_head = 128 を以下のように分割:

    Temporal軸 (position_ids[0]):
        次元 [0:48] → 24 pairs of (cos, sin)
        θ = 5,000,000^(-2i/128) for i=0..23

    Height軸 (position_ids[1]):
        次元 [48:88] → 20 pairs of (cos, sin)
        θ = 5,000,000^(-2i/128) for i=24..43

    Width軸 (position_ids[2]):
        次元 [88:128] → 20 pairs of (cos, sin)
        θ = 5,000,000^(-2i/128) for i=44..63

    インターリーブ配置:
        実際の次元順: [T0, H0, W0, T1, H1, W1, ..., T23, ...]

    ========================================
    Shape
    ========================================
    入力:
        q: (B, N_heads, T, 128)
        k: (B, N_kv_heads, T, 128)
        position_ids: (3, B, T)

    出力:
        q_rotated: (B, N_heads, T, 128)
        k_rotated: (B, N_kv_heads, T, 128)
    """
    d = q.shape[-1]  # 128

    # 周波数計算
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2, device=q.device).float() / d))
    # freqs: (64,) - 64 pairs

    # 3軸それぞれの位置で角度を計算
    # position_ids[axis]: (B, T) → 各軸の位置
    # freqs: (64,) → 各次元の周波数

    # Temporal軸: 次元 [0:48] → freqs[0:24]
    # Height軸:  次元 [48:88] → freqs[24:44]
    # Width軸:   次元 [88:128] → freqs[44:64]

    for axis in range(3):
        pos = position_ids[axis]  # (B, T)
        if axis == 0:
            dim_start, dim_end, freq_start, freq_end = 0, 48, 0, 24
        elif axis == 1:
            dim_start, dim_end, freq_start, freq_end = 48, 88, 24, 44
        else:
            dim_start, dim_end, freq_start, freq_end = 88, 128, 44, 64

        axis_freqs = freqs[freq_start:freq_end]  # (num_pairs,)
        angles = pos.unsqueeze(-1).float() * axis_freqs  # (B, T, num_pairs)

        cos_vals = torch.cos(angles)  # (B, T, num_pairs)
        sin_vals = torch.sin(angles)  # (B, T, num_pairs)

        # Q に回転適用
        q_slice = q[..., dim_start:dim_end]  # (B, N_heads, T, 2*num_pairs)
        q_even = q_slice[..., 0::2]  # 偶数次元
        q_odd = q_slice[..., 1::2]   # 奇数次元
        q_rotated_even = q_even * cos_vals.unsqueeze(1) - q_odd * sin_vals.unsqueeze(1)
        q_rotated_odd = q_even * sin_vals.unsqueeze(1) + q_odd * cos_vals.unsqueeze(1)
        q[..., dim_start:dim_end:2] = q_rotated_even
        q[..., dim_start + 1:dim_end:2] = q_rotated_odd

        # K に同様の回転適用
        k_slice = k[..., dim_start:dim_end]
        k_even = k_slice[..., 0::2]
        k_odd = k_slice[..., 1::2]
        k_rotated_even = k_even * cos_vals.unsqueeze(1) - k_odd * sin_vals.unsqueeze(1)
        k_rotated_odd = k_even * sin_vals.unsqueeze(1) + k_odd * cos_vals.unsqueeze(1)
        k[..., dim_start:dim_end:2] = k_rotated_even
        k[..., dim_start + 1:dim_end:2] = k_rotated_odd

    return q, k


# ============================================================
# 6. SwiGLU MLP
# ============================================================

class Qwen3ASRTextMLP(nn.Module):
    """
    SwiGLU MLP

    ========================================
    数式
    ========================================
    SwiGLU(x) = SiLU(gate_proj(x)) * up_proj(x)
    output = down_proj(SwiGLU(x))

    SiLU(x) = x * sigmoid(x)

    ========================================
    Shape
    ========================================
    入力:  (B, T, D_hidden=4096)
    中間:  (B, T, D_inter=22016)
    出力:  (B, T, D_hidden=4096)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size          # 4096
        self.intermediate_size = config.intermediate_size  # 22016

        # Gate projection: x → gate_proj(x) を SiLU で活性化
        self.gate_proj = nn.Linear(
            config.hidden_size,       # 4096
            config.intermediate_size,  # 22016
            bias=False,
        )

        # Up projection: x → up_proj(x)
        self.up_proj = nn.Linear(
            config.hidden_size,       # 4096
            config.intermediate_size,  # 22016
            bias=False,
        )

        # Down projection: 中間 → 出力
        self.down_proj = nn.Linear(
            config.intermediate_size,  # 22016
            config.hidden_size,       # 4096
            bias=False,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ========================================
        Shape
        ========================================
        入力:
            x: (B, T, D_hidden=4096)

        出力:
            output: (B, T, D_hidden=4096)
        """
        # Gate: SiLU(gate_proj(x))
        gate = self.act_fn(self.gate_proj(x))
        # gate: (B, T, 22016)

        # Up: up_proj(x)
        up = self.up_proj(x)
        # up: (B, T, 22016)

        # SwiGLU: element-wise乗算
        intermediate = gate * up
        # intermediate: (B, T, 22016)

        # Down: down_proj(intermediate)
        output = self.down_proj(intermediate)
        # output: (B, T, 4096)

        return output


# ============================================================
# 7. RMSNorm
# ============================================================

class Qwen3ASRTextRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    ========================================
    数式
    ========================================
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    LayerNormと比較:
    - LayerNorm: (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    - RMSNorm: 平均引き算なし、biasなし → 計算効率が高い

    ========================================
    Shape
    ========================================
    入力/出力: (..., D)  - 最終次元に対して正規化
    weight: (D,) - 学習可能なスケールパラメータ
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力:  (..., D)
        出力:  (..., D)
        """
        # x^2 の平均 (最終次元)
        variance = x.pow(2).mean(-1, keepdim=True)
        # variance: (..., 1)

        # 正規化
        x = x * torch.rsqrt(variance + self.eps)
        # x: (..., D)

        # スケーリング
        return x * self.weight


# ============================================================
# 入出力shape一覧表
# ============================================================
"""
========================================
Text Decoder Shape遷移 (1.7Bモデルの例)
========================================

| 段階                    | テンソル名        | Shape                    | 説明                          |
|------------------------|------------------|--------------------------|-------------------------------|
| Token IDs              | input_ids        | (B, T_text)              | int64 トークンID               |
| Token Embedding        | inputs_embeds    | (B, T_text, 4096)        | 埋め込みベクトル                |
| Audio Scatter後        | inputs_embeds    | (B, T_combined, 4096)    | Audio特徴統合済み               |
| MRoPE Position IDs     | position_ids     | (3, B, T_combined)       | 3軸: Temporal, H, W           |
| Q射影                  | q                | (B, 32, T, 128)          | 32ヘッド × 128次元             |
| K射影                  | k                | (B, 32, T, 128)          | 32 KVヘッド (1.7B)            |
| V射影                  | v                | (B, 32, T, 128)          | 32 KVヘッド (1.7B)            |
| QK-Norm後              | q, k             | 同上                      | RMSNorm適用済み                |
| RoPE適用後             | q, k             | 同上                      | 位置情報エンコード済み           |
| Attention Weights      | attn_weights     | (B, 32, T, T)            | Causal + Padding mask         |
| Attention Output       | attn_output      | (B, 32, T, 128)          | V × weights                   |
| Multi-head結合         | attn_output      | (B, T, 4096)             | 32 × 128 = 4096               |
| O射影                  | output           | (B, T, 4096)             | 出力射影                       |
| SwiGLU Gate            | gate             | (B, T, 22016)            | SiLU(gate_proj(x))            |
| SwiGLU Up              | up               | (B, T, 22016)            | up_proj(x)                    |
| SwiGLU intermediate    | intermediate     | (B, T, 22016)            | gate * up                     |
| SwiGLU Output          | mlp_output       | (B, T, 4096)             | down_proj(intermediate)       |
| 32層Decoder後          | hidden_states    | (B, T, 4096)             | 全レイヤー通過                  |
| Final RMSNorm          | hidden_states    | (B, T, 4096)             | 最終正規化                     |
| LM Head                | logits           | (B, T, 151936)           | 語彙分布                       |

========================================
1.7Bモデル vs 0.6Bモデルのパラメータ比較
========================================

| パラメータ            | 1.7B              | 0.6B              |
|----------------------|--------------------|--------------------|
| hidden_size (D)      | 4096               | 1536               |
| intermediate_size    | 22016              | 8960               |
| num_hidden_layers    | 32                 | 28                 |
| num_attention_heads  | 32                 | 12                 |
| num_key_value_heads  | 32 (MHA)           | 4 (GQA, ratio=3)  |
| head_dim             | 128                | 128                |
| vocab_size           | 151936             | 151936             |
| max_position_embeddings | 128000          | 128000             |
| rope_theta           | 5,000,000          | 5,000,000          |
| rms_norm_eps         | 1e-6               | 1e-6               |
"""
