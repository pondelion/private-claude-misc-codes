"""
Qwen2.5-Omni Thinker - 簡略化疑似コード
=========================================

マルチモーダル統合 LLM (Thinker = "脳")
テキスト/音声/画像/動画の特徴を統合してテキストを生成

TMRoPE (Time-aligned Multimodal RoPE) による統一的な位置エンコーディング

公式実装: modeling_qwen2_5_omni_low_VRAM_mode.py (Lines 1378-2540)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict


# ============================================
# TMRoPE (Time-aligned Multimodal RoPE)
# ============================================

class MultimodalRotaryEmbedding(nn.Module):
    """
    TMRoPE: Time-aligned Multimodal Rotary Position Embedding

    RoPE を3軸に分解して、異なるモダリティの位置情報を統一的にエンコード

    3軸: (temporal, height, width)
        - テキスト: 3軸同一 → 標準1D-RoPEと等価
        - 音声: 3軸同一、1 temporal ID = 40ms (音声エンコーダ出力の1トークンに対応)
        - 画像: temporal固定、height/widthは空間位置
        - 動画: temporalが時間増分 (1 ID = 40ms)、height/widthは空間位置
    """

    def __init__(self, dim: int, max_position: int = 32768, base: float = 10000.0):
        """
        パラメータ:
            dim: 回転埋め込みの次元 (= head_dim)
            max_position: 最大位置数 (32768)
            base: 基底周波数
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力:
            x: (B, L, D) - 入力テンソル (shapeのみ使用)
            position_ids: (3, B, L) - 3軸の位置ID
                [0]: temporal - 時間位置
                [1]: height - 高さ位置
                [2]: width - 幅位置

        出力:
            cos: (3, B, L, head_dim) - コサイン成分
            sin: (3, B, L, head_dim) - サイン成分

        ★ 標準RoPEとの違い:
            標準RoPE: position_ids (B, L) → cos/sin (B, L, dim)
            TMRoPE:   position_ids (3, B, L) → cos/sin (3, B, L, dim)
                      3軸を独立に計算
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


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TMRoPE をQuery と Key に適用

    ★ コアの仕組み:
        head_dim をセクションに分割し、各セクションに異なる軸の位置を適用
        mrope_section = [m1, m2, m3] where m1+m2+m3 = head_dim//2

        チャンネル [0:m1*2] → temporal 軸の cos/sin を適用
        チャンネル [m1*2:(m1+m2)*2] → height 軸の cos/sin を適用
        チャンネル [(m1+m2)*2:] → width 軸の cos/sin を適用

    入力:
        q: (B, num_heads, L, head_dim) - Query
        k: (B, num_heads, L, head_dim) - Key
        cos: (3, B, L, head_dim) - 3軸のコサイン
        sin: (3, B, L, head_dim) - 3軸のサイン
        mrope_section: [m1, m2, m3] - 各軸に割り当てるチャンネル数

    出力:
        q_rotated: (B, num_heads, L, head_dim)
        k_rotated: (B, num_heads, L, head_dim)
    """

    # セクションを2倍 (cos/sinペア)
    mrope_section_doubled = [s * 2 for s in mrope_section]
    # 例: [16, 24, 24] → [32, 48, 48]

    # cos/sin を3軸のセクションに分割
    cos_sections = torch.split(cos, mrope_section_doubled, dim=-1)
    sin_sections = torch.split(sin, mrope_section_doubled, dim=-1)
    # cos_sections: List of (3, B, L, section_dim) × 3

    # 各セクションから対応する軸を選択
    cos_combined = torch.cat([
        cos_sections[0][0:1, ...],   # temporal軸 → セクション0
        cos_sections[1][1:2, ...],   # height軸 → セクション1
        cos_sections[2][2:3, ...],   # width軸 → セクション2
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
# Thinker の主要コンポーネント
# ============================================

class ThinkerDecoderLayer(nn.Module):
    """
    Thinker LLM の単一 Transformer Decoder レイヤー

    構成: RMSNorm → Self-Attention (TMRoPE) → RMSNorm → SwiGLU MLP
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 14336,
    ):
        """
        パラメータ:
            hidden_size: 隠れ次元 (4096)
            num_heads: Queryヘッド数 (32)
            num_kv_heads: Key/Valueヘッド数 (8, GQA)
            intermediate_size: FFN中間次元 (14336)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = num_heads // num_kv_heads  # 32 // 8 = 4

        # Self-Attention
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # SwiGLU MLP
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

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
            hidden_states: (B, L, 4096) - 入力隠れ状態
            cos: (3, B, L, head_dim) - TMRoPEコサイン
            sin: (3, B, L, head_dim) - TMRoPEサイン
            mrope_section: [m1, m2, m3] - 各軸のチャンネル割当
            attention_mask: (B, 1, L, L) - 因果マスク (optional)
            past_key_value: KVキャッシュ (optional)

        出力:
            hidden_states: (B, L, 4096) - 出力隠れ状態
            past_key_value: 更新されたKVキャッシュ
        """
        B, L, D = hidden_states.shape
        residual = hidden_states

        # ========================================
        # 1. RMSNorm + Self-Attention
        # ========================================
        hidden_states = self.input_layernorm(hidden_states)
        # hidden_states: (B, L, 4096)

        # Q, K, V の計算
        q = self.q_proj(hidden_states)
        # q: (B, L, 32 * 128) = (B, L, 4096)

        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # k, v: (B, L, 8 * 128) = (B, L, 1024)

        # ヘッド分割
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, 32, L, 128)

        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # k, v: (B, 8, L, 128)

        # ★ TMRoPE 適用
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
        # q: (B, 32, L, 128) - 位置情報が埋め込まれたQuery
        # k: (B, 8, L, 128)  - 位置情報が埋め込まれたKey

        # GQA (Grouped Query Attention): K/Vをグループ数分繰り返す
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
            # k, v: (B, 32, L, 128)

        # KVキャッシュの結合 (推論時)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_past_key_value = (k, v)

        # Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # attn_weights: (B, 32, L, L_kv)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (B, 32, L, 128)

        # ヘッド結合
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        # attn_output: (B, L, 4096)

        attn_output = self.o_proj(attn_output)
        # attn_output: (B, L, 4096)

        hidden_states = residual + attn_output
        # hidden_states: (B, L, 4096)

        # ========================================
        # 2. RMSNorm + SwiGLU MLP
        # ========================================
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states: (B, L, 4096)

        # SwiGLU: gate * up
        gate = F.silu(self.gate_proj(hidden_states))
        # gate: (B, L, 14336)
        up = self.up_proj(hidden_states)
        # up: (B, L, 14336)
        hidden_states = gate * up
        # hidden_states: (B, L, 14336)

        hidden_states = self.down_proj(hidden_states)
        # hidden_states: (B, L, 4096)

        hidden_states = residual + hidden_states
        # hidden_states: (B, L, 4096)

        return hidden_states, new_past_key_value


class ThinkerForConditionalGeneration(nn.Module):
    """
    Qwen2.5-Omni Thinker

    マルチモーダル入力を統合してテキストを生成する LLM

    構成:
        - Audio Tower: Whisperベース Audio Encoder
        - Visual: ViTベース Vision Encoder
        - Text Model: Qwen2.5-7B Transformer Decoder (32層)
        - LM Head: テキスト生成用の線形層

    Low-VRAM モード:
        Audio Tower と Visual を CPU に配置し、
        必要時のみ GPU に転送して処理後に CPU に戻す
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        intermediate_size: int = 14336,
        vocab_size: int = 151643,
        mrope_section: Optional[List[int]] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # mrope_section: 各軸のチャンネル割当
        # head_dim = 128, mrope_section = [16, 24, 24] × 2 = [32, 48, 48]
        if mrope_section is None:
            mrope_section = [16, 24, 24]
        self.mrope_section = mrope_section

        # ========================================
        # モデルコンポーネント
        # ========================================

        # テキスト埋め込み
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # TMRoPE
        self.rotary_emb = MultimodalRotaryEmbedding(
            dim=hidden_size // num_heads,  # 128
        )

        # Transformer Decoder レイヤー
        self.layers = nn.ModuleList([
            ThinkerDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
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
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> torch.Tensor:
        """
        TMRoPE の位置ID計算

        入力:
            input_ids: (B, L) - トークンID
            image_grid_thw: (num_images, 3) - 画像の [T, H, W]
            video_grid_thw: (num_videos, 3) - 動画の [T, H, W]
            audio_feature_lengths: (num_audios,) - 音声トークン数
            use_audio_in_video: 動画内音声を使用するか

        出力:
            position_ids: (3, B, L) - 3軸の位置ID
                [0]: temporal - 時間位置
                [1]: height - 高さ位置
                [2]: width - 幅位置

        ★ 位置ID割当ルール:

        テキスト:
            temporal = height = width = 同一の連番
            → 標準1D-RoPEと等価

        音声:
            temporal = height = width = 同一の連番
            1 ID = 40ms (Audio Encoder出力の1トークン)

        画像:
            temporal = 固定値 (画像開始位置)
            height = 空間高さ座標 [0, 1, ..., H-1]
            width = 空間幅座標 [0, 1, ..., W-1]

        動画:
            temporal = フレーム時刻 (1 ID = 40ms, 動的調整)
            height = 空間高さ座標
            width = 空間幅座標

        動画+音声 (インターリーブ):
            2秒ごとのチャンクに分割
            各チャンク内: 視覚トークン → 聴覚トークン の順序
            temporal IDは実時間に基づいて連続

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
                token_id = input_ids[b, i].item()

                # テキストトークン: 3軸同一
                position_ids[0, b, i] = text_pos  # temporal
                position_ids[1, b, i] = text_pos  # height
                position_ids[2, b, i] = text_pos  # width
                text_pos += 1

        # 画像トークンの場合の例:
        # position_ids[0, b, img_start:img_end] = img_temporal (固定)
        # position_ids[1, b, img_start:img_end] = [0,0,1,1,0,0,1,1,...] (height)
        # position_ids[2, b, img_start:img_end] = [0,1,0,1,0,1,0,1,...] (width)

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
        Thinker のフォワードパス

        入力:
            input_ids: (B, L) - トークンID (inputs_embedsと排他)
            inputs_embeds: (B, L, 4096) - 入力埋め込み (マルチモーダル統合済み)
            position_ids: (3, B, L) - TMRoPE位置ID
            attention_mask: (B, L) - アテンションマスク
            past_key_values: KVキャッシュリスト (推論時)
            labels: (B, L) - 教師ラベル (学習時)

        出力:
            Dict {
                'logits': (B, L, 151643) - ボキャブラリ全体の確率
                'hidden_states': (B, L, 4096) - 最終隠れ状態
                'loss': scalar - Cross-Entropy Loss (学習時)
                'past_key_values': KVキャッシュ (推論時)
            }

        Low-VRAMモードの処理フロー (実際の実装):
            1. Visual と Audio Tower を CPU に移動
            2. CUDA キャッシュクリア
            3. (Prefill時のみ):
               a. Audio Tower を GPU に移動 → 音声エンコード → CPU に戻す → キャッシュクリア
               b. Visual を GPU に移動 → 画像/動画エンコード → CPU に戻す → キャッシュクリア
               c. masked_scatter で特徴を統合
            4. Transformer Decoder 実行
            5. LM Head でロジット計算
        """

        B = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        L = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]

        # ========================================
        # Step 1: 入力埋め込み
        # ========================================
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # inputs_embeds: (B, L, 4096)

        # ========================================
        # Step 2: TMRoPE 計算
        # ========================================
        if position_ids is None:
            position_ids = self.get_rope_index(input_ids)
        # position_ids: (3, B, L)

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        # cos, sin: (3, B, L, 128)

        # ========================================
        # Step 3: Transformer Decoder
        # ========================================
        hidden_states = inputs_embeds
        new_past_key_values = []

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
            # hidden_states: (B, L, 4096)
            new_past_key_values.append(new_past_kv)

        # ========================================
        # Step 4: 最終 LayerNorm
        # ========================================
        hidden_states = self.norm(hidden_states)
        # hidden_states: (B, L, 4096)

        # ========================================
        # Step 5: LM Head
        # ========================================
        logits = self.lm_head(hidden_states)
        # logits: (B, L, 151643)

        # ========================================
        # Step 6: Loss 計算 (学習時)
        # ========================================
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # shift_logits: (B, L-1, 151643)
            # shift_labels: (B, L-1)

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'loss': loss,
            'past_key_values': new_past_key_values,
        }


# ============================================
# 使用例
# ============================================

def example_tmrope():
    """
    TMRoPE の使用例

    MultimodalRotaryEmbedding と apply_multimodal_rotary_pos_emb を
    実際にインスタンス化・実行して形状を確認する
    """

    head_dim = 128
    mrope_section = [16, 24, 24]  # temporal, height, width のチャンネル割当
    B, L = 1, 20

    # --- TMRoPE インスタンス化 ---
    rotary_emb = MultimodalRotaryEmbedding(dim=head_dim)

    # --- テキストのみの位置ID: 3軸同一 → 標準1D-RoPEと等価 ---
    position_ids_text = torch.arange(L).unsqueeze(0).expand(3, B, L)
    # position_ids_text: (3, 1, 20) - 3軸すべて [0, 1, 2, ..., 19]

    x_dummy = torch.randn(B, L, 4096)  # shape参照用
    cos, sin = rotary_emb(x_dummy, position_ids_text)
    assert cos.shape == (3, B, L, head_dim)
    assert sin.shape == (3, B, L, head_dim)

    # --- Q, K に TMRoPE 適用 ---
    num_heads = 32
    q = torch.randn(B, num_heads, L, head_dim)
    k = torch.randn(B, num_heads, L, head_dim)

    q_rot, k_rot = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
    assert q_rot.shape == (B, num_heads, L, head_dim)
    assert k_rot.shape == (B, num_heads, L, head_dim)

    # --- 画像の位置ID例: temporal固定、height/widthは空間座標 ---
    # テキスト5トークン + 画像2×2=4トークン + テキスト3トークン = 12トークン
    L_mixed = 12
    pos_ids_mixed = torch.zeros(3, B, L_mixed, dtype=torch.long)
    # テキスト部分 (0-4): 3軸同一
    for i in range(5):
        pos_ids_mixed[:, :, i] = i
    # 画像部分 (5-8): temporal固定=5, height/widthは空間座標
    pos_ids_mixed[0, :, 5:9] = 5        # temporal: 固定
    pos_ids_mixed[1, :, 5:9] = torch.tensor([0, 0, 1, 1])  # height
    pos_ids_mixed[2, :, 5:9] = torch.tensor([0, 1, 0, 1])  # width
    # テキスト部分 (9-11): 前のmax+1から連番
    for i in range(3):
        pos_ids_mixed[:, :, 9 + i] = 6 + i

    x_mixed = torch.randn(B, L_mixed, 4096)
    cos_m, sin_m = rotary_emb(x_mixed, pos_ids_mixed)
    assert cos_m.shape == (3, B, L_mixed, head_dim)

    print(f"[TMRoPE 使用例]")
    print(f"  head_dim={head_dim}, mrope_section={mrope_section}")
    print(f"  テキストのみ: position_ids {position_ids_text.shape} → cos/sin {cos.shape}")
    print(f"  Q回転後: {q_rot.shape}, K回転後: {k_rot.shape}")
    print()
    print(f"  画像混在入力 (L={L_mixed}):")
    print(f"    temporal: {pos_ids_mixed[0, 0].tolist()}")
    print(f"    height:   {pos_ids_mixed[1, 0].tolist()}")
    print(f"    width:    {pos_ids_mixed[2, 0].tolist()}")
    print(f"    → テキスト: 3軸同一, 画像: temporal固定・h/w空間座標")


def example_thinker_forward():
    """
    Thinker のフォワードパスの使用例

    ThinkerForConditionalGeneration を縮小サイズでインスタンス化し、
    実際にフォワードパスを実行して形状を確認する
    """

    # --- 縮小版 Thinker ---
    thinker = ThinkerForConditionalGeneration(
        hidden_size=256,       # 実モデルは4096
        num_layers=2,          # 実モデルは32
        num_heads=4,           # 実モデルは32
        num_kv_heads=2,        # 実モデルは8
        intermediate_size=512, # 実モデルは14336
        vocab_size=1000,       # 実モデルは151643
        mrope_section=[16, 16, 16],  # head_dim=64 に合わせる
    )
    thinker.eval()

    B, L = 1, 30
    head_dim = 256 // 4  # = 64

    # --- 入力: トークンID ---
    input_ids = torch.randint(0, 1000, (B, L))

    # --- TMRoPE 位置ID (テキストのみ: 3軸同一) ---
    position_ids = torch.arange(L).unsqueeze(0).expand(3, B, L)
    # position_ids: (3, 1, 30)

    # --- フォワードパス (学習時: labels付き) ---
    labels = input_ids.clone()
    labels[:, :10] = -100  # 最初の10トークンはシステムプロンプト等 → 無視

    with torch.no_grad():
        outputs = thinker(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
        )

    logits = outputs['logits']
    hidden_states = outputs['hidden_states']
    loss = outputs['loss']

    assert logits.shape == (B, L, 1000)
    assert hidden_states.shape == (B, L, 256)

    # --- フォワードパス (推論時: inputs_embeds) ---
    inputs_embeds = torch.randn(B, L, 256)
    with torch.no_grad():
        outputs_emb = thinker(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
    assert outputs_emb['logits'].shape == (B, L, 1000)

    # --- KVキャッシュ付き推論 (自己回帰) ---
    # Step 1: Prefill
    with torch.no_grad():
        outputs_prefill = thinker(
            input_ids=input_ids[:, :20],
            position_ids=position_ids[:, :, :20],
        )
    past_kv = outputs_prefill['past_key_values']
    assert len(past_kv) == 2  # num_layers=2
    assert past_kv[0][0].shape == (B, 4, 20, head_dim)  # K: (B, num_heads, L, head_dim)

    # Step 2: Decode (1トークンずつ)
    next_ids = input_ids[:, 20:21]
    next_pos = position_ids[:, :, 20:21]
    with torch.no_grad():
        outputs_decode = thinker(
            input_ids=next_ids,
            position_ids=next_pos,
            past_key_values=past_kv,
        )
    assert outputs_decode['logits'].shape == (B, 1, 1000)
    # KVキャッシュのK長が21に増加
    assert outputs_decode['past_key_values'][0][0].shape == (B, 4, 21, head_dim)

    print(f"[Thinker フォワードパス例]")
    print(f"  モデル: hidden={256}, layers={2}, heads={4}, kv_heads={2}")
    print(f"  入力: input_ids {input_ids.shape}, position_ids {position_ids.shape}")
    print()
    print(f"  学習時 (labels付き):")
    print(f"    logits:        {logits.shape}  (B, L, vocab_size)")
    print(f"    hidden_states: {hidden_states.shape}  (B, L, hidden_size)")
    print(f"    loss:          {loss.item():.4f}")
    print()
    print(f"  推論時 (KVキャッシュ):")
    print(f"    Prefill: input (1,20) → KV cache K={past_kv[0][0].shape}")
    print(f"    Decode:  input (1,1)  → KV cache K={outputs_decode['past_key_values'][0][0].shape}")
    print(f"    出力logits: {outputs_decode['logits'].shape}")


if __name__ == "__main__":
    example_tmrope()
    example_thinker_forward()
