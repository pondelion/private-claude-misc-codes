"""
CosyVoice3 Language Model (Qwen2ベース) - 簡略化疑似コード
============================================================

テキストトークンとプロンプト音声トークンから、
音声トークン列を自己回帰的に生成するLLMモジュール。

論文: CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
公式実装: cosyvoice/llm/llm.py

CosyVoice2との違い:
- モデルサイズ: 0.5B → 0.5B / 1.5B (スケールアップ)
- 音声トークン語彙: 6561 + 3特殊 → 6561 + 200追加トークン
- DiffRO後処理に対応 (Gumbel-Softmax対応)
- Bistream推論対応 (テキストと音声を交互に生成)

Shape Convention
============================================================
B: バッチサイズ
L_text: テキストトークン長
L_prompt_speech: プロンプト音声トークン長
T_speech: 生成音声トークン長
D_text: テキストエンコーダ出力次元 (512等)
D_llm: LLM隠れ次元 (896)
Q: 音声トークン語彙サイズ (6561)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Generator


class CosyVoice3LM(nn.Module):
    """
    CosyVoice3 Language Model

    アーキテクチャ全体像:
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  テキストトークン (1, L_text)                              │
    │       ↓                                                  │
    │  Text Embedding (vocab → 512)                            │
    │       ↓                                                  │
    │  Conformer Text Encoder (6層)                             │
    │       ↓                                                  │
    │  Affine Layer (512 → 896)                                │
    │       ↓                                                  │
    │  text_embeds: (1, L_text, 896)                           │
    │                                                          │
    │  プロンプト音声トークン (1, L_prompt_speech)                │
    │       ↓                                                  │
    │  Speech Embedding (6561 → 896)                           │
    │       ↓                                                  │
    │  prompt_speech_embeds: (1, L_prompt_speech, 896)          │
    │                                                          │
    │  LLM入力の構築:                                           │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │ [SOS] [text_embeds] [prompt_speech_embeds] [生成部分] │ │
    │  │  (1)    (L_text)     (L_prompt_speech)      (自己回帰)│ │
    │  └─────────────────────────────────────────────────────┘ │
    │       ↓                                                  │
    │  Qwen2 Decoder (22層/28層 Transformer)                   │
    │       ↓                                                  │
    │  LLM Decoder Head (896 → 6561+α)                         │
    │       ↓                                                  │
    │  音声トークン logits → サンプリング (RAS)                   │
    │       ↓                                                  │
    │  speech_tokens: (1, T_speech)                            │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        text_encoder_input_size: int = 512,   # テキストエンコーダ入力次元
        llm_input_size: int = 896,            # LLM入力次元
        llm_output_size: int = 896,           # LLM出力次元
        speech_token_size: int = 6561,        # 音声トークン語彙サイズ
        text_token_size: int = 151936,        # テキストトークン語彙サイズ
        num_text_encoder_layers: int = 6,     # テキストエンコーダ層数
        num_llm_layers: int = 22,             # Qwen2レイヤー数 (0.5B:22, 1.5B:28)
    ):
        super().__init__()

        self.speech_token_size = speech_token_size

        # ========================================
        # 1. テキスト埋め込み + テキストエンコーダ
        # ========================================
        # テキストトークン → 埋め込みベクトル
        self.text_embedding = nn.Embedding(
            num_embeddings=text_token_size,    # ~151K
            embedding_dim=text_encoder_input_size,  # 512
        )
        # 入力: (B, L_text) → 出力: (B, L_text, 512)

        # Conformer Text Encoder
        self.text_encoder = ConformerEncoder(
            input_size=text_encoder_input_size,  # 512
            output_size=text_encoder_input_size,  # 512
            num_layers=num_text_encoder_layers,   # 6
            num_heads=4,
            linear_units=2048,
        )
        # 入力: (B, L_text, 512) → 出力: (B, L_text, 512)

        # テキストエンコーダ出力 → LLM入力次元に射影
        self.text_encoder_affine_layer = nn.Linear(
            text_encoder_input_size,  # 512
            llm_input_size,           # 896
        )
        # 入力: (B, L_text, 512) → 出力: (B, L_text, 896)

        # ========================================
        # 2. 音声トークン埋め込み
        # ========================================
        self.speech_embedding = nn.Embedding(
            num_embeddings=speech_token_size + 200,  # 6561 + 200 追加トークン
            embedding_dim=llm_input_size,            # 896
        )
        # 入力: (B, L_speech) → 出力: (B, L_speech, 896)

        # ========================================
        # 3. 特殊トークン埋め込み
        # ========================================
        # SOS (Start of Sequence) / TASK_ID トークン
        # CosyVoice3ではspeech_embeddingの特殊インデックスを使用
        self.sos_token_id = speech_token_size      # 6561
        self.eos_token_id = speech_token_size + 1  # 6562
        self.fill_token_id = speech_token_size + 2 # 6563

        # ========================================
        # 4. Qwen2 LLMデコーダ
        # ========================================
        self.llm = Qwen2ForCausalLM(
            hidden_size=llm_input_size,       # 896
            num_layers=num_llm_layers,        # 22 (0.5B) or 28 (1.5B)
            num_attention_heads=14,           # 0.5B: 14, 1.5B: 12
            num_key_value_heads=2,            # GQA (Grouped Query Attention)
            intermediate_size=4864,           # 0.5B: 4864
        )
        # 入力: (B, T_total, 896) → 出力: (B, T_total, 896)

        # ========================================
        # 5. LLMデコーダヘッド (音声トークン予測)
        # ========================================
        self.llm_decoder = nn.Linear(
            llm_output_size,                  # 896
            speech_token_size + 200,          # 6761
        )
        # 入力: (B, T, 896) → 出力: (B, T, 6761)

    def forward(
        self,
        text_token_ids: torch.Tensor,         # (B, L_text)
        text_token_ids_len: torch.Tensor,     # (B,)
        speech_tokens: torch.Tensor,          # (B, T_speech)
        speech_tokens_len: torch.Tensor,      # (B,)
    ) -> Dict[str, torch.Tensor]:
        """
        学習時のフォワードパス

        入力:
            text_token_ids: (B, L_text) - テキストトークンID
            text_token_ids_len: (B,) - テキスト長 (パディング対応)
            speech_tokens: (B, T_speech) - 正解音声トークン
            speech_tokens_len: (B,) - 音声トークン長

        出力:
            loss: スカラー - 交差エントロピーロス
            acc: スカラー - トークン予測精度

        ========================================
        学習時のシーケンス構成
        ========================================

        入力シーケンス (LLMへの入力):
        ┌─────┬──────────────┬─────────────────────────┐
        │ SOS │ text_embeds  │ speech_token_embeds[:-1] │
        │ (1) │ (L_text)     │ (T_speech - 1)          │
        └─────┴──────────────┴─────────────────────────┘
        Total: 1 + L_text + T_speech - 1 = L_text + T_speech

        ターゲット (予測対象):
        ┌──────────────────────┬────────────────────────┐
        │ IGNORE (テキスト部分) │ speech_tokens + [EOS]  │
        │ (-1 × L_text+1)     │ (T_speech)             │
        └──────────────────────┴────────────────────────┘
        ※ テキスト部分はロス計算から除外 (IGNORE_ID = -1)
        """
        # Step 1: テキストエンコーディング
        text_embeds = self.text_embedding(text_token_ids)
        # text_embeds: (B, L_text, 512)

        text_embeds = self.text_encoder(text_embeds)
        # text_embeds: (B, L_text, 512)

        text_embeds = self.text_encoder_affine_layer(text_embeds)
        # text_embeds: (B, L_text, 896)

        # Step 2: 音声トークン埋め込み
        speech_embeds = self.speech_embedding(speech_tokens)
        # speech_embeds: (B, T_speech, 896)

        # Step 3: SOS トークン
        sos_embed = self.speech_embedding(
            torch.full((text_embeds.shape[0], 1), self.sos_token_id,
                       device=text_embeds.device)
        )
        # sos_embed: (B, 1, 896)

        # Step 4: LLM入力を構築
        # [SOS, text_embeds, speech_embeds[:-1]]
        lm_input = torch.cat([
            sos_embed,                    # (B, 1, 896)
            text_embeds,                  # (B, L_text, 896)
            speech_embeds[:, :-1, :],     # (B, T_speech-1, 896)
        ], dim=1)
        # lm_input: (B, 1 + L_text + T_speech - 1, 896)

        # Step 5: Qwen2 LLMでフォワード
        lm_output = self.llm(lm_input)
        # lm_output: (B, 1 + L_text + T_speech - 1, 896)

        # Step 6: 音声トークン予測
        logits = self.llm_decoder(lm_output)
        # logits: (B, 1 + L_text + T_speech - 1, 6761)

        # Step 7: ロス計算 (テキスト部分はIGNORE)
        # 音声部分のみでCross-Entropy計算
        speech_logits = logits[:, 1 + text_embeds.shape[1]:, :]
        # speech_logits: (B, T_speech - 1, 6761)

        # ターゲット: speech_tokens[:, 1:] + EOS
        loss = F.cross_entropy(
            speech_logits.reshape(-1, speech_logits.shape[-1]),
            speech_tokens[:, 1:].reshape(-1),
            ignore_index=-1,
        )

        return {'loss': loss}

    @torch.inference_mode()
    def inference(
        self,
        text_token_ids: torch.Tensor,          # (1, L_text)
        prompt_speech_tokens: torch.Tensor,    # (1, L_prompt_speech)
        max_length: int = 4096,                # 最大生成長
        top_p: float = 0.8,                    # Nucleus sampling
        top_k: int = 25,                       # Top-K sampling
    ) -> torch.Tensor:
        """
        推論: テキストから音声トークン列を自己回帰生成

        入力:
            text_token_ids: (1, L_text) - テキストトークンID
            prompt_speech_tokens: (1, L_prompt_speech) - プロンプト音声トークン
            max_length: 最大生成長 (デフォルト4096)
            top_p: Nucleus samplingの閾値
            top_k: Top-K samplingの上位K個

        出力:
            generated_tokens: (1, T_speech) - 生成された音声トークン
                各値 ∈ [0, 6560]
                T_speech は可変 (EOS検出まで)

        ========================================
        推論時のシーケンス構成
        ========================================

        初期コンテキスト:
        ┌─────┬──────────┬──────────────────┐
        │ SOS │ text     │ prompt_speech    │
        │ (1) │ (L_text) │ (L_prompt_speech)│
        └─────┴──────────┴──────────────────┘

        自己回帰生成 (1トークンずつ):
        ┌─────┬──────────┬──────────────────┬──────────────┐
        │ SOS │ text     │ prompt_speech    │ generated    │
        │ (1) │ (L_text) │ (L_prompt_speech)│ (1,2,...,t)  │
        └─────┴──────────┴──────────────────┴──────────────┘
        ※ KV cacheにより既存部分は再計算不要
        """
        # Step 1: テキストエンコーディング
        text_embeds = self.text_embedding(text_token_ids)
        text_embeds = self.text_encoder(text_embeds)
        text_embeds = self.text_encoder_affine_layer(text_embeds)
        # text_embeds: (1, L_text, 896)

        # Step 2: プロンプト音声の埋め込み
        prompt_embeds = self.speech_embedding(prompt_speech_tokens)
        # prompt_embeds: (1, L_prompt_speech, 896)

        # Step 3: 初期コンテキスト構築
        sos_embed = self.speech_embedding(
            torch.tensor([[self.sos_token_id]], device=text_embeds.device)
        )
        # sos_embed: (1, 1, 896)

        context = torch.cat([
            sos_embed,       # (1, 1, 896)
            text_embeds,     # (1, L_text, 896)
            prompt_embeds,   # (1, L_prompt_speech, 896)
        ], dim=1)
        # context: (1, 1 + L_text + L_prompt_speech, 896)

        # Step 4: KVキャッシュで初期コンテキストを処理
        hidden, kv_cache = self.llm.forward_with_cache(context)
        # hidden: (1, 1 + L_text + L_prompt_speech, 896)
        # kv_cache: 各レイヤーの (key, value) テンソル

        # 最後の位置から最初のトークンを予測
        logits = self.llm_decoder(hidden[:, -1:, :])
        # logits: (1, 1, 6761)

        # Step 5: 自己回帰生成ループ
        generated_tokens = []

        for step in range(max_length):
            # RAS (Repetition-Aware Sampling)
            token_id = ras_sampling(
                logits[:, -1, :self.speech_token_size],
                top_p=top_p,
                top_k=top_k,
                generated_tokens=generated_tokens,
                win_size=10,   # 繰り返し検出ウィンドウ
                tau_r=0.1,     # 繰り返しペナルティ
            )
            # token_id: (1,) - サンプリングされたトークンID

            # EOS検出
            if token_id.item() == self.eos_token_id:
                break

            generated_tokens.append(token_id.item())

            # 次ステップの入力
            next_embed = self.speech_embedding(token_id.unsqueeze(0))
            # next_embed: (1, 1, 896)

            # KVキャッシュで1ステップ推論
            hidden, kv_cache = self.llm.forward_one_step(
                next_embed, kv_cache
            )
            # hidden: (1, 1, 896)

            logits = self.llm_decoder(hidden)
            # logits: (1, 1, 6761)

        result = torch.tensor(generated_tokens, device=text_embeds.device)
        return result.unsqueeze(0)
        # 出力: (1, T_speech) - 生成された音声トークン


class Qwen2ForCausalLM(nn.Module):
    """
    Qwen2ベースのTransformer Decoder

    事前学習済みQwen2をベースに使用。
    Grouped Query Attention (GQA) で効率的な推論。

    0.5Bモデル:
      - 22レイヤー, 14ヘッド, 2 KVヘッド, 隠れ次元896
    1.5Bモデル:
      - 28レイヤー, 12ヘッド, 2 KVヘッド, 隠れ次元1536
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_layers: int = 22,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D_llm)
    ) -> torch.Tensor:
        """
        入力: x (B, T, 896) - 埋め込みシーケンス
        出力: (B, T, 896) - LLM出力
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward_with_cache(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List]:
        """初期コンテキスト処理 + KVキャッシュ生成"""
        kv_cache = []
        for layer in self.layers:
            x, cache = layer.forward_with_cache(x)
            kv_cache.append(cache)
        return self.norm(x), kv_cache

    def forward_one_step(
        self,
        x: torch.Tensor,       # (B, 1, D_llm)
        kv_cache: List,
    ) -> Tuple[torch.Tensor, List]:
        """KVキャッシュを使った1ステップ推論"""
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, cache = layer.forward_one_step(x, kv_cache[i])
            new_cache.append(cache)
        return self.norm(x), new_cache


class Qwen2DecoderLayer(nn.Module):
    """
    Qwen2デコーダレイヤー

    構成:
    1. RMSNorm → GQA Self-Attention → 残差接続
    2. RMSNorm → SwiGLU FFN → 残差接続
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
    ):
        super().__init__()

        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,       # 14
            num_kv_heads=num_key_value_heads,    # 2 (GQA)
        )
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力/出力: (B, T, 896)
        """
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    14個のクエリヘッドに対して2個のKVヘッドを共有。
    メモリ効率とスループットの向上。

    入力: (B, T, 896)
    内部:
      Q: (B, 14, T, 64) - 14ヘッド × 64次元
      K: (B, 2, T, 64)  - 2 KVヘッド × 64次元
      V: (B, 2, T, 64)  - 2 KVヘッド × 64次元
      KVヘッドをクエリヘッド数に拡張 (repeat_interleave)
    出力: (B, T, 896)
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_kv_heads: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads  # 64
        self.num_kv_groups = num_heads // num_kv_heads  # 7

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim)       # 896 → 896
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)    # 896 → 128
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)    # 896 → 128
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size)       # 896 → 896

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, 14, T, 64)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # k: (B, 2, T, 64)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v: (B, 2, T, 64)

        # KVヘッドをクエリヘッド数に拡張: 2 → 14 (各KVヘッドを7回繰り返し)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)  # (B, 14, T, 64)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)  # (B, 14, T, 64)

        # Scaled Dot-Product Attention (因果マスク付き)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # attn_out: (B, 14, T, 64)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        # attn_out: (B, T, 896)

        return self.o_proj(attn_out)
        # 出力: (B, T, 896)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network

    SwiGLU(x) = (xW_1) ⊗ SiLU(xW_gate) を適用後 W_2 で射影

    入力: (B, T, 896)
    中間: (B, T, 4864)  ← gate と up を並列計算
    出力: (B, T, 896)
    """

    def __init__(self, hidden_size: int = 896, intermediate_size: int = 4864):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)  # 896 → 4864
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)    # 896 → 4864
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)  # 4864 → 896

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 896)
        gate = F.silu(self.gate_proj(x))  # (B, T, 4864)
        up = self.up_proj(x)               # (B, T, 4864)
        return self.down_proj(gate * up)   # (B, T, 896)
        # SwiGLU: SiLU(xW_gate) ⊗ (xW_up) → W_down


class ConformerEncoder(nn.Module):
    """
    Conformer Text Encoder

    テキスト埋め込みをエンコードする軽量Conformer。
    Self-Attention + Convolution + FFN の組み合わせ。

    入力: (B, L_text, 512)
    出力: (B, L_text, 512)

    構成:
    - 6レイヤー
    - 4ヘッド
    - FFN中間次元: 2048
    - Depthwise Separable Convolution
    """

    def __init__(
        self,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_dim: int = 2048,
        conv_kernel_size: int = 31,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(dim, num_heads, ff_dim, conv_kernel_size)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L_text, 512)
        for layer in self.layers:
            x = layer(x)
        return x
        # 出力: (B, L_text, 512)


class ConformerBlock(nn.Module):
    """
    Conformer Block = FFN(½) + MHSA + Conv + FFN(½) + LayerNorm

    入力/出力: (B, T, dim)
    """

    def __init__(self, dim: int, num_heads: int, ff_dim: int, conv_kernel_size: int):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(ff_dim, dim), nn.Dropout(0.1),
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 1),  # Pointwise expand
            nn.GLU(dim=1),                # (B, dim, T)
            nn.Conv1d(dim, dim, conv_kernel_size, padding=conv_kernel_size // 2, groups=dim),  # Depthwise
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),      # Pointwise compress
            nn.Dropout(0.1),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(ff_dim, dim), nn.Dropout(0.1),
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        x = x + 0.5 * self.ff1(x)                                    # Half-step FFN
        x_norm = self.attn_norm(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]                # MHSA
        x_norm = self.conv_norm(x).transpose(1, 2)                    # (B, dim, T)
        x = x + self.conv(x_norm).transpose(1, 2)                    # Convolution
        x = x + 0.5 * self.ff2(x)                                    # Half-step FFN
        return self.final_norm(x)
        # 出力: (B, T, dim)


def ras_sampling(
    logits: torch.Tensor,          # (B, Q) - 音声トークンの未正規化スコア
    top_p: float = 0.8,            # Nucleus sampling閾値
    top_k: int = 25,               # Top-K sampling
    generated_tokens: List[int] = None,  # これまで生成されたトークン
    win_size: int = 10,            # 繰り返し検出ウィンドウサイズ
    tau_r: float = 0.1,            # 繰り返しペナルティ温度
) -> torch.Tensor:
    """
    RAS (Repetition-Aware Sampling)

    通常のTop-P/Top-K samplingに加えて、
    直近のwin_size個のトークン内での繰り返しを検出し、
    繰り返しトークンにペナルティを与える。

    入力:
        logits: (B, Q) - 各音声トークンの未正規化スコア
            B: バッチサイズ (通常1)
            Q: 音声トークン語彙サイズ (6561)

    出力:
        token_id: (B,) - サンプリングされたトークンID

    処理:
    1. 直近win_sizeトークンで繰り返しをチェック
    2. 繰り返しトークンのlogitsにペナルティ (tau_rで温度調整)
    3. Top-K → Top-P フィルタリング
    4. Softmax → Categorical分布からサンプリング
    """
    # 繰り返しペナルティ
    if generated_tokens and len(generated_tokens) >= win_size:
        recent = generated_tokens[-win_size:]
        for token in set(recent):
            count = recent.count(token)
            if count > 1:
                logits[:, token] -= count * tau_r

    # Top-K フィルタリング
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    # top_k_values: (B, top_k)

    # Softmax
    probs = F.softmax(top_k_values, dim=-1)
    # probs: (B, top_k)

    # Top-P (Nucleus) フィルタリング
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # サンプリング
    sampled_idx = torch.multinomial(sorted_probs, 1)
    # sampled_idx: (B, 1)

    token_id = top_k_indices.gather(-1, sorted_indices.gather(-1, sampled_idx))
    return token_id.squeeze(-1)
    # token_id: (B,)
