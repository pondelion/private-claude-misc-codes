"""
Qwen2.5-Omni Talker - 簡略化疑似コード
========================================

Thinkerの隠れ状態から音声コードトークンを生成する (Talker = "口")

Thinkerの高レベル表現を受け取り、音声コーデック(qwen-tts-tokenizer)の
離散トークンを自己回帰的に生成

公式実装: modeling_qwen2_5_omni_low_VRAM_mode.py (Lines 2934-3108)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


class TalkerForConditionalGeneration(nn.Module):
    """
    Qwen2.5-Omni Talker

    Thinkerの隠れ状態(テキストのセマンティック表現)を受け取り、
    音声コードトークンを自己回帰的に生成するDual-Track Transformer Decoder

    アーキテクチャ:
        Thinker隠れ状態 (B, L_text, 4096)
        + codecトークン埋め込み (B, 1, hidden)   ← 前ステップの生成トークン
        → 加算融合
        → thinker_to_talker_proj (線形射影)
        → Talker Transformer Decoder (32層)
        → codec_head → 音声コードトークン予測

    Dual-Track:
        - Track 1: Thinkerの隠れ状態 (セマンティック情報)
        - Track 2: 生成済みcodecトークン (音響情報)
        → 両方を加算で融合して次トークンを予測

    ★ Thinkerの隠れ状態は1トークンずつ消費される
        → テキスト生成と音声生成が並行してストリーミング可能
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        talker_hidden_size: int = 2048,
        num_layers: int = 32,
        num_heads: int = 16,
        codebook_size: int = 8295,
        embedding_size: int = 4096,
    ):
        """
        パラメータ:
            hidden_size: Thinkerの隠れ次元 (4096)
            talker_hidden_size: Talker内部の隠れ次元 (2048)
            num_layers: Talker Transformerレイヤー数 (32)
            num_heads: アテンションヘッド数 (16)
            codebook_size: 音声コードブックサイズ (8295)
            embedding_size: Thinker出力の埋め込み次元 (4096)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.codebook_size = codebook_size

        # ========================================
        # 特殊トークン
        # ========================================
        self.codec_bos_token_id = 8292    # 音声コード開始トークン
        self.codec_eos_token_id = 8294    # 音声コード終了トークン
        self.codec_pad_token_id = 8293    # パディング
        self.codec_mask_token_id = 8291   # マスク

        self.text_bos_token_id = 0
        self.text_eos_token_id = 1
        self.text_pad_token_id = 2

        # ========================================
        # コンポーネント
        # ========================================

        # Thinker → Talker の次元変換
        self.thinker_to_talker_proj = nn.Linear(embedding_size, talker_hidden_size)
        # 入力: (B, L, 4096) → 出力: (B, L, 2048)

        # コードトークン埋め込み
        self.codec_embed_tokens = nn.Embedding(codebook_size, embedding_size)
        # codebook_size → 4096

        # Talker Transformer Decoder
        self.model = TalkerTransformerModel(
            hidden_size=talker_hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # 音声コードヘッド
        self.codec_head = nn.Linear(talker_hidden_size, codebook_size, bias=False)
        # (B, 1, 2048) → (B, 1, 8295)

    def forward(
        self,
        input_ids: torch.Tensor,
        thinker_reply_part: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        cache_position: Optional[torch.Tensor] = None,
        input_text_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Talker のフォワードパス

        入力:
            input_ids: (B, 1) - 前ステップで生成されたcodecトークンID
            thinker_reply_part: (B, L_remaining, 4096) - 残りのThinker隠れ状態
                ★ 各ステップで先頭1トークンが消費され、残りが返される
            inputs_embeds: (B, L, hidden) - 入力埋め込み (初期化時のみ使用)
            position_ids: (3, B, L) - TMRoPE位置ID
            attention_mask: (B, L) - アテンションマスク
            past_key_values: KVキャッシュ
            cache_position: (L,) - キャッシュ内の位置
            input_text_ids: (B, L) - TMRoPE計算用のテキストトークンID

        出力:
            Dict {
                'logits': (B, 1, codebook_size) - 次のcodecトークンの確率分布
                'thinker_reply_part': (B, L_remaining-1, 4096) - 残りのThinker隠れ状態
                'past_key_values': KVキャッシュ
            }
        """

        # ========================================
        # Step 1: Prefill 初期化 (最初のステップ)
        # ========================================
        if cache_position is not None and cache_position[0] == 0:
            # 初期化: Thinkerの履歴コンテキスト + BOS/PADトークン

            # inputs_embeds の最後の2位置にBOS/PADを配置
            # [-2] → codec_pad_token 埋め込み
            # [-1] → codec_bos_token 埋め込み
            bos_embed = self.codec_embed_tokens(
                torch.tensor([self.codec_bos_token_id])
            )
            pad_embed = self.codec_embed_tokens(
                torch.tensor([self.codec_pad_token_id])
            )

            if inputs_embeds is not None:
                inputs_embeds[:, -1, :] += bos_embed
                inputs_embeds[:, -2, :] += pad_embed

            # inputs_embeds: (B, L_context, 4096) - Thinkerの全コンテキスト + BOS

        # ========================================
        # Step 2: 自己回帰ステップ (2回目以降)
        # ========================================
        else:
            # 前ステップのcodecトークンを埋め込み
            codec_embeds = self.codec_embed_tokens(input_ids)
            # codec_embeds: (B, 1, 4096)

            # ★ Thinker隠れ状態と加算融合
            # Thinkerの先頭1トークンをcodec埋め込みに加算
            thinker_current = thinker_reply_part[:, :1, :]
            # thinker_current: (B, 1, 4096) - 現在のThinkerトークン

            inputs_embeds = codec_embeds + thinker_current
            # inputs_embeds: (B, 1, 4096)
            # → codecの音響情報 + Thinkerのセマンティック情報 の融合

            # Thinker隠れ状態を1トークン消費
            thinker_reply_part = thinker_reply_part[:, 1:, :]
            # thinker_reply_part: (B, L_remaining-1, 4096)

        # ========================================
        # Step 3: Thinker → Talker 次元変換
        # ========================================
        talker_input = self.thinker_to_talker_proj(inputs_embeds)
        # talker_input: (B, 1, 2048) ← 4096 → 2048

        # ========================================
        # Step 4: Talker Transformer
        # ========================================
        outputs = self.model(
            inputs_embeds=talker_input,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        hidden_states = outputs['hidden_states']
        # hidden_states: (B, 1, 2048)

        # ========================================
        # Step 5: Codec Head
        # ========================================
        logits = self.codec_head(hidden_states).float()
        # logits: (B, 1, 8295)

        return {
            'logits': logits,
            'thinker_reply_part': thinker_reply_part,
            'past_key_values': outputs.get('past_key_values'),
        }

    def generate(
        self,
        thinker_hidden_states: torch.Tensor,
        text_token_ids: torch.Tensor,
        speaker: str = "Chelsie",
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 40,
        top_p: float = 0.8,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> torch.Tensor:
        """
        音声コードトークンの自己回帰生成

        入力:
            thinker_hidden_states: (B, L_text, 4096) - Thinkerの隠れ状態
            text_token_ids: (B, L_text) - Thinkerが生成したテキストトークン
            speaker: str - 話者名 ("Chelsie", "Ethan" 等)
            max_new_tokens: int - 最大生成トークン数 (4096 ≈ 21秒)
            do_sample: bool - サンプリングするか
            top_k: int - Top-K サンプリング
            top_p: float - Top-P (Nucleus) サンプリング
            temperature: float - 温度パラメータ
            repetition_penalty: float - 繰り返しペナルティ

        出力:
            codec_tokens: (B, L_codec) - 生成された音声コードトークン系列

        生成パラメータの意味:
            max_new_tokens=4096: ~21秒の音声 (1トークン ≈ 5ms)
            top_k=40: 上位40トークンからサンプリング
            top_p=0.8: 累積確率80%以内からサンプリング
            temperature=0.9: やや高め (多様性確保)
            repetition_penalty=1.05: 軽い繰り返し抑制
        """

        B = thinker_hidden_states.shape[0]
        device = thinker_hidden_states.device

        # 生成済みトークンを保持
        generated_tokens = []
        thinker_remaining = thinker_hidden_states

        # BOS トークンから開始
        current_token = torch.tensor(
            [[self.codec_bos_token_id]], device=device
        ).expand(B, 1)

        # EOS トークンID
        eos_token_ids = {self.codec_eos_token_id, 8294}

        # 自己回帰ループ
        past_key_values = None
        for step in range(max_new_tokens):
            # フォワードパス
            outputs = self.forward(
                input_ids=current_token,
                thinker_reply_part=thinker_remaining,
                past_key_values=past_key_values,
            )

            logits = outputs['logits'][:, -1, :]
            # logits: (B, 8295)

            thinker_remaining = outputs['thinker_reply_part']
            past_key_values = outputs['past_key_values']

            # サンプリング
            if do_sample:
                # 温度スケーリング
                logits = logits / temperature

                # Top-K フィルタリング
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered

                # Top-P (Nucleus) フィルタリング
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # サンプリング
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_indices.gather(1, next_token)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # next_token: (B, 1)
            generated_tokens.append(next_token)

            # EOS チェック
            if next_token.item() in eos_token_ids:
                break

            # Thinker隠れ状態が尽きた場合
            if thinker_remaining.shape[1] == 0:
                break

            current_token = next_token

        # 生成トークンを結合
        codec_tokens = torch.cat(generated_tokens, dim=1)
        # codec_tokens: (B, L_codec)

        return codec_tokens


class TalkerTransformerModel(nn.Module):
    """
    Talker 内部の Transformer Decoder

    Thinkerと同様の構造だが、隠れ次元が小さい
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_layers: int = 32,
        num_heads: int = 16,
        intermediate_size: int = 8192,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TalkerDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        入力:
            inputs_embeds: (B, L, 2048)
        出力:
            Dict { 'hidden_states': (B, L, 2048), 'past_key_values': ... }
        """
        hidden_states = inputs_embeds
        new_past = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, kv = layer(hidden_states, past_kv)
            new_past.append(kv)

        hidden_states = self.norm(hidden_states)
        return {'hidden_states': hidden_states, 'past_key_values': new_past}


class TalkerDecoderLayer(nn.Module):
    """Talker の単一 Decoder レイヤー (簡略版)"""

    def __init__(self, hidden_size=2048, num_heads=16, intermediate_size=8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, past_kv=None):
        # Self-Attention + SwiGLU (簡略化)
        residual = x
        x = self.norm1(x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=1)
            v = torch.cat([past_kv[1], v], dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = residual + self.o_proj(attn)

        residual = x
        x = self.norm2(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        return x, (k, v)


# ============================================
# 使用例
# ============================================

def example_talker():
    """
    Talker の使用例

    TalkerForConditionalGeneration を縮小サイズでインスタンス化し、
    Dual-Track の融合・自己回帰生成の流れを実際に実行して確認する
    """

    # --- 縮小版 Talker ---
    # 実モデル: hidden=4096, talker_hidden=2048, layers=32, codebook=8295
    talker = TalkerForConditionalGeneration(
        hidden_size=256,
        talker_hidden_size=128,
        num_layers=2,
        num_heads=4,
        codebook_size=100,
        embedding_size=256,
    )
    talker.eval()

    B = 1
    L_text = 15  # Thinkerが生成したテキストトークン数

    # --- Thinkerの隠れ状態 (テキスト生成結果) ---
    thinker_hidden_states = torch.randn(B, L_text, 256)
    # thinker_hidden_states: (1, 15, 256) - Thinkerの最終隠れ状態

    # --- コンポーネントの形状確認 ---
    # thinker_to_talker_proj: 256 → 128
    proj_out = talker.thinker_to_talker_proj(thinker_hidden_states)
    assert proj_out.shape == (B, L_text, 128)

    # codec_embed_tokens: codebook_size → 256
    dummy_codec = torch.tensor([[talker.codec_bos_token_id]])
    codec_emb = talker.codec_embed_tokens(dummy_codec)
    assert codec_emb.shape == (1, 1, 256)

    # --- Dual-Track 融合の確認 ---
    # codecトークン埋め込み + Thinker隠れ状態の先頭1トークンを加算
    thinker_current = thinker_hidden_states[:, :1, :]  # (1, 1, 256)
    fused = codec_emb + thinker_current                 # (1, 1, 256) - 加算融合
    assert fused.shape == (1, 1, 256)

    # 射影
    talker_input = talker.thinker_to_talker_proj(fused)  # (1, 1, 128)
    assert talker_input.shape == (1, 1, 128)

    # Talker Transformer
    with torch.no_grad():
        transformer_out = talker.model(inputs_embeds=talker_input)
    assert transformer_out['hidden_states'].shape == (1, 1, 128)

    # codec_head
    logits = talker.codec_head(transformer_out['hidden_states'])
    assert logits.shape == (1, 1, 100)  # (B, 1, codebook_size)

    # --- 自己回帰生成 ---
    text_token_ids = torch.randint(0, 100, (B, L_text))

    with torch.no_grad():
        codec_tokens = talker.generate(
            thinker_hidden_states=thinker_hidden_states,
            text_token_ids=text_token_ids,
            max_new_tokens=20,  # 短く制限
            do_sample=True,
            top_k=10,
            temperature=0.9,
        )
    # codec_tokens: (1, L_codec) - 生成された音声コードトークン
    assert codec_tokens.shape[0] == B
    assert codec_tokens.shape[1] <= 20

    print(f"[Talker 使用例]")
    print(f"  モデル: hidden=256, talker_hidden=128, layers=2, codebook=100")
    print()
    print(f"  Thinker隠れ状態: {thinker_hidden_states.shape}  (B, L_text, hidden)")
    print()
    print(f"  [Dual-Track 融合]")
    print(f"    codec_embed:         {codec_emb.shape}      (B, 1, embedding_size)")
    print(f"    thinker_current:     {thinker_current.shape} (B, 1, hidden) ← 先頭1トークン")
    print(f"    加算融合:            {fused.shape}           codec + thinker")
    print(f"    thinker_to_talker:   {talker_input.shape}    (B, 1, talker_hidden)")
    print(f"    Transformer出力:     {transformer_out['hidden_states'].shape}")
    print(f"    codec_head logits:   {logits.shape}          (B, 1, codebook_size)")
    print()
    print(f"  [自己回帰生成]")
    print(f"    入力 L_text={L_text} → 生成 codec_tokens {codec_tokens.shape}")
    print(f"    生成トークン: {codec_tokens[0].tolist()}")
    print()
    print(f"  [実モデルの生成パラメータ]")
    print(f"    max_new_tokens=4096 (~21秒), top_k=40, top_p=0.8")
    print(f"    temperature=0.9, repetition_penalty=1.05")
    print(f"    EOS: codec_eos_token_id=8294")
    print(f"    Speaker: 'Chelsie'(女性), 'Ethan'(男性)")


if __name__ == "__main__":
    example_talker()
