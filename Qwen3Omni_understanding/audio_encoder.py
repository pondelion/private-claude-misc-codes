"""
Qwen3-Omni Audio Encoder: AuT (Audio Transformer)
===================================================

20M時間の教師あり音声データでスクラッチ学習された
attention-encoder-decoder ベースの音声エンコーダ。

Qwen2.5-Omni の Whisper-large-v3 を完全に置き換える。

主な差分 (vs Qwen2.5-Omni):
    - Whisper → AuT (スクラッチ学習, encoder-decoder)
    - ダウンサンプリング: Conv1D stride=2×2 (4倍) → Conv2D×3 (8倍)
    - ブロックアテンション: 固定2秒 → 動的1-8秒ウィンドウ
    - トークンレート: 25Hz (40ms) → 12.5Hz (80ms)
    - 学習データ: 80% 中英ASR, 10% 他言語ASR, 10% 音声理解

アーキテクチャ (Figure 3 in paper):
    Encoder:
        3× Downsampling Conv2D (8倍ダウンサンプリング)
        32× Self-Attention Layer (flash attention, 動的ウィンドウ 1-8秒)
    Decoder:
        8× {Decoder Self-Attention + Decoder Cross-Attention}

入力:
    - 16kHz 音声 → 128チャネルメルスペクトログラム (25ms window, 10ms hop)
    - input_features: (B, 128, T_mel) - FBank特徴量

出力:
    - AuT Hidden: (B, T_tokens, d_model) - 12.5Hz トークン列
    - 各トークンは約80ms分の音声情報を表現

パラメータ数: ~650M (0.6B)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# Downsampling Conv2D Block
# ============================================

class DownsamplingConv2d(nn.Module):
    """
    2D畳み込みによるダウンサンプリングブロック

    AuTでは3段のConv2Dで合計8倍ダウンサンプリング:
        Stage 1: stride=2 → 2倍ダウン
        Stage 2: stride=2 → 2倍ダウン (累計4倍)
        Stage 3: stride=2 → 2倍ダウン (累計8倍)

    入力shape: (B, C_in, freq_bins, T_mel)
    出力shape: (B, C_out, freq_bins // stride, T_mel // stride)

    ※ Whisper (Qwen2.5-Omni) は Conv1D stride=2 ×2 で4倍だったが、
      AuT は Conv2D ×3 で8倍ダウンサンプリング → トークンレート 12.5Hz
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        """
        入力: (B, C_in, F, T)
        出力: (B, C_out, F//stride, T//stride)
        """
        x = self.conv(x)       # (B, C_out, F', T')
        # LayerNorm は最後の次元に適用するため転置
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1)   # (B, F', T', C)
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)   # (B, C, F', T')
        return x


class DownsamplingStack(nn.Module):
    """
    3段の Conv2D ダウンサンプリングスタック

    メルスペクトログラム → 8倍ダウンサンプリング
    最終的に周波数軸をフラットにして d_model 次元にする

    入力: (B, 128, T_mel) - メルスペクトログラム
    出力: (B, T_mel//8, d_model) - ダウンサンプリング済みトークン列
    """

    def __init__(self, num_mel_bins=128, d_model=768):
        super().__init__()
        # 3段のConv2D: 各 stride=2 → 合計 8倍
        self.conv1 = DownsamplingConv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = DownsamplingConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = DownsamplingConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # 周波数軸をフラットにして d_model に射影
        freq_after_3_downs = num_mel_bins // 8  # 128 // 8 = 16
        self.proj = nn.Linear(256 * freq_after_3_downs, d_model)

    def forward(self, input_features):
        """
        入力: input_features (B, 128, T_mel)
        出力: (B, T_mel//8, d_model)
        """
        # Conv2D 用に reshape: (B, 1, 128, T_mel)
        x = input_features.unsqueeze(1)  # (B, 1, 128, T_mel)

        x = self.conv1(x)  # (B, 64, 64, T_mel//2)
        x = self.conv2(x)  # (B, 128, 32, T_mel//4)
        x = self.conv3(x)  # (B, 256, 16, T_mel//8)

        B, C, F, T = x.shape
        # (B, C, F, T) → (B, T, C*F) → 射影 → (B, T, d_model)
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x = self.proj(x)  # (B, T_mel//8, d_model)

        return x


# ============================================
# Positional Encoding
# ============================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦波位置エンコーディング

    Whisper と同様の正弦波PE。
    max_source_positions で最大系列長を制限。
    """

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, length):
        """
        入力: length (int)
        出力: (length, d_model) - 位置エンコーディング
        """
        return self.pe[:length]


# ============================================
# Encoder Self-Attention Layer
# ============================================

class AuTEncoderLayer(nn.Module):
    """
    AuT Encoder の Self-Attention Layer

    Flash Attention + 動的ウィンドウサイズ (1-8秒)
    - オフラインタスク: ウィンドウ 1-8秒のパターンで prefill キャッシング
    - リアルタイム: ブロック単位でストリーミング処理

    Qwen2.5-Omni の固定2秒ブロックから動的ウィンドウに変更。
    """

    def __init__(self, d_model=768, num_heads=12, ffn_dim=3072, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        """
        入力: x (B, T, d_model)
        出力: x (B, T, d_model)

        実モデルでは flash attention + cu_seqlens でブロック単位処理
        """
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


# ============================================
# Decoder Layer (Cross-Attention + Self-Attention)
# ============================================

class AuTDecoderLayer(nn.Module):
    """
    AuT Decoder Layer

    Decoder Self-Attention + Decoder Cross-Attention
    8層スタックで ASR / 音声理解の両タスクに対応

    入力:
        x: (B, T_dec, d_model) - デコーダ入力
        encoder_hidden: (B, T_enc, d_model) - エンコーダ出力
    """

    def __init__(self, d_model=768, num_heads=12, ffn_dim=3072, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, encoder_hidden, self_attn_mask=None, cross_attn_mask=None):
        """
        入力:
            x: (B, T_dec, d_model)
            encoder_hidden: (B, T_enc, d_model)
        出力:
            x: (B, T_dec, d_model)
        """
        # Self-Attention
        residual = x
        x = self.layer_norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = residual + x

        # Cross-Attention (encoder_hidden を参照)
        residual = x
        x = self.layer_norm2(x)
        x, _ = self.cross_attn(x, encoder_hidden, encoder_hidden, attn_mask=cross_attn_mask)
        x = residual + x

        # FFN
        residual = x
        x = self.layer_norm3(x)
        x = self.ffn(x)
        x = residual + x

        return x


# ============================================
# AuT Encoder
# ============================================

class AuTEncoder(nn.Module):
    """
    AuT (Audio Transformer) Encoder

    3× Downsampling Conv2D + 32× Self-Attention Layer

    入力: input_features (B, 128, T_mel) - 128チャネルメルスペクトログラム
          feature_lens   (B,)             - 各音声の有効フレーム数

    出力: encoder_hidden (B, T_tokens, d_model)
          T_tokens = T_mel // 8  (8倍ダウンサンプリング)

    トークンレート: 12.5Hz (80ms/token)
    - 10ms frame shift × 8倍ダウン = 80ms/token
    """

    def __init__(
        self,
        num_mel_bins=128,
        d_model=768,
        encoder_layers=32,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        output_dim=1024,
        max_source_positions=10000,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # 3× Conv2D ダウンサンプリング (合計8倍)
        self.downsampling = DownsamplingStack(num_mel_bins, d_model)

        # 位置エンコーディング
        self.positional_embedding = SinusoidalPositionalEncoding(d_model, max_source_positions)

        # 32× Self-Attention Layer
        self.layers = nn.ModuleList([
            AuTEncoderLayer(d_model, encoder_attention_heads, encoder_ffn_dim)
            for _ in range(encoder_layers)
        ])

        # 出力射影
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, input_features, feature_lens):
        """
        入力:
            input_features: (B, 128, T_mel) - メルスペクトログラム
            feature_lens:   (B,)            - 各音声の有効メルフレーム数

        出力:
            encoder_output: (B, T_tokens, output_dim)
            T_tokens = T_mel // 8

        処理フロー:
            (B, 128, T_mel)
              → 3× Conv2D ダウンサンプリング: (B, T_mel//8, d_model)
              → + 位置エンコーディング
              → 32× Self-Attention: (B, T_tokens, d_model)
              → LayerNorm + Linear: (B, T_tokens, output_dim)
        """
        # ダウンサンプリング
        x = self.downsampling(input_features)  # (B, T_mel//8, d_model)

        T_tokens = x.shape[1]

        # 位置エンコーディング追加
        pos_emb = self.positional_embedding(T_tokens)  # (T_tokens, d_model)
        x = x + pos_emb.unsqueeze(0)

        # 32× Self-Attention
        for layer in self.layers:
            x = layer(x)

        # 出力射影
        x = self.layer_norm(x)
        x = self.output_proj(x)  # (B, T_tokens, output_dim)

        return x


# ============================================
# AuT Decoder
# ============================================

class AuTDecoder(nn.Module):
    """
    AuT Decoder (8層)

    Decoder Self-Attention + Cross-Attention で
    encoder 出力を参照しながら ASR / 音声理解タスクに対応

    AuT の学習時に使用。推論時は encoder 出力を LLM に渡す。
    """

    def __init__(
        self,
        d_model=768,
        decoder_layers=8,
        decoder_attention_heads=12,
        decoder_ffn_dim=3072,
        vocab_size=50000,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = SinusoidalPositionalEncoding(d_model, 5000)

        self.layers = nn.ModuleList([
            AuTDecoderLayer(d_model, decoder_attention_heads, decoder_ffn_dim)
            for _ in range(decoder_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, encoder_hidden):
        """
        入力:
            input_ids:      (B, T_dec) - デコーダ入力トークンID
            encoder_hidden: (B, T_enc, d_model) - エンコーダ出力

        出力:
            logits: (B, T_dec, vocab_size)
        """
        x = self.embed_tokens(input_ids)  # (B, T_dec, d_model)
        T_dec = x.shape[1]
        pos_emb = self.positional_embedding(T_dec)
        x = x + pos_emb.unsqueeze(0)

        for layer in self.layers:
            x = layer(x, encoder_hidden)

        x = self.layer_norm(x)
        logits = self.lm_head(x)  # (B, T_dec, vocab_size)
        return logits


# ============================================
# Complete AuT Model
# ============================================

class AudioTransformer(nn.Module):
    """
    AuT (Audio Transformer) 完全モデル

    Encoder-Decoder アーキテクチャ
    - Encoder: 3× Conv2D + 32× Self-Attention → 12.5Hz トークン列
    - Decoder: 8× (Self-Attn + Cross-Attn) → ASR / 音声理解

    Qwen3-Omni では Encoder 出力を LLM (Thinker) に渡す。
    Decoder は AuT の事前学習時に使用。

    パラメータ: ~650M
    """

    def __init__(
        self,
        num_mel_bins=128,
        d_model=768,
        encoder_layers=32,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        decoder_layers=8,
        decoder_attention_heads=12,
        decoder_ffn_dim=3072,
        output_dim=1024,
    ):
        super().__init__()
        self.encoder = AuTEncoder(
            num_mel_bins=num_mel_bins,
            d_model=d_model,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            output_dim=output_dim,
        )
        self.decoder = AuTDecoder(
            d_model=d_model,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
        )

    def get_encoder_output(self, input_features, feature_lens):
        """
        Qwen3-Omni の推論時に使用: Encoder 出力のみ返す

        入力:
            input_features: (B, 128, T_mel)
            feature_lens:   (B,)

        出力:
            encoder_output: (B, T_tokens, output_dim)
            T_tokens = T_mel // 8, 12.5Hz
        """
        return self.encoder(input_features, feature_lens)


# ============================================
# 使用例
# ============================================

def example_aut_encoder():
    """
    AuT Encoder の使用例

    実際にモジュールをインスタンス化し、ダミー入力で
    フォワードパスを実行して各ステージの形状を確認する
    """

    # --- 縮小版で初期化 (実モデルは encoder_layers=32) ---
    encoder = AuTEncoder(
        num_mel_bins=128,
        d_model=256,          # 実モデルは768
        encoder_layers=4,     # 実モデルは32
        encoder_attention_heads=4,
        encoder_ffn_dim=512,
        output_dim=512,       # 実モデルは1024
    )
    encoder.eval()

    # --- ダミー入力: 5秒の音声 ---
    duration_sec = 5.0
    T_mel = int(duration_sec * 100)  # 10ms hop → 500 フレーム
    B = 1

    input_features = torch.randn(B, 128, T_mel)
    feature_lens = torch.tensor([T_mel])

    # --- ダウンサンプリング確認 ---
    with torch.no_grad():
        ds_out = encoder.downsampling(input_features)
    T_tokens = T_mel // 8  # 500 // 8 = 62
    assert ds_out.shape == (B, T_tokens, 256), f"Expected (1, {T_tokens}, 256), got {ds_out.shape}"

    # --- フルフォワードパス ---
    with torch.no_grad():
        output = encoder(input_features, feature_lens)
    assert output.shape == (B, T_tokens, 512)

    # --- 結果表示 ---
    print(f"[AuT Encoder 使用例]")
    print(f"  入力: input_features {input_features.shape}  (B, mel_bins, T_mel)")
    print(f"         feature_lens  {feature_lens.shape}   = [{T_mel}]")
    print(f"  音声長: {duration_sec}秒")
    print()
    print(f"  3× Conv2D ダウンサンプリング後: {ds_out.shape}  (B, T_mel//8, d_model)")
    print(f"  32× Self-Attention後:            同shape")
    print(f"  LayerNorm+出力射影:             {output.shape}  (B, T_tokens, output_dim)")
    print()
    print(f"  ダウンサンプリング率: {T_mel} → {T_tokens} ({T_mel/T_tokens:.1f}倍)")
    print(f"  トークンレート: 12.5Hz (80ms/token)")
    print(f"  1トークン ≈ {duration_sec * 1000 / T_tokens:.1f}ms")
    print()
    print(f"  [vs Qwen2.5-Omni (Whisper)]")
    print(f"    ダウンサンプリング: Conv1D ×2 (4倍) → Conv2D ×3 (8倍)")
    print(f"    トークンレート: 25Hz → 12.5Hz")
    print(f"    エンコーダ層数: 12 → 32")
    print(f"    ウィンドウ: 固定2秒 → 動的1-8秒")

    # --- バッチ入力 ---
    B2 = 2
    T_mel_max = 600
    input_features_2 = torch.randn(B2, 128, T_mel_max)
    feature_lens_2 = torch.tensor([300, 600])  # 3秒, 6秒

    with torch.no_grad():
        output_2 = encoder(input_features_2, feature_lens_2)

    T_tokens_max = T_mel_max // 8
    assert output_2.shape == (B2, T_tokens_max, 512)

    print()
    print(f"  [バッチ入力 (2音声)]")
    print(f"    音声1: 300フレーム (3秒) → {300//8} トークン")
    print(f"    音声2: 600フレーム (6秒) → {600//8} トークン")
    print(f"    パディング済み出力: {output_2.shape}")


if __name__ == "__main__":
    example_aut_encoder()
