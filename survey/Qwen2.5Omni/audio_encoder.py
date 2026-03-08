"""
Qwen2.5-Omni Audio Encoder - 簡略化疑似コード
===============================================

Whisper-large-v3 ベースの音声エンコーダ
16kHz音声 → 128チャネルメルスペクトログラム → 特徴ベクトル系列

公式実装: modeling_qwen2_5_omni_low_VRAM_mode.py (Lines 844-1014)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class SinusoidsPositionEmbedding(nn.Module):
    """
    正弦波ベースの位置エンコーディング (Whisper方式)

    学習不要の固定位置エンコーディング
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # max_len: 最大シーケンス長
        # d_model: 埋め込み次元

        position = torch.arange(0, max_len).unsqueeze(1).float()
        # position: (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        # div_term: (d_model // 2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数インデックス
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数インデックス
        # pe: (max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, length: int) -> torch.Tensor:
        """
        入力:
            length: int - 必要なシーケンス長

        出力:
            pe: (length, d_model) - 位置エンコーディング
        """
        return self.pe[:length]


class AudioEncoderLayer(nn.Module):
    """
    Audio Encoder の単一 Transformer レイヤー

    Whisper の Encoder Layer と同一構造
    ブロックワイズアテンション対応 (2秒ブロック)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )

        # Feed-Forward Network
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

        # Layer Normalization (Pre-Norm)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        入力:
            hidden_states: (total_tokens, d_model) - パック済みトークン
                total_tokens: 全チャンクの有効トークン数合計
                d_model: 768 (Whisper-large-v3)

            cu_seqlens: (num_chunks + 1,) - 累積シーケンス長
                Flash Attention用のチャンク境界情報
                例: [0, 50, 100, 145] → 3チャンク (50, 50, 45 トークン)

        出力:
            hidden_states: (total_tokens, d_model)

        注意:
            ブロックワイズアテンション: 各チャンク(2秒)内でのみアテンション計算
            cu_seqlens により異なる長さのチャンクを効率的にバッチ処理
        """

        residual = hidden_states
        # residual: (total_tokens, d_model)

        # ----------------------------------------
        # 1. Pre-Norm + Self-Attention
        # ----------------------------------------
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # hidden_states: (total_tokens, d_model)

        # Flash Attention (cu_seqlens使用時)
        # 実際にはflash_attn_varlen_funcを使用してブロック内アテンション
        # ここでは標準的なアテンションで簡略化
        hidden_states_unsqueeze = hidden_states.unsqueeze(1)  # (total_tokens, 1, d_model)
        attn_out, _ = self.self_attn(
            query=hidden_states_unsqueeze,
            key=hidden_states_unsqueeze,
            value=hidden_states_unsqueeze,
        )
        hidden_states = attn_out.squeeze(1)
        # hidden_states: (total_tokens, d_model)

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        # hidden_states: (total_tokens, d_model)

        # ----------------------------------------
        # 2. Pre-Norm + Feed-Forward
        # ----------------------------------------
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        # hidden_states: (total_tokens, d_model)

        hidden_states = F.gelu(self.fc1(hidden_states))
        # hidden_states: (total_tokens, dim_feedforward=3072)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        # hidden_states: (total_tokens, d_model=768)

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        # hidden_states: (total_tokens, d_model)

        return hidden_states


class AudioEncoder(nn.Module):
    """
    Qwen2.5-Omni Audio Encoder

    Whisper-large-v3 から初期化された音声エンコーダ
    16kHz音声をメルスペクトログラムに変換し、特徴ベクトル系列にエンコード

    アーキテクチャ:
        Conv1D (128 → 768, kernel=3) + GELU
        → Conv1D (768 → 768, kernel=3, stride=2) + GELU    ← 2倍ダウン
        → 正弦波位置エンコーディング追加
        → Transformer Encoder × 12 レイヤー (ブロックワイズアテンション)
        → AvgPool1D (stride=2)                              ← さらに2倍ダウン
        → LayerNorm
        → Linear (768 → 1024)                              ← LLM入力次元へ射影

    合計ダウンサンプリング: 4倍
        元のメル10msフレーム → 出力1トークン ≈ 40ms
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        d_model: int = 768,
        encoder_layers: int = 12,
        encoder_attention_heads: int = 12,
        encoder_ffn_dim: int = 3072,
        output_dim: int = 1024,
        max_source_positions: int = 1500,
        n_window: int = 50,
        dropout: float = 0.0,
    ):
        """
        パラメータ:
            num_mel_bins: メルスペクトログラムのビン数 (128)
            d_model: Transformer内部の隠れ次元 (768)
            encoder_layers: Transformerレイヤー数 (12)
            encoder_attention_heads: アテンションヘッド数 (12)
            encoder_ffn_dim: FFN中間次元 (3072)
            output_dim: 出力特徴次元 (1024, Thinker LLMの入力次元に合わせる)
            max_source_positions: 最大位置数 (1500)
            n_window: チャンク処理のウィンドウサイズ (50, 2秒=50フレーム)
            dropout: ドロップアウト率
        """
        super().__init__()

        self.d_model = d_model
        self.n_window = n_window

        # ========================================
        # 1. 畳み込み層 (メルスペクトログラム → 特徴系列)
        # ========================================
        self.conv1 = nn.Conv1d(
            in_channels=num_mel_bins,   # 128
            out_channels=d_model,       # 768
            kernel_size=3,
            padding=1,
        )
        # 入力: (B, 128, T_mel)
        # 出力: (B, 768, T_mel)  ← サイズ不変

        self.conv2 = nn.Conv1d(
            in_channels=d_model,        # 768
            out_channels=d_model,       # 768
            kernel_size=3,
            stride=2,                   # ★ 2倍ダウンサンプリング
            padding=1,
        )
        # 入力: (B, 768, T_mel)
        # 出力: (B, 768, T_mel//2)  ← 2倍ダウン

        # ========================================
        # 2. 位置エンコーディング (正弦波)
        # ========================================
        self.positional_embedding = SinusoidsPositionEmbedding(
            max_len=max_source_positions,
            d_model=d_model,
        )

        # ========================================
        # 3. Transformer Encoder
        # ========================================
        self.layers = nn.ModuleList([
            AudioEncoderLayer(
                d_model=d_model,
                num_heads=encoder_attention_heads,
                dim_feedforward=encoder_ffn_dim,
                dropout=dropout,
            )
            for _ in range(encoder_layers)
        ])

        # ========================================
        # 4. 後処理
        # ========================================
        self.ln_post = nn.LayerNorm(d_model)

        self.avg_pooler = nn.AvgPool1d(
            kernel_size=2,
            stride=2,                   # ★ さらに2倍ダウンサンプリング
        )
        # 入力: (B, d_model, T_mel//2)
        # 出力: (B, d_model, T_mel//4)  ← 合計4倍ダウン

        self.proj = nn.Linear(d_model, output_dim)
        # 入力: (B, T_mel//4, 768)
        # 出力: (B, T_mel//4, 1024)

        # ========================================
        # 5. 特殊トークン
        # ========================================
        self.audio_bos_eos_token = nn.Embedding(2, output_dim)
        # index 0: BOS (Beginning Of Speech)
        # index 1: EOS (End Of Speech)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        ダウンサンプリング後の出力長を計算

        入力:
            input_lengths: (B,) - メルスペクトログラムのフレーム数

        出力:
            output_lengths: (B,) - エンコーダ出力のトークン数

        計算:
            1. Conv2 (stride=2): (input_lengths - 1) // 2 + 1
            2. AvgPool (stride=2): (after_conv - 2) // 2 + 1
            合計: 約 input_lengths // 4
        """
        # Conv2 (stride=2) 後の長さ
        after_conv = (input_lengths - 1) // 2 + 1

        # AvgPool (stride=2) 後の長さ
        output_lengths = (after_conv - 2) // 2 + 1

        return output_lengths

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Audio Encoder のフォワードパス

        入力:
            input_features: (B, 128, T_mel) - メルスペクトログラム
                B: バッチサイズ (音声数)
                128: メルビン数
                T_mel: フレーム数 (= 音声秒数 × 100, 10msホップ)

            feature_lens: (B,) - 各音声の有効フレーム数

        出力:
            audio_tokens: (total_tokens, output_dim) - 全音声の結合トークン
                total_tokens: 全音声のトークン数合計 (+ BOS/EOS)
                output_dim: 1024

        処理フロー:
            1. チャンク分割 (n_window * 2 フレームごと)
            2. Conv1D × 2 (2倍ダウン)
            3. 位置エンコーディング追加
            4. Transformer Encoder (ブロックワイズアテンション)
            5. 音声ごとに結合 → AvgPool (2倍ダウン) → LayerNorm → 射影
            6. BOS/EOSトークン付加
        """
        B = input_features.shape[0]

        # ========================================
        # Step 1: チャンク分割
        # ========================================
        # 各音声を n_window * 2 フレームのチャンクに分割
        # 最後のチャンクは短い場合がある
        chunk_size = self.n_window * 2  # 100 フレーム = 1秒
        # (実際には 2秒 = n_window=50 × 2 = 100 フレーム @ 10ms/frame)

        all_chunks = []     # パディング済みチャンクリスト
        all_masks = []      # マスクリスト
        chunk_lens = []     # 各チャンクの有効長

        for b in range(B):
            valid_len = feature_lens[b].item()
            feat = input_features[b, :, :valid_len]
            # feat: (128, valid_len)

            num_chunks = (valid_len + chunk_size - 1) // chunk_size
            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, valid_len)
                chunk = feat[:, start:end]
                # chunk: (128, chunk_len) where chunk_len <= chunk_size
                all_chunks.append(chunk)
                chunk_lens.append(end - start)

        # ========================================
        # Step 2: パディングとバッチ化
        # ========================================
        max_chunk_len = max(chunk_lens)
        num_chunks_total = len(all_chunks)

        padded_features = torch.zeros(num_chunks_total, 128, max_chunk_len)
        padded_mask = torch.zeros(num_chunks_total, 1, max_chunk_len)  # Conv1用マスク

        for i, (chunk, clen) in enumerate(zip(all_chunks, chunk_lens)):
            padded_features[i, :, :clen] = chunk
            padded_mask[i, :, :clen] = 1.0

        # padded_features: (num_chunks, 128, max_chunk_len)
        # padded_mask: (num_chunks, 1, max_chunk_len)

        # ========================================
        # Step 3: 畳み込み層
        # ========================================

        # Conv1: (num_chunks, 128, max_chunk_len) → (num_chunks, 768, max_chunk_len)
        hidden = F.gelu(self.conv1(padded_features))
        # hidden: (num_chunks, 768, max_chunk_len)

        # マスク適用 (パディング部分をゼロに)
        hidden = hidden * padded_mask
        # hidden: (num_chunks, 768, max_chunk_len)

        # Conv2 (stride=2): (num_chunks, 768, max_chunk_len) → (num_chunks, 768, max_chunk_len//2)
        hidden = F.gelu(self.conv2(hidden))
        # hidden: (num_chunks, 768, max_chunk_len_after_conv)
        # max_chunk_len_after_conv = (max_chunk_len - 1) // 2 + 1

        # Conv後のチャンク長を計算
        aftercnn_lens = [(clen - 1) // 2 + 1 for clen in chunk_lens]

        # Conv後のマスク
        max_aftercnn = hidden.shape[2]
        padded_mask_after_cnn = torch.zeros(num_chunks_total, max_aftercnn, dtype=torch.bool)
        for i, alen in enumerate(aftercnn_lens):
            padded_mask_after_cnn[i, :alen] = True

        # ========================================
        # Step 4: 位置エンコーディング
        # ========================================

        # (num_chunks, 768, T_after_conv) → (num_chunks, T_after_conv, 768)
        hidden = hidden.permute(0, 2, 1)
        # hidden: (num_chunks, T_after_conv, 768)

        # 正弦波位置エンコーディング追加
        seq_len = hidden.shape[1]
        pos_emb = self.positional_embedding(seq_len)
        # pos_emb: (T_after_conv, 768)

        hidden = hidden + pos_emb.unsqueeze(0)
        # hidden: (num_chunks, T_after_conv, 768) + 位置情報

        # ========================================
        # Step 5: Flash Attention用パッキング
        # ========================================

        # 有効トークンのみ抽出 (パディング除去)
        packed_hidden = hidden[padded_mask_after_cnn]
        # packed_hidden: (total_valid_tokens, 768)
        # total_valid_tokens = sum(aftercnn_lens)

        # cu_seqlens の計算 (累積シーケンス長)
        cu_seqlens = torch.zeros(num_chunks_total + 1, dtype=torch.int32)
        for i, alen in enumerate(aftercnn_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + alen
        # cu_seqlens: (num_chunks + 1,)
        # 例: [0, 50, 100, 145] → チャンク長 50, 50, 45

        # ========================================
        # Step 6: Transformer Encoder
        # ========================================

        for layer in self.layers:
            packed_hidden = layer(packed_hidden, cu_seqlens=cu_seqlens)
            # packed_hidden: (total_valid_tokens, 768)
            # 各チャンク内でのみアテンション (ブロックワイズ)

        # ========================================
        # Step 7: 音声ごとの後処理
        # ========================================

        # チャンクを音声ごとに結合
        all_audio_tokens = []
        chunk_idx = 0

        for b in range(B):
            valid_len = feature_lens[b].item()
            num_chunks_b = (valid_len + chunk_size - 1) // chunk_size

            # この音声の全チャンクを結合
            audio_hidden_parts = []
            for c in range(num_chunks_b):
                start = cu_seqlens[chunk_idx].item()
                end = cu_seqlens[chunk_idx + 1].item()
                audio_hidden_parts.append(packed_hidden[start:end])
                chunk_idx += 1

            audio_hidden = torch.cat(audio_hidden_parts, dim=0)
            # audio_hidden: (T_after_conv_total, 768)

            # AvgPool (stride=2): さらに2倍ダウンサンプリング
            # (T, 768) → (1, 768, T) → AvgPool → (1, 768, T//2) → (T//2, 768)
            audio_hidden_pooled = self.avg_pooler(
                audio_hidden.unsqueeze(0).permute(0, 2, 1)
            ).permute(0, 2, 1).squeeze(0)
            # audio_hidden_pooled: (T_after_conv_total // 2, 768)
            # ≈ (T_mel // 4, 768) ← 合計4倍ダウン

            # LayerNorm
            audio_hidden_pooled = self.ln_post(audio_hidden_pooled)
            # audio_hidden_pooled: (T_mel//4, 768)

            # 線形射影 (768 → 1024)
            audio_tokens = self.proj(audio_hidden_pooled)
            # audio_tokens: (T_mel//4, 1024)

            # BOS/EOSトークン付加
            bos_token = self.audio_bos_eos_token(
                torch.tensor([0], device=audio_tokens.device)
            )
            # bos_token: (1, 1024)

            eos_token = self.audio_bos_eos_token(
                torch.tensor([1], device=audio_tokens.device)
            )
            # eos_token: (1, 1024)

            audio_tokens = torch.cat([bos_token, audio_tokens, eos_token], dim=0)
            # audio_tokens: (T_mel//4 + 2, 1024)

            all_audio_tokens.append(audio_tokens)

        # 全音声のトークンを結合
        token_audio = torch.cat(all_audio_tokens, dim=0)
        # token_audio: (total_audio_tokens, 1024)
        # total_audio_tokens = Σ(T_mel_i//4 + 2) for each audio

        return token_audio


# ============================================
# 使用例
# ============================================

def example_audio_encoder():
    """
    Audio Encoder の使用例

    実際にモジュールをインスタンス化し、ダミー入力で
    フォワードパスを実行して各ステージの形状を確認する
    """

    # --- 初期化 ---
    encoder = AudioEncoder(
        num_mel_bins=128,
        d_model=768,
        encoder_layers=12,
        encoder_attention_heads=12,
        encoder_ffn_dim=3072,
        output_dim=1024,
        max_source_positions=1500,
        n_window=50,
    )
    encoder.eval()

    # --- ダミー入力: 5秒の音声 ---
    duration_sec = 5.0
    T_mel = int(duration_sec * 100)  # 10ms hop → 500 フレーム
    B = 1

    input_features = torch.randn(B, 128, T_mel)
    # input_features: (1, 128, 500)

    feature_lens = torch.tensor([T_mel])
    # feature_lens: (1,) = [500]

    # --- フォワードパス ---
    with torch.no_grad():
        output = encoder(input_features, feature_lens)
    # output: (total_tokens, 1024) = (BOS + T_mel//4 + EOS, 1024)

    # --- 各ステージの形状確認 ---
    # Conv1: サイズ不変
    conv1_out = torch.nn.functional.gelu(encoder.conv1(input_features))
    assert conv1_out.shape == (B, 768, T_mel)

    # Conv2: stride=2 でダウンサンプリング
    conv2_out = torch.nn.functional.gelu(encoder.conv2(conv1_out))
    after_conv2 = (T_mel - 1) // 2 + 1
    assert conv2_out.shape == (B, 768, after_conv2)

    # 位置エンコーディング
    pos_emb = encoder.positional_embedding(after_conv2)
    assert pos_emb.shape == (after_conv2, 768)

    # AvgPool: さらにstride=2
    after_avgpool = (after_conv2 - 2) // 2 + 1

    # チャンク分割の確認
    chunk_size = 50 * 2  # n_window * 2 = 100フレーム
    num_chunks = (T_mel + chunk_size - 1) // chunk_size

    # 出力形状の確認
    expected_tokens = after_avgpool + 2  # +2 for BOS/EOS
    assert output.shape == (expected_tokens, 1024), \
        f"Expected ({expected_tokens}, 1024), got {output.shape}"

    # --- 結果表示 ---
    print(f"[Audio Encoder 使用例]")
    print(f"  入力: input_features {input_features.shape}  (B, mel_bins, T_mel)")
    print(f"         feature_lens  {feature_lens.shape}   = [{T_mel}]")
    print(f"  音声長: {duration_sec}秒")
    print()
    print(f"  Conv1 後:       {conv1_out.shape}  (B, 768, {T_mel})")
    print(f"  Conv2 後:       {conv2_out.shape}  (B, 768, {after_conv2}) stride=2")
    print(f"  位置エンコ:     ({after_conv2}, 768)")
    print(f"  Transformer×12: packed (total_tokens, 768)")
    print(f"  AvgPool後:      ({after_avgpool}, 768) stride=2")
    print(f"  LayerNorm+射影: ({after_avgpool}, 1024)")
    print(f"  BOS/EOS付加:    ({expected_tokens}, 1024)")
    print(f"  出力:           {output.shape}")
    print()
    print(f"  チャンク分割: {num_chunks}チャンク × {chunk_size}フレーム")
    print(f"  合計ダウンサンプリング: {T_mel} → {after_avgpool} (約{T_mel/after_avgpool:.1f}倍)")
    print(f"  1トークン ≈ {duration_sec * 1000 / after_avgpool:.1f}ms")

    # --- バッチ入力 (複数音声) ---
    B2 = 2
    T_mel_1 = 300  # 3秒
    T_mel_2 = 500  # 5秒
    T_mel_max = max(T_mel_1, T_mel_2)

    # パディング済み入力
    input_features_2 = torch.randn(B2, 128, T_mel_max)
    feature_lens_2 = torch.tensor([T_mel_1, T_mel_2])

    with torch.no_grad():
        output_2 = encoder(input_features_2, feature_lens_2)

    # 2音声分のトークンが結合される
    after_conv2_1 = (T_mel_1 - 1) // 2 + 1
    after_avgpool_1 = (after_conv2_1 - 2) // 2 + 1
    after_conv2_2 = (T_mel_2 - 1) // 2 + 1
    after_avgpool_2 = (after_conv2_2 - 2) // 2 + 1
    expected_total = (after_avgpool_1 + 2) + (after_avgpool_2 + 2)

    print()
    print(f"  [バッチ入力 (2音声)]")
    print(f"  音声1: {T_mel_1}フレーム → {after_avgpool_1}+2 = {after_avgpool_1+2} トークン")
    print(f"  音声2: {T_mel_2}フレーム → {after_avgpool_2}+2 = {after_avgpool_2+2} トークン")
    print(f"  結合出力: {output_2.shape}  (total={expected_total}, 1024)")


if __name__ == "__main__":
    example_audio_encoder()
