"""
Qwen3-Omni Code2Wav: 軽量因果 ConvNet ボコーダ (200Mパラメータ)
================================================================

Talker+MTP が生成したマルチコードブック離散音声トークン (RVQ) を
24kHz 音声波形に変換する軽量因果 ConvNet ボコーダ。

Qwen2.5-Omni の DiT (Flow-Matching) + BigVGAN を完全に置き換える。

主な差分 (vs Qwen2.5-Omni Token2Wav):
    - DiT (Flow-Matching) + BigVGAN → 単一の軽量因果 ConvNet (200M)
    - マルチステップ拡散サンプリング → シングルフォワードパス
    - ブロック単位処理 (chunk=48, 左右コンテキスト) → フレーム単位因果処理
    - 単一コードブック入力 → マルチコードブック RVQ 入力
    - FLOPs 大幅削減、ハードウェアアクセラレーション親和性向上
    - ストリーミング: 最初のコーデックフレームから即座に波形出力 (80msフレーム単位)

アーキテクチャ:
    1. CodebookEmbedding:
        マルチコードブック RVQ インデックスを埋め込みベクトルに変換・集約
        各コードブックの Embedding を合算 → (B, T_codec, embed_dim)

    2. CausalConvBlock (×N スタック):
        因果 (左のみ) 1D畳み込みブロック
        - 因果パディング: kernel_size - 1 の左パディング (右パディング 0)
        - 未来のコンテキストを一切使用しない
        - ConvTranspose1d によるアップサンプリング層を含む
        - 拡散モデルのように反復デノイズ不要

    3. Code2WavModel:
        CodebookEmbedding + CausalConvBlock スタック + 出力 Conv1d
        RVQ フレーム (12.5Hz) → 24kHz 波形 (アップサンプリング比 1920)

入力:
    - マルチコードブック離散音声トークン: (B, num_codebooks, T_codec)
    - 12.5Hz (80ms/フレーム), RVQ 構造
    - Talker + MTP (Multi-Token Prediction) から出力

出力:
    - 音声波形: (B, 1, T_codec * 1920) at 24kHz
    - 1 RVQフレーム (80ms) → 1920 サンプル (24000 * 0.08)

ストリーミング:
    - 因果構造のため、最初のコーデックフレーム到着時点で即座に波形生成可能
    - フレーム単位 (80ms) で逐次的に波形を出力
    - Qwen2.5-Omni のようなブロックコンテキスト待機が不要
    - 初回発話遅延を大幅に削減

パラメータ数: ~200M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ============================================
# Codebook Embedding (マルチコードブック RVQ)
# ============================================

class CodebookEmbedding(nn.Module):
    """
    マルチコードブック RVQ トークンの埋め込み層

    RVQ (Residual Vector Quantization) の各コードブックに対して
    独立した Embedding テーブルを持ち、合算して単一の埋め込みベクトルにする。

    Qwen2.5-Omni との差分:
        - Qwen2.5-Omni: 単一コードブック (1つの Embedding テーブル)
        - Qwen3-Omni: マルチコードブック RVQ (num_codebooks 個の Embedding テーブル)

    RVQ の仕組み:
        各コードブックは前段の量子化残差をエンコード。
        codebook 0: 粗い音声特徴 (基本周波数, エネルギー等)
        codebook 1: codebook 0 の残差 (より細かい特徴)
        codebook 2: codebook 1 の残差 (さらに細かい特徴)
        ...
        合算することで全コードブックの情報を統合。

    入力: (B, num_codebooks, T_codec) - 各コードブックのトークンインデックス
    出力: (B, T_codec, embed_dim)     - 全コードブックの合算埋め込み
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 2048,
        embed_dim: int = 512,
    ):
        """
        パラメータ:
            num_codebooks: RVQ コードブック数 (4)
            codebook_size: 各コードブックの語彙サイズ (2048)
            embed_dim: 埋め込み次元 (512)
        """
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim

        # 各コードブックに独立した Embedding テーブル
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim)
            for _ in range(num_codebooks)
        ])

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        マルチコードブック RVQ トークンを埋め込みベクトルに変換

        入力:
            codes: (B, num_codebooks, T_codec) - RVQ トークンインデックス
                   各コードブック k の codes[:, k, :] は [0, codebook_size) の整数

        出力:
            embedding: (B, T_codec, embed_dim)
                       全コードブックの埋め込みを合算

        処理:
            embedding = sum_{k=0}^{num_codebooks-1} Embedding_k(codes[:, k, :])
        """
        B, K, T = codes.shape
        assert K == self.num_codebooks, (
            f"コードブック数の不一致: 入力 {K}, 期待 {self.num_codebooks}"
        )

        # 各コードブックの埋め込みを合算
        combined = torch.zeros(B, T, self.embed_dim, device=codes.device)
        for k in range(self.num_codebooks):
            combined = combined + self.embeddings[k](codes[:, k, :])
            # self.embeddings[k](codes[:, k, :]) : (B, T, embed_dim)

        return combined
        # combined: (B, T_codec, embed_dim)


# ============================================
# Causal Conv1d Block
# ============================================

class CausalConv1d(nn.Module):
    """
    因果 (左のみ) 1D畳み込み

    通常の Conv1d は左右対称にパディングするため未来の情報を使用する。
    CausalConv1d は左側のみにパディングし、未来の情報を一切使用しない。

    パディング戦略:
        - 左パディング: kernel_size - 1
        - 右パディング: 0
        → 出力の各時刻 t は入力の [t - kernel_size + 1, t] のみに依存

    これによりストリーミング推論時にフレーム到着順で逐次処理可能。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, groups: int = 1):
        """
        パラメータ:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数
            kernel_size: カーネルサイズ
            dilation: ダイレーション (受容野拡大)
            groups: グループ畳み込み
        """
        super().__init__()
        # 因果パディング量: (kernel_size - 1) * dilation
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: (B, C_in, T)
        出力: (B, C_out, T)  ※ 系列長は不変

        因果パディング: 左に causal_padding, 右に 0
        """
        # 左のみパディング
        x = F.pad(x, (self.causal_padding, 0))
        # x: (B, C_in, T + causal_padding)

        x = self.conv(x)
        # x: (B, C_out, T)
        return x


class CausalConvBlock(nn.Module):
    """
    因果畳み込みブロック (Code2Wav の基本ブロック)

    構成:
        CausalConv1d → LayerNorm → GELU → CausalConv1d → LayerNorm → 残差接続

    特徴:
        - 全ての畳み込みが因果的 (左パディングのみ)
        - 拡散モデルのような反復デノイズが不要 (シングルパス)
        - ConvNet ベースのためハードウェアアクセラレーション親和性が高い
        - BatchNorm ではなく LayerNorm を使用 (ストリーミング安定性)

    Qwen2.5-Omni の DiT Layer との対比:
        DiT: Self-Attention + Cross-Attention + FFN (時間ステップ条件付け)
        CausalConvBlock: CausalConv1d × 2 + 残差接続 (条件付け不要)
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        """
        パラメータ:
            channels: チャンネル数 (入出力同一)
            kernel_size: カーネルサイズ (7)
            dilation: ダイレーション (受容野拡大用)
        """
        super().__init__()

        # 2層の因果畳み込み + 残差接続
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)  # LayerNorm 相当 (GroupNorm groups=1)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=1)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: (B, C, T)
        出力: (B, C, T) - 残差接続により形状不変

        処理:
            residual = x
            x → CausalConv1d → GroupNorm → GELU
              → CausalConv1d → GroupNorm
            x = x + residual  (残差接続)
        """
        residual = x

        x = self.conv1(x)     # (B, C, T)
        x = self.norm1(x)     # (B, C, T)
        x = F.gelu(x)         # (B, C, T)

        x = self.conv2(x)     # (B, C, T)
        x = self.norm2(x)     # (B, C, T)

        x = x + residual      # 残差接続
        x = F.gelu(x)

        return x


# ============================================
# Causal Upsample Block (ConvTranspose1d)
# ============================================

class CausalUpsampleBlock(nn.Module):
    """
    因果アップサンプリングブロック

    ConvTranspose1d によるアップサンプリング + CausalConvBlock
    RVQ フレームレート (12.5Hz) を段階的に 24kHz まで引き上げる。

    アップサンプリング比 1920 の分解例:
        Stage 0: ×8  (12.5Hz → 100Hz)
        Stage 1: ×8  (100Hz → 800Hz)
        Stage 2: ×6  (800Hz → 4800Hz)
        Stage 3: ×5  (4800Hz → 24000Hz)
        合計: 8 × 8 × 6 × 5 = 1920

    ※ 上記は一例。実装は異なる因数分解を使用する可能性がある。

    各アップサンプリング段の後に CausalConvBlock でスムージング。
    """

    def __init__(self, in_channels: int, out_channels: int, upsample_rate: int,
                 kernel_size: int = 7, num_conv_blocks: int = 3):
        """
        パラメータ:
            in_channels: 入力チャンネル数
            out_channels: 出力チャンネル数
            upsample_rate: アップサンプリング倍率
            kernel_size: 因果畳み込みのカーネルサイズ
            num_conv_blocks: スムージング用 CausalConvBlock の数
        """
        super().__init__()
        self.upsample_rate = upsample_rate

        # ConvTranspose1d でアップサンプリング
        # kernel_size = upsample_rate * 2, stride = upsample_rate
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=upsample_rate * 2,
            stride=upsample_rate,
            padding=upsample_rate // 2,
        )

        # スムージング用因果畳み込みブロック
        # ダイレーションを段階的に増加させて受容野を拡大
        self.conv_blocks = nn.ModuleList([
            CausalConvBlock(out_channels, kernel_size=kernel_size, dilation=2**i)
            for i in range(num_conv_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: (B, C_in, T)
        出力: (B, C_out, T * upsample_rate)

        処理:
            x → GELU → ConvTranspose1d (×upsample_rate)
              → CausalConvBlock × num_conv_blocks (スムージング)
        """
        x = F.gelu(x)
        x = self.upsample_conv(x)
        # x: (B, C_out, T * upsample_rate)

        for block in self.conv_blocks:
            x = block(x)
        # x: (B, C_out, T * upsample_rate) - スムージング済み

        return x


# ============================================
# Code2Wav Model (完全モデル)
# ============================================

class Code2WavModel(nn.Module):
    """
    Qwen3-Omni Code2Wav - 軽量因果 ConvNet ボコーダ

    マルチコードブック RVQ フレーム (12.5Hz) → 24kHz 音声波形

    パイプライン (シングルフォワードパス):
        1. CodebookEmbedding: RVQ トークン → 埋め込みベクトル
        2. 入力射影: embed_dim → model_channels (Conv1d)
        3. CausalUpsampleBlock × 4: 段階的アップサンプリング (×1920)
        4. 出力 Conv1d: model_channels → 1 (波形)

    vs Qwen2.5-Omni Token2Wav:
        ┌────────────────────┬──────────────────────────────────┐
        │ Qwen2.5-Omni       │ Qwen3-Omni                       │
        ├────────────────────┼──────────────────────────────────┤
        │ DiT + BigVGAN       │ 単一 CausalConvNet               │
        │ 2段階パイプライン   │ 1段階 (シングルフォワードパス)    │
        │ 拡散サンプリング    │ 決定論的フォワードパス           │
        │ (10 Euler steps)   │ (1 step)                         │
        │ 非因果ブロック処理  │ 因果フレーム処理                 │
        │ chunk=48 + ctx      │ フレーム単位 (80ms)              │
        │ 単一コードブック    │ マルチコードブック RVQ            │
        │ 高FLOPs            │ 低FLOPs (200Mパラメータ)         │
        └────────────────────┴──────────────────────────────────┘

    ストリーミング動作:
        因果畳み込みにより、フレーム t の出力は フレーム [0, t] のみに依存。
        → 最初の RVQ フレーム到着時点から波形生成開始可能。
        → Qwen2.5-Omni のように右コンテキスト (lookahead) 待機が不要。

    パラメータ数: ~200M
    """

    def __init__(
        self,
        num_codebooks: int = 4,
        codebook_size: int = 2048,
        embed_dim: int = 512,
        model_channels: int = 1024,
        upsample_rates: Optional[List[int]] = None,
        conv_kernel_size: int = 7,
        num_conv_blocks_per_stage: int = 3,
        sample_rate: int = 24000,
        codec_frame_rate: float = 12.5,
    ):
        """
        パラメータ:
            num_codebooks: RVQ コードブック数 (4)
            codebook_size: 各コードブックの語彙サイズ (2048)
            embed_dim: コードブック埋め込み次元 (512)
            model_channels: ConvNet の基本チャンネル数 (1024)
            upsample_rates: 各段のアップサンプリング倍率 [8, 8, 6, 5]
                → 合計 8×8×6×5 = 1920
                → 12.5Hz × 1920 = 24000Hz
            conv_kernel_size: 因果畳み込みのカーネルサイズ (7)
            num_conv_blocks_per_stage: 各アップサンプリング段の畳み込みブロック数 (3)
            sample_rate: 出力サンプリングレート (24000)
            codec_frame_rate: 入力コーデックフレームレート (12.5)
        """
        super().__init__()

        if upsample_rates is None:
            # 8 × 8 × 6 × 5 = 1920
            # 1 RVQフレーム (80ms @ 12.5Hz) → 1920 サンプル (80ms @ 24kHz)
            upsample_rates = [8, 8, 6, 5]

        self.num_codebooks = num_codebooks
        self.sample_rate = sample_rate
        self.codec_frame_rate = codec_frame_rate
        self.hop_size = int(sample_rate / codec_frame_rate)  # 1920
        total_upsample = 1
        for r in upsample_rates:
            total_upsample *= r
        assert total_upsample == self.hop_size, (
            f"アップサンプリング率の積 ({total_upsample}) が "
            f"hop_size ({self.hop_size}) と一致しない"
        )

        # ================================================
        # 1. CodebookEmbedding: RVQ → 埋め込み
        # ================================================
        self.codebook_embedding = CodebookEmbedding(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            embed_dim=embed_dim,
        )

        # ================================================
        # 2. 入力射影: embed_dim → model_channels
        # ================================================
        self.input_conv = CausalConv1d(embed_dim, model_channels, kernel_size=conv_kernel_size)
        self.input_norm = nn.GroupNorm(1, model_channels)

        # ================================================
        # 3. アップサンプリングスタック (×4段)
        # ================================================
        self.upsample_blocks = nn.ModuleList()
        channels = model_channels
        for i, rate in enumerate(upsample_rates):
            out_channels = channels // 2 if i < len(upsample_rates) - 1 else channels // 2
            self.upsample_blocks.append(
                CausalUpsampleBlock(
                    in_channels=channels,
                    out_channels=out_channels,
                    upsample_rate=rate,
                    kernel_size=conv_kernel_size,
                    num_conv_blocks=num_conv_blocks_per_stage,
                )
            )
            channels = out_channels
        # チャンネル遷移: 1024 → 512 → 256 → 128 → 64

        # ================================================
        # 4. 出力 Conv1d: channels → 1 (波形)
        # ================================================
        self.output_conv = CausalConv1d(channels, 1, kernel_size=conv_kernel_size)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Code2Wav のフォワードパス (シングルパス、拡散不要)

        入力:
            codes: (B, num_codebooks, T_codec) - RVQ トークンインデックス
                   B: バッチサイズ
                   num_codebooks: コードブック数 (例: 4)
                   T_codec: コーデックフレーム数 (12.5Hz)

        出力:
            waveform: (B, 1, T_codec * 1920) - 音声波形 at 24kHz

        処理フロー:
            (B, num_codebooks, T_codec)
              → CodebookEmbedding: (B, T_codec, embed_dim)
              → transpose: (B, embed_dim, T_codec)
              → CausalConv1d + Norm: (B, model_channels, T_codec)
              → UpsampleBlock ×4: (B, channels, T_codec * 1920)
              → CausalConv1d: (B, 1, T_codec * 1920)
              → tanh: [-1, 1] の波形

        ※ Qwen2.5-Omni Token2Wav との比較:
            Token2Wav: code → DiT(10 steps) → mel → BigVGAN → waveform
            Code2Wav:  codes → ConvNet(1 pass) → waveform
        """
        # 1. コードブック埋め込み
        x = self.codebook_embedding(codes)
        # x: (B, T_codec, embed_dim)

        # 2. Conv1d 入力のため転置
        x = x.transpose(1, 2)
        # x: (B, embed_dim, T_codec)

        # 3. 入力射影
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        # x: (B, model_channels, T_codec)

        # 4. 段階的アップサンプリング
        for block in self.upsample_blocks:
            x = block(x)
        # Stage 0: (B, 512, T * 8)
        # Stage 1: (B, 256, T * 64)
        # Stage 2: (B, 128, T * 384)
        # Stage 3: (B, 64,  T * 1920)

        # 5. 出力波形
        x = self.output_conv(x)
        # x: (B, 1, T_codec * 1920)

        x = torch.tanh(x)
        # x: (B, 1, T_codec * 1920) - [-1, 1] の音声波形

        return x

    def forward_streaming(
        self,
        codes: torch.Tensor,
        frame_index: int = 0,
    ) -> torch.Tensor:
        """
        ストリーミングフォワードパス (フレーム単位)

        因果畳み込みにより、各フレームの出力は過去のフレームのみに依存。
        最初のフレームから即座に波形を出力可能。

        入力:
            codes: (B, num_codebooks, T_chunk) - RVQ フレーム (チャンク)
                   T_chunk: 今回処理するフレーム数 (1以上)
            frame_index: 現在のフレーム位置 (累積フレーム数)

        出力:
            waveform_chunk: (B, 1, T_chunk * 1920) - 波形チャンク at 24kHz

        ストリーミング動作:
            Qwen2.5-Omni:
                DiT が chunk=48 コード + 左24/右12 コンテキスト必要
                → 最低 48+12=60 コード (4.8秒) 待機してから波形生成開始
                BigVGAN が左20/右20 メルフレームコンテキスト必要
                → 追加の待機

            Qwen3-Omni (Code2Wav):
                因果ConvNet → 右コンテキスト (lookahead) 不要
                → 最初の1フレーム (80ms) から即座に波形生成可能
                → first-token-to-speech レイテンシ大幅削減

        ※ 実装では因果畳み込みの内部状態 (キャッシュ) を管理し、
           逐次的にフレームを処理する。ここでは簡略化のため
           累積入力に対する full forward を実行して該当部分を切り出す。
        """
        # 簡略化実装: 全コンテキストを使用してフォワード → チャンク切り出し
        waveform = self.forward(codes)
        # waveform: (B, 1, T_total * 1920)

        # チャンク切り出し (最後の T_chunk フレーム分)
        chunk_samples = codes.shape[2] * self.hop_size
        if frame_index > 0:
            # 累積入力の場合、最新チャンク分のみ切り出す
            # 実際のストリーミング実装ではキャッシュベースで効率化
            start_sample = frame_index * self.hop_size
            waveform_chunk = waveform[:, :, start_sample:start_sample + chunk_samples]
        else:
            waveform_chunk = waveform[:, :, :chunk_samples]

        return waveform_chunk


# ============================================
# 使用例
# ============================================

def example_code2wav():
    """
    Code2Wav の使用例

    CodebookEmbedding, CausalConvBlock, Code2WavModel を
    実際にインスタンス化し、フォワードパスを実行して形状を確認する。

    ※ パラメータは実モデル (200M) より縮小して動作確認用。
    """

    print("=" * 70)
    print("Qwen3-Omni Code2Wav (軽量因果ConvNetボコーダ) 使用例")
    print("=" * 70)

    # ========================================
    # パラメータ設定 (縮小版)
    # ========================================
    num_codebooks = 4       # 実モデルと同じ
    codebook_size = 2048    # 実モデルと同じ
    embed_dim = 128         # 実モデルは 512
    model_channels = 256    # 実モデルは 1024
    upsample_rates = [8, 8, 6, 5]  # 合計 1920
    codec_frame_rate = 12.5
    sample_rate = 24000
    hop_size = int(sample_rate / codec_frame_rate)  # 1920

    B = 1
    T_codec = 25  # 25フレーム = 2秒 (12.5Hz × 2秒)
    duration_sec = T_codec / codec_frame_rate

    # ========================================
    # 1. CodebookEmbedding 単体テスト
    # ========================================
    print()
    print("[1] CodebookEmbedding テスト")
    print("-" * 40)

    codebook_emb = CodebookEmbedding(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        embed_dim=embed_dim,
    )
    codebook_emb.eval()

    # ダミー RVQ トークン
    codes = torch.randint(0, codebook_size, (B, num_codebooks, T_codec))
    print(f"  入力 codes: {codes.shape}  (B, num_codebooks, T_codec)")

    with torch.no_grad():
        emb = codebook_emb(codes)
    assert emb.shape == (B, T_codec, embed_dim), (
        f"形状不一致: 期待 ({B}, {T_codec}, {embed_dim}), 実際 {emb.shape}"
    )
    print(f"  出力 embedding: {emb.shape}  (B, T_codec, embed_dim)")
    print(f"  → {num_codebooks}個のコードブック埋め込みを合算")

    # ========================================
    # 2. CausalConv1d 因果性テスト
    # ========================================
    print()
    print("[2] CausalConv1d 因果性テスト")
    print("-" * 40)

    causal_conv = CausalConv1d(in_channels=64, out_channels=64, kernel_size=7)
    causal_conv.eval()

    test_input = torch.randn(1, 64, 20)
    with torch.no_grad():
        test_output = causal_conv(test_input)
    assert test_output.shape == (1, 64, 20), (
        f"因果畳み込みで系列長が変化: 入力 20, 出力 {test_output.shape[2]}"
    )
    print(f"  入力: {test_input.shape}  → 出力: {test_output.shape}")
    print(f"  系列長不変 (因果パディング: 左{causal_conv.causal_padding}, 右0)")

    # 因果性検証: 入力の末尾を変更しても前方の出力は不変
    test_input_modified = test_input.clone()
    test_input_modified[:, :, 15:] = torch.randn(1, 64, 5)  # 後半5フレームを変更
    with torch.no_grad():
        test_output_modified = causal_conv(test_input_modified)
    # 前半15フレームの出力は不変であるべき (因果性)
    # ※ kernel_size=7 なので、変更位置15から影響を受けるのは位置15以降のみ
    assert torch.allclose(test_output[:, :, :15], test_output_modified[:, :, :15], atol=1e-6), (
        "因果性の違反: 未来の入力変更が過去の出力に影響している"
    )
    print(f"  因果性検証: 入力[15:]変更 → 出力[:15]不変 ... OK")
    print(f"  → 未来のコンテキストを使用していないことを確認")

    # ========================================
    # 3. CausalConvBlock 単体テスト
    # ========================================
    print()
    print("[3] CausalConvBlock テスト")
    print("-" * 40)

    conv_block = CausalConvBlock(channels=64, kernel_size=7, dilation=1)
    conv_block.eval()

    block_input = torch.randn(1, 64, 20)
    with torch.no_grad():
        block_output = conv_block(block_input)
    assert block_output.shape == (1, 64, 20), (
        f"CausalConvBlock で形状変化: {block_output.shape}"
    )
    print(f"  入力: {block_input.shape}  → 出力: {block_output.shape}")
    print(f"  残差接続により形状不変")

    # ========================================
    # 4. CausalUpsampleBlock 単体テスト
    # ========================================
    print()
    print("[4] CausalUpsampleBlock テスト")
    print("-" * 40)

    upsample_block = CausalUpsampleBlock(
        in_channels=128, out_channels=64,
        upsample_rate=8, kernel_size=7, num_conv_blocks=2,
    )
    upsample_block.eval()

    up_input = torch.randn(1, 128, 25)
    with torch.no_grad():
        up_output = upsample_block(up_input)
    expected_T = 25 * 8  # 200
    print(f"  入力: {up_input.shape}  (B, 128, 25)")
    print(f"  出力: {up_output.shape}  (B, 64, {up_output.shape[2]})")
    print(f"  ×8 アップサンプリング: 25 → {up_output.shape[2]}")

    # ========================================
    # 5. Code2WavModel 完全パイプライン
    # ========================================
    print()
    print("[5] Code2WavModel 完全パイプライン")
    print("-" * 40)

    model = Code2WavModel(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        embed_dim=embed_dim,
        model_channels=model_channels,        # 実モデルは 1024
        upsample_rates=upsample_rates,
        conv_kernel_size=7,
        num_conv_blocks_per_stage=2,           # 実モデルは 3
        sample_rate=sample_rate,
        codec_frame_rate=codec_frame_rate,
    )
    model.eval()

    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  縮小版パラメータ数: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  ※ 実モデル: ~200M パラメータ")
    print()

    # フォワードパス (非ストリーミング)
    codes_input = torch.randint(0, codebook_size, (B, num_codebooks, T_codec))
    print(f"  入力 codes: {codes_input.shape}  (B={B}, codebooks={num_codebooks}, T={T_codec})")
    print(f"  音声長: {duration_sec:.1f}秒 ({T_codec}フレーム @ {codec_frame_rate}Hz)")

    with torch.no_grad():
        waveform = model(codes_input)

    expected_samples = T_codec * hop_size
    assert waveform.shape == (B, 1, expected_samples), (
        f"波形形状不一致: 期待 ({B}, 1, {expected_samples}), 実際 {waveform.shape}"
    )
    assert waveform.min() >= -1.0 and waveform.max() <= 1.0, (
        f"波形値が [-1, 1] の範囲外: min={waveform.min():.4f}, max={waveform.max():.4f}"
    )

    actual_duration = waveform.shape[2] / sample_rate
    print()
    print(f"  出力 waveform: {waveform.shape}  (B, 1, N_samples)")
    print(f"  サンプル数: {waveform.shape[2]:,}  ({actual_duration:.1f}秒 @ {sample_rate}Hz)")
    print(f"  波形値範囲: [{waveform.min():.4f}, {waveform.max():.4f}]  (tanh出力)")
    print(f"  アップサンプリング: {T_codec} → {waveform.shape[2]:,} (×{hop_size})")

    # ========================================
    # 6. ストリーミングシミュレーション
    # ========================================
    print()
    print("[6] ストリーミング生成シミュレーション")
    print("-" * 40)

    # フレーム単位で逐次処理 (5フレームずつ)
    chunk_size = 5  # 5フレーム = 400ms
    total_frames = T_codec
    accumulated_codes = None
    streaming_waveforms = []

    print(f"  チャンクサイズ: {chunk_size}フレーム ({chunk_size / codec_frame_rate * 1000:.0f}ms)")
    print()

    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk_codes = codes_input[:, :, start:end]

        if accumulated_codes is None:
            accumulated_codes = chunk_codes
        else:
            accumulated_codes = torch.cat([accumulated_codes, chunk_codes], dim=2)

        with torch.no_grad():
            waveform_chunk = model.forward_streaming(
                accumulated_codes, frame_index=start
            )

        chunk_samples = waveform_chunk.shape[2]
        chunk_ms = chunk_samples / sample_rate * 1000
        streaming_waveforms.append(waveform_chunk)

        print(f"    フレーム [{start:3d}:{end:3d}] → "
              f"波形チャンク {waveform_chunk.shape} "
              f"({chunk_samples:,} samples, {chunk_ms:.0f}ms)")

    # ストリーミング結果を結合
    full_streaming_waveform = torch.cat(streaming_waveforms, dim=2)
    print()
    print(f"  結合波形: {full_streaming_waveform.shape}")
    print(f"  サンプル数: {full_streaming_waveform.shape[2]:,} "
          f"({full_streaming_waveform.shape[2] / sample_rate:.1f}秒)")

    # ========================================
    # Qwen2.5-Omni vs Qwen3-Omni 比較
    # ========================================
    print()
    print("=" * 70)
    print("[比較] Qwen2.5-Omni Token2Wav vs Qwen3-Omni Code2Wav")
    print("=" * 70)
    print()
    print("  項目                 Qwen2.5-Omni             Qwen3-Omni")
    print("  " + "-" * 66)
    print("  アーキテクチャ       DiT + BigVGAN            CausalConvNet (単一)")
    print("  サンプリング         Flow-Matching (10 steps) シングルフォワードパス")
    print("  入力形式             単一コードブック         マルチコードブック RVQ")
    print("  パラメータ数         DiT+BigVGAN (大)         ~200M (軽量)")
    print("  処理方式             ブロック単位             フレーム単位 (因果)")
    print("  右コンテキスト       必要 (lookahead=12)      不要 (因果)")
    print(f"  コーデックレート     ~25Hz (1コードブック)    {codec_frame_rate}Hz (RVQ)")
    print(f"  出力レート           24kHz                    {sample_rate}Hz")
    print(f"  フレームあたり       ~960 samples             {hop_size} samples")
    print("  初回発話遅延         chunk(48)+右ctx(12)待機  1フレーム(80ms)で即出力")
    print("  ハードウェア親和性   Transformer (DiT)        ConvNet (高い)")


if __name__ == "__main__":
    example_code2wav()
