"""
CosyVoice3 Vocoder (Causal HiFT) - 簡略化疑似コード
======================================================

メルスペクトログラムから音声波形を生成するボコーダモジュール。
HiFi-GAN + Neural Source Filter (NSF) の統合アーキテクチャ。

論文: CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
公式実装:
  - cosyvoice/hifigan/generator.py (CausalHiFTGenerator)
  - cosyvoice/hifigan/f0_predictor.py (CausalConvRNNF0Predictor)
  - cosyvoice/hifigan/hifigan.py (HiFiGan - 学習ラッパー)

特徴:
- Causal構造: ストリーミング推論対応
- NSF (Neural Source Filter): F0予測に基づく源信号生成
- Snake活性化関数: 周期的な活性化で音声波形の周期性を捕捉
- ISTFT合成: 最終段でiSTFTによる波形合成

Shape Convention
============================================================
B: バッチサイズ
T_mel: メルスペクトログラムのフレーム数
T_audio: 出力波形サンプル数 (≈ T_mel × hop_size)
D_mel: メル周波数ビン数 (80)
C_base: ベースチャンネル数 (512)
hop_size: ホップサイズ (≈ 240, = 8×5×3×2 アップサンプリング)
sr: サンプリングレート (24000 Hz)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CausalHiFTGenerator(nn.Module):
    """
    Causal HiFi-GAN + Source Filter ボコーダ

    アーキテクチャ全体像:
    ┌──────────────────────────────────────────────────────────┐
    │ メルスペクトログラム (B, 80, T_mel)                        │
    │     ↓                                                    │
    │ ┌─────────────────────┐   ┌────────────────────────┐    │
    │ │ F0 Predictor        │   │ 初期Causal Conv        │    │
    │ │ (CausalConvRNN)     │   │ (80 → 512)             │    │
    │ │                     │   │                        │    │
    │ │ mel → F0値           │   │ mel → 初期特徴          │    │
    │ │ (B, 1, T_mel)       │   │ (B, 512, T_mel)        │    │
    │ └─────────┬───────────┘   └───────────┬────────────┘    │
    │           ↓                           ↓                  │
    │ ┌─────────────────────┐   ┌────────────────────────┐    │
    │ │ Source Generator     │   │ Upsampling Block 1     │    │
    │ │ (NSF)               │   │ (×8)                   │    │
    │ │                     │   │ 512 → 256              │    │
    │ │ F0 → サイン波+ノイズ  │   │ (B, 256, T×8)          │    │
    │ │ (B, 9, T_mel×8)    │   └───────────┬────────────┘    │
    │ └─────────┬───────────┘               ↓                  │
    │           │               ┌────────────────────────┐    │
    │           │               │ + ResBlock (Snake活性化) │    │
    │           │               │ + Source Signal Mix     │    │
    │           │               └───────────┬────────────┘    │
    │           ↓                           ↓                  │
    │         (各段で源信号を混合)                               │
    │                           ┌────────────────────────┐    │
    │                           │ Upsampling Block 2     │    │
    │                           │ (×5)                   │    │
    │                           │ 256 → 128              │    │
    │                           │ (B, 128, T×40)         │    │
    │                           └───────────┬────────────┘    │
    │                                       ↓                  │
    │                           ┌────────────────────────┐    │
    │                           │ Upsampling Block 3     │    │
    │                           │ (×3)                   │    │
    │                           │ 128 → 64               │    │
    │                           │ (B, 64, T×120)         │    │
    │                           └───────────┬────────────┘    │
    │                                       ↓                  │
    │                           ┌────────────────────────┐    │
    │                           │ ISTFT Synthesis        │    │
    │                           │ 64 → (mag, phase)      │    │
    │                           │ → 波形                  │    │
    │                           └───────────┬────────────┘    │
    │                                       ↓                  │
    │                           waveform (B, T_audio)          │
    │                           T_audio ≈ T_mel × 240          │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        mel_dim: int = 80,
        base_channels: int = 512,
        upsample_rates: list = [8, 5, 3],          # 合計120×
        upsample_kernel_sizes: list = [16, 10, 6],
        resblock_kernel_sizes: list = [3, 7, 11],
        resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        nb_harmonics: int = 8,                       # 高調波数
        sample_rate: int = 24000,
        nsf_alpha: float = 0.1,                      # NSF寄与率
        nsf_sigma: float = 0.003,                    # ノイズレベル
        istft_n_fft: int = 16,                       # ISTFT FFTサイズ
        istft_hop_length: int = 4,                   # ISTFT ホップ長
    ):
        super().__init__()

        self.upsample_rates = upsample_rates
        self.sample_rate = sample_rate
        self.nsf_alpha = nsf_alpha

        # ========================================
        # 1. F0 Predictor (基本周波数予測)
        # ========================================
        self.f0_predictor = CausalConvRNNF0Predictor(
            mel_dim=mel_dim,
        )
        # 入力: mel (B, 80, T_mel) → 出力: f0 (B, 1, T_mel)

        # ========================================
        # 2. 源信号生成器 (NSF)
        # ========================================
        self.source_generator = SourceGenerator(
            sample_rate=sample_rate,
            nb_harmonics=nb_harmonics,  # 8高調波
            sigma=nsf_sigma,
        )
        # 入力: f0 (B, 1, T_mel) → 出力: source (B, 1+nb_harmonics, T_audio)
        #   チャンネル0: 基本波, チャンネル1-8: 高調波

        # ========================================
        # 3. 初期畳み込み
        # ========================================
        self.conv_pre = CausalConv1d(
            in_channels=mel_dim,     # 80
            out_channels=base_channels,  # 512
            kernel_size=7,
        )
        # 入力: (B, 80, T_mel) → 出力: (B, 512, T_mel)

        # ========================================
        # 4. アップサンプリングブロック
        # ========================================
        channels = [base_channels]  # [512]
        for i, rate in enumerate(upsample_rates):
            channels.append(base_channels // (2 ** (i + 1)))
        # channels = [512, 256, 128, 64]

        self.upsample_blocks = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.source_mixers = nn.ModuleList()

        for i, (rate, kernel_size) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            # アップサンプリング (転置畳み込み)
            self.upsample_blocks.append(
                CausalConvTranspose1d(
                    in_channels=channels[i],      # 512/256/128
                    out_channels=channels[i + 1],  # 256/128/64
                    kernel_size=kernel_size,
                    stride=rate,
                )
            )

            # 残差ブロック (Snake活性化)
            self.resblocks.append(
                ResBlockWithSnake(
                    channels=channels[i + 1],
                    kernel_sizes=resblock_kernel_sizes,
                    dilation_sizes=resblock_dilation_sizes,
                )
            )

            # 源信号ミキサー
            self.source_mixers.append(
                SourceMixer(
                    in_channels=1 + nb_harmonics,  # 9
                    out_channels=channels[i + 1],   # 256/128/64
                )
            )

        # ========================================
        # 5. ISTFT合成層
        # ========================================
        self.istft_layer = ISTFTSynthesis(
            in_channels=channels[-1],  # 64
            n_fft=istft_n_fft,         # 16
            hop_length=istft_hop_length,  # 4
        )
        # 入力: (B, 64, T×120) → 出力: (B, T_audio)
        # T_audio ≈ T×120 × (n_fft / hop_length) / 2

    def forward(
        self,
        mel: torch.Tensor,  # (B, 80, T_mel)
    ) -> torch.Tensor:
        """
        メルスペクトログラム → 音声波形

        入力:
            mel: (B, 80, T_mel) - メルスペクトログラム
                B: バッチサイズ
                80: メル周波数ビン
                T_mel: メルフレーム数

        出力:
            waveform: (B, T_audio) - 音声波形
                T_audio ≈ T_mel × 240
                24kHz サンプリングレート

        処理フロー:
            1. F0予測: mel → f0 (B, 1, T_mel)
            2. 源信号生成: f0 → source (B, 9, T_audio)
            3. 初期Conv: mel → h (B, 512, T_mel)
            4. 3段アップサンプリング:
               - ×8: (B, 256, T×8) + source mix
               - ×5: (B, 128, T×40) + source mix
               - ×3: (B, 64, T×120) + source mix
            5. ISTFT: (B, 64, T×120) → (B, T_audio)
        """
        # Step 1: F0予測
        f0 = self.f0_predictor(mel)
        # f0: (B, 1, T_mel) - 基本周波数 (Hz)
        #   有声区間: 80-800 Hz, 無声区間: 0 Hz

        # Step 2: 源信号生成 (NSF)
        source_signals = self.source_generator(f0)
        # source_signals: (B, 9, T_audio)
        #   チャンネル0: 基本波サイン (f0)
        #   チャンネル1-8: 高調波サイン (2f0, 3f0, ..., 9f0)
        #   無声区間: ガウスノイズ

        # Step 3: 初期畳み込み
        h = self.conv_pre(mel)
        # h: (B, 512, T_mel)

        # Step 4: 3段アップサンプリング
        total_upsample = 1
        for i in range(len(self.upsample_rates)):
            # アップサンプリング
            h = F.leaky_relu(h, 0.1)
            h = self.upsample_blocks[i](h)
            # ステップi後:
            #   i=0: (B, 256, T_mel×8)
            #   i=1: (B, 128, T_mel×40)
            #   i=2: (B, 64, T_mel×120)

            total_upsample *= self.upsample_rates[i]

            # 源信号ミキシング
            # 源信号を現在の解像度にリサンプル
            source_at_scale = F.interpolate(
                source_signals, size=h.shape[-1], mode='linear'
            )
            # source_at_scale: (B, 9, T_current)

            source_mixed = self.source_mixers[i](source_at_scale)
            # source_mixed: (B, channels[i+1], T_current)

            h = h + self.nsf_alpha * source_mixed
            # h: (B, channels[i+1], T_current)

            # 残差ブロック (Snake活性化)
            h = self.resblocks[i](h)
            # h: (B, channels[i+1], T_current)

        # Step 5: ISTFT合成
        waveform = self.istft_layer(h)
        # waveform: (B, T_audio)

        return waveform


class CausalConvRNNF0Predictor(nn.Module):
    """
    F0予測器 (因果畳み込み + RNN)

    メルスペクトログラムから基本周波数 (F0) を予測。
    ストリーミング推論のため因果構造。

    入力: mel (B, 80, T_mel)
    出力: f0 (B, 1, T_mel) - Hz単位のF0値
          有声区間: 80-800Hz, 無声区間: 0Hz
    """

    def __init__(
        self,
        mel_dim: int = 80,
        hidden_dim: int = 256,
        num_rnn_layers: int = 2,
    ):
        super().__init__()

        # 因果畳み込み → RNN → F0予測
        self.conv_layers = nn.Sequential(
            CausalConv1d(mel_dim, hidden_dim, kernel_size=5),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=5),
            nn.ReLU(),
        )
        # (B, 80, T) → (B, 256, T)

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
        )
        # (B, T, 256) → (B, T, 256)

        self.f0_proj = nn.Linear(hidden_dim, 1)
        # (B, T, 256) → (B, T, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        入力: mel (B, 80, T_mel)
        出力: f0 (B, 1, T_mel)
        """
        h = self.conv_layers(mel)
        # h: (B, 256, T_mel)

        h = h.transpose(1, 2)
        # h: (B, T_mel, 256)

        h, _ = self.rnn(h)
        # h: (B, T_mel, 256)

        f0 = self.f0_proj(h)
        # f0: (B, T_mel, 1)

        f0 = F.relu(f0)  # F0は非負
        f0 = f0.transpose(1, 2)
        # f0: (B, 1, T_mel)

        return f0


class SourceGenerator(nn.Module):
    """
    Neural Source Filter (NSF) 源信号生成器

    F0値からサイン波ベースの源信号を生成。
    有声区間: サイン波 (基本波 + 高調波)
    無声区間: ガウスノイズ

    入力: f0 (B, 1, T_mel) - Hz単位のF0
    出力: source (B, 1+nb_harmonics, T_audio)
          チャンネル0: 基本波, チャンネル1-N: 高調波

    T_audio = T_mel × (total_upsample_rate)
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        nb_harmonics: int = 8,
        sine_amp: float = 0.1,
        sigma: float = 0.003,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.nb_harmonics = nb_harmonics
        self.sine_amp = sine_amp
        self.sigma = sigma

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        入力:
            f0: (B, 1, T_mel) - F0値 (Hz)

        出力:
            source: (B, 1+nb_harmonics, T_audio) - 源信号
                チャンネル0: sin(2π × f0 × t)
                チャンネルk: sin(2π × (k+1)f0 × t)
                無声 (f0=0): ガウスノイズ (σ=0.003)
        """
        # F0をオーディオレートにアップサンプル
        f0_upsampled = F.interpolate(
            f0, scale_factor=self.sample_rate // 25,  # ≈960
            mode='nearest'
        )
        # f0_upsampled: (B, 1, T_audio)

        # 位相累積
        phase = torch.cumsum(f0_upsampled / self.sample_rate, dim=-1)
        # phase: (B, 1, T_audio) - 正規化位相 [0, ...]

        # 基本波 + 高調波
        harmonics = []
        for k in range(1, self.nb_harmonics + 2):  # k=1,...,9
            sine = self.sine_amp * torch.sin(2 * torch.pi * k * phase)
            # sine: (B, 1, T_audio)

            # 無声区間 (f0=0) はノイズに置換
            voiced_mask = (f0_upsampled > 0).float()
            noise = self.sigma * torch.randn_like(sine)
            signal = voiced_mask * sine + (1 - voiced_mask) * noise

            harmonics.append(signal)

        source = torch.cat(harmonics, dim=1)
        # source: (B, 1+nb_harmonics, T_audio) = (B, 9, T_audio)

        return source


class ResBlockWithSnake(nn.Module):
    """
    残差ブロック (Snake活性化関数)

    Snake(x) = x + (1/α) × sin²(αx)
    周期的な活性化により音声波形の周期性を効果的に捕捉。

    入力/出力: (B, C, T)
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: list = [3, 7, 11],
        dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for k, dilations in zip(kernel_sizes, dilation_sizes):
            layers = []
            for d in dilations:
                layers.extend([
                    SnakeActivation(channels),
                    CausalConv1d(channels, channels, kernel_size=k, dilation=d),
                ])
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: x (B, C, T)
        出力: (B, C, T) - 各ブランチの平均
        """
        output = torch.zeros_like(x)
        for block in self.blocks:
            output = output + block(x)
        return output / len(self.blocks)


class SnakeActivation(nn.Module):
    """
    Snake活性化関数

    Snake(x) = x + (1/α) × sin²(αx)

    αは学習可能パラメータ (チャンネルごと)
    周期的な特性により、音声信号の周期構造を捕捉

    入力/出力: (B, C, T)
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.full((1, channels, 1), alpha_init)
        )
        # alpha: (1, C, 1) - 学習可能

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2
        # 出力: (B, C, T)


class ISTFTSynthesis(nn.Module):
    """
    ISTFT合成層

    最終的な特徴マップからiSTFTで波形を合成。
    magnitude と phase を予測し、iSTFTで波形復元。

    入力: features (B, C, T)
    出力: waveform (B, T_audio)
    """

    def __init__(
        self,
        in_channels: int = 64,
        n_fft: int = 16,
        hop_length: int = 4,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        # magnitude予測 (n_fft//2 + 1 周波数ビン)
        self.mag_proj = nn.Conv1d(in_channels, n_fft // 2 + 1, 1)
        # phase予測
        self.phase_proj = nn.Conv1d(in_channels, n_fft // 2 + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力: x (B, 64, T)
        出力: waveform (B, T_audio)
              T_audio ≈ T × hop_length
        """
        mag = torch.exp(self.mag_proj(x))
        # mag: (B, n_fft//2+1, T)

        phase = self.phase_proj(x)
        # phase: (B, n_fft//2+1, T)

        # 複素STFT表現
        stft = mag * torch.exp(1j * phase)
        # stft: (B, n_fft//2+1, T)

        # iSTFTで波形合成
        waveform = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=False,
        )
        # waveform: (B, T_audio)

        return waveform


class CausalConv1d(nn.Module):
    """因果的1D畳み込み (左パディングのみ)"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """因果的1D転置畳み込み (アップサンプリング用)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv_t = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride,
        )
        self.stride = stride

    def forward(self, x):
        return self.conv_t(x)


class SourceMixer(nn.Module):
    """源信号をフィルタ特徴に混合するモジュール"""

    def __init__(self, in_channels: int = 9, out_channels: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """
        入力: source (B, 9, T) - 源信号 (基本波+8高調波)
        出力: (B, out_channels, T) - 混合済み信号
        """
        return self.conv(source)
