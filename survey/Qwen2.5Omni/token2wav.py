"""
Qwen2.5-Omni Token2Wav - 簡略化疑似コード
============================================

音声コードトークンから音声波形への変換パイプライン
DiT (Diffusion Transformer) + BigVGAN (Neural Vocoder)

2段階パイプライン:
    Stage 1: DiT (Flow-Matching) → メルスペクトログラム
    Stage 2: BigVGAN → 音声波形 (24kHz)

ストリーミング対応: チャンク単位での段階的生成

公式実装: modeling_qwen2_5_omni_low_VRAM_mode.py (Lines 4165-4530)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Generator, Tuple


class DiTModel(nn.Module):
    """
    Diffusion Transformer (DiT) - Flow-Matching 方式

    音声コードトークンからメルスペクトログラムを生成

    Flow-Matching:
        ノイズからメルスペクトログラムへの確率フローを学習
        Euler ODE Solver で段階的にデノイズ

    スライディングウィンドウブロックアテンション:
        受容野を4ブロック (2 lookback + 1 current + 1 lookahead) に制限
        → ストリーミング生成が可能
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        mel_dim: int = 160,
        cond_dim: int = 512,
    ):
        """
        パラメータ:
            hidden_size: Transformer隠れ次元 (1024)
            num_layers: Transformerレイヤー数 (12)
            num_heads: アテンションヘッド数 (16)
            mel_dim: メルスペクトログラム次元 (160)
            cond_dim: 条件付け次元 (512)
        """
        super().__init__()

        self.mel_dim = mel_dim

        # コード埋め込み
        self.code_embedding = nn.Embedding(8295, cond_dim)

        # ノイズ + 条件 → hidden
        self.input_proj = nn.Linear(mel_dim + cond_dim, hidden_size)

        # Transformer ブロック
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # 出力射影
        self.output_proj = nn.Linear(hidden_size, mel_dim)

        # 時間ステップ埋め込み (ノイズレベル)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        noise: torch.Tensor,
        conditioning: torch.Tensor,
        reference_mel: Optional[torch.Tensor],
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        DiT のフォワードパス (1ステップ)

        入力:
            noise: (B, T_mel, 160) - ノイズ (またはデノイズ途中のメル)
            conditioning: (B, T_cond, cond_dim) - 条件ベクトル (コード埋め込み)
            reference_mel: (B, T_ref, 160) - 参照メルスペクトログラム (話者情報)
            timestep: (B,) or scalar - 現在のタイムステップ [0, 1]

        出力:
            velocity: (B, T_mel, 160) - 速度場 (Flow-Matching)
        """
        B, T, D = noise.shape

        # 時間ステップ埋め込み
        t_emb = self.time_embed(timestep.view(-1, 1))
        # t_emb: (B, hidden_size)

        # ノイズと条件を結合
        # conditioning をアップサンプルしてメルの長さに合わせる
        cond_upsampled = F.interpolate(
            conditioning.transpose(1, 2),
            size=T,
            mode='nearest',
        ).transpose(1, 2)
        # cond_upsampled: (B, T_mel, cond_dim)

        x = torch.cat([noise, cond_upsampled], dim=-1)
        # x: (B, T_mel, 160 + cond_dim)

        x = self.input_proj(x) + t_emb.unsqueeze(1)
        # x: (B, T_mel, hidden_size)

        # Transformer ブロック (スライディングウィンドウアテンション)
        for layer in self.layers:
            x = layer(x)
        # x: (B, T_mel, hidden_size)

        # 速度場の予測
        velocity = self.output_proj(x)
        # velocity: (B, T_mel, 160)

        return velocity

    def sample(
        self,
        conditioning: torch.Tensor,
        reference_mel: Optional[torch.Tensor],
        code: torch.Tensor,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float = -1.0,
    ) -> torch.Tensor:
        """
        Flow-Matching によるメルスペクトログラム生成

        入力:
            conditioning: (B, T_cond, cond_dim) - 条件
            reference_mel: (B, T_ref, 160) - 参照メル
            code: (B, L_codec) - 音声コードトークン
            num_steps: int - デノイズステップ数 (10)
            guidance_scale: float - ガイダンス強度 (0.5)
            sway_coefficient: float - スウェイ係数 (-1.0)

        出力:
            mel_spectrogram: (B, T_mel, 160) - 生成メルスペクトログラム

        Flow-Matching サンプリング (Euler ODE Solver):
            t=0 (ノイズ) → t=1 (クリーンなメル) へ段階的に移動
            各ステップ: x_{t+dt} = x_t + v(x_t, t) * dt
            v = velocity field (モデルが予測)
        """
        B = code.shape[0]
        device = code.device

        # コード埋め込み
        cond = self.code_embedding(code)
        # cond: (B, L_codec, cond_dim)

        # 出力メル長を推定 (コード数 × アップサンプリング比)
        repeats = 4  # 1コード ≈ 4メルフレーム
        T_mel = code.shape[1] * repeats

        # 初期ノイズ
        noise = torch.randn(B, T_mel, self.mel_dim, device=device)
        # noise: (B, T_mel, 160)

        x = noise

        # Euler ODE Solver
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = step * dt
            # Sway scheduling (適応的ステップサイズ)
            if sway_coefficient >= 0:
                t_adjusted = t + sway_coefficient * (t * (1 - t))
            else:
                t_adjusted = t

            timestep = torch.tensor([t_adjusted], device=device)

            # 速度場の予測
            velocity = self.forward(x, cond, reference_mel, timestep)
            # velocity: (B, T_mel, 160)

            # Classifier-Free Guidance (optional)
            if guidance_scale != 1.0:
                # 条件なしの予測
                velocity_uncond = self.forward(x, torch.zeros_like(cond),
                                                reference_mel, timestep)
                velocity = velocity_uncond + guidance_scale * (velocity - velocity_uncond)

            # Euler ステップ
            x = x + velocity * dt
            # x: (B, T_mel, 160) - 更新されたメル

        # x: (B, T_mel, 160) - 生成メルスペクトログラム
        return x


class BigVGANModel(nn.Module):
    """
    BigVGAN - Neural Vocoder

    メルスペクトログラムから音声波形を生成

    修正版BigVGAN: チャンク単位生成に対応
    固定受容野のため、自然にストリーミング処理が可能
    """

    def __init__(
        self,
        mel_dim: int = 160,
        upsample_rates: list = None,
        upsample_initial_channel: int = 1536,
    ):
        """
        パラメータ:
            mel_dim: 入力メルスペクトログラム次元 (160)
            upsample_rates: アップサンプリング率 [8, 8, 2, 2]
                → 合計 8×8×2×2 = 256 (1メルフレーム → 256サンプル)
                ※ 実際は240 (24kHz / 100fps = 240サンプル/フレーム)
            upsample_initial_channel: 初期チャンネル数 (1536)
        """
        super().__init__()

        if upsample_rates is None:
            upsample_rates = [8, 8, 2, 2]

        self.mel_dim = mel_dim
        self.upsample_rate = 240  # 1メルフレーム → 240サンプル at 24kHz

        # 入力畳み込み
        self.conv_pre = nn.Conv1d(mel_dim, upsample_initial_channel, 7, 1, 3)

        # アップサンプリングブロック
        self.ups = nn.ModuleList()
        channels = upsample_initial_channel
        for rate in upsample_rates:
            self.ups.append(
                nn.ConvTranspose1d(
                    channels, channels // 2,
                    kernel_size=rate * 2, stride=rate,
                    padding=rate // 2,
                )
            )
            channels = channels // 2

        # 出力畳み込み
        self.conv_post = nn.Conv1d(channels, 1, 7, 1, 3)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        メルスペクトログラムから音声波形を生成

        入力:
            mel_spectrogram: (B, T_mel, 160) - メルスペクトログラム

        出力:
            waveform: (B, 1, T_mel * 240) - 音声波形 at 24kHz

        例:
            mel: (1, 100, 160) → 100フレーム
            waveform: (1, 1, 24000) → 100 × 240 = 24000サンプル = 1秒
        """
        # (B, T, mel_dim) → (B, mel_dim, T) for 1D conv
        x = mel_spectrogram.transpose(1, 2)
        # x: (B, 160, T_mel)

        x = self.conv_pre(x)
        # x: (B, 1536, T_mel)

        # アップサンプリング
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            # 段階的にアップサンプル: T → T*8 → T*64 → T*128 → T*256
        # x: (B, channels, T_mel * 256)

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        # x: (B, 1, T_mel * upsample_total)

        x = torch.tanh(x)
        # x: (B, 1, T_samples) - [-1, 1] の音声波形

        return x


class Token2WavModel(nn.Module):
    """
    Qwen2.5-Omni Token2Wav

    音声コードトークンから音声波形への完全なパイプライン

    パイプライン:
        コードトークン (B, L_codec)
        → DiT (Flow-Matching): コード → メルスペクトログラム
        → BigVGAN (Vocoder): メル → 音声波形

    ストリーミングモード:
        コードチャンクが到着するたびに段階的に変換
        オーバーラップコンテキストでシームレスな結合
    """

    def __init__(self):
        super().__init__()

        self.code2wav_dit_model = DiTModel(
            hidden_size=1024,
            num_layers=12,
            num_heads=16,
            mel_dim=160,
        )
        self.code2wav_bigvgan_model = BigVGANModel(mel_dim=160)

    def forward(
        self,
        code: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
        return_audio_in_chunk: bool = False,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float = -1.0,
    ):
        """
        Token2Wav のフォワードパス

        入力:
            code: (B, L_codec) - 音声コードトークン
            conditioning: (B, T, dim) - 条件ベクトル (optional)
            reference_mel: (B, T, 160) - 参照メル (optional)
            return_audio_in_chunk: bool - ストリーミングモードか
            num_steps: int - DiTの推論ステップ数
            guidance_scale: float - ガイダンス強度
            sway_coefficient: float - スウェイ係数

        出力:
            非ストリーミング: waveform (1, N_samples) at 24kHz
            ストリーミング: Generator[torch.Tensor] - チャンクごとの波形
        """

        if not return_audio_in_chunk:
            return self._forward_full(code, conditioning, reference_mel,
                                       num_steps, guidance_scale, sway_coefficient)
        else:
            return self._forward_streaming(code, conditioning, reference_mel,
                                            num_steps, guidance_scale, sway_coefficient)

    def _forward_full(
        self,
        code: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        reference_mel: Optional[torch.Tensor],
        num_steps: int,
        guidance_scale: float,
        sway_coefficient: float,
    ) -> torch.Tensor:
        """
        非ストリーミング (一括) 生成

        処理フロー:
            1. DiT: コード → メルスペクトログラム
            2. (DiT削除 + CUDAキャッシュクリア) ← Low-VRAMモード
            3. BigVGAN: メル → 波形
        """

        # ========================================
        # Stage 1: DiT (Flow-Matching)
        # ========================================
        mel_spectrogram = self.code2wav_dit_model.sample(
            conditioning=conditioning,
            reference_mel=reference_mel,
            code=code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )
        # mel_spectrogram: (B, T_mel, 160)

        # Low-VRAM最適化: DiTモデルを削除してメモリ解放
        # del self.code2wav_dit_model
        # torch.cuda.empty_cache()

        # ========================================
        # Stage 2: BigVGAN (Vocoder)
        # ========================================
        waveform = self.code2wav_bigvgan_model(mel_spectrogram)
        # waveform: (B, 1, N_samples)
        # N_samples = T_mel × 240

        return waveform.squeeze(0)
        # waveform: (1, N_samples) at 24kHz

    def _forward_streaming(
        self,
        code: torch.Tensor,
        conditioning: Optional[torch.Tensor],
        reference_mel: Optional[torch.Tensor],
        num_steps: int,
        guidance_scale: float,
        sway_coefficient: float,
    ) -> Generator[torch.Tensor, None, None]:
        """
        ストリーミング生成

        2段階のチャンクストリーミング:

        Stage 1: DiT ストリーミング
            dit_chunk_size=48 コードずつ処理
            左コンテキスト: 24コード (lookback)
            右コンテキスト: 12コード (lookahead)
            → メルチャンクを段階的に出力

        Stage 2: BigVGAN ストリーミング
            vocoder_left_context=20 メルフレーム
            vocoder_right_context=20 メルフレーム
            vocoder_upsample_rate=240 (1メルフレーム → 240サンプル)
            → 波形チャンクを段階的に出力
        """

        # DiT ストリーミングパラメータ
        dit_chunk_size = 48        # コードチャンクサイズ
        dit_left_context = 24      # 左コンテキスト (lookback)
        dit_right_context = 12     # 右コンテキスト (lookahead)

        # BigVGAN ストリーミングパラメータ
        vocoder_left_context = 20   # メルフレーム
        vocoder_right_context = 20  # メルフレーム
        vocoder_upsample_rate = 240 # 1メル → 240サンプル

        B, L_codec = code.shape

        # ========================================
        # Stage 1: DiT チャンク処理
        # ========================================

        # 初期ノイズ (全長分を事前生成)
        repeats = 4  # 1コード → 4メルフレーム
        total_mel_len = L_codec * repeats
        noise = torch.randn(1, total_mel_len, 160, device=code.device)
        # noise: (1, 30000, 160) - 事前生成ノイズ

        code_buffer = torch.tensor([], device=code.device, dtype=code.dtype)
        mel_buffer = []
        mel_offset = 0

        for chunk_start in range(0, L_codec, dit_chunk_size):
            chunk_end = min(chunk_start + dit_chunk_size, L_codec)
            code_chunk = code[:, chunk_start:chunk_end]
            # code_chunk: (1, <=48)

            # バッファに追加
            code_buffer = torch.cat([code_buffer.unsqueeze(0) if code_buffer.dim() == 1 else code_buffer,
                                      code_chunk], dim=1) if code_buffer.numel() > 0 else code_chunk

            # チャンクが十分な長さになったら処理
            required_len = dit_left_context + dit_chunk_size + dit_right_context
            if code_buffer.shape[1] >= required_len or chunk_end == L_codec:

                # DiT sample_chunk で部分的にメル生成
                mel_start = mel_offset
                mel_duration = code_buffer.shape[1] * repeats
                mel_end = mel_start + mel_duration

                noise_chunk = noise[:, mel_start:mel_end, :]
                # noise_chunk: (1, chunk_mel_len, 160)

                # Flow-Matching サンプリング
                mel_chunk = self.code2wav_dit_model.sample(
                    conditioning=conditioning,
                    reference_mel=reference_mel,
                    code=code_buffer,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
                # mel_chunk: (1, chunk_mel_len, 160)

                # 左右コンテキストをトリム
                left_trim = dit_left_context * repeats if mel_offset > 0 else 0
                right_trim = dit_right_context * repeats if chunk_end < L_codec else 0
                if right_trim > 0:
                    mel_trimmed = mel_chunk[:, left_trim:-right_trim, :]
                else:
                    mel_trimmed = mel_chunk[:, left_trim:, :]

                mel_buffer.append(mel_trimmed)
                mel_offset = mel_end - (dit_left_context + dit_right_context) * repeats

                # ========================================
                # Stage 2: BigVGAN チャンク処理
                # ========================================

                # メルバッファが十分な長さなら波形に変換
                mel_concat = torch.cat(mel_buffer, dim=1)
                # mel_concat: (1, accumulated_mel_len, 160)

                if mel_concat.shape[1] > vocoder_left_context + vocoder_right_context:
                    waveform_chunk = self.code2wav_bigvgan_model(mel_concat)
                    # waveform_chunk: (1, 1, mel_len * 240)

                    # コンテキストトリム
                    left_samples = vocoder_left_context * vocoder_upsample_rate
                    right_samples = vocoder_right_context * vocoder_upsample_rate

                    if right_samples > 0:
                        waveform_trimmed = waveform_chunk[:, :, left_samples:-right_samples]
                    else:
                        waveform_trimmed = waveform_chunk[:, :, left_samples:]

                    yield waveform_trimmed.squeeze()
                    # waveform_trimmed: (N_samples,) at 24kHz

                    # バッファを更新 (コンテキスト分を保持)
                    mel_buffer = [mel_concat[:, -vocoder_left_context - vocoder_right_context:, :]]

                # コードバッファを更新 (左コンテキスト分を保持)
                code_buffer = code_buffer[:, -(dit_left_context + dit_right_context):]


# ============================================
# 使用例
# ============================================

def example_token2wav():
    """
    Token2Wav の使用例

    DiTModel, BigVGANModel, Token2WavModel を実際にインスタンス化し、
    各ステージのフォワードパスを実行して形状を確認する
    """

    # ========================================
    # DiTModel 単体の確認
    # ========================================
    dit = DiTModel(
        hidden_size=256,   # 実モデルは1024
        num_layers=2,      # 実モデルは12
        num_heads=4,       # 実モデルは16
        mel_dim=160,
        cond_dim=128,      # 実モデルは512
    )
    dit.eval()

    B = 1
    L_codec = 50  # 50コードトークン
    repeats = 4
    T_mel = L_codec * repeats  # 200メルフレーム

    # DiT フォワードパス (1ステップ)
    noise = torch.randn(B, T_mel, 160)
    code = torch.randint(0, 100, (B, L_codec))
    cond = dit.code_embedding(code)  # (1, 50, 128)
    timestep = torch.tensor([0.5])

    with torch.no_grad():
        velocity = dit(noise, cond, reference_mel=None, timestep=timestep)
    assert velocity.shape == (B, T_mel, 160)

    # DiT サンプリング (Flow-Matching)
    with torch.no_grad():
        mel_generated = dit.sample(
            conditioning=cond,
            reference_mel=None,
            code=code,
            num_steps=5,       # 高速化のため5ステップ
            guidance_scale=0.5,
        )
    assert mel_generated.shape == (B, T_mel, 160)

    # ========================================
    # BigVGANModel 単体の確認
    # ========================================
    bigvgan = BigVGANModel(mel_dim=160)
    bigvgan.eval()

    with torch.no_grad():
        waveform = bigvgan(mel_generated)
    # waveform: (B, 1, T_mel * upsample_total)
    assert waveform.shape[0] == B
    assert waveform.shape[1] == 1

    # ========================================
    # Token2WavModel (パイプライン全体)
    # ========================================
    token2wav = Token2WavModel()
    token2wav.eval()

    code_input = torch.randint(0, 100, (B, L_codec))

    # 非ストリーミング
    with torch.no_grad():
        waveform_full = token2wav(
            code=code_input,
            num_steps=5,
            guidance_scale=0.5,
            return_audio_in_chunk=False,
        )
    # waveform_full: (1, N_samples)

    N_samples = waveform_full.shape[-1]
    duration_sec = N_samples / 24000

    print(f"[Token2Wav 使用例]")
    print()
    print(f"  [DiT (Flow-Matching)]")
    print(f"    コード埋め込み: code {code.shape} → cond {cond.shape}")
    print(f"    ノイズ:         {noise.shape}  (B, T_mel, mel_dim)")
    print(f"    速度場 (1step): {velocity.shape}")
    print(f"    サンプリング:   {mel_generated.shape}  (5 Euler steps)")
    print()
    print(f"  [BigVGAN (Vocoder)]")
    print(f"    入力メル:       {mel_generated.shape}  (B, T_mel, 160)")
    print(f"    出力波形:       {waveform.shape}  (B, 1, N_samples)")
    print()
    print(f"  [Token2Wav パイプライン]")
    print(f"    入力:  code {code_input.shape}  ({L_codec}コードトークン)")
    print(f"    出力:  waveform {waveform_full.shape}  ({N_samples}サンプル ≈ {duration_sec:.2f}秒 @24kHz)")
    print()
    print(f"  [ストリーミングパラメータ (実モデル)]")
    print(f"    DiT: chunk=48コード, left_ctx=24, right_ctx=12")
    print(f"    BigVGAN: left_ctx=20メルフレーム, right_ctx=20")
    print(f"    upsample: 1メルフレーム → 240サンプル")


if __name__ == "__main__":
    example_token2wav()
