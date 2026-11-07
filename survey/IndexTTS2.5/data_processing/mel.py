"""Slaney Mel filterbank・STFT振幅・log圧縮をNumPy/PyTorchで計算する。

外部IndexTTSのmel_spectrogramやlibrosaの特徴抽出関数を呼ばない。
公式と同じreflect padding、Hann窓、sqrt(real²+imag²+1e-9)、log clampを使う。
B=発話数、S=波形sample数、K=n_fft/2+1、F=Mel bin数、T=STFT時刻数。
"""
from __future__ import annotations

from torch import nn
from torch.nn import functional as functional
import numpy as np
import torch


def slaney_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float,
                      fmax: float | None) -> np.ndarray:
    """Slaney周波数尺度・面積正規化のMel行列[F,K]をfloat32で構築する。

    Args:
        sample_rate: 入力音声のサンプリング周波数、単位はHz。
        n_fft: FFTの点数。周波数bin数はn_fft // 2 + 1。
        n_mels: Mel周波数bin数。
        fmin: Melフィルタの最低周波数、単位はHz。
        fmax: Melフィルタの最高周波数。NoneではNyquist周波数。

    Returns:
        float32のMelフィルタ行列[F,K]。Fはn_mels、Kはn_fft // 2 + 1。
    """
    maximum = sample_rate / 2 if fmax is None else fmax
    step = np.log(6.4) / 27

    def hz_to_mel(value: float) -> float:
        """1kHz未満は線形、それ以上は対数尺度へ変換する。

        Args:
            value: 周波数のNumPy配列。任意shapeを保ち、値の単位HzをMel尺度へ変換する。

        Returns:
            入力と同shapeのMel周波数尺度のNumPy配列。
        """
        return value / (
            200 / 3) if value < 1000 else 15 + np.log(value / 1000) / step

    points = np.linspace(hz_to_mel(fmin), hz_to_mel(maximum), n_mels + 2)
    frequencies = (200 / 3) * points
    logarithmic = points >= 15
    frequencies[logarithmic] = 1000 * np.exp(step * (points[logarithmic] - 15))
    fft_frequencies = np.fft.rfftfreq(n=n_fft, d=1 / sample_rate)
    ramps = frequencies[:, None] - fft_frequencies[None]
    spacing = np.diff(frequencies)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for index in range(n_mels):
        lower = -ramps[index] / spacing[index]
        upper = ramps[index + 2] / spacing[index + 1]
        weights[index] = np.maximum(0, np.minimum(lower, upper))
    weights *= (2 / (frequencies[2:] - frequencies[:-2]))[:, None]
    return weights


class MelSpectrogram(nn.Module):
    """vocoderの設定値を保持する、波形→log Melの前処理。"""

    def __init__(self,
                 sampling_rate: int,
                 n_fft: int,
                 win_size: int,
                 hop_size: int,
                 num_mels: int,
                 fmin: float = 0,
                 fmax: float | None = None,
                 center: bool = False) -> None:
        """公式configのFFT/窓/hop/filterbank定義をそのまま受け取る。

        Args:
            sampling_rate: 音声のサンプリング周波数、単位はHz。
            n_fft: FFTの点数。周波数bin数はn_fft // 2 + 1。
            win_size: STFT窓長、単位は波形サンプル。
            hop_size: STFTの時間方向の移動幅、単位は波形サンプル。
            num_mels: Mel周波数bin数。
            fmin: Melフィルタの最低周波数、単位はHz。
            fmax: Melフィルタの最高周波数。NoneではNyquist周波数。
            center: STFTで各フレームの中心に窓を配置するか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.n_fft, self.win_size, self.hop_size = n_fft, win_size, hop_size
        self.center = center
        self.register_buffer(
            "basis",
            torch.from_numpy(
                slaney_filterbank(sampling_rate, n_fft, num_mels, fmin, fmax)))
        self.register_buffer("window", torch.hann_window(win_size))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """mono波形[B,S]→log Mel[B,F,T]。

        B/Tの意味を変えず特徴軸だけを投影。

        Args:
            waveform: 波形[B,S]。Bは発話数、Sは波形サンプル数。

        Returns:
            対数Mel[B,F,T]。Bは発話数、FはMel bin数、TはSTFTフレーム数。
        """
        # 1. STFTのframe境界を公開実装と合わせる。[B,S]→[B,S+2pad]。
        padding = int((self.n_fft - self.hop_size) / 2)
        padded = functional.pad(waveform[:, None], (padding, padding),
                                mode="reflect")[:, 0]
        # 2. 複素STFT[B,K,T]から振特徴次元数を計算する。
        # 時間窓ごとにFFTする。波形[B,S]→複素スペクトル[B,K,T]。K=n_fft//2+1、TはSTFTフレーム数。
        spectrum = torch.stft(padded,
                              self.n_fft,
                              hop_length=self.hop_size,
                              win_length=self.win_size,
                              window=self.window,
                              center=self.center,
                              pad_mode="reflect",
                              normalized=False,
                              onesided=True,
                              return_complex=True)
        magnitude = torch.sqrt(
            torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
        # 3. 周波数bin KをMel bin Fへ投影し、log clampする。
        return torch.log(torch.clamp(self.basis @ magnitude, min=1e-5))
