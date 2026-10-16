"""Parakeet の音声前処理: waveform -> log-Mel + SpecAugment。

Shape convention: B=batch, S=sample, F=mel bin, T=10 ms frame.
外部音声ライブラリに依存せず、torch.stft と三角 Mel filter bank で実際に計算する。
"""
from __future__ import annotations

import torch
from torch import nn


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    """Hzを人間の聴覚尺度に近いMelへ写す。入力と出力のshapeは同じ。"""
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    """MelからHzへ戻す。filter bankの境界周波数を作るために使う。"""
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """三角Mel filter bank ``(F=n_mels, K=n_fft//2+1)`` を構築する。

    各行が1つのMel band、各列がSTFTの周波数binを表す。隣接する3境界
    ``left-center-right`` の間を線形に結び、centerで重み1になる三角形を作る。
    """
    # Mel上で等間隔に区切ってからHz/STFT binへ戻すことで、低域を細かく、
    # 高域を粗く見る人間の周波数知覚を近似する。
    mel_edges = torch.linspace(_hz_to_mel(torch.tensor(0.0)), _hz_to_mel(torch.tensor(sample_rate / 2)), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)
    bins = torch.floor((n_fft + 1) * hz_edges / sample_rate).long().clamp(max=n_fft // 2)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(n_mels):
        left, center, right = bins[m : m + 3].tolist()
        if center > left:
            fb[m, left:center] = torch.arange(center - left) / (center - left)
        if right > center:
            fb[m, center:right] = torch.arange(right - center, 0, -1) / (right - center)
    return fb


class LogMelFrontend(nn.Module):
    """16 kHz waveform ``(B,S)`` -> normalized log-Mel ``(B,F,T)``.

    ``lengths (B,)`` is in samples; returned lengths are valid 10 ms frames.
    The paper/model configuration uses F=128, 25 ms windows and 10 ms stride.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 512,
                 win_length: int = 400, hop_length: int = 160):
        super().__init__()
        self.sample_rate, self.n_mels = sample_rate, n_mels
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("fb", mel_filterbank(n_fft, n_mels, sample_rate), persistent=False)

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """padding済みwaveformを、有効長を保ったまま正規化log-Melへ変換する。

        Args:
            waveform: ``(B,S_max)``。末尾はzero padding。
            lengths: ``(B,)``。padding前のsample数。
        Returns:
            features: ``(B,F,T_max)``。
            frame_lengths: ``(B,)``。各sampleの有効frame数。
        """
        # STFT: complex spectrum (B,K,T) -> power spectrum (B,K,T)。
        spec = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length,
                          self.window, center=True, return_complex=True).abs().square()  # (B,K,T)
        # Mel filter bankを周波数軸Kへ掛け、log powerへ変換する。
        x = torch.einsum("fk,bkt->bft", self.fb.to(spec.dtype), spec).clamp_min(1e-10).log()
        frame_lengths = torch.div(lengths, self.hop_length, rounding_mode="floor") + 1  # (B,)
        frame_lengths.clamp_(max=x.shape[-1])
        # padding frameを統計量へ混ぜないためのvalid mask (B,T)。
        steps = torch.arange(x.shape[-1], device=x.device)[None, :]
        valid = steps < frame_lengths[:, None]  # (B,T)
        denom = valid.sum(1).clamp_min(1).to(x.dtype)[:, None]
        # NeMo設定の normalize="per_feature" に対応。各sample・各Mel binを
        # 時間方向だけで標準化する。mean/varはいずれも(B,F)。
        mean = (x * valid[:, None]).sum(2) / denom
        var = ((x - mean[:, :, None]).square() * valid[:, None]).sum(2) / denom
        x = (x - mean[:, :, None]) / var.clamp_min(1e-5).sqrt()[:, :, None]  # per-feature normalization
        return x.masked_fill(~valid[:, None], 0.0), frame_lengths


class SpecAugment(nn.Module):
    """Training-only frequency/time masking. Input/output ``(B,F,T)`` unchanged."""

    def __init__(self, freq_masks: int = 2, freq_width: int = 27, time_masks: int = 10, time_ratio: float = .05):
        super().__init__()
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_ratio = time_masks, time_ratio

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """有効領域内へランダムな長方形maskを入れる。

        frequency maskはMel bin方向、time maskはframe方向を0にする。
        token labelを変えずに音響的な欠損を作り、noiseや発音変動への頑健性を促す。
        ``eval()``では何も変更しない。
        """
        if not self.training:
            return x
        out = x.clone()
        for b in range(x.shape[0]):
            # sampleごとに異なるmaskを生成する。教材実装では読みやすさを優先してloopする。
            for _ in range(self.freq_masks):
                width = int(torch.randint(0, min(self.freq_width, x.shape[1]) + 1, ()).item())
                start = int(torch.randint(0, max(1, x.shape[1] - width + 1), ()).item())
                out[b, start:start + width] = 0
            for _ in range(self.time_masks):
                maximum = max(1, int(lengths[b].item() * self.time_ratio))
                width = int(torch.randint(0, maximum + 1, ()).item())
                start = int(torch.randint(0, max(1, int(lengths[b]) - width + 1), ()).item())
                out[b, :, start:start + width] = 0
        return out
