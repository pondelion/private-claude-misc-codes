"""BigVGANのanti-alias filterとsample rate変換。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、C=channel数、T=変換前sample数、r=倍率。波形[B,C,T]→[B,C,rT]等。
"""
from __future__ import annotations

from .filter import LowPassFilter1d, kaiser_sinc_filter1d
from torch import nn
from torch.nn import functional as F
import torch


class UpSample1d(nn.Module):
    """zero insertion相当の転置Convとlow-pass filterで時間軸を拡大する。
    """

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        """zero insertion相当の転置Convとlow-pass filterで時間軸を拡大するための層と設定を構築する。

        Args:
            ratio: 時間方向の整数拡大率または縮小率。
            kernel_size: 畳み込みまたはフィルタのカーネル長。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (int(6 * ratio // 2) *
                            2 if kernel_size is None else kernel_size)
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size -
                                                  self.stride) // 2
        self.pad_right = (self.pad * self.stride +
                          (self.kernel_size - self.stride + 1) // 2)
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # 入力特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """zero insertion相当の転置Convとlow-pass filterで時間軸を拡大する。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは入力時間長。

        Returns:
            拡大した特徴[B,C,T*ratio]。Bは発話数、Cはチャネル数、ratioは拡大倍率。
        """
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]

        return x


class DownSample1d(nn.Module):
    """low-pass filterを適用して時間sampleを間引く。
    """

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        """low-pass filterを適用して時間sampleを間引くための層と設定を構築する。

        Args:
            ratio: 時間方向の整数拡大率または縮小率。
            kernel_size: 畳み込みまたはフィルタのカーネル長。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (int(6 * ratio // 2) *
                            2 if kernel_size is None else kernel_size)
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """low-pass filterを適用して時間sampleを間引く。

        入力[B,C,T]→出力[B,C,Tout]。B=発話数、C=channel数、Toutはstrideとfilterのpaddingで決まる。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは入力時間長。

        Returns:
            間引いた特徴[B,C,Tout]。Bは発話数、Cはチャネル数、Toutはフィルタとstrideで決まる長さ。
        """
        xx = self.lowpass(x)

        return xx
