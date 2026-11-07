"""BigVGANのanti-alias filterとsample rate変換。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、C=channel数、T=変換前sample数、r=倍率。波形[B,C,T]→[B,C,rT]等。
"""
from __future__ import annotations

from typing import Any

from .resample import UpSample1d, DownSample1d
from torch import nn
import torch


class Activation1d(nn.Module):
    """upsample、周期活性化、downsampleでaliasを抑える。
    """

    def __init__(self,
                 activation: Any,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12) -> None:
        """upsample、周期活性化、downsampleでaliasを抑えるための層と設定を構築する。

        Args:
            activation: 活性化関数の名前または構築済みモジュール。
            up_ratio: 活性化の前に時間軸を拡大する整数倍率。
            down_ratio: 活性化の後に時間軸を縮小する整数倍率。
            up_kernel_size: 拡大用low-passフィルタの長さ。
            down_kernel_size: 縮小用low-passフィルタの長さ。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # 入力特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """upsample、周期活性化、downsampleでaliasを抑える。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。

        Returns:
            活性化後の特徴[B,C,Tout]。Bは発話数、Cはチャネル数、Toutは拡大・縮小率で決まる。両倍率が等しければTと同長。
        """
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x
