"""BigVGANのanti-alias filterとsample rate変換。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、C=channel数、T=変換前sample数、r=倍率。波形[B,C,T]→[B,C,rT]等。
"""
from __future__ import annotations

from typing import Any
import math

from einops import rearrange
from torch import nn
from torch import sinc
from torch.nn import functional as F
import torch


def kaiser_sinc_filter1d(cutoff: float, half_width: float,
                         kernel_size: Any) -> torch.Tensor:
    """cutoff/遷移帯域/窓長からKaiser窓付きsinc filter[1,1,K]を作る。

    Args:
        cutoff: サンプリング周波数で正規化した遮断周波数。
        half_width: フィルタの遷移帯域幅の半分。
        kernel_size: 畳み込みまたはフィルタのカーネル長。

    Returns:
        フィルタ係数[1,1,K]。最初の2軸は畳み込み用の単一出力・入力チャネル、Kはkernel_size。
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # 遷移帯域とカーネル長からKaiser窓の形状係数を求める。
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # 遮断周波数に応じてsincの幅と面積を調整する。
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # フィルタ係数の和を1へ正規化し、入力の直流成分の大きさを維持する。
        filter_ /= filter_.sum()
        filter = rearrange(filter_, "k -> 1 1 k")

    return filter


class LowPassFilter1d(nn.Module):
    """Kaiser窓付きsincフィルタで高周波成分を抑える。"""

    def __init__(self,
                 cutoff: float = 0.5,
                 half_width: float = 0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12) -> None:
        """Kaiser窓付きsincフィルタで高周波成分を抑えるための層と設定を構築する。

        Args:
            cutoff: サンプリング周波数で正規化した遮断周波数。
            half_width: フィルタの遷移帯域幅の半分。
            stride: 畳み込み窓を移動する間隔。
            padding: 左右に追加する長さ。フィルタではpaddingの有効・無効。
            padding_mode: padding領域の値の作り方。
            kernel_size: 畳み込みまたはフィルタのカーネル長。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # 入力特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Kaiser窓付きsincフィルタで高周波成分を抑える。

        入力[B,C,T]へgrouped filterを適用し[B,C,Tout]を返す。B=発話数、C=channel数、
        Toutはstrideとpaddingから決まる。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは入力時間長。

        Returns:
            フィルタ後の特徴[B,C,Tout]。Bは発話数、Cはチャネル数、Toutはpaddingとstrideで決まる長さ。
        """
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x,
                       self.filter.expand(C, -1, -1),
                       stride=self.stride,
                       groups=C)

        return out
