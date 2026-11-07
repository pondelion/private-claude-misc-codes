"""BigVGANの固定pretrainedモデル部品。周期活性化をローカルで実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、C=特徴channel数、T=時間sample数。入力/出力[B,C,T]。
"""
from __future__ import annotations

from torch import nn
from torch import sin, pow
from torch.nn import Parameter
import torch


class Snake(nn.Module):
    """x + sin(alpha*x)^2 / alphaによる周期活性化を適用する。"""

    def __init__(self,
                 in_features: int,
                 alpha: float = 1.0,
                 alpha_trainable: bool = True,
                 alpha_logscale: bool = False) -> None:
        """x + sin(alpha*x)^2 / alphaによる周期活性化を適用するための層と設定を構築する。

        Args:
            in_features: 入力チャネル数。
            alpha: 周期活性化の周波数係数の初期値。
            alpha_trainable: 周期係数を学習可能にするか。
            alpha_logscale: 周期係数を対数空間で保持するか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # 周期活性化の周波数係数alphaを初期化する。
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # 対数表現はゼロで初期化し、exp後の係数を1にする。
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # 通常表現は係数を1で初期化する。
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x + sin(alpha*x)^2 / alphaによる周期活性化を適用する。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。

        Returns:
            周期活性化後の特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(
            -1)  # 係数[C]を[1,C,1]へ展開し、発話軸B・時間軸Tへbroadcastする。
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """周波数係数alphaと振幅係数betaを分離した周期活性化を適用する。"""

    def __init__(self,
                 in_features: int,
                 alpha: float = 1.0,
                 alpha_trainable: bool = True,
                 alpha_logscale: bool = False) -> None:
        """周波数係数alphaと振幅係数betaを分離した周期活性化を適用するための層と設定を構築する。

        Args:
            in_features: 入力チャネル数。
            alpha: 周期活性化の周波数係数の初期値。
            alpha_trainable: 周期係数を学習可能にするか。
            alpha_logscale: 周期係数を対数空間で保持するか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # 周期活性化の周波数係数alphaを初期化する。
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # 対数表現はゼロで初期化し、exp後の係数を1にする。
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # 通常表現は係数を1で初期化する。
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """周波数係数alphaと振幅係数betaを分離した周期活性化を適用する。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。

        Returns:
            周期活性化後の特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(
            -1)  # 係数[C]を[1,C,1]へ展開し、発話軸B・時間軸Tへbroadcastする。
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
