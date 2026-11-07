"""S2MelのWaveNet出力分岐。条件付き膨張Convと残差/skipを実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=時間位置数、D=特徴次元数。
"""
from __future__ import annotations

from typing import Any

from torch import nn
import torch

from models.common import fused_add_tanh_sigmoid_multiply
from models.s2mel.convolution import SConv1d


class WN(torch.nn.Module):
    """時刻条件を加えた膨張Convのgate、残差、skipを集約する。
    """

    def __init__(self,
                 hidden_channels: Any,
                 kernel_size: int,
                 dilation_rate: int,
                 n_layers: int,
                 gin_channels: int = 0,
                 p_dropout: float = 0,
                 causal: bool = False) -> None:
        """時刻条件を加えた膨張Convのgate、残差、skipを集約するための層と設定を構築する。

        Args:
            hidden_channels: WaveNetの内部チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            dilation_rate: 層の深さに応じたdilationの倍率。
            n_layers: WaveNet層数。
            gin_channels: WaveNetへ加える条件特徴のチャネル数。
            p_dropout: WaveNet内のdropout確率。
            causal: 未来の時刻を参照しない処理にするか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(WN, self).__init__()
        conv1d_type = SConv1d
        assert (kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = conv1d_type(gin_channels,
                                          2 * hidden_channels * n_layers,
                                          1,
                                          norm='weight_norm')

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = conv1d_type(hidden_channels,
                                   2 * hidden_channels,
                                   kernel_size,
                                   dilation=dilation,
                                   padding=padding,
                                   norm='weight_norm',
                                   causal=causal)
            self.in_layers.append(in_layer)

            # 最終層には後続層への残差分岐が不要なため、skip出力だけを作る。
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = conv1d_type(hidden_channels,
                                         res_skip_channels,
                                         1,
                                         norm='weight_norm',
                                         causal=causal)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self,
                x: torch.Tensor,
                x_mask: Any,
                g: Any = None,
                **kwargs: Any) -> torch.Tensor:
        """時刻条件を加えた膨張Convのgate、残差、skipを集約する。

        Args:
            x: WaveNet入力[B,C,T]。Bは発話数、Cは内部チャネル数、Tは時間長。
            x_mask: 有効位置mask[B,1,T]。Bは発話数、Tは時間長。真が有効。
            g: 任意のWaveNet条件[B,Cg,1]または[B,Cg,T]。Bは発話数、Cgは条件の特徴次元数、Tは時間長。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            残差・skipを統合した特徴[B,C,T]。Bは発話数、Cは内部チャネル数、Tは時間長。
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:,
                        cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l,
                                                   n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        """推論用にweight_g/weight_vを実重みへ統合する。

        演算上の重みは同じ。

        Returns:
            None。正規化を統合した重みへインスタンスを更新する。
        """
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)
