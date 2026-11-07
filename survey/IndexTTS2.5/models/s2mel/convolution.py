"""WaveNetで使う正規化付きConvと、左右padding/stride処理。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=時間位置数、D=特徴次元数。
"""
from __future__ import annotations

from typing import Any
import math
import typing as tp
import warnings

from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm
import einops
import torch


class ConvLayerNorm(nn.LayerNorm):
    """畳み込み特徴のチャネル軸を末尾へ移してLayerNormを計算する。"""

    def __init__(self, normalized_shape: tp.Union[int, tp.List[int],
                                                  torch.Size],
                 **kwargs: Any) -> None:
        """畳み込み特徴のチャネル軸を末尾へ移してLayerNormを計算するための層と設定を構築する。

        Args:
            normalized_shape: LayerNormが正規化する末尾軸の大きさ。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x: torch.Tensor) -> None:
        """畳み込み特徴のチャネル軸を末尾へ移してLayerNormを計算する。

        入力/出力[B,C,T]。正規化計算では[B,T,C]へ変換する。 B=発話数、C=channel数、T=時間長。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            None。参照元のこの未使用分岐には戻り値がなく、その挙動を保持している。対象の公開S2Melはこの正規化方式を選択しない。
        """
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return


CONV_NORMALIZATIONS = frozenset([
    'none', 'weight_norm', 'spectral_norm', 'time_layer_norm', 'layer_norm',
    'time_group_norm'
])


def apply_parametrization_norm(module: nn.Module,
                               norm: str = 'none') -> nn.Module:
    """指定したweight/spectral normをConvへ適用する。

    層の演算定義は保持する。

    Args:
        module: 構築済みのPyTorchモジュール。
        norm: 使用する正規化方式または構築済み正規化層。

    Returns:
        指定した重み正規化を適用したモジュール。
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # 重みの再パラメータ化が不要な方式では元の畳み込みを返す。
        # 出力特徴の正規化は別の層で処理する。
        return module


def get_norm_module(module: nn.Module,
                    causal: bool = False,
                    norm: str = 'none',
                    **norm_kwargs: Any) -> nn.Module:
    """指定した方式と因果性の制約に合う正規化層を返す。

    Args:
        module: 構築済みのPyTorchモジュール。
        causal: 未来の時刻を参照しない処理にするか。
        norm: 使用する正規化方式または構築済み正規化層。
        **norm_kwargs: 正規化層のコンストラクタへ渡す追加引数。

    Returns:
        特徴正規化層、または正規化しないnn.Identity。
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor,
                                 kernel_size: int,
                                 stride: int,
                                 padding_total: int = 0) -> int:
    """最後の畳み込み窓を満たすために必要な右padding長を求める。

    Args:
        x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。
        kernel_size: 畳み込みまたはフィルタのカーネル長。
        stride: 畳み込み窓を移動する間隔。
        padding_total: 通常の畳み込みで追加する左右paddingの合計。

    Returns:
        右端へ追加する波形または特徴のサンプル数。
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size -
                                                         padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor,
                   kernel_size: int,
                   stride: int,
                   padding_total: int = 0) -> torch.Tensor:
    """末尾の畳み込み窓が途中で切れないように右側をpaddingする。

    Args:
        x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。
        kernel_size: 畳み込みまたはフィルタのカーネル長。
        stride: 畳み込み窓を移動する間隔。
        padding_total: 通常の畳み込みで追加する左右paddingの合計。

    Returns:
        末尾へ必要なpaddingを追加した特徴[B,C,Tpad]。Bは発話数、Cは特徴チャネル数、Tpadは補完後長。
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride,
                                                 padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(x: torch.Tensor,
          paddings: tp.Tuple[int, int],
          mode: str = 'zero',
          value: float = 0.0) -> torch.Tensor:
    """短い入力へのreflect処理を含め、時間軸の左右をpaddingする。

    Args:
        x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。
        paddings: 左側・右側へ追加または除去する長さの組。
        mode: paddingの方法。reflectでは短い入力も反射可能な長さへ補う。
        value: value特徴[B,M,D]。Bは発話数、Mはkey数、Dは特徴次元数。

    Returns:
        左右を補完した特徴[B,C,T+Pl+Pr]。Bは発話数、Cは特徴チャネル数、Tは入力長、Pl・Prは左右padding長。
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left,
                                                      padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]) -> torch.Tensor:
    """時間軸の左右から指定したpaddingを取り除く。

    Args:
        x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。
        paddings: 左側・右側へ追加または除去する長さの組。

    Returns:
        左右のpaddingを除いた特徴[B,C,T-Pl-Pr]。Bは発話数、Cは特徴チャネル数、Tは入力長、Pl・Prは除去長。
    """
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left,
                                                      padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    """1次元畳み込みの重み正規化と出力正規化を管理する。"""

    def __init__(self,
                 *args: Any,
                 causal: bool = False,
                 norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 **kwargs: Any) -> None:
        """1次元畳み込みの重み正規化と出力正規化を管理するための層と設定を構築する。

        Args:
            causal: 未来の時刻を参照しない処理にするか。
            norm: 使用する正規化方式または構築済み正規化層。
            norm_kwargs: 正規化層のコンストラクタへ渡す追加引数。
            *args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs),
                                               norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1次元畳み込みの重み正規化と出力正規化を管理する。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            畳み込みまたはLSTM処理後の特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、
            Toutはカーネル・stride・paddingで決まる長さ。2次元層では周波数軸Foutも保持する。
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Module):
    """2次元畳み込みの重み正規化と出力正規化を管理する。"""

    def __init__(self,
                 *args: Any,
                 norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 **kwargs: Any) -> None:
        """2次元畳み込みの重み正規化と出力正規化を管理するための層と設定を構築する。

        Args:
            norm: 使用する正規化方式または構築済み正規化層。
            norm_kwargs: 正規化層のコンストラクタへ渡す追加引数。
            *args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs),
                                               norm)
        self.norm = get_norm_module(self.conv,
                                    causal=False,
                                    norm=norm,
                                    **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """2次元畳み込みの重み正規化と出力正規化を管理する。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            畳み込みまたはLSTM処理後の特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、
            Toutはカーネル・stride・paddingで決まる長さ。2次元層では周波数軸Foutも保持する。
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """1次元転置畳み込みの重み正規化と出力正規化を管理する。"""

    def __init__(self,
                 *args: Any,
                 causal: bool = False,
                 norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 **kwargs: Any) -> None:
        """1次元転置畳み込みの重み正規化と出力正規化を管理するための層と設定を構築する。

        Args:
            causal: 未来の時刻を参照しない処理にするか。
            norm: 使用する正規化方式または構築済み正規化層。
            norm_kwargs: 正規化層のコンストラクタへ渡す追加引数。
            *args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """1次元転置畳み込みの重み正規化と出力正規化を管理する。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            畳み込みまたはLSTM処理後の特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、
            Toutはカーネル・stride・paddingで決まる長さ。2次元層では周波数軸Foutも保持する。
        """
        x = self.convtr(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):
    """2次元転置畳み込みの重み正規化と出力正規化を管理する。"""

    def __init__(self,
                 *args: Any,
                 norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 **kwargs: Any) -> None:
        """2次元転置畳み込みの重み正規化と出力正規化を管理するための層と設定を構築する。

        Args:
            norm: 使用する正規化方式または構築済み正規化層。
            norm_kwargs: 正規化層のコンストラクタへ渡す追加引数。
            *args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr,
                                    causal=False,
                                    norm=norm,
                                    **norm_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """2次元転置畳み込みの重み正規化と出力正規化を管理する。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            畳み込みまたはLSTM処理後の特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、
            Toutはカーネル・stride・paddingで決まる長さ。2次元層では周波数軸Foutも保持する。
        """
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """strideと因果性に応じてpaddingした後に時間畳み込みを適用する。"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 causal: bool = False,
                 norm: str = 'none',
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 pad_mode: str = 'reflect',
                 **kwargs: Any) -> None:
        """strideと因果性に応じてpaddingした後に時間畳み込みを適用するための層と設定を構築する。

        Args:
            in_channels: 入力チャネル数。
            out_channels: 出力チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            stride: 畳み込み窓を移動する間隔。
            dilation: 畳み込みのdilation。複数指定では各層の整数列。
            groups: 畳み込みで分割するチャネルのグループ数。
            bias: 射影・畳み込みにバイアスを付けるか。
            causal: 未来の時刻を参照しない処理にするか。
            norm: 使用する正規化方式または構築済み正規化層。
            norm_kwargs: 正規化層のコンストラクタへ渡す追加引数。
            pad_mode: padding領域の値の作り方。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        # strideとdilationを同時に大きくする設定を検出する。
        if stride > 1 and dilation > 1:
            warnings.warn(
                'SConv1d has been initialized with stride > 1 and dilation > 1'
                f' (kernel_size={kernel_size} stride={stride}, '
                f'dilation={dilation}).')
        self.conv = NormConv1d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               causal=causal,
                               norm=norm,
                               norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """strideと因果性に応じてpaddingした後に時間畳み込みを適用する。

        入力[B,Cin,T]→出力[B,Cout,Tout]。Cin/Cout=channel数。
        Toutはstrideと右padding規則から決まる。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            畳み込みまたはLSTM処理後の特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、
            Toutはカーネル・stride・paddingで決まる長さ。2次元層では周波数軸Foutも保持する。
        """
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size -
                       1) * dilation + 1  # dilationを含めた実効カーネル長を計算する。
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride,
                                                     padding_total)
        if self.causal:
            # 因果設定では必要な文脈を左側へ補う。
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # strideによって左右の補完・除去量が異なる場合を扱う。
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding),
                      mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """時間方向の転置畳み込み後に余分なpadding領域を除く。"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 causal: bool = False,
                 norm: str = 'none',
                 trim_right_ratio: float = 1.0,
                 norm_kwargs: tp.Dict[str, tp.Any] = {},
                 **kwargs: Any) -> None:
        """時間方向の転置畳み込み後に余分なpadding領域を除くための層と設定を構築する。

        Args:
            in_channels: 入力チャネル数。
            out_channels: 出力チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            stride: 畳み込み窓を移動する間隔。
            causal: 未来の時刻を参照しない処理にするか。
            norm: 使用する正規化方式または構築済み正規化層。
            trim_right_ratio: 転置畳み込みの余剰長を右側から除く割合。
            norm_kwargs: 正規化層のコンストラクタへ渡す追加引数。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          causal=causal,
                                          norm=norm,
                                          norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            ("`trim_right_ratio` != 1.0 only "
                "makes sense for causal convolutions")
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """時間方向の転置畳み込み後に余分なpadding領域を除く。

        入力[B,Cin,T]→出力[B,Cout,Tout]。Cin/Cout=channel数。
        転置Conv後に左右paddingを除いてToutを決める。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            畳み込みまたはLSTM処理後の特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、
            Toutはカーネル・stride・paddingで決まる長さ。2次元層では周波数軸Foutも保持する。
        """
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # ここでは固定paddingだけを除き、末尾窓を満たす追加paddingは残す。
        # 追加paddingは最終出力を必要長へ切り出す段階で除く。
        # この層単独では対応するencoder入力長を保持していない。
        # そのため途中段で必要なサンプルを切り捨てない。
        if self.causal:
            # 設定した割合に従い、余剰paddingを右側から除く。
            # trim_right_ratio=1なら余剰分をすべて右側から除く。
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # strideによって左右の補完・除去量が異なる場合を扱う。
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class SLSTM(nn.Module):
    """時間軸をLSTMで処理し、設定に応じて入力を残差加算する。"""

    def __init__(self,
                 dimension: int,
                 num_layers: int = 2,
                 skip: bool = True) -> None:
        """時間軸をLSTMで処理し、設定に応じて入力を残差加算するための層と設定を構築する。

        Args:
            dimension: LSTMの入力・内部チャネル数。
            num_layers: 積み重ねる層数。
            skip: LSTM出力へ入力を加える残差接続を使うか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)
        self.hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """時間軸をLSTMで処理し、設定に応じて入力を残差加算する。

        Args:
            x: 畳み込み特徴[B,C,T]。Bは発話数、Cは特徴チャネル数、Tは時間長。2次元層では[B,C,F,T]でFは周波数長。

        Returns:
            畳み込みまたはLSTM処理後の特徴[B,Cout,Tout]。Bは発話数、Coutは出力の特徴次元数、
            Toutはカーネル・stride・paddingで決まる長さ。2次元層では周波数軸Foutも保持する。
        """
        x = x.permute(2, 0, 1)
        if self.training:
            y, _ = self.lstm(x)
        else:
            y, self.hidden = self.lstm(x, self.hidden)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y
