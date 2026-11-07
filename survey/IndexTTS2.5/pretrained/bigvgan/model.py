"""BigVGANネットワーク全体。外部vocoder実装をimportせず、checkpointだけを読む。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、F=Mel bin数、M=Mel長、S=出力sample数。入力[B,F,M]→mono波形[B,1,S]。
"""
from __future__ import annotations

from typing import Any
import json

from . import activations
from .act import Activation1d as TorchActivation1d
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils import weight_norm
import torch


class AttrDict(dict):
    """BigVGAN設定を辞書と属性の両形式で参照する。"""

    def __init__(self, values: dict[str, Any]) -> None:
        """JSON設定を登録し、公式と同じ属性アクセスを提供する。

        Args:
            values: JSONから読み込んだ設定辞書。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__(values)
        self.__dict__ = self


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """時間長を維持する対称Conv padding数を計算する。

    Args:
        kernel_size: 畳み込みまたはフィルタのカーネル長。
        dilation: 畳み込みのdilation。複数指定では各層の整数列。

    Returns:
        左右それぞれに追加するpadding長。
    """
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(module: nn.Module,
                 mean: float = 0.0,
                 std: float = 0.01) -> None:
    """BigVGANのConv重みを初期化する。

    pretrained読込時には重みで置換される。

    Args:
        module: 構築済みのPyTorchモジュール。
        mean: 重み初期化の平均値。
        std: 重み初期化の標準偏差。

    Returns:
        None。対象モジュールの重み・バイアスを更新する。
    """
    if 'Conv' in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


def load_hparams_from_json(path: Any) -> AttrDict:
    """JSONからBigVGANの特徴チャネル数・upsample倍率・活性化等の設定を読む。

    Args:
        path: 読み書きするファイルのパス。

    Returns:
        JSONの内容を属性でも参照できるAttrDict。
    """
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class AMPBlock1(torch.nn.Module):
    """各dilationの2段の畳み込みと周期活性化を残差接続する。"""

    def __init__(self,
                 h: AttrDict,
                 channels: int,
                 kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5),
                 activation: str = None) -> None:
        """各dilationの2段の畳み込みと周期活性化を残差接続するための層と設定を構築する。

        channels=Convの特徴channel数。

        Args:
            h: BigVGANのチャネル数・拡大率・活性化等の設定。
            channels: 畳み込みの特徴チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            dilation: 畳み込みのdilation。複数指定では各層の整数列。
            activation: 活性化関数の名前または構築済みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()

        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )) for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                )) for _ in range(len(dilation))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2)  # weight normを解除する対象層数を記録する。

        # 周期活性化にはローカルのPyTorch実装を使用する。
        if self.h.get("use_cuda_kernel", False):
            raise ValueError(
                "CUDA extension is disabled; use the local PyTorch path")
        else:
            Activation1d = TorchActivation1d

        # 各畳み込みの前に置くalias抑制付き周期活性化。
        if activation == "snake":
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.Snake(
                    channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == "snakebeta":
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.SnakeBeta(
                    channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(
                ("activation incorrectly specified. check "
                 "the config file and look for 'activation'."))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """各dilationの2段の畳み込みと周期活性化を残差接続する。

        入力/出力[B,C,T]。複数dilationのConvを残差接続し時間長を維持する。 B=発話数、C=channel数、T=sample数。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。

        Returns:
            残差加算後の特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。
        """
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self) -> None:
        """推論用にweight_g/weight_vを実重みへ統合する。

        演算上の重みは同じ。

        Returns:
            None。正規化を統合した重みへインスタンスを更新する。
        """
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """各dilationの畳み込みと周期活性化を残差接続する。"""

    def __init__(self,
                 h: AttrDict,
                 channels: int,
                 kernel_size: int = 3,
                 dilation: tuple = (1, 3, 5),
                 activation: str = None) -> None:
        """各dilationの畳み込みと周期活性化を残差接続するための層と設定を構築する。

        channels=Convの特徴channel数。

        Args:
            h: BigVGANのチャネル数・拡大率・活性化等の設定。
            channels: 畳み込みの特徴チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            dilation: 畳み込みのdilation。複数指定では各層の整数列。
            activation: 活性化関数の名前または構築済みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()

        self.h = h

        self.convs = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d),
                )) for d in dilation
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # weight normを解除する対象層数を記録する。

        # 周期活性化にはローカルのPyTorch実装を使用する。
        if self.h.get("use_cuda_kernel", False):
            raise ValueError(
                "CUDA extension is disabled; use the local PyTorch path")
        else:
            Activation1d = TorchActivation1d

        # 各畳み込みの前に置くalias抑制付き周期活性化。
        if activation == "snake":
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.Snake(
                    channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == "snakebeta":
            self.activations = nn.ModuleList([
                Activation1d(activation=activations.SnakeBeta(
                    channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(
                ("activation incorrectly specified. check "
                 "the config file and look for 'activation'."))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """各dilationの畳み込みと周期活性化を残差接続する。

        入力/出力[B,C,T]。周期活性化とConvを残差接続して時間長を維持する。 B=発話数、C=channel数、T=sample数。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。

        Returns:
            残差加算後の特徴[B,C,T]。Bは発話数、Cはチャネル数、Tは時間長。
        """
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

    def remove_weight_norm(self) -> None:
        """推論用にweight_g/weight_vを実重みへ統合する。

        演算上の重みは同じ。

        Returns:
            None。正規化を統合した重みへインスタンスを更新する。
        """
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(nn.Module):
    """Melを段階的に拡大し、周期活性化付き残差層から波形を生成する。"""

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False) -> None:
        """Melを段階的に拡大し、周期活性化付き残差層から波形を生成するための層と設定を構築する。

        Args:
            h: BigVGANのチャネル数・拡大率・活性化等の設定。
            use_cuda_kernel: 外部専用CUDA kernelを使うか。本実装ではFalseのみ対応。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # 周期活性化にはローカルのPyTorch実装を使用する。
        # 実験用は公式torch経路をローカル実装で使う。CUDA拡張のcompileは行わない。
        if use_cuda_kernel:
            raise ValueError("Use the implemented PyTorch anti-alias path")
        Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Melまたは入力特徴を内部チャネル数へ射影する先頭畳み込み。
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # 設定からAMPBlockの種類を選ぶ。既定は2段畳み込みのAMPBlock1。
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(
                "Incorrect resblock class specified in hyperparameters. "
                f"Got {h.resblock}")

        # 時間拡大は転置畳み込みで行う。この層自体にはalias抑制を加えない。
        self.ups = nn.ModuleList()
        for i, (u,
                k) in enumerate(zip(h.upsample_rates,
                                    h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList([
                    weight_norm(
                        ConvTranspose1d(
                            h.upsample_initial_channel // (2**i),
                            h.upsample_initial_channel // (2**(i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2,
                        ))
                ]))

        # 異なるカーネル長のAMP残差ブロックで周期的な特徴を抽出する。
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation))

        # 内部特徴をmono波形の1チャネルへ射影する末尾畳み込み。
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake" else
            (activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
             if h.activation == "snakebeta" else None))
        if activation_post is None:
            raise NotImplementedError(
                ("activation incorrectly specified. check "
                 "the config file and look for 'activation'."))

        self.activation_post = Activation1d(activation=activation_post)

        # 最終畳み込みのbiasは設定で選び、省略時は有効にする。
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))

        # 公開モデルと同じ規則で畳み込み重みを初期化する。
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # 設定が有効なら最終波形にtanhを適用する。省略時は有効。
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Melまたは入力特徴を内部チャネル数へ射影する先頭畳み込み。
        """Melを段階的に拡大し、周期活性化付き残差層から波形を生成する。

        Args:
            x: Mel[B,F,M]。Bは発話数、FはMel bin数、MはMelフレーム数。

        Returns:
            波形[B,1,S]。Bは発話数、1はmonoチャネル、Sは生成サンプル数。
        """
        # Mel[B,F,M]→内部特徴[B,C0,M]。Bは発話数、FはMel bin数、MはMel長、C0は初期チャネル数。
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # 転置畳み込みで時間長を拡大し、特徴チャネル数を減らす。
            for i_up in range(len(self.ups[i])):
                # 転置畳み込みで時間を拡大する。[B,Cin,Ti]→[B,Cout,Ti_next]。Ti_nextは段iのstride・
                # kernel・paddingで決まる。
                x = self.ups[i][i_up](x)
            # 異なるカーネルのAMP出力を加算し、ブロック数で平均する。
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # 内部特徴をmono波形の1チャネルへ射影する末尾畳み込み。
        x = self.activation_post(x)
        # 最終特徴[B,Clast,S]→mono波形[B,1,S]。Clastは最終内部特徴の次元数、Sは波形サンプル数。
        x = self.conv_post(x)
        # 最終波形へtanhを適用する。
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # tanh無効時は値域を[-1,1]へclipする。

        return x

    def remove_weight_norm(self) -> None:
        """推論用にweight_g/weight_vを実重みへ統合する。

        演算上の重みは同じ。

        Returns:
            None。正規化を統合した重みへインスタンスを更新する。
        """
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass
