"""CAMPPlusの話者embedding抽出器。畳み込み・統計pooling・投影をローカル実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=fbank frame数、F=80bin、C=中間特徴の次元数。入力[B,T,F]→話者embedding[B,192]。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .layers import (
    BasicResBlock,
    CAMDenseTDNNBlock,
    DenseLayer,
    StatsPool,
    TDNNLayer,
    TransitLayer,
    get_nonlinear,
)
from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch


class FCM(nn.Module):
    """fbankの周波数軸をConvで圧縮し時間TDNNの入力へ変換する。
    """

    def __init__(self,
                 block: Any = BasicResBlock,
                 num_blocks: int = [2, 2],
                 m_channels: int = 32,
                 feat_dim: int = 80) -> None:
        """fbankの周波数軸をConvで圧縮し時間TDNNの入力へ変換するための層と設定を構築する。

        Args:
            block: 構築する残差ブロックのクラス。
            num_blocks: 残差ブロック数。段ごとの指定では整数のリスト。
            m_channels: 周波数畳み込みの内部チャネル数。
            feat_dim: 入力音響特徴の次元数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=2)
        self.layer2 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[1],
                                       stride=2)

        self.conv2 = nn.Conv2d(m_channels,
                               m_channels,
                               kernel_size=3,
                               stride=(2, 1),
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block: Any, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        """CAMPPlusの残差Conv層を指定block数とstrideで構築する。

        Args:
            block: 構築する残差ブロックのクラス。
            planes: 残差ブロックの出力チャネル数。
            num_blocks: 残差ブロック数。段ごとの指定では整数のリスト。
            stride: 畳み込み窓を移動する間隔。

        Returns:
            指定個数の残差ブロックを接続したnn.Sequential。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fbankの周波数軸をConvで圧縮し時間TDNNの入力へ変換する。

        Args:
            x: fbank[B,F,T]。Bは発話数、Fは周波数bin数、Tは時間長。

        Returns:
            特徴[B,C*Fout,T]。Bは発話数、Cは畳み込みチャネル数、Foutは圧縮後の周波数長、Tは時間長。
        """
        # fbankへ畳み込みチャネル軸を追加する。[B,F,T]→[B,1,F,T]。Bは発話数、Fは周波数bin数、Tは時間長。
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = rearrange(out, "b c f t -> b (c f) t")
        return out


class CAMPPlus(nn.Module):
    """fbankの局所文脈抽出と統計poolingから話者embeddingを計算する。
    """

    def __init__(self,
                 feat_dim: int = 80,
                 embedding_size: int = 512,
                 growth_rate: int = 32,
                 bn_size: int = 4,
                 init_channels: int = 128,
                 config_str: str = 'batchnorm-relu',
                 memory_efficient: bool = True) -> None:
        """fbankの局所文脈抽出と統計poolingから話者embeddingを計算するための層と設定を構築する。

        Args:
            feat_dim: 入力音響特徴の次元数。
            embedding_size: 話者埋め込みの出力次元数。
            growth_rate: 各dense TDNN層で追加するチャネル数。
            bn_size: ボトルネックの特徴次元数をgrowth_rateから決める倍率。
            init_channels: TDNN先頭のチャネル数。
            config_str: BatchNormやReLU等を並べる順序の指定。
            memory_efficient: 中間活性を再計算するgradient checkpointingを使うか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(channels,
                           init_channels,
                           5,
                           stride=2,
                           dilation=1,
                           padding=-1,
                           config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size,
                dilation) in enumerate(zip((12, 24, 16), (3, 3, 3),
                                           (1, 2, 2))):
            block = CAMDenseTDNNBlock(num_layers=num_layers,
                                      in_channels=channels,
                                      out_channels=growth_rate,
                                      bn_channels=bn_size * growth_rate,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      config_str=config_str,
                                      memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(channels,
                             channels // 2,
                             bias=False,
                             config_str=config_str))
            channels //= 2

        self.xvector.add_module('out_nonlinear',
                                get_nonlinear(config_str, channels))

        self.xvector.add_module('stats', StatsPool())
        self.xvector.add_module(
            'dense',
            DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fbankの局所文脈抽出と統計poolingから話者embeddingを計算する。

        Args:
            x: fbank[B,T,F]。Bは発話数、Tはフレーム数、Fはfeat_dim。

        Returns:
            話者埋め込み[B,E]。Bは発話数、Eはembedding_size。対象公開重みではE=192。
        """
        # fbank[B,T,F]→畳み込み入力[B,F,T]。Bは発話数、Tはフレーム数、Fは周波数bin数。
        x = x.permute(0, 2, 1)
        x = self.head(x)
        x = self.xvector(x)
        return x
