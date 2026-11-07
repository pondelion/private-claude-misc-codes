"""ConvNeXt/Vocosの局所実装。時間Conv→特徴FFN→残差を計算する。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=codec時刻数、C=Conv特徴次元数、D=FFN出力の特徴次元数。入力[B,C,T]→出力[B,T,D]。
"""
from __future__ import annotations

from typing import Any, Optional

from torch import nn
import torch


class ConvNeXtBlock(nn.Module):
    """時間方向のdepthwise畳み込みと特徴FFNを残差接続する。"""

    def __init__(self,
                 dim: int,
                 intermediate_dim: int,
                 layer_scale_init_value: float,
                 adanorm_num_embeddings: Optional[int] = None) -> None:
        """時間方向のdepthwise畳み込みと特徴FFNを残差接続するための層と設定を構築する。

        Args:
            dim: 特徴の次元数。
            intermediate_dim: 特徴FFNの中間次元数。
            layer_scale_init_value: 残差分岐に掛ける学習可能な係数の初期値。
            adanorm_num_embeddings: AdaLayerNormの条件カテゴリ数。
                Noneでは通常のLayerNorm。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3,
                                groups=dim)  # 特徴チャネルごとに独立した時間畳み込み。
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim,
                                 intermediate_dim)  # 時間長を維持し、各時刻の特徴を全結合層で混合する。
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
                      if layer_scale_init_value > 0 else None)

    def forward(
            self,
            x: torch.Tensor,
            cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """時間方向のdepthwise畳み込みと特徴FFNを残差接続する。

        特徴[B,C,T]→[B,C,T]。Conv後は[B,T,C]で特徴FFNを計算する。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cは内部チャネル数、Tは時間長。
            cond_embedding_id: AdaLayerNormの条件ID[B]。Bは発話数。Noneでは無条件。

        Returns:
            残差加算後の特徴[B,C,T]。Bは発話数、Cは内部チャネル数、Tは時間長。
        """
        residual = x
        x = self.dwconv(x)
        # FFN用に特徴軸を末尾へ移す。[B,C,T]→[B,T,C]。Bは発話数、Cは内部特徴の次元数、Tは時間長。
        x = x.transpose(1, 2)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # 畳み込み用の軸順へ戻す。[B,T,C]→[B,C,T]。Bは発話数、Cは内部特徴の次元数、Tは時間長。
        x = x.transpose(1, 2)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """条件IDから得るscaleとshiftでLayerNorm出力を変調する。"""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 eps: float = 1e-06) -> None:
        """条件IDから得るscaleとshiftでLayerNorm出力を変調するための層と設定を構築する。

        eps=正規化分母の安定化係数。

        Args:
            num_embeddings: 条件IDのカテゴリ数。
            embedding_dim: 条件から生成する正規化係数の次元数。
            eps: 正規化の分母へ加える安定化定数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings,
                                  embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings,
                                  embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor,
                cond_embedding_id: torch.Tensor) -> torch.Tensor:
        """条件IDから得るscaleとshiftでLayerNorm出力を変調する。

        特徴[B,T,D]と条件ID[B]→[B,T,D]。B=発話数、T=位置数、D=特徴次元数。

        Args:
            x: 特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            cond_embedding_id: AdaLayerNormの条件ID[B]。Bは発話数。Noneでは無条件。

        Returns:
            条件で変調した特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim, ), eps=self.eps)
        x = x * scale + shift
        return x


class VocosBackbone(nn.Module):
    """入力畳み込み、ConvNeXt層群、最終LayerNormで特徴を変換する。"""

    def __init__(self,
                 input_channels: int,
                 dim: int,
                 intermediate_dim: int,
                 num_layers: int,
                 layer_scale_init_value: Optional[float] = None,
                 adanorm_num_embeddings: Optional[int] = None) -> None:
        """入力畳み込み、ConvNeXt層群、最終LayerNormで特徴を変換するための層と設定を構築する。

        Args:
            input_channels: 入力チャネル数。
            dim: 特徴の次元数。
            intermediate_dim: 特徴FFNの中間次元数。
            num_layers: 積み重ねる層数。
            layer_scale_init_value: 残差分岐に掛ける学習可能な係数の初期値。
            adanorm_num_embeddings: AdaLayerNormの条件カテゴリ数。
                Noneでは通常のLayerNorm。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                adanorm_num_embeddings=adanorm_num_embeddings,
            ) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Conv/Linearの重みをtruncated normal、biasをゼロで初期化する。

        Args:
            m: 初期化対象のPyTorchモジュール。

        Returns:
            None。対象モジュールの重み・バイアスを更新する。
        """
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """入力畳み込み、ConvNeXt層群、最終LayerNormで特徴を変換する。

        Args:
            x: 特徴[B,C,T]。Bは発話数、Cは入力の特徴次元数、Tは時間長。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            特徴[B,T,D]。Bは発話数、Tは時間長、DはVocos内部特徴の次元数。
        """
        bandwidth_id = kwargs.get("bandwidth_id", None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x
