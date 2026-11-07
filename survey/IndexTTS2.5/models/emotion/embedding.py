"""感情frontendのConformer部品。射影・位置情報・畳み込み・attentionを局所実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=入力特徴の時間長、D=特徴次元数、H=head数。hidden[B,T,D]、mask[B,1,T]。
"""
from __future__ import annotations

from typing import Tuple, Union
import math

from torch.nn import functional as F
import torch


class PositionalEncoding(torch.nn.Module):
    """sin/cos絶対位置特徴を入力へ加算する。"""

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False) -> None:
        """sin/cos絶対位置特徴を入力へ加算するための層と設定を構築する。

        Args:
            d_model: 位置特徴または隠れ状態の特徴次元数。
            dropout_rate: dropout確率。
            max_len: 位置表またはmaskの最大位置数。
            reverse: 位置表の並びを逆向きにするか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) *
            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int,
                      torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """sin/cos絶対位置特徴を入力へ加算する。

        Args:
            x: 特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            位置特徴を加算した入力[B,T,D]と位置特徴[1,T,D]または[B,T,D]の組。Bは発話数、Tは対象長、
            Dは位置埋め込みの次元数。
        """

        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        """指定した開始位置から必要な長さの位置特徴を取り出す。

        Args:
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。
            size: 取得する位置特徴の時間長。
            apply_dropout: 位置特徴にもdropoutを適用するか。

        Returns:
            位置特徴[1,T,D]または[B,T,D]。Bは発話数、Tはsize、Dは位置埋め込みの次元数。offsetが発話別なら発話軸を持つ。
        """
        # Union型の添字に関するPyTorchの参照情報。
        # 参照資料: https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size < self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        # 全発話に共通する整数offsetから位置を切り出す。
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:
            assert offset + size < self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:  # 発話ごとのoffset[B]を使い、GPU上で位置をまとめて取得する。
            assert torch.max(offset) + size < self.max_len
            # 位置index[B,T]。Bは発話数、Tは今回取得する位置数。
            index = offset.unsqueeze(1) + \
                torch.arange(0, size).to(offset.device)
            flag = index > 0
            # 負のoffsetを位置表の先頭へ丸める。
            index = index * flag
            # 位置特徴[B,T,D]。Bは発話数、Tは取得長、Dはd_model。
            pos_emb = F.embedding(index, self.pe[0])

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """入力特徴と相対位置attention用の位置特徴を返す。"""

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000) -> None:
        """入力特徴と相対位置attention用の位置特徴を返すための層と設定を構築する。

        Args:
            d_model: 位置特徴または隠れ状態の特徴次元数。
            dropout_rate: dropout確率。
            max_len: 位置表またはmaskの最大位置数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int,
                      torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """入力特徴と相対位置attention用の位置特徴を返す。

        Args:
            x: 特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            dropout後の入力[B,T,D]と相対位置特徴[1,T,D]または[B,T,D]の組。Bは発話数、Tは対象長、
            Dは位置埋め込みの次元数。
        """
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(torch.nn.Module):
    """位置特徴を加えずdropoutだけを適用する。"""

    def __init__(self, d_model: int, dropout_rate: float) -> None:
        """位置特徴を加えずdropoutだけを適用するための層と設定を構築する。

        Args:
            d_model: 位置特徴または隠れ状態の特徴次元数。
            dropout_rate: dropout確率。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        offset: Union[int,
                      torch.Tensor] = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """位置特徴を加えずdropoutだけを適用する。

        Args:
            x: 特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            dropout後の入力[B,T,D]とゼロ位置特徴[1,T,D]の組。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        """指定した開始位置から必要な長さの位置特徴を取り出す。

        Args:
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。
            size: 取得する位置特徴の時間長。

        Returns:
            位置特徴[1,T,D]または[B,T,D]。Bは発話数、Tはsize、Dは位置埋め込みの次元数。offsetが発話別なら発話軸を持つ。
        """
        return torch.zeros(1, size, self.d_model)
