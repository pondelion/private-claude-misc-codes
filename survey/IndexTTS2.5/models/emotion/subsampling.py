"""感情frontendのConformer部品。射影・位置情報・畳み込み・attentionを局所実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=入力特徴の時間長、D=特徴次元数、H=head数。hidden[B,T,D]、mask[B,1,T]。
"""
from __future__ import annotations

from typing import Tuple, Union

from einops import rearrange
import torch


class BaseSubsampling(torch.nn.Module):
    """入力層の時間圧縮率と位置情報のインターフェースを定義する。
    """

    def __init__(self) -> None:
        """入力層の時間圧縮率と位置情報のインターフェースを定義するための層と設定を構築する。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        """指定した開始位置から必要な長さの位置特徴を取り出す。

        Args:
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。
            size: 取得する位置特徴の時間長。

        Returns:
            位置特徴[1,T,D]または[B,T,D]。Bは発話数、Tはsize、Dは位置埋め込みの次元数。offsetが発話別なら発話軸を持つ。
        """
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """時間長を維持して入力特徴を線形射影する。"""

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module) -> None:
        """時間長を維持して入力特徴を線形射影するための層と設定を構築する。

        Args:
            idim: 入力フレームの特徴次元数。
            odim: 出力フレームの特徴次元数。
            dropout_rate: dropout確率。
            pos_enc_class: 構築済みの位置埋め込みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """時間長を維持して入力特徴を線形射影する。

        Args:
            x: 音声特徴[B,Tin,Din]。Bは発話数、Tinは入力時間長、Dinは入力の特徴次元数。
            x_mask: 有効位置mask[B,1,T]。Bは発話数、Tは時間長。真が有効。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            特徴[B,Tout,Dout]、位置特徴[1,Tout,Dout]または[B,Tout,Dout]、
            有効mask[B,1,Tout]の順の組。Bは発話数、Toutは畳み込み後時間長、Doutは出力の特徴次元数。
            Toutは各カーネルとstrideの整数式で決まり、単純な割り算とは限らない。
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling3(BaseSubsampling):
    """2次元畳み込みで時間長を約3分の1へ圧縮する。"""

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module) -> None:
        """2次元畳み込みで時間長を約3分の1へ圧縮するための層と設定を構築する。

        Args:
            idim: 入力フレームの特徴次元数。
            odim: 出力フレームの特徴次元数。
            dropout_rate: dropout確率。
            pos_enc_class: 構築済みの位置埋め込みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, odim, 5, 3),
                                        torch.nn.ReLU())
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 2) // 3), odim))
        self.pos_enc = pos_enc_class
        # 各畳み込み層の右文脈を、入力フレーム単位へ換算して加算する。
        # 右文脈の増分=(カーネル長-1)×その層の入力間隔。
        self.subsampling_rate = 3
        # 右文脈を入力フレーム単位で計算する: 4 = (5 - 1) * 1。
        self.right_context = 4

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2次元畳み込みで時間長を約3分の1へ圧縮する。

        Args:
            x: 音声特徴[B,Tin,Din]。Bは発話数、Tinは入力時間長、Dinは入力の特徴次元数。
            x_mask: 有効位置mask[B,1,T]。Bは発話数、Tは時間長。真が有効。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            特徴[B,Tout,Dout]、位置特徴[1,Tout,Dout]または[B,Tout,Dout]、
            有効mask[B,1,Tout]の順の組。Bは発話数、Toutは畳み込み後時間長、Doutは出力の特徴次元数。
            Toutは各カーネルとstrideの整数式で決まり、単純な割り算とは限らない。
        """
        # 音声特徴へ畳み込みチャネル軸を追加する。[B,Tin,Fin]→[B,1,Tin,Fin]。Bは発話数、Tinは時間長、
        # Finは入力の特徴次元数。
        x = x.unsqueeze(1)
        # 時間・周波数を畳み込む。[B,1,Tin,Fin]→[B,C,Tconv,Fconv]。Cは畳み込みの出力チャネル数、
        # TconvとFconvはカーネル・
        # strideで決まる長さ。
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(rearrange(x, "b c t f -> b t (c f)"))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:3]


class Conv2dSubsampling2(BaseSubsampling):
    """2次元畳み込みで時間長を約2分の1へ圧縮する。"""

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module) -> None:
        """2次元畳み込みで時間長を約2分の1へ圧縮するための層と設定を構築する。

        Args:
            idim: 入力フレームの特徴次元数。
            odim: 出力フレームの特徴次元数。
            dropout_rate: dropout確率。
            pos_enc_class: 構築済みの位置埋め込みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # 各畳み込み層の右文脈を、入力フレーム単位へ換算して加算する。
        # 右文脈の増分=(カーネル長-1)×その層の入力間隔。
        self.subsampling_rate = 2
        # 右文脈を入力フレーム単位で計算する: 2 = (3 - 1) * 1。
        self.right_context = 2

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2次元畳み込みで時間長を約2分の1へ圧縮する。

        Args:
            x: 音声特徴[B,Tin,Din]。Bは発話数、Tinは入力時間長、Dinは入力の特徴次元数。
            x_mask: 有効位置mask[B,1,T]。Bは発話数、Tは時間長。真が有効。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            特徴[B,Tout,Dout]、位置特徴[1,Tout,Dout]または[B,Tout,Dout]、
            有効mask[B,1,Tout]の順の組。Bは発話数、Toutは畳み込み後時間長、Doutは出力の特徴次元数。
            Toutは各カーネルとstrideの整数式で決まり、単純な割り算とは限らない。
        """
        # 音声特徴へ畳み込みチャネル軸を追加する。[B,Tin,Fin]→[B,1,Tin,Fin]。Bは発話数、Tinは時間長、
        # Finは入力の特徴次元数。
        x = x.unsqueeze(1)
        # 時間・周波数を畳み込む。[B,1,Tin,Fin]→[B,C,Tconv,Fconv]。Cは畳み込みの出力チャネル数、
        # TconvとFconvはカーネル・
        # strideで決まる長さ。
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(rearrange(x, "b c t f -> b t (c f)"))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2]


class Conv2dSubsampling4(BaseSubsampling):
    """2次元畳み込みで時間長を約4分の1へ圧縮する。"""

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module) -> None:
        """2次元畳み込みで時間長を約4分の1へ圧縮するための層と設定を構築する。

        Args:
            idim: 入力フレームの特徴次元数。
            odim: 出力フレームの特徴次元数。
            dropout_rate: dropout確率。
            pos_enc_class: 構築済みの位置埋め込みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # 各畳み込み層の右文脈を、入力フレーム単位へ換算して加算する。
        # 右文脈の増分=(カーネル長-1)×その層の入力間隔。
        self.subsampling_rate = 4
        # 右文脈を入力フレーム単位で計算する: 6 = (3 - 1) * 1 + (3 - 1) * 2。
        self.right_context = 6

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2次元畳み込みで時間長を約4分の1へ圧縮する。

        Args:
            x: 音声特徴[B,Tin,Din]。Bは発話数、Tinは入力時間長、Dinは入力の特徴次元数。
            x_mask: 有効位置mask[B,1,T]。Bは発話数、Tは時間長。真が有効。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            特徴[B,Tout,Dout]、位置特徴[1,Tout,Dout]または[B,Tout,Dout]、
            有効mask[B,1,Tout]の順の組。Bは発話数、Toutは畳み込み後時間長、Doutは出力の特徴次元数。
            Toutは各カーネルとstrideの整数式で決まり、単純な割り算とは限らない。
        """
        # 音声特徴へ畳み込みチャネル軸を追加する。[B,Tin,Fin]→[B,1,Tin,Fin]。Bは発話数、Tinは時間長、
        # Finは入力の特徴次元数。
        x = x.unsqueeze(1)
        # 時間・周波数を畳み込む。[B,1,Tin,Fin]→[B,C,Tconv,Fconv]。Cは畳み込みの出力チャネル数、
        # TconvとFconvはカーネル・
        # strideで決まる長さ。
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(rearrange(x, "b c t f -> b t (c f)"))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2]


class Conv2dSubsampling6(BaseSubsampling):
    """2次元畳み込みで時間長を約6分の1へ圧縮する。"""

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module) -> None:
        """2次元畳み込みで時間長を約6分の1へ圧縮するための層と設定を構築する。

        Args:
            idim: 入力フレームの特徴次元数。
            odim: 出力フレームの特徴次元数。
            dropout_rate: dropout確率。
            pos_enc_class: 構築済みの位置埋め込みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3),
                                      odim)
        self.pos_enc = pos_enc_class
        # 右文脈を入力フレーム単位で計算する: 10 = (3 - 1) * 1 + (5 - 1) * 2。
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2次元畳み込みで時間長を約6分の1へ圧縮する。

        Args:
            x: 音声特徴[B,Tin,Din]。Bは発話数、Tinは入力時間長、Dinは入力の特徴次元数。
            x_mask: 有効位置mask[B,1,T]。Bは発話数、Tは時間長。真が有効。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            特徴[B,Tout,Dout]、位置特徴[1,Tout,Dout]または[B,Tout,Dout]、
            有効mask[B,1,Tout]の順の組。Bは発話数、Toutは畳み込み後時間長、Doutは出力の特徴次元数。
            Toutは各カーネルとstrideの整数式で決まり、単純な割り算とは限らない。
        """
        # 周波数と時間を畳み込むため[B,Tin,Fin]→[B,1,Tin,Fin]へ変換する。Bは発話数、Tinは時間長、
        # Finは入力の特徴次元数。
        x = x.unsqueeze(1)
        # 時間・周波数を畳み込む。[B,1,Tin,Fin]→[B,C,Tconv,Fconv]。Cは畳み込みの出力チャネル数、
        # TconvとFconvはカーネル・
        # strideで決まる長さ。
        x = self.conv(x)
        b, c, t, f = x.size()
        # 周波数とチャネルを結合して射影する。[B,C,Tconv,Fconv]→[B,Tconv,C*Fconv]→[B,Tconv,Dout]。
        # Doutは出力の特徴次元数。
        x = self.linear(rearrange(x, "b c t f -> b t (c f)"))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 4::3]


class Conv2dSubsampling8(BaseSubsampling):
    """2次元畳み込みで時間長を約8分の1へ圧縮する。"""

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module) -> None:
        """2次元畳み込みで時間長を約8分の1へ圧縮するための層と設定を構築する。

        Args:
            idim: 入力フレームの特徴次元数。
            odim: 出力フレームの特徴次元数。
            dropout_rate: dropout確率。
            pos_enc_class: 構築済みの位置埋め込みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 右文脈を入力フレーム単位で計算する: 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4。
        self.right_context = 14

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """2次元畳み込みで時間長を約8分の1へ圧縮する。

        Args:
            x: 音声特徴[B,Tin,Din]。Bは発話数、Tinは入力時間長、Dinは入力の特徴次元数。
            x_mask: 有効位置mask[B,1,T]。Bは発話数、Tは時間長。真が有効。
            offset: 位置表の開始位置。整数、または発話ごとの整数[B]。

        Returns:
            特徴[B,Tout,Dout]、位置特徴[1,Tout,Dout]または[B,Tout,Dout]、
            有効mask[B,1,Tout]の順の組。Bは発話数、Toutは畳み込み後時間長、Doutは出力の特徴次元数。
            Toutは各カーネルとstrideの整数式で決まり、単純な割り算とは限らない。
        """
        # 周波数と時間を畳み込むため[B,Tin,Fin]→[B,1,Tin,Fin]へ変換する。Bは発話数、Tinは時間長、
        # Finは入力の特徴次元数。
        x = x.unsqueeze(1)
        # 時間・周波数を畳み込む。[B,1,Tin,Fin]→[B,C,Tconv,Fconv]。Cは畳み込みの出力チャネル数、
        # TconvとFconvはカーネル・
        # strideで決まる長さ。
        x = self.conv(x)
        b, c, t, f = x.size()
        # 周波数とチャネルを結合して射影する。[B,C,Tconv,Fconv]→[B,Tconv,C*Fconv]→[B,Tconv,Dout]。
        # Doutは出力の特徴次元数。
        x = self.linear(rearrange(x, "b c t f -> b t (c f)"))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, 2::2][:, :, 2::2][:, :, 2::2]
