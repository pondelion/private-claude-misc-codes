"""感情frontendのConformer部品。射影・位置情報・畳み込み・attentionを局所実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=入力特徴の時間長、D=特徴次元数、H=head数。hidden[B,T,D]、mask[B,1,T]。
"""
from __future__ import annotations

from typing import Optional, Tuple

from torch import nn
import torch

from models.common import make_pad_mask
from models.emotion.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from models.emotion.embedding import (
    NoPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
)
from models.emotion.subsampling import (
    Conv2dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    LinearNoSubsampling,
)


class PositionwiseFeedForward(torch.nn.Module):
    """各時刻の特徴へ独立に2段のFFNを適用する。"""

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU()
    ) -> None:
        """各時刻の特徴へ独立に2段のFFNを適用するための層と設定を構築する。

        Args:
            idim: 入力フレームの特徴次元数。
            hidden_units: 内部表現の特徴次元数。
            dropout_rate: dropout確率。
            activation: 活性化関数の名前または構築済みモジュール。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """各時刻の特徴へ独立に2段のFFNを適用する。

        Args:
            xs: 入力特徴[B,T,Din]。Bは発話数、Tは時間長、Dinは入力の特徴次元数。

        Returns:
            FFN出力[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ConvolutionModule(nn.Module):
    """pointwise畳み込み、GLU、時間depthwise畳み込みを適用する。"""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 bias: bool = True) -> None:
        """pointwise畳み込み、GLU、時間depthwise畳み込みを適用するための層と設定を構築する。

        Args:
            channels: 畳み込みの特徴チャネル数。
            kernel_size: 畳み込みまたはフィルタのカーネル長。
            activation: 活性化関数の名前または構築済みモジュール。
            bias: 射影・畳み込みにバイアスを付けるか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # lorderが正なら左側だけを補う因果畳み込みを使う。
        # 因果設定では過去lorderフレームをcacheまたはpaddingで補う。
        # 左側の補完後も現在区間の出力時間長を保つ。
        # 非因果設定では左右対称のpaddingを使う。
        # 非因果設定では対称paddingのためカーネル長を奇数にする。
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.lorder = 0

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        self.use_layer_norm = True
        self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """pointwise畳み込み、GLU、時間depthwise畳み込みを適用する。

        Args:
            x: 入力特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            mask_pad: 畳み込み用の有効位置mask[B,1,T]。Bは発話数、Tは時間長。空Tensorではmaskなし。
            cache: 過去の畳み込み入力[B,D,Kc]。Bは発話数、Dは特徴次元数、Kcは保持する過去長。
                空Tensorでは過去なし。

        Returns:
            特徴[B,T,D]と新しい畳み込みcache[B,D,K]。Bは発話数、Tは入力長、Dは特徴次元数、Kは保持する過去長。
            非因果設定ではcacheは空。
        """
        # 時間と特徴の軸を交換する。[B,T,D]→[B,D,T]。
        x = x.transpose(1, 2)  # Bは発話数、Dは特徴チャネル数、Tは今回の時間長。

        # 発話末尾のpadding位置をゼロにして畳み込みへの混入を防ぐ。
        if mask_pad.size(2) > 0:  # 有効位置maskが空でない場合。
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # 過去cacheが空なら必要な左文脈をゼロで補う。
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # cacheと入力の発話数が一致することを確認する。
                # cacheと入力の特徴次元数が一致することを確認する。
                assert cache.size(1) == x.size(1)
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # 非因果経路では過去cacheは不要だが、出力の型をTensorに統一する。
            # JIT向けの署名互換のため、未使用cacheを空Tensorで表す。
            # この空Tensorは音声特徴の代用品ではない。
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # 特徴を値側とgate側へ分割し、sigmoid gateを乗算する。
        x = self.pointwise_conv1(x)  # pointwise射影で[B,D,T]→[B,2D,T]へ拡張する。
        x = nn.functional.glu(x, dim=1)  # GLUで[B,2D,T]→[B,D,T]へ戻す。

        # チャネルごとに独立した時間畳み込みを適用する。
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # 発話末尾のpadding位置をゼロにして畳み込みへの混入を防ぐ。
        if mask_pad.size(2) > 0:  # 有効位置maskが空でない場合。
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache


class ConformerEncoderLayer(nn.Module):
    """FFN、attention、時間畳み込みを残差接続する。"""

    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: Optional[nn.Module] = None,
                 feed_forward_macaron: Optional[nn.Module] = None,
                 conv_module: Optional[nn.Module] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 concat_after: bool = False) -> None:
        """FFN、attention、時間畳み込みを残差接続するための層と設定を構築する。

        Args:
            size: 隠れ状態の特徴次元数。
            self_attn: 構築済みのself-attention層。
            feed_forward: attention後の構築済みFFN。
            feed_forward_macaron: attention前の構築済みFFN。Noneでは無効。
            conv_module: 構築済みの時間畳み込み分岐。Noneでは無効。
            dropout_rate: dropout確率。
            normalize_before: 各残差分岐の前にLayerNormを適用するか。
            concat_after: attention入力と出力を連結して射影するか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # FFNの残差分岐に使う正規化を用意する。
        # attentionの残差分岐に使う正規化を用意する。
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            # 時間畳み込みの残差分岐に使う正規化を用意する。
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)
            self.norm_final = nn.LayerNorm(size,
                                           eps=1e-5)  # ブロック末尾の特徴正規化を用意する。
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """FFN、attention、時間畳み込みを残差接続する。

        Args:
            x: 入力特徴[B,T,D]。Bは発話数、Tは時間位置数、Dは特徴次元数。
            mask:有効位置mask[B,1,M]または[B,N,M]。Bは発話数、Nはquery数、Mはkey数。0は参照禁止。
                空Tensorではmaskなし。
            pos_emb: 位置特徴[1,Tpos,D]または[B,Tpos,D]。Bは発話数、Tposは位置表の長さ、Dは特徴次元数。
            mask_pad: 畳み込み用の有効位置mask[B,1,T]。Bは発話数、Tは時間長。空Tensorではmaskなし。
            att_cache: 過去のattention cache[B,H,K,2d]。Bは発話数、Hはhead数、Kは過去長、
                dはattention headごとの特徴次元数。空Tensorでは過去なし。
            cnn_cache: 過去の畳み込み入力[B,D,Kc]。Bは発話数、Dは特徴次元数、Kcは過去長。
                空Tensorでは過去なし。

        Returns:
            特徴[B,T,D]、入力mask、更新attention cache[B,H,K,2d]、更新畳み込みcache[B,D,Kc]の組。
            Bは発話数、Tは入力長、Dは特徴次元数、Hはhead数、KとKcは各cache長、d=D/H。
        """

        # Macaron設定時だけattention前にも半重みのFFNを適用する。
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # 複数headのself-attentionを計算し、入力へ残差加算する。
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # 時間方向の局所文脈を畳み込み分岐で抽出する。
        # 畳み込み分岐が無効なら空cacheを返し、有効なら実際のcacheで更新する。
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # 各時刻の特徴へFFNを適用して残差加算する。
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class BaseEncoder(torch.nn.Module):
    """入力射影と位置特徴を適用し、encoder層を順に実行する。"""

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.0,
                 input_layer: str = 'conv2d',
                 pos_enc_layer_type: str = 'abs_pos',
                 normalize_before: bool = True,
                 concat_after: bool = False) -> None:
        """入力射影と位置特徴を適用し、encoder層を順に実行するための層と設定を構築する。

        Args:
            input_size: 入力特徴の次元数。
            output_size: 出力特徴の次元数。
            attention_heads: attention head数。
            linear_units: Conformer FFNの中間次元数。
            num_blocks: 残差ブロック数。段ごとの指定では整数のリスト。
            dropout_rate: dropout確率。
            input_layer: Conformer先頭の線形射影または時間圧縮層の種類。
            pos_enc_layer_type: 絶対位置または相対位置等の位置情報の方式。
            normalize_before: 各残差分岐の前にLayerNormを適用するか。
            concat_after: attention入力と出力を連結して射影するか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d2":
            subsampling_class = Conv2dSubsampling2
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)

    def output_size(self) -> int:
        """encoderの出力の特徴次元数を整数で返す。

        Returns:
            encoderの出力特徴次元数。
        """
        return self._output_size

    def forward(self, xs: torch.Tensor,
                xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """入力射影と位置特徴を適用し、encoder層を順に実行する。

        Args:
            xs: 入力特徴[B,T,Din]。Bは発話数、Tは時間長、Dinは入力の特徴次元数。
            xs_lens: 各入力発話の有効フレーム数[B]。Bは発話数。

        Returns:
            encoder特徴[B,Tenc,Dout]と有効mask[B,1,Tenc]の組。Bは発話数、Tencは時間圧縮後の長さ、
            Doutはencoder出力の特徴次元数。
        """
        T = xs.size(1)
        # 有効位置mask[B,1,Tin]。Bは発話数、Tinは入力特徴長。
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        chunk_masks = masks
        mask_pad = masks  # 圧縮後mask[B,1,Tenc]。Bは発話数、Tencは入力層の畳み込み後の時間長。
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # encoder層内では時間長と有効maskを変えない。
        # 入力層で作ったmaskを出力特徴とともに返す。
        # 後段でこのmaskを使ってpaddingの参照を防ぐ。
        return xs, masks


class ConformerEncoder(BaseEncoder):
    """感情参照特徴をConformer層群で文脈化する。"""

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.0,
                 input_layer: str = 'conv2d',
                 pos_enc_layer_type: str = 'rel_pos',
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 macaron_style: bool = False,
                 use_cnn_module: bool = True,
                 cnn_module_kernel: int = 15) -> None:
        """感情参照特徴をConformer層群で文脈化するための層と設定を構築する。

        Args:
            input_size: 入力特徴の次元数。
            output_size: 出力特徴の次元数。
            attention_heads: attention head数。
            linear_units: Conformer FFNの中間次元数。
            num_blocks: 残差ブロック数。段ごとの指定では整数のリスト。
            dropout_rate: dropout確率。
            input_layer: Conformer先頭の線形射影または時間圧縮層の種類。
            pos_enc_layer_type: 絶対位置または相対位置等の位置情報の方式。
            normalize_before: 各残差分岐の前にLayerNormを適用するか。
            concat_after: attention入力と出力を連結して射影するか。
            macaron_style: attentionの前にも半重みのFFNを置くか。
            use_cnn_module: Conformerの時間畳み込み分岐を使うか。
            cnn_module_kernel: Conformer時間畳み込みのカーネル長。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """

        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate, input_layer,
                         pos_enc_layer_type, normalize_before, concat_after)

        activation = torch.nn.SiLU()

        # attentionの構成を選ぶ。相対位置方式では専用の位置scoreを使う。
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            dropout_rate,
        )

        # Conformerの前後に配置するFFNを構築する。
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # 設定に応じて時間畳み込み分岐を構築する。
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
        )

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
