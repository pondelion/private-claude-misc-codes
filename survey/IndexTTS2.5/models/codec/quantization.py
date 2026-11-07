"""FVQと残差量子化。検索、STE、codebook損失、lookupを実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、D=入力の特徴次元数、Q=最近傍検索に使う特徴の次元数、T=codec長、N=量子化器数。特徴[B,D,T]、ID[N,B,T]。
"""
from __future__ import annotations

from typing import Any

from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
import torch


def WNConv1d(*args: Any, **kwargs: Any) -> nn.Conv1d:
    """Conv1dにweight normalizationを適用し、weight_g/weight_vで重みを保持する。

    Args:
        *args: 対象コンストラクタへ渡す位置引数、または階層化したモデル設定。
        **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

    Returns:
        weight normalizationを適用したnn.Conv1d。
    """
    return weight_norm(nn.Conv1d(*args, **kwargs))


class FactorizedVectorQuantize(nn.Module):
    """入力を検索空間へ射影し、最近傍codeと量子化損失を求める。"""

    def __init__(self,
                 input_dim: int,
                 codebook_size: int,
                 codebook_dim: int,
                 commitment: float = 0.005,
                 codebook_loss_weight: float = 1.0,
                 use_l2_normlize: bool = True) -> None:
        """入力を検索空間へ射影し、最近傍codeと量子化損失を求めるための層と設定を構築する。

        input_dim=入力の特徴次元数、codebook_size=codebookの語彙数、codebook_dim=量子化検索空間の次元数、
        commitment=encoderのcommitment損失重み、codebook_loss_weight=codebook更新損失の重み。

        Args:
            input_dim: 入力特徴の次元数。
            codebook_size: 量子化codebookに登録する離散IDの個数。
            codebook_dim: 最近傍検索とcodebook埋め込みの次元数。
            commitment: encoderを選択codeへ近づける損失の係数。
            codebook_loss_weight: codebookをencoder出力へ近づける損失の係数。
            use_l2_normlize: 検索前に特徴とcodebookをL2正規化するか。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim,
                                       self.codebook_dim,
                                       kernel_size=1)
            self.out_project = WNConv1d(self.codebook_dim,
                                        self.input_dim,
                                        kernel_size=1)

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """入力を検索空間へ射影し、最近傍codeと量子化損失を求める。

        Args:
            z: 量子化前の特徴[B,D,T]。Bは発話数、Dは入力の特徴次元数、Tは量子化位置数。

        Returns:
            量子化特徴[B,D,T]、commitment損失[B]、codebook損失[B]、ID[B,T]、
            射影前量子化特徴[B,Q,T]の順の組。Bは発話数、Dは入力の特徴次元数、Tは量子化長、Qは最近傍検索に使う特徴の次元数。
        """

        # 入力特徴[B,D,T]を検索空間[B,Q,T]へ射影する。D=入力の特徴次元数、Q=最近傍検索に使う特徴の次元数。
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)

        # encoder更新用のcommitment損失とcodebook更新用の損失を分けて計算する。
        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2]) *
                self.commitment)
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2]) *
                self.codebook_loss_weight)
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()

        # 最近傍検索に使う特徴の次元数から元の特徴次元数へ戻す。[B,Q,T]→[B,D,T]。Bは発話数、Qは最近傍検索に使う特徴の次元数、
        # Tは量子化長、Dは元の特徴次元数。
        z_q = self.out_project(z_q)

        return z_q, commit_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """code ID[B,T]をcodebookでlookupし特徴[B,T,Q]へ変換する。

        Q=最近傍検索に使う特徴の次元数。

        Args:
            embed_id: codebookの離散ID[B,T]。Bは発話数、Tは量子化位置数。

        Returns:
            codebook lookup値[B,T,Q]。Bは発話数、Tは量子化長、Qはcodebook特徴次元数。
        """
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """ID[B,T]をlookupし、特徴[B,Q,T]へ軸を入れ替える。

        Q=最近傍検索に使う特徴の次元数。

        Args:
            embed_id: codebookの離散ID[B,T]。Bは発話数、Tは量子化位置数。

        Returns:
            codebook lookup値[B,Q,T]。Bは発話数、Qはcodebook特徴次元数、Tは量子化長。
        """
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(
            self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """特徴[B,Q,T]とcodebook[K,Q]の距離から最近傍ID[B,T]とlookup値[B,Q,T]を得る。

        Args:
            latents: 検索空間の特徴[B,Q,T]。Bは発話数、Qは検索特徴次元数、Tは量子化位置数。

        Returns:
            lookup値[B,Q,T]と最近傍ID[B,T]の組。Bは発話数、Qはcodebook特徴次元数、Tは量子化長。
        """
        # 全フレームを最近傍検索の行に並べる。[B,Q,T]→[B*T,Q]。Bは発話数、Qは最近傍検索に使う特徴の次元数、Tは量子化長。
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # 検索特徴とcodebookを特徴軸でL2正規化する。
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # 各フレームと各codeの二乗距離[B*T,K]を計算する。
        # L2正規化時はcos類似度と同じ順位で最近傍codeを選択する。
        dist = (encodings.pow(2).sum(1, keepdim=True) -
                2 * encodings @ codebook.t() +
                codebook.pow(2).sum(1, keepdim=True).t())
        # 各行で距離最小のcodeを選び、発話と時間の軸へ戻す。[B*T]→[B,T]。
        indices = rearrange((-dist).max(1)[1],
                            "(b t) -> b t",
                            b=latents.size(0))
        z_q = self.decode_code(indices)

        return z_q, indices

    def vq2emb(self, vq: torch.Tensor, out_proj: bool = True) -> torch.Tensor:
        """離散ID列をcodebookでlookupし、必要なら元の特徴次元数Dへ射影して返す。

        Args:
            vq: 離散codebook ID[B,T]。Bは発話数、Tは量子化位置数。
            out_proj: 検索空間から元の特徴次元への出力射影を適用するか。

        Returns:
            lookup後の特徴[B,D,T]。Bは発話数、Dは出力射影後の特徴次元数、Tは量子化長。
        """
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb

    def latent2dist(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """特徴とcodebookの距離、最近傍ID、量子化特徴を計算する。

        距離のcodebook軸はK。

        Args:
            latents: 検索空間の特徴[B,Q,T]。Bは発話数、Qは検索特徴次元数、Tは量子化位置数。

        Returns:
            負の距離[B,T,K]、最近傍ID[B,T]、lookup特徴[B,Q,T]の組。Bは発話数、Tは量子化長、
            Kはcodebook語彙数、Qは最近傍検索に使う特徴の次元数。
        """
        # 全フレームを最近傍検索の行に並べる。[B,Q,T]→[B*T,Q]。Bは発話数、Qは最近傍検索に使う特徴の次元数、Tは量子化長。
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        # 検索特徴とcodebookを特徴軸でL2正規化する。
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # 各フレームと各codeの二乗距離[B*T,K]を計算する。
        # L2正規化時はcos類似度と同じ順位で最近傍codeを選択する。
        dist = (
            encodings.pow(2).sum(1, keepdim=True) -
            2 * encodings @ codebook.t() +
            # 距離[B*T,K]。Bは発話数、Tは量子化長、Kはcodebook語彙数。
            codebook.pow(2).sum(1, keepdim=True).t())

        # 各行で距離最小のcodeを選び、発話と時間の軸へ戻す。[B*T]→[B,T]。
        indices = rearrange((-dist).max(1)[1],
                            "(b t) -> b t",
                            b=latents.size(0))
        # codeごとの距離を発話・時間へ分ける。[B*T,K]→[B,T,K]。Kは語彙数。
        dist = rearrange(dist, "(b t) k -> b t k", b=latents.size(0))
        z_q = self.decode_code(indices)

        return -dist, indices, z_q


class ResidualVQ(nn.Module):
    """残差を順に量子化し、各codebookの出力を加算する。"""

    def __init__(self,
                 input_dim: int = 256,
                 num_quantizers: int = 8,
                 codebook_size: int = 1024,
                 codebook_dim: int = 256,
                 quantizer_type: str = 'vq',
                 quantizer_dropout: float = 0.5,
                 **kwargs: Any) -> None:
        """残差を順に量子化し、各codebookの出力を加算するための層と設定を構築する。

        input_dim=入力の特徴次元数、codebook_size=codebookの語彙数、codebook_dim=量子化検索空間の次元数。

        Args:
            input_dim: 入力特徴の次元数。
            num_quantizers: 残差量子化器の個数。
            codebook_size: 量子化codebookに登録する離散IDの個数。
            codebook_dim: 最近傍検索とcodebook埋め込みの次元数。
            quantizer_type: 量子化器の種類。対象codecはfvqを指定する。
            quantizer_dropout: 学習中に使用する量子化器数を減らす発話の割合。
            **kwargs: 下位モジュールまたはtokenizerへ渡す追加のキーワード引数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = quantizer_type
        self.quantizer_dropout = quantizer_dropout

        # EnhancedCodecが固定指定するFVQを実装する。他のcodec族へ置換しない。
        if quantizer_type != "fvq":
            raise ValueError("IndexTTS EnhancedCodec uses factorized VQ")
        VQ = FactorizedVectorQuantize

        self.quantizers = nn.ModuleList([
            VQ(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                **kwargs,
            ) for _ in range(num_quantizers)
        ])

    def forward(
        self,
        z: torch.Tensor,
        n_quantizers: int = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """残差を順に量子化し、各codebookの出力を加算する。

        Args:
            z: 量子化前の特徴[B,D,T]。Bは発話数、Dは入力の特徴次元数、Tは量子化位置数。
            n_quantizers: 使用する量子化器数。Noneでは全量子化器を使う。

        Returns:
            合計特徴[B,D,T]、ID[N,B,T]、commitment損失[N]、codebook損失[N]、
            各量子化特徴[N,B,D,T]の順の組。Nは使用量子化器数。Bは発話数、Dは入力の特徴次元数、Tは量子化長。
        """

        quantized_out = 0.0
        residual = z

        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        if self.training:
            n_quantizers = torch.ones((z.shape[0], )) * self.num_quantizers + 1
            dropout = torch.randint(1, self.num_quantizers + 1, (z.shape[0], ))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            (z_q_i, commit_loss_i, codebook_loss_i, indices_i,
             z_e_i) = quantizer(residual)

            # 発話ごとの使用量子化器数から、有効な量子化出力だけを加算する。
            mask = (torch.full((z.shape[0], ), fill_value=i, device=z.device)
                    < n_quantizers)
            quantized_out = quantized_out + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commit_loss_i = (commit_loss_i * mask).mean()
            codebook_loss_i = (codebook_loss_i * mask).mean()

            all_commit_losses.append(commit_loss_i)
            all_codebook_losses.append(codebook_loss_i)
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)

        # 量子化器軸Nqを先頭へ追加する。損失は[Nq]、IDは[Nq,B,T]、各特徴は[Nq,B,D,T]。
        (all_commit_losses, all_codebook_losses, all_indices,
         all_quantized) = map(
             torch.stack,
             (all_commit_losses, all_codebook_losses, all_indices,
              all_quantized),
         )

        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )

    def vq2emb(self,
               vq: torch.Tensor,
               n_quantizers: int | None = None) -> torch.Tensor:
        """離散ID列をcodebookでlookupし、必要なら元の特徴次元数Dへ射影して返す。

        Args:
            vq: semantic ID[Nq,B,T]。Nqは量子化器数、Bは発話数、Tは量子化長。
            n_quantizers: 使用する量子化器数。Noneでは全量子化器を使う。

        Returns:
            lookup後の特徴[B,D,T]。Bは発話数、Dは出力射影後の特徴次元数、Tは量子化長。
        """
        quantized_out = 0.0
        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        for idx, quantizer in enumerate(self.quantizers):
            if idx >= n_quantizers:
                break
            quantized_out += quantizer.vq2emb(vq[idx])
        return quantized_out

    def latent2dist(
            self,
            z: torch.Tensor,
            n_quantizers: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """特徴とcodebookの距離、最近傍ID、量子化特徴を計算する。

        距離のcodebook軸はK。

        Args:
            z: 量子化前の特徴[B,D,T]。Bは発話数、Dは入力の特徴次元数、Tは量子化位置数。
            n_quantizers: 使用する量子化器数。Noneでは全量子化器を使う。

        Returns:
            負の距離[Nq,B,T,K]とID[Nq,B,T]の組。Nqは使用量子化器数、Bは発話数、Tは量子化長、Kはcodebook語彙数。
        """
        quantized_out = 0.0
        residual = z

        all_dists = []
        all_indices = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            dist_i, indices_i, z_q_i = quantizer.latent2dist(residual)
            all_dists.append(dist_i)
            all_indices.append(indices_i)

            quantized_out = quantized_out + z_q_i
            residual = residual - z_q_i

        # 量子化器別の距離を積む。[B,T,K]のリスト→[Nq,B,T,K]。Nqは量子化器数。
        all_dists = torch.stack(all_dists)
        # 量子化器別のIDを積む。[B,T]のリスト→[Nq,B,T]。Nqは量子化器数。
        all_indices = torch.stack(all_indices)

        return all_dists, all_indices
