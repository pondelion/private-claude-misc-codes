"""length regulatorの任意VQ分岐。連続値射影、距離検索、STEを実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=時間位置数、D=特徴次元数。
"""
from __future__ import annotations

from einops import rearrange
from torch import nn
from torch.nn import functional as F
import torch

from models.codec.quantization import WNConv1d


class VectorQuantizeLegacy(nn.Module):
    """入力特徴と同じ次元のcodebookで最近傍codeを選択する。"""

    def __init__(self, input_dim: int, codebook_size: int) -> None:
        """入力特徴と同じ次元のcodebookで最近傍codeを選択するための層と設定を構築する。

        input_dim=入力の特徴次元数、codebook_size=codebookの語彙数。

        Args:
            input_dim: 入力特徴の次元数。
            codebook_size: 量子化codebookに登録する離散IDの個数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, input_dim)

    def forward(
        self,
        z: torch.Tensor,
        z_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """入力特徴と同じ次元のcodebookで最近傍codeを選択する。

        Args:
            z: 量子化前の特徴[B,D,T]。Bは発話数、Dは入力の特徴次元数、Tは量子化位置数。
            z_mask: 有効位置mask[B,T]。Bは発話数、Tは量子化位置数。Noneでは全位置。

        Returns:
            量子化特徴[B,D,T]、ID[B,T]、入力特徴[B,D,T]、commitment損失[]、codebook損失[]の順の組。
            Bは発話数、Tは時間位置数、Dは特徴次元数。[]はscalar。
        """

        z_e = z
        z_q, indices = self.decode_latents(z)

        if z_mask is not None:
            commitment_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean(1) *
                z_mask).sum() / z_mask.sum()
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean(1) *
                z_mask).sum() / z_mask.sum()
        else:
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z_e.detach())
        z_q = (z_e + (z_q - z_e).detach())  # 順伝播は選択code、逆伝播は入力特徴への恒等勾配とするSTE。

        return z_q, indices, z_e, commitment_loss, codebook_loss

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
        # 位置ごとの最近傍検索へ並べる。[B,Q,T]→[B*T,Q]。Bは発話数、Qはcodebookの埋め込み次元数、Tは時間長。
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook[K,Q]。Kは語彙数、Qは検索特徴次元数。

        # 検索特徴とcodebookをL2正規化し、方向の近さでcodeを選ぶ。
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # 全位置と全codeの二乗距離[B*T,K]を計算する。
        dist = (encodings.pow(2).sum(1, keepdim=True) -
                2 * encodings @ codebook.t() +
                codebook.pow(2).sum(1, keepdim=True).t())
        # 最近傍codeの行番号を発話と時間へ戻す。[B*T]→[B,T]。
        indices = rearrange((-dist).max(1)[1],
                            "(b t) -> b t",
                            b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class VectorQuantize(nn.Module):
    """低次元・L2正規化空間で最近傍codeを選択する。"""

    def __init__(self, input_dim: int, codebook_size: int,
                 codebook_dim: int) -> None:
        """低次元・L2正規化空間で最近傍codeを選択するための層と設定を構築する。

        input_dim=入力の特徴次元数、codebook_size=codebookの語彙数、codebook_dim=量子化検索空間の次元数。

        Args:
            input_dim: 入力特徴の次元数。
            codebook_size: 量子化codebookに登録する離散IDの個数。
            codebook_dim: 最近傍検索とcodebook埋め込みの次元数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(
        self,
        z: torch.Tensor,
        z_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """低次元・L2正規化空間で最近傍codeを選択する。

        Args:
            z: 量子化前の特徴[B,D,T]。Bは発話数、Dは入力の特徴次元数、Tは量子化位置数。
            z_mask: 有効位置mask[B,T]。Bは発話数、Tは量子化位置数。Noneでは全位置。

        Returns:
            量子化特徴[B,D,T]、ID[B,T]、検索特徴[B,Q,T]、commitment損失[]、codebook損失[]の順の組。
            Bは発話数、Dは入力の特徴次元数、Tは量子化長、Qは最近傍検索に使う特徴の次元数。[]はscalar。
        """

        # 入力の特徴次元数Dから最近傍検索に使う特徴の次元数Qへ1×1畳み込みで射影する。
        # 入力特徴[B,D,T]→検索特徴[B,Q,T]。Bは発話数、Dは入力の特徴次元数、Tは量子化長、Qは最近傍検索に使う特徴の次元数。
        z_e = self.in_proj(z)  # 検索特徴[B,Q,T]。Bは発話数、Qは最近傍検索に使う特徴の次元数、Tは量子化長。
        z_q, indices = self.decode_latents(z_e)

        if z_mask is not None:
            commitment_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean(1) *
                z_mask).sum() / z_mask.sum()
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean(1) *
                z_mask).sum() / z_mask.sum()
        else:
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z_e.detach())

        z_q = (z_e + (z_q - z_e).detach())  # 順伝播は選択code、逆伝播は入力特徴への恒等勾配とするSTE。

        # 検索空間から入力の特徴次元数へ射影する。[B,Q,T]→[B,D,T]。
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

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
        # 位置ごとの最近傍検索へ並べる。[B,Q,T]→[B*T,Q]。Bは発話数、Qはcodebookの埋め込み次元数、Tは時間長。
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook[K,Q]。Kは語彙数、Qは検索特徴次元数。

        # 検索特徴とcodebookをL2正規化し、方向の近さでcodeを選ぶ。
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # 全位置と全codeの二乗距離[B*T,K]を計算する。
        dist = (encodings.pow(2).sum(1, keepdim=True) -
                2 * encodings @ codebook.t() +
                codebook.pow(2).sum(1, keepdim=True).t())
        # 最近傍codeの行番号を発話と時間へ戻す。[B*T]→[B,T]。
        indices = rearrange((-dist).max(1)[1],
                            "(b t) -> b t",
                            b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices
