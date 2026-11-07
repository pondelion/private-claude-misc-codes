"""音声patchの読み取り・情報量制約・局所生成を担うVoxCPM2の部品。

3つの役割:
    LocalEncoder: patch[B,S,P,C]を[B,S,Denc]へ圧縮する。各patchの
        先頭にlearned tokenを追加し、patch内の非因果attentionで集約する。
    ScalarQuantizer: TSLM特徴を低次元へ射影し、tanh→roundで有限水準へ
        制限して元特徴次元数へ戻す。音声codecのbook IDを生成する処理ではない。
    LocalDiT: noisy patchと直前patch、LM条件、flow時刻を受け、
        同じpatch shapeのvelocityを出力する。波形を直接生成しない。

軸と接続:
    B=発話数、S=LM位置数、P=patch内frame数、C=VAE特徴次元数、Denc=局所幅。
    LocDiTではN=B*Sをpatchのbatch数とし、入力/出力を[N,C,P]にする。
    時刻token1個とLM条件token2個の後へpatch tokenを並べ、patch位置だけ
    を出力する。局所Transformerのattentionは時間AR用のcausalではない。

使い方:
    各クラスはTransformerConfig等でrandom構築できるnn.Moduleで、
    model.pyとflow_matching.pyが組み合わせて使う。本ファイル単体に
    CLIやデータ読み込みはない。PyTorchとeinopsが必要。
    実データからの学習入口はtraining_example.py。
"""
import math

import torch
from einops import rearrange
from torch import nn

from transformer import Transformer, TransformerConfig


class ScalarQuantizer(nn.Module):
    """公式のround(tanh(x)*scale)/scale。scale=9なら最大19水準。

    論文の「9 levels」と公開コードのscale引数は同義ではない。
    このFSQは音声codebook IDsをLMが分類予測する機構ではない。
    """

    def __init__(self,
                 dim: int,
                 latent_dim: int = 512,
                 scale: int = 9) -> None:
        """入出力の特徴次元数dimと量子化する特徴の次元数latent_dim、丸め刻み1/scaleを設定する。

        Args:
            dim: 特徴の次元数。
            latent_dim: 音声latentの1フレーム当たりの特徴次元数C。
            scale: FSQの丸め刻みを決める正の係数。刻み幅は1/scale。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.input = nn.Linear(dim, latent_dim)
        self.output = nn.Linear(latent_dim, dim)
        self.scale = scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden[B,S,D]を半離散state[B,S,D]へ変換する。

        B=発話数、S=LM位置数、D=dim、Q=latent_dim。[B,S,Q]で各軸を独立に丸め、
        学習時だけstraight-through勾配を使う。

        Args:
            hidden: 隠れ状態[B,S,D]。Bは発話数または音声フレーム数、Sは位置数、Dは内部特徴の次元数。

        Returns:
            半離散特徴[B,S,D]。Bは発話数、SはLM位置数、Dは外部特徴次元数。
        """
        bounded = self.input(hidden).tanh()  # [B,S,D] -> [B,S,Q]
        discrete = torch.round(bounded * self.scale) / self.scale
        quantized = bounded + (discrete -
                               bounded).detach() if self.training else discrete
        return self.output(quantized)  # [B,S,Q] -> [B,S,D]


class LocalEncoder(nn.Module):
    """1patch内の連続frameを、learned CLSで1つのLM入力へ集約する。"""

    def __init__(self,
                 config: TransformerConfig,
                 latent_dim: int = 64) -> None:
        """patch内Transformer設定とVAE frameの特徴次元数latent_dimを受ける。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。
            latent_dim: 音声latentの1フレーム当たりの特徴次元数C。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.input = nn.Linear(latent_dim, config.dim)
        self.special = nn.Parameter(torch.randn(1, 1, 1, config.dim))
        self.transformer = Transformer(config)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """連続patch[B,S,P,C]を埋込[B,S,Denc]へ集約する。

        B=発話数、S=LM位置数、P=patch内frame数（通常4）、C=latent_dim、Denc=config.dim。
        attentionはP+1位置内で完結し、S位置間を直接混ぜない。

        Args:
            patches: 連続音声patch[B,S,P,C]。Bは発話数、SはLM列長、Pはpatch内フレーム数、
                CはVAEのlatentの特徴次元数。

        Returns:
            集約特徴[B,S,Denc]。Bは発話数、SはLM位置数、Dencは局所encoder出力の特徴次元数。
        """
        batch, positions, _, _ = patches.shape
        hidden = self.input(patches)  # [B,S,P,C] -> [B,S,P,Denc]
        # 同じlearned CLSを各patchへ展開。[B,S,1,Denc]+[B,S,P,Denc]。
        hidden = torch.cat(
            (self.special.expand(batch, positions, 1, -1), hidden), dim=2)
        hidden = rearrange(hidden, "b s p d -> (b s) p d")  # [B*S,P+1,Denc]
        encoded = self.transformer(hidden, causal=False)  # [B*S,P+1,Denc]。
        # CLSだけを抜き、patch-batch軸を復元。[B*S,Denc]->[B,S,Denc]。
        return rearrange(encoded[:, 0], "(b s) d -> b s d", b=batch)


class TimeEmbedding(nn.Module):
    """正弦/余弦と2層MLPでCFM時刻を特徴vectorへ変換する。"""

    def __init__(self, dim: int) -> None:
        """偶数の出力の特徴次元数dimを指定し、時刻MLPを構築する。

        Args:
            dim: 特徴の次元数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        if dim % 2 or dim < 4:
            raise ValueError("time embedding width must be even and >=4")
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """CFM時刻[N]を埋込[N,D]へ変換する。

        N=patch処理数、D=dim。内部位相は[N,D/2]、sin/cos連結で[N,D]となる。時刻には1000を掛ける。

        Args:
            time: flow時刻[N]。Nは処理patch数。

        Returns:
            時刻特徴[N,D]。Nは処理patch数、Dは出力の特徴次元数。
        """
        frequencies = torch.exp(
            torch.arange(self.dim // 2, device=time.device).float() *
            (-math.log(10000) / (self.dim // 2 - 1)))
        phase = 1000 * time.float()[:, None] * frequencies[None]
        return self.mlp(
            torch.cat((phase.sin(), phase.cos()),
                      dim=-1).to(self.mlp[0].weight.dtype))


class LocalDiT(nn.Module):
    """semantic/residual/timeの3prefixと前patchを条件にvelocityを予測する。"""

    def __init__(self,
                 config: TransformerConfig,
                 latent_dim: int = 64) -> None:
        """patch内Transformerと、各latent frameの特徴次元数を設定する。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。
            latent_dim: 音声latentの1フレーム当たりの特徴次元数C。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.config = config
        self.input = nn.Linear(latent_dim, config.dim)
        self.condition = nn.Linear(latent_dim, config.dim)
        self.time = TimeEmbedding(config.dim)
        self.delta_time = TimeEmbedding(config.dim)
        self.transformer = Transformer(config)
        self.output = nn.Linear(config.dim, latent_dim)

    def forward(
        self,
        noisy: torch.Tensor,
        mu: torch.Tensor,
        time: torch.Tensor,
        previous_patch: torch.Tensor,
        delta_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """noisy/previous_patch:[N,C,P]からvelocity[N,C,P]を予測する。

        N=B*Sは全LM位置をbatch化した個数。LocDiTのattention範囲は1patch。B=発話数、S=LM位置数、
        C=VAE latentの特徴次元数（通常64）、P=patch内frame数。
        mu:[N,2*Dd]はsemantic/residual別の条件、time/delta_time:[N]。Dd=config.dim。
        delta_timeはoptional mean-flow区間幅で、既定はゼロ。

        Args:
            noisy: ノイズを加えた現在のpatch状態[N,C,P]。Nは処理patch数、CはVAEのlatentの特徴次元数、
                Pはpatch内フレーム数。
            mu: semantic・residual条件[N,2D]。Nは処理patch数、DはLocalDiTの内部特徴の次元数。
            time: flow時刻[N]。Nは処理patch数。
            previous_patch: 直前の音声patch[N,C,P]。Nは処理patch数、
                CはVAEのlatentの特徴次元数、Pはpatch内フレーム数。
            delta_time: 任意の積分区間幅[N]。Nは処理patch数。Noneでは0。

        Returns:
            予測速度[N,C,P]。Nは処理patch数、CはVAEのlatentの特徴次元数、Pはpatch内長。
        """
        if delta_time is None:
            delta_time = torch.zeros_like(time)
        lm_tokens = rearrange(mu, "n (k d) -> n k d",
                              k=2)  # [N,2Dd]->[N,2,Dd]。
        time_token = self.time(time) + self.delta_time(delta_time)  # [N,Ddit]
        # 各[N,C,P]->[N,P,C]->[N,P,Dd]。clean/noisyは独立した射影を使う。
        clean_tokens = self.condition(
            rearrange(previous_patch, "n c p -> n p c"))
        noisy_tokens = self.input(rearrange(noisy, "n c p -> n p c"))
        prefix_length = 3 + previous_patch.size(-1)
        tokens = torch.cat(
            (lm_tokens, time_token[:, None], clean_tokens, noisy_tokens),
            dim=1)
        # [N,2+1+P+P,Ddit]。full attentionで現在noisy patchのP位置だけを出力。
        hidden = self.transformer(tokens,
                                  causal=False)[:, prefix_length:]  # [N,P,Dd]。
        # [N,P,Dd]->Linear(Dd,C)->[N,P,C]->[N,C,P]。
        return rearrange(self.output(hidden), "n p c -> n c p")
