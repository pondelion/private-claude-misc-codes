"""semantic条件のMel時間軸への整列。token/F0/量子化分岐も局所実装。

参照: index-tts revision d9e41aac89fd00b3d71497fddb287b7f24613712。
各層・演算をこのファイルに実装し、外部のモデル本体はimportしない。
B=発話数、T=時間位置数、D=特徴次元数。
"""
from __future__ import annotations

from typing import Tuple

from torch import nn
from torch.nn import functional as F
import numpy as np
import torch

from models.common import sequence_mask
from models.s2mel.regulator_quantization import VectorQuantize

f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0: torch.Tensor | None, f0_bin: int) -> torch.Tensor:
    """F0周波数[B,T]をMel尺度で離散化し、設定したF0 binの整数ID[B,T]へ変換する。

    Args:
        f0: 任意の基本周波数[B,T]。Bは発話数、Tは条件時間長。Noneでは使用しない。
        f0_bin: 基本周波数を離散化するbin数。

    Returns:
        離散化した基本周波数の整数ID[B,T]。Bは発話数、Tはf0フレーム数。
    """
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)

    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


class InterpolateRegulator(nn.Module):
    """semantic特徴をMel長へ補間しConvと有効長maskを適用する。
    """

    def __init__(self,
                 channels: int,
                 sampling_ratios: Tuple,
                 is_discrete: bool = False,
                 in_channels: int = None,
                 vector_quantize: bool = False,
                 codebook_size: int = 1024,
                 out_channels: int = None,
                 groups: int = 1,
                 n_codebooks: int = 1,
                 quantizer_dropout: float = 0.0,
                 f0_condition: bool = False,
                 n_f0_bins: int = 512) -> None:
        """semantic特徴をMel長へ補間しConvと有効長maskを適用するための層と設定を構築する。

        channels=Convの特徴channel数、in_channels=入力channel数、
        codebook_size=codebookの語彙数、out_channels=出力channel数。

        Args:
            channels: 畳み込みの特徴チャネル数。
            sampling_ratios: regulator内の時間拡大率を指定する列。
            is_discrete: semantic条件を連続特徴でなく離散IDとして受け取るか。
            in_channels: 入力チャネル数。
            vector_quantize: regulator内で量子化するか。
            codebook_size: 量子化codebookに登録する離散IDの個数。
            out_channels: 出力チャネル数。
            groups: 畳み込みで分割するチャネルのグループ数。
            n_codebooks: 離散条件に使用するcodebookの個数。
            quantizer_dropout: 学習中に使用する量子化器数を減らす発話の割合。
            f0_condition: 基本周波数による条件付けを有効にするか。
            n_f0_bins: 基本周波数の離散bin数。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        else:
            self.interpolate = False
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)
        self.is_discrete = is_discrete

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

        self.n_codebooks = n_codebooks
        if n_codebooks > 1:
            self.extra_codebooks = nn.ModuleList([
                nn.Embedding(codebook_size, channels)
                for _ in range(n_codebooks - 1)
            ])
            self.extra_codebook_mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, channels))
                for _ in range(n_codebooks - 1)
            ])
        self.quantizer_dropout = quantizer_dropout

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_condition = f0_condition
            self.n_f0_bins = n_f0_bins
            self.f0_bins = torch.arange(2, 1024, 1024 // n_f0_bins)
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
        else:
            self.f0_condition = False

        if not is_discrete:
            self.content_in_proj = nn.Linear(in_channels, channels)
            if vector_quantize:
                self.vq = VectorQuantize(channels, codebook_size, 8)

    def forward(
        self,
        x: torch.Tensor,
        ylens: torch.Tensor = None,
        n_quantizers: int | None = None,
        f0: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
               | None, torch.Tensor | None]:
        # 学習時の設定に応じて一部の条件tokenを無効化する。
        """semantic特徴をMel長へ補間しConvと有効長maskを適用する。

        入力[B,Ts,Din]→条件[B,max(ylens),C]と有効長等のtuple。Ts=入力長、C=出力の特徴次元数。

        Args:
            x:連続条件[B,T,D]、または離散ID[B,T]／[B,Nq,T]。Bは発話数、Tは条件長、Dは特徴次元数、
                Nqはcodebook数。
            ylens: 整列先のMelフレーム数[B]。Bは発話数。
            n_quantizers: 使用する量子化器数。Noneでは全量子化器を使う。
            f0: 任意の基本周波数[B,T]。Bは発話数、Tは条件時間長。Noneでは使用しない。

        Returns:
            整列特徴[B,M,C]、出力長[B]、code IDまたはNone、commitment損失またはNone、
            codebook損失またはNoneの組。Bは発話数、Mは最大出力長、Cは出力の特徴次元数。連続条件の既定経路では後半3値はNone。
        """
        if self.training:
            n_quantizers = torch.ones((x.shape[0], )) * self.n_codebooks
            dropout = torch.randint(1, self.n_codebooks + 1, (x.shape[0], ))
            n_dropout = int(x.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(x.device)
            # 発話ごとに有効にするcodebookの個数を決める。
        else:
            n_quantizers = torch.ones((x.shape[0], ), device=x.device) * (
                self.n_codebooks if n_quantizers is None else n_quantizers)
        if self.is_discrete:
            if self.n_codebooks > 1:
                assert len(x.size()) == 3
                x_emb = self.embedding(x[:, 0])
                for i, emb in enumerate(self.extra_codebooks):
                    x_emb = x_emb + (n_quantizers > i +
                                     1)[..., None, None] * emb(x[:, i + 1])
                    # 使わないcodebookには専用のmask埋め込みを加える。

                x = x_emb
            elif self.n_codebooks == 1:
                if len(x.size()) == 2:
                    x = self.embedding(x)
                else:
                    x = self.embedding(x[:, 0])
        else:
            x = self.content_in_proj(x)
        # 入力条件は[B,T,D]。Bは発話数、Tはsemantic長、Dは特徴次元数。
        # 出力長から有効mask[B,M,1]を作る。Bは発話数、Mは最大Mel長。末尾1を特徴次元数へbroadcastする。
        mask = sequence_mask(ylens).unsqueeze(-1)
        if self.interpolate:
            # 連続特徴[B,Ts,C]→[B,C,Ts]へ移し、Mel長Mへ補間する。[B,C,M]。
            # Tsはsemantic token列の長さ、Cは特徴次元数。
            x = F.interpolate(x.transpose(1, 2).contiguous(),
                              size=ylens.max(),
                              mode='nearest')
        else:
            x = x.transpose(1, 2).contiguous()
            mask = mask[:, :x.size(2), :]
            ylens = ylens.clamp(max=x.size(2)).long()
        if self.f0_condition:
            if f0 is None:
                x = x + self.f0_mask.unsqueeze(-1)
            else:

                quantized_f0 = f0_to_coarse(f0, self.n_f0_bins)
                quantized_f0 = quantized_f0.clamp(0, self.n_f0_bins - 1).long()
                f0_emb = self.f0_embedding(quantized_f0)
                # F0埋め込み[B,Tf,C]を[B,C,Tf]へ転置し、semantic条件と同じMel長Mへ補間する。[B,C,M]。
                f0_emb = F.interpolate(f0_emb.transpose(1, 2).contiguous(),
                                       size=ylens.max(),
                                       mode='nearest')
                x = x + f0_emb
        # 時間畳み込みで[B,Cin,M]→[B,Cout,M]、転置して[B,M,Cout]。Coutは出力条件の特徴次元数。
        out = self.model(x).transpose(1, 2).contiguous()
        if hasattr(self, 'vq'):
            out_q, commitment_loss, codebook_loss, codes, out, = self.vq(
                out.transpose(1, 2))
            # 量子化特徴をMel条件の軸順へ戻す。[B,Cout,M]→[B,M,Cout]。
            out_q = out_q.transpose(1, 2)
            return out_q * mask, ylens, codes, commitment_loss, codebook_loss
        olens = ylens
        return out * mask, olens, None, None, None
