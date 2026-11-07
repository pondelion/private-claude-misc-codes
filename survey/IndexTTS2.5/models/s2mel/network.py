"""semantic-to-Mel全体をローカルのregulator・CFM・DiTから構築する。"""
from __future__ import annotations

from typing import Any

from torch import nn
import torch

from models.s2mel.flow import CFM
from models.s2mel.regulator import InterpolateRegulator


class SemanticToMelNetwork(nn.Module):
    """length regulatorと条件付きflowモデルを構築してS2Melを接続する。"""

    def __init__(self,
                 args: Any,
                 use_emovec: bool = False,
                 use_gpt_latent: bool = False) -> None:
        """長さ調整と条件付きflow matchingのネットワークを構築する。

        Args:
            args: length_regulator、DiT、cfm等の階層を持つモデル設定。
            use_emovec: 元の設定との互換用。現行の計算では使用しない。
            use_gpt_latent: GPT潜在特徴の追加射影を登録するか。現行のforwardではこの射影を呼び出さない。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        # 1. semantic特徴をMelの時間長へ整列するネットワークを構築する。
        config = args.length_regulator
        regulator = InterpolateRegulator(
            channels=config.channels,
            sampling_ratios=config.sampling_ratios,
            is_discrete=config.is_discrete,
            in_channels=getattr(config, "in_channels", None),
            vector_quantize=getattr(config, "vector_quantize", False),
            codebook_size=config.content_codebook_size,
            n_codebooks=getattr(config, "n_codebooks", 1),
            quantizer_dropout=getattr(config, "quantizer_dropout", 0.0),
            f0_condition=getattr(config, "f0_condition", False),
            n_f0_bins=getattr(config, "n_f0_bins", 512),
        )
        # 2. 長さ調整とMel速度推定を登録する。保存重みのキーは公開重みと対応させる。
        self.models = nn.ModuleDict({
            "cfm": CFM(args),
            "length_regulator": regulator
        })
        if use_gpt_latent:
            self.models["gpt_layer"] = nn.Sequential(nn.Linear(1280, 256),
                                                     nn.Linear(256, 128),
                                                     nn.Linear(128, 1024))

    def forward(
        self,
        mel: torch.Tensor,
        target_lengths: torch.Tensor,
        prompt_lengths: torch.Tensor,
        condition: torch.Tensor,
        speaker: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GT Melと整列済みsemantic条件から速度回帰損失を計算する。

        B=発話数、F=Mel bin数、T=Mel長、C=semantic条件の特徴次元数。 戻り値はscalar lossと、
        CFMの補助再構成値[B,F,T]。

        Args:
            mel: Mel特徴[B,F,M]。Bは発話数、FはMel bin数、MはMelフレーム数。
            target_lengths: 各発話のMel有効フレーム数[B]。Bは発話数。
            prompt_lengths: 各発話の参照Melフレーム数[B]。Bは発話数。
            condition: 整列したsemantic条件[B,T,C]。Bは発話数、TはMel長、Cは条件の特徴次元数。
            speaker: 話者埋め込み[B,192]。Bは発話数、192はCAMPPlusの出力の特徴次元数。

        Returns:
            CFM本体の戻り値。scalarの速度回帰損失[]と、対応する予測Mel状態[B,F,T]の組。Bは発話数、FはMel bin数、
            Tは全Mel長。
        """
        # 長さ調整済みの条件[B,T,C]を使う。GT Mel[B,F,T]から速度回帰を学習する。
        # 戻り値はscalar損失と、対応する推定終点Mel[B,F,T]。
        return self.models["cfm"](mel, target_lengths, prompt_lengths,
                                  condition, speaker)
