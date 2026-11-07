"""公式EnhancedCodecをrandom初期化し、学習APIへ接続する。

実モデルはmodels.codec.enhancedに定義したローカルEnhancedCodecである。
ConvNeXt/Vocos、FVQの正規化・weight norm・初期化、量子化、時間圧縮・復元を
公式の演算構造を基準に実装する。core属性でforward/quantize/decodeへアクセスする。
設定はmodels/config.pyの方式で公式semantic_codec section全体を渡す。
旧独自実装のcheckpointは互換ではないため、再学習または公式重みを使用する。

軸: B=発話数、T50=約50Hz特徴長、U=圧縮後長、D=特徴次元数、Nq=量子化器数。
単一FVQならNq=1で、公開quantizeが返すIDは[B,U]となる。
"""
from __future__ import annotations

from typing import Any

from torch import nn
import torch

from models.config import resolve_config


class SemanticCodec(nn.Module):
    """公式codecの処理を維持し、学習lossの戻り値だけを辞書にまとめる。"""

    def __init__(self, **spec: Any) -> None:
        """公式設定全体でモデルをrandom構築する。

        pretrained重みは取得しない。

        Args:
            **spec: 公式設定の参照先、または展開済み設定と容量variant。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        super().__init__()
        from models.codec.enhanced import EnhancedCodec

        self.configuration = resolve_config("semantic_codec", spec)
        self.core = EnhancedCodec(**self.configuration)

    @property
    def vocabulary(self) -> int:
        """公式codebookの語彙数を返す。

        Returns:
            semantic codebookの語彙数。
        """
        return self.core.codebook_size

    def encode(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """特徴[B,T50,D]→量子化特徴[B,D,U]、ID[B,U]または[Nq,B,U]。

        公式quantizeは量子化特徴を[B,U,D]で返すので、接続APIの軸へ転置する。量子化lossは公式quantizeの戻り値に存在せず、
        ここでも架空の値を返さない。

        Args:
            features: 正規化音声特徴[B,T50,1024]。Bは発話数、T50は約50Hzのフレーム数。

        Returns:
            量子化特徴[B,D,U]とID[B,U]または[Nq,B,U]。Bは発話数、Dは特徴次元数、Uは圧縮後長、Nqは量子化器数。
        """
        # 1. 公式quantizeを直接実行。stride Conv/Vocos/FVQの演算は変更しない。
        ids, time_first = self.core.quantize(features)
        # 2. 特徴次元数と時間の軸だけ入れ替える。[B,U,D]→[B,D,U]。
        return time_first.transpose(1, 2), ids

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        """semantic IDから連続特徴を復元する。

        Args:
            ids: semantic ID[B,U]または[Nq,B,U]。Bは発話数、Uは圧縮長、Nqは量子化器数。

        Returns:
            復元特徴[B,Trec,D]。Bは発話数、Trecはcodecによる復元後長、Dは特徴次元数。2倍圧縮構成ではTrec=2U。
        """
        return self.core.decode(ids)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """特徴[B,T50,D]をローカルcodecで再構成し、各出力とscalar lossを返す。

        奇数長の切り詰め、FVQ損失の重み、復元処理はローカルのcodecで計算する。lossの合算と辞書への整理だけ
        が学習用adapterの処理である。

        Args:
            features: 正規化音声特徴[B,T50,1024]。Bは発話数、T50は約50Hzのフレーム数。

        Returns:
            再構成特徴、code ID、量子化損失、再構成損失、合計損失を格納した辞書。特徴は[B,Trec,D]、IDは[Nq,B,U]、
            損失はscalar[]。Bは発話数、Trecは復元長、Dは特徴次元数、Nqは量子化器数、Uは圧縮長。
        """
        # 1. 公式forwardの4出力を取得する。reconstruction:[B,Teven,D]。
        reconstruction, vq_loss, ids, reconstruction_loss = self.core(features)
        # 2. 公式が計算したscalar損失を合算する。モデルの演算は追加しない。
        return {
            "loss": vq_loss + reconstruction_loss,
            "reconstruction": reconstruction,
            "codes": ids,
            "quantization_loss": vq_loss,
            "reconstruction_loss": reconstruction_loss
        }
