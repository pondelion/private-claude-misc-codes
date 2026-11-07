"""本文・参照・GT音声のIDを、Dual-ARのteacher forcing入力へ変換する。

準備するデータ:
    本文側は公式tokenizerのchat templateで処理済みint64 ID[L]。
    音声側は対応codecが抽出したint64 IDs[N,T]。L=text token数、
    N=book数、T=音声frame数。波形や未token化文字列を直接渡すAPIではない。
    DataFrameのpath列と参照あり/なしの例はtraining_example.pyにまとめる。

構築する表現:
    main_ids[S]はtext IDとglobal semantic IDを並べた主track。
    codes[S,N]は同じ位置に対応するbook内ID。音声frameのq0にだけ
    semantic_beginを加えmainへ入れるので、codecのID空間と混同しない。
    supervise[S]はtargetだけTrue、valid[S]はpadding以外True。
    Sは本文と参照/target音声を合計した位置数であり、純粋な音声長ではない。

呼び出し:
    builder = SequenceBuilder(config)
    builder.add_text(prefix_ids)
    builder.add_audio(target_codes, supervise=True)
    builder.add_text(torch.tensor([config.eos_id]), supervise=True)
    sample = builder.finish()
    手動構築の代わりにSpeechDataset(dataframe, config)を使ってもよい。
    collate_sequencesは複数sampleを右paddingし[B,Smax]と[B,Smax,N]へ
    まとめる。B=発話数、Smax=最大混在長。device移動は学習側が行う。
    本ファイルはPyTorchを使い、metadataはpandas DataFrameを受け取る。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import pandas as pd

    from dual_ar import DualARConfig


class FishSequence(TypedDict):
    """1発話の混在列。S=総位置数、N=codebook数。

    main_ids/supervise/valid:[S]、codes:[S,N]。
    main_idsはglobal tokenizer ID、codesは各book内IDであり別のID空間。

    Attributes:
        main_ids: 本文・semantic共通語彙のID[S]。Sは混在列の位置数。
        codes: codebook内の音声ID[S,N]。Sは全位置数、Nはbook数。
        supervise: 教師対象の位置がTrueのmask[S]。Sは全位置数。
        valid: padding以外がTrueのmask[S]。Sは全位置数。
    """

    main_ids: torch.Tensor
    codes: torch.Tensor
    supervise: torch.Tensor
    valid: torch.Tensor


class SequenceBuilder:
    """文字列のchat template/tokenizer処理と、音声ID配置を混同しない。

    add_textへ渡すのは使用tokenizerで処理済みの実ID。reference音声も
    add_audio(..., supervise=False)でMCF入力には入れ、lossからだけ除外する。
    """

    def __init__(self, config: DualARConfig) -> None:
        """語彙band/book数設定を受け取り、text/audio segmentの格納先を作る。

        Args:
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        self.config = config
        self.main: list[torch.Tensor] = []
        self.codes: list[torch.Tensor] = []
        self.supervise: list[torch.Tensor] = []

    def add_text(self, ids: torch.Tensor, supervise: bool = False) -> None:
        """text IDs[L]を追加する。

        L=追加token数、superviseはCE対象かどうか。対応するcode trackは[L,N]のゼロ。N=book数で、
        text位置ではMCFに使わない。

        Args:
            ids: 追加する本文ID[L]。Lはsegmentのtoken数。
            supervise: このsegmentを損失対象にするかを表すbool。

        Returns:
            None。本文segmentのIDとmaskを内部リストへ追加する。
        """
        ids = ids.long()
        self.main.append(ids)
        self.codes.append(
            torch.zeros(ids.numel(),
                        self.config.num_codebooks,
                        dtype=torch.long))
        self.supervise.append(
            torch.full((ids.numel(), ), supervise, dtype=torch.bool))

    def add_audio(self,
                  codec_ids: torch.Tensor,
                  supervise: bool = False) -> None:
        """codec内IDs[N,T]をLM入力用[T,N]へ変換して追加する。

        N=book数、T=約21.53Hzの音声frame数。main track[T]はq0へsemantic band offsetを足した
        global ID。referenceはsupervise=Falseで条件だけに使う。

        Args:
            codec_ids: 音声codec ID[N,T]。Nはcodebook数、Tは音声フレーム数。
            supervise: このsegmentを損失対象にするかを表すbool。

        Returns:
            None。音声segmentのIDまたはpatchとmaskを内部リストへ追加する。
        """
        # 保存codec形式[N,T] -> LM形式[T,N]。Nは10、Tは約21Hzのframe数。
        if (codec_ids.ndim != 2
                or codec_ids.size(0) != self.config.num_codebooks
                or codec_ids.size(1) == 0):
            raise ValueError("expected codec IDs [num_codebooks,audio_frames]")
        semantic_ids = codec_ids[0]  # [T]。先頭bookのsemantic ID列。
        if (semantic_ids.min() < 0
                or semantic_ids.max() >= self.config.semantic_size):
            raise ValueError("semantic code outside configured semantic band")
        self.codes.append(codec_ids.T.long())
        self.main.append(codec_ids[0].long() + self.config.semantic_begin)
        self.supervise.append(
            torch.full((codec_ids.size(1), ), supervise, dtype=torch.bool))

    def finish(self) -> FishSequence:
        """全segmentを位置軸で連結し、FishSequenceのshapeで返す。

        Returns:
            連結済みのFishSequence。ID[S]、code[S,N]、各mask[S]。Sは混在列の総長、Nはbook数。
        """
        # 各segmentの[Lpart]/[Lpart,N]を連結 -> main[S], codes[S,N]。
        ids = torch.cat(self.main)
        codes = torch.cat(self.codes)
        if codes.min() < 0 or codes.max() >= self.config.codebook_size:
            raise ValueError(
                "codec ID outside configured shared codebook vocabulary")
        return {
            "main_ids": ids,
            "codes": codes,
            "supervise": torch.cat(self.supervise),
            "valid": torch.ones_like(ids, dtype=torch.bool)
        }


class SpeechDataset(Dataset[FishSequence]):
    """DataFrame: prefix_ids_path, target_codes_path、任意reference系2列。

    referenceありの場合prefix_idsはsystem reference直前まで、after_reference_ids
    はreference直後の区切り・user text・assistant headerを含める。
    tokenizerのspeaker/instruction書式は公式chat templateに従って事前保存する。
    """

    def __init__(self, dataframe: pd.DataFrame, config: DualARConfig) -> None:
        """cacheのpath列を持つmetadataと、token/bookの構成設定を保持する。

        Args:
            dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        self.rows, self.config = dataframe.to_dict("records"), config

    def __len__(self) -> int:
        """metadataにあるtarget発話数を返す。

        Returns:
            登録された発話数。
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> FishSequence:
        """1発話のprefix/reference/targetを読んでloss mask付きでpackする。

        Args:
            index: DataFrameから取得する発話の行番号。

        Returns:
            本文・参照・targetを連結したFishSequence。main ID[S]、code[S,N]、各mask[S]。Sは全位置数、
            Nはbook数。
        """
        row = self.rows[index]

        def load(column: str) -> torch.Tensor:
            """指定path列のtext[L]またはcodec IDs[N,T]をCPU int64で読む。

            L=text token数、N=book数、T=音声frame数。列ごとの契約はREADMEに記す。

            Args:
                column: 現在行からTensorファイルを読む列名。

            Returns:
                本文ID[L]またはcodec ID[N,T]。Lは本文長、Nはbook数、Tは音声フレーム数。
            """
            return torch.load(row[column],
                              map_location="cpu",
                              weights_only=True).long()

        builder = SequenceBuilder(self.config)
        builder.add_text(load("prefix_ids_path"))
        if isinstance(row.get("reference_codes_path"),
                      str) and row["reference_codes_path"]:
            builder.add_audio(load("reference_codes_path"), supervise=False)
            builder.add_text(load("after_reference_ids_path"))
        builder.add_audio(load("target_codes_path"), supervise=True)
        builder.add_text(torch.tensor([self.config.eos_id]), supervise=True)
        return builder.finish()


def collate_sequences(samples: list[FishSequence]) -> dict[str, torch.Tensor]:
    """可変長混在列を右paddingしてbatch化する。

    main_ids/supervise/valid:[B,Smax]、codes:[B,Smax,N]。B=発話数、Smax=batch内最大位置数、
    N=book数。paddingのmaskはFalse。

    Args:
        samples: 1発話ずつのpacked列を格納したリスト。

    Returns:
        main_ids・各mask[B,Smax]とcodes[B,Smax,N]のbatch辞書。Bは発話数、Smaxは最大列長、Nはbook数。
    """
    maximum = max(s["main_ids"].numel() for s in samples)
    batch = {}
    for key in samples[0]:
        if key == "codes":
            # [S,N]の位置軸だけをSmaxへ伸ばしてからB個stackする。
            batch[key] = torch.stack([
                F.pad(s[key], (0, 0, 0, maximum - s[key].size(0)))
                for s in samples
            ])
        else:
            # ID/mask各[S]->[Smax]をstackし[B,Smax]へまとめる。
            batch[key] = torch.stack([
                F.pad(s[key], (0, maximum - s[key].size(0))) for s in samples
            ])
    return batch
