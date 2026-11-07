"""抽出済みAudioVAE latentを用いるDataFrame Dataset / sequence packer。

text_ids:[L]、latent:[N,4,64]、任意reference:[Nr,4,64]を.ptで保持する。
text tokenizerとAudioVAEは固定し、LM/LocEnc/LocDiTをスクラッチ学習できる。

データがmodel入力になるまで:
    LatentDatasetが1行ずつ.ptを読み、pack_sequenceがreference/本文/targetを
    1本の混在列へ配置する。text位置にはゼロpatch、audio位置にはゼロtext IDを
    置き、audio_maskでどちらのembeddingを使うかを指定する。
    collate_sequencesは異なる長さの列を右paddingし、validで実位置を区別する。

DataFrameの契約:
    必須列text_ids_pathとlatent_pathは、それぞれ本文int64[L]とtarget
    float32[N,P,C]のTensor単体を保存したpath。任意reference_pathは
    float32[Nr,P,C]の参照。L=本文長、N/Nr=音声patch数、P=patch内frame数、
    C=VAEのlatentの特徴次元数。公式接続ではP=4、C=64。全行で同じtokenizer/VAEを使う。
    reference_pathの空欄は参照なし。抽出手順とCLIはtraining_example.py。

使用例:
    dataset = LatentDataset(dataframe, config)
    batch = collate_sequences([dataset[0], dataset[1]])
    losses = model(**batch)  # device移動は呼び出し側で行う。
    target音声だけloss_mask=1、最後のtarget patchだけstop_labels=1にする。
    生成prefixはpack_sequence(text_ids, None, config, reference)で作れる。
    本ファイルは音声抽出・tokenizer処理をせず、CLIも持たない。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import pandas as pd

    from model import VoxConfig


class VoxSequence(TypedDict):
    """1発話のpacked列。S=LM位置数、P=patch内frame数、C=VAE特徴次元数。

    patches:[S,P,C]、他のtensor:[S]。valid/audio_maskはbool、
    loss_maskはtarget patchの重み、stop_labelsはcontinue/stopの整数。

    Attributes:
        text_ids: 本文trackの整数ID[S]。Sは混在列の位置数。
        patches: 音声trackの連続patch[S,P,C]。Sは全位置数、Pはpatch内長、CはVAEのlatentの特徴次元数。
        audio_mask: 音声patch位置がTrueのmask[S]。Sは全位置数。
        loss_mask: target音声位置の損失重み[S]。Sは全位置数。
        stop_labels: 最後のtarget patchだけが1の終了GT[S]。Sは全位置数。
        valid: padding以外がTrueのmask[S]。Sは全位置数。
    """

    text_ids: torch.Tensor
    patches: torch.Tensor
    audio_mask: torch.Tensor
    loss_mask: torch.Tensor
    stop_labels: torch.Tensor
    valid: torch.Tensor


def pack_sequence(
    text_ids: torch.Tensor,
    target: torch.Tensor | None,
    config: VoxConfig,
    reference: torch.Tensor | None = None,
) -> VoxSequence:
    """[REF_START, ref audio, REF_END, text, AUDIO_START, target, AUDIO_END]。

    targetの最後のpatchをstop=1、他のtarget patchを0にする。参照・本文・境界tokenはlossから除く。
    音声のPADはloss_mask=0。target=Noneなら生成prefixをAUDIO_STARTまで構築する。text_ids:[L]、
    target:[N,P,C]、reference:[Nr,P,C]。L=本文token数、N/Nr=target/reference patch数、
    P=patch内frame数、C=VAEのlatentの特徴次元数。戻り値のS位置軸はreference境界・本文・audio境界を含む総長。

    Args:
        text_ids: 本文ID[L]。Lは本文token数。
        target: 任意のGT音声patch[N,P,C]。Nはtarget patch数、Pはpatch内長、
            CはVAEのlatentの特徴次元数。Noneでは生成prefixだけを作る。
        config: モデルの層数・特徴次元数・条件付け等を保持した設定。
        reference: 任意の参照patch[Nr,P,C]。Nrは参照patch数、Pはpatch内フレーム数、
            CはVAEのlatentの特徴次元数。

    Returns:
        1発話のVoxSequence。patch[S,P,C]、他のID・mask・終了GTは[S]。Sは本文・参照を含む全位置数、
        Pはpatch長、CはVAEのlatentの特徴次元数。
    """
    source = target if target is not None else reference
    device = source.device if source is not None else text_ids.device
    dtype = source.dtype if source is not None else torch.float32
    blank = torch.zeros(1,
                        config.patch_size,
                        config.latent_dim,
                        device=device,
                        dtype=dtype)
    token_parts, patch_parts, modality_parts, loss_parts = [], [], [], []

    def add_text(ids: torch.Tensor) -> None:
        """text IDs[Lpart]を追加する。

        Lpart=追加text位置数、audio trackはゼロ。

        Args:
            ids: 追加する本文ID[L]。Lはsegmentのtoken数。

        Returns:
            None。本文segmentのIDとmaskを内部リストへ追加する。
        """
        token_parts.append(ids)
        patch_parts.append(blank.expand(ids.numel(), -1,
                                        -1))  # [1,P,C]->[Lpart,P,C]。
        modality_parts.append(
            torch.zeros(ids.numel(), device=device, dtype=torch.bool))
        loss_parts.append(torch.zeros(ids.numel(), device=device))

    def add_audio(values: torch.Tensor, supervise: bool) -> None:
        """音声patch列[Npart,P,C]とtarget教師の有無を追加する。

        Npart=追加patch数、P=patch内frame数、C=VAEのlatentの特徴次元数。text trackのIDはゼロ。

        Args:
            values: 音声patch[Npart,P,C]。Npartは追加patch数、Pはpatch内フレーム数、
                CはVAE特徴次元数。
            supervise: このsegmentを損失対象にするかを表すbool。

        Returns:
            None。音声segmentのIDまたはpatchとmaskを内部リストへ追加する。
        """
        if (values.ndim != 3 or values.size(0) == 0
                or values.shape[1:] != (config.patch_size, config.latent_dim)):
            raise ValueError(
                "audio must be nonempty [N,patch_size,latent_dim]")
        token_parts.append(
            torch.zeros(values.size(0), device=device, dtype=torch.long))
        patch_parts.append(values)
        modality_parts.append(
            torch.ones(values.size(0), device=device, dtype=torch.bool))
        loss_parts.append(
            torch.full((values.size(0), ), float(supervise), device=device))

    def token(value: int) -> torch.Tensor:
        """境界IDのscalar整数を、連結用int64 tensor[1]へ変換する。

        Args:
            value: 追加する境界tokenの整数ID。

        Returns:
            追加する境界IDの整数Tensor[1]。1はsegment内のtoken数。
        """
        return torch.tensor([value], device=device, dtype=torch.long)

    if reference is not None:
        add_text(token(config.ref_start))
        add_audio(reference, False)
        add_text(token(config.ref_end))
    add_text(text_ids.to(device=device, dtype=torch.long))
    add_text(token(config.audio_start))
    if target is not None:
        add_audio(target, True)
        add_text(token(config.audio_end))
    # 全segmentを位置軸でつなぐ。ids:[S]、patches:[S,P,C]。
    ids, patches = torch.cat(token_parts), torch.cat(patch_parts)
    labels = torch.zeros_like(ids)
    if target is not None:
        labels[-2] = 1
    return {
        "text_ids": ids,
        "patches": patches,
        "audio_mask": torch.cat(modality_parts),
        "loss_mask": torch.cat(loss_parts),
        "stop_labels": labels,
        "valid": torch.ones_like(ids, dtype=torch.bool)
    }


class LatentDataset(Dataset[VoxSequence]):
    """DataFrameのtext/latent/reference cacheをVoxSequenceへ変換する。"""

    def __init__(self, dataframe: pd.DataFrame, config: VoxConfig) -> None:
        """text_ids_path/latent_pathと任意reference_pathを含むmetadataを保持する。

        Args:
            dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
            config: モデルの層数・特徴次元数・条件付け等を保持した設定。

        Returns:
            None。層・設定を登録する。ここでは学習・音声生成を行わない。
        """
        self.rows, self.config = dataframe.to_dict("records"), config

    def __len__(self) -> int:
        """metadata内の発話数を返す。

        Returns:
            登録された発話数。
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> VoxSequence:
        """cacheのtext[L]・latent[N,P,C]・任意reference[Nr,P,C]をpackする。

        Args:
            index: DataFrameから取得する発話の行番号。

        Returns:
            本文・参照・targetを連結したVoxSequence。patch[S,P,C]、各ID・mask[S]。Sは全位置数、
            Pはpatch内長、CはVAEのlatentの特徴次元数。
        """
        row = self.rows[index]
        # tensorだけをロード。Python pickle objectの実行は不要。
        text = torch.load(row["text_ids_path"],
                          map_location="cpu",
                          weights_only=True).long()
        target = torch.load(row["latent_path"],
                            map_location="cpu",
                            weights_only=True).float()
        ref = row.get("reference_path")
        reference = (torch.load(ref, map_location="cpu",
                                weights_only=True).float()
                     if isinstance(ref, str) and ref else None)
        return pack_sequence(text, target, self.config, reference)


def collate_sequences(samples: list[VoxSequence]) -> dict[str, torch.Tensor]:
    """発話を右paddingしてbatch化する。

    B=発話数、Smax=最大LM位置数。patches:[B,Smax,P,C]、他の各tensor:[B,Smax]。P=patch内frame数、
    C=VAEのlatentの特徴次元数。paddingのvalid/loss_maskは0。

    Args:
        samples: 1発話ずつのpacked列を格納したリスト。

    Returns:
        patches[B,Smax,P,C]と各ID・mask[B,Smax]のbatch辞書。Bは発話数、Smaxは最大列長、Pはpatch内長、
        CはVAEのlatentの特徴次元数。
    """
    maximum = max(sample["text_ids"].numel() for sample in samples)
    batch = {}
    for key in samples[0]:
        if key == "patches":
            # [S,P,C]の先頭位置軸だけSmaxへ延長し、B個積む。[B,Smax,P,C]。
            batch[key] = torch.stack([
                F.pad(s[key], (0, 0, 0, 0, 0, maximum - s[key].size(0)))
                for s in samples
            ])
        else:
            # ID/mask各[S]->[Smax]をstackして[B,Smax]にする。
            batch[key] = torch.stack([
                F.pad(s[key], (0, maximum - s[key].size(0))) for s in samples
            ])
    return batch
