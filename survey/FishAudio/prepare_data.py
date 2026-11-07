"""音声path+書き起こしの2列から、Fish Dual-ARのGT IDとpromptを作る。

入力と前提:
    CSV必須列はaudio_pathとtextのみ。音声は発話単位へ分割済みとする。
    speaker_id/感情ラベルは不要。存在する追加列は保持するが推測はしない。
    感情指示の追加教師なしでも、本文からcodec IDを予測するCE学習はできる。
    指示と感情を対応づける別の学習まで、この処理で出来たことにはしない。

処理:
    公式codecがmono化/resampleして実音声をIDs[N,T]へ変換する。
    N=codebook数、T=約21.53Hzのframe数。本文は公式Conversationで
    system→user→assistant/voice headerへ整形する。prefixはint64[L]。
    L=prompt token数。GT音声はprefixへ混ぜず、別のcacheへ保存する。

必要資産:
    公式fish-speechとその依存、ローカルS2-Pro config/tokenizer/codec.pth。
    LM本体の重みはロードせず、実codec/tokenizerだけを使う。自動取得は無効。
    Python依存は公式依存に加えpandas/torchaudio。勝手にinstallしない。
    --repoで未installの公式repoを指定できる。

使用例:
    python prepare_data.py --metadata data/raw.csv \
        --model-dir checkpoints/s2-pro \
        --output data/cache --device cuda:0
    python training_example.py --metadata data/cache/train.parquet \
        --config data/cache/config_scratch.json --output checkpoints/fish.pt

出力:
    tensors/*.pt、train.parquet、config_scratch.json、preparation.json。
    Parquet内のpathはメタデータファイルの親基準。config_scratchは実語彙とbook定義を反映した
    小型scratch学習例で、公開S2-Proの容量・品質を再現する設定ではない。
    音声を無断でcropせず、長すぎる行は拒否する。既存outputも上書きしない。

保存形式:
    embedding/latent/ID列などの数値配列はTensor単体の.ptで保存する。
    dtypeとshapeを維持し、CSVの文字列表現へは変換しない。
    pathやsample数等の中間メタデータはParquet（pyarrow、ZSTD圧縮）へ保存する。
    raw表の入力はCSV/Parquet両方に対応する。pyarrowは依存として必要だが、
    このスクリプトが追加installしたりCSVへ自動fallbackしたりはしない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio


@torch.inference_mode()
def prepare(
    dataframe: pd.DataFrame,
    source_dir: Path,
    output: Path,
    model_dir: Path,
    codec_path: Path,
    device: str = "cuda:0",
    max_seconds: float = 30.0,
) -> None:
    """実音声からGT codec ID、本文から公式書式prefixを1発話ずつ保存する。

    dataframeの必須列はaudio_path/text。source_dirは音声の相対path基準。
    入力書き起こしを感情指示やspeaker tagで勝手に変更しない。

    Args:
        dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
        source_dir: 入力表の相対音声パスを解決する基準ディレクトリ。
        output: 生成するcache、checkpoint、または保存先ディレクトリ。
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        codec_path: 事前配置した学習済み音声codec重みのパス。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。
        max_seconds: 1発話として受け付ける音声長の上限、単位は秒。

    Returns:
        None。抽出Tensorを.pt、対応するメタデータをParquetへ保存する。
    """
    if not {"audio_path", "text"}.issubset(
            dataframe.columns) or dataframe.empty:
        raise ValueError("Nonempty table with audio_path and text required")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from fish_speech.conversation import Conversation, Message, TextPart
    from fish_speech.models.text2semantic.inference import (
        encode_audio,
        load_codec_model,
    )
    from fish_speech.models.text2semantic.llama import BaseModelArgs
    from fish_speech.tokenizer import FishTokenizer, IM_END_TOKEN

    values = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8"))
    if values.get("model_type") not in {"dual_ar", "fish_qwen3_omni"}:
        raise ValueError("Expected S2-Pro Dual-AR assets")
    config = BaseModelArgs.from_pretrained(str(model_dir))
    tokenizer = FishTokenizer.from_pretrained(str(model_dir))
    # 自作DualARは先頭book ID+offset方式。非連続語彙を黙って誤変換しない。
    mapping = tokenizer.semantic_id_to_token_id
    if not mapping or any(
            mapping.get(index) != tokenizer.semantic_begin_id + index
            for index in range(len(mapping))):
        raise ValueError(
            "Scratch DualAR requires a contiguous semantic ID band")
    codec = load_codec_model(codec_path, device, precision=torch.float32)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tensors").mkdir()
    rows = []
    for index, row in enumerate(dataframe.to_dict("records")):
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Missing transcript at row {index}")
        path = Path(str(row["audio_path"]))
        path = path if path.is_absolute() else source_dir / path
        # encode_audioへ渡す前に実長/有限性を確認する。sampleは捨てない。
        waveform, rate = torchaudio.load(str(path))  # [C,S]。
        if (waveform.numel() == 0 or not torch.isfinite(waveform).all()
                or waveform.size(-1) / rate > max_seconds):
            raise ValueError(f"Empty/nonfinite/too long; split first: {path}")
        codes = encode_audio(path, codec, device).cpu().long()  # [N,T]。
        if (codes.ndim != 2 or codes.size(0) != config.num_codebooks
                or codes.size(1) == 0 or codes.min() < 0
                or codes.max() >= config.codebook_size
                or codes[0].max() >= len(mapping)):
            raise ValueError(
                f"Codec/config vocabulary mismatch at row {index}")
        conversation = Conversation(messages=[
            Message(role="system",
                    parts=[
                        TextPart(
                            text="convert the provided text to speech",
                            cal_loss=False,
                        )
                    ],
                    cal_loss=False),
            Message(role="user",
                    parts=[TextPart(
                        text=row["text"],
                        cal_loss=False,
                    )],
                    cal_loss=False),
            Message(role="assistant",
                    parts=[],
                    modality="voice",
                    add_im_end=False,
                    cal_loss=False),
        ])
        # 公式が作るtrack[(N+1),L]の主trackだけ保存。prefixに音声はない。
        packed, _, _ = conversation.encode_for_inference(
            tokenizer,
            num_codebooks=config.num_codebooks,
        )
        prefix = packed[0].cpu().long()  # [L]。L=system/user/assistant位置数。
        prefix_path = Path("tensors") / f"{index:08d}_prefix.pt"
        codes_path = Path("tensors") / f"{index:08d}_codes.pt"
        torch.save(prefix, output / prefix_path)
        torch.save(codes.contiguous(), output / codes_path)
        rows.append({
            **row,
            "audio_path":
            os.path.relpath(path.resolve(), output.resolve()),
            "prefix_ids_path":
            prefix_path.as_posix(),
            "target_codes_path":
            codes_path.as_posix(),
        })
    pd.DataFrame(rows).to_parquet(output / "train.parquet",
                                  index=False,
                                  engine="pyarrow",
                                  compression="zstd")
    # tokenizer.vocab_sizeはadded tokenを含まない場合があるため最大IDから取る。
    vocab = max(tokenizer.get_vocab().values()) + 1
    small = {
        "dim": 256,
        "intermediate": 768,
        "layers": 4,
        "heads": 4,
        "kv_heads": 2,
        "head_dim": 64
    }
    scratch_config = {
        "vocab_size": vocab,
        "semantic_begin": tokenizer.semantic_begin_id,
        "semantic_size": len(mapping),
        "codebook_size": config.codebook_size,
        "num_codebooks": config.num_codebooks,
        "eos_id": tokenizer.get_token_id(IM_END_TOKEN),
        "slow": small,
        "fast": {
            **small, "layers": 2
        },
    }
    (output / "config_scratch.json").write_text(
        json.dumps(scratch_config, indent=2),
        encoding="utf-8",
    )
    (output / "preparation.json").write_text(json.dumps(
        {
            "model_dir": str(model_dir),
            "codec": str(codec_path),
            "reference": "none",
            "emotion_labels": "not required / not generated",
            "sample_rate": int(codec.sample_rate),
        },
        indent=2),
                                             encoding="utf-8")


def read_source_table(path: Path) -> pd.DataFrame:
    """raw表をCSV/Parquetから読む。

    数値配列を文字列へ変換しない。必須列はaudio_path/text。Parquetにはpyarrowが必要。CSVは利用者の元表の入力だけに対応し、
    中間データの保存には使わない。

    Args:
        path: 読み書きするファイルのパス。

    Returns:
        1行が1発話のDataFrame。必須列はaudio_pathとtext。
    """
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, keep_default_na=False, dtype={"text": str})
    raise ValueError("Input metadata must be .parquet or .csv")


def main() -> None:
    """CSVと公式codec/tokenizerのpathを読み、学習cacheを作るCLI。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.repo:
        sys.path.insert(0, str(args.repo.resolve()))
    if args.output.exists():
        raise FileExistsError(args.output)
    prepare(read_source_table(args.metadata),
            args.metadata.resolve().parent, args.output,
            args.model_dir.resolve(), args.codec
            or args.model_dir / "codec.pth", args.device, args.max_seconds)


if __name__ == "__main__":
    main()
