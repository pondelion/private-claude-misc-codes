"""音声pathと書き起こしだけの表からIndexTTSの学習cacheを生成する。

モデル本体と設定はローカルのモデル実装と公式configを使う。旧独自モデルのcheckpointと
compact言語IDのcacheは再利用せず、この版で再学習・再抽出する。

入力CSV（1行=切り出し済みの1発話、必須列は2つだけ）:
    audio_path: 音声ファイル。相対pathは入力CSVの親ディレクトリ基準。
    text: その音声で実際に話している本文。
    speaker_id等の追加列は保持するが、必須とせず推測もしない。
    言語は--languageでdataset全体に指定する。未知の感情ラベルは要求しない。

処理順と依存:
    音声読込→mono化→16kHz resample→公式W2v-BERT hidden[17]の正規化。
    t2s/allではさらに公式CAMPPlusと感情Conformer/Perceiverでembeddingを
    抽出し、実tokenizerで本文をID化する。感情embeddingは音声から計算する
    1024次元ベクトルであり、happy等の正解ラベルではない。
    Melは公開設定に従い、ローカルのMelSpectrogramで計算する。
    音声を勝手に切って本文を残すことは
    せず、長さ上限を超える行はエラーにする。事前に発話単位へ分割すること。

必要資産:
    W2v-BERT、CAMPPlus、感情encoder、BigVGANの重み、tokenizer、設定と統計。
    data_processing/resources.pyで固定モデルだけを構築し、読み込む。
    感情encoderの重みはGPT checkpointから抽出するが、生成GPTは構築しない。
    自動download、compile、DeepSpeed、Qwen感情ラベル推定は無効にする。
    対象データの追加アノテーションは不要だが、固定特徴抽出器の重みは必要。

scratch学習の順序:
    1. --stage codecでraw特徴を抽出し、codec.parquetでcodecを学習する。
    2. そのcheckpointを--codec-checkpointへ渡し--stage allで再抽出する。
       固定した自作codecでcodesを作る。ランダムな未学習codecは使わない。
    3. t2s.parquet/s2mel.parquetで各モデルを学習する。s2melの特徴は同じcodecで
       復元したものなので、生成時の入力表現に揃う。

使用例（各surveyディレクトリで実行）:
    python prepare_data.py --metadata data/raw.csv --language JA         --
    model-dir checkpoints/IndexTTS-2.5 --stage codec --output data/cache
    python training_example.py --stage codec         --metadata
    data/cache/codec.parquet         --config configs/config_codec_example.json
    --output checkpoints/codec.pt
    python prepare_data.py --metadata data/raw.csv --language JA         --
    model-dir checkpoints/IndexTTS-2.5 --stage all --output data/cache2
    --codec-checkpoint checkpoints/codec.pt

出力:
    stage別Parquet、Tensor単体の.pt、preparation.json（言語/語彙/資産記録）。
    cache pathはメタデータファイルの親基準で記録する。training_example.pyはこの規則で読む。
    既存outputは置換しない。やり直す場合は別ディレクトリを指定する。
    ここでは実行例を提供するだけで、作成時にモデル実行はしていない。

保存形式:
    embedding/latent/ID列などの数値配列はTensor単体の.ptで保存する。
    dtypeとshapeを維持し、CSVの文字列表現へは変換しない。
    pathやsample数等の中間メタデータはParquet（pyarrow、ZSTD圧縮）へ保存する。
    raw表の入力はCSV/Parquet両方に対応する。pyarrowは依存として必要だが、
    このスクリプトが追加installしたりCSVへ自動fallbackしたりはしない。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import argparse
import json
import os
import sys

import pandas as pd
import torch
import torchaudio

from data_processing.frontend import LANGUAGES, IndexTTSFrontend, language_id
from models.codec.model import SemanticCodec

if TYPE_CHECKING:
    from data_processing.resources import FrontendAssets


def load_pretrained_assets(model_dir: Path, device: str) -> FrontendAssets:
    """固定特徴抽出器とvocoderのローカル資産を読み込む。

    Args:
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

    Returns:
        ローカル重みを読み込んだFrontendAssets。
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from prepare_assets import check_assets
    from data_processing.resources import load_assets

    check_assets(model_dir)
    return load_assets(model_dir, device)


def read_audio(path: Path, max_seconds: float) -> tuple[torch.Tensor, int]:
    """波形[C,S]をmono[1,S]へ集約する。

    C=channel数、S=sample数。

    Args:
        path: 読み書きするファイルのパス。
        max_seconds: 1発話として受け付ける音声長の上限、単位は秒。

    Returns:
        mono波形[1,S]とサンプリング周波数の組。1は音声チャネル数、Sは波形サンプル数。
    """
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Empty/nonfinite waveform: {path}")
    if waveform.size(-1) / sample_rate > max_seconds:
        raise ValueError(f"Split long audio and transcript first: {path}")
    # channel軸を平均する。時間sampleを切り捨てない。[C,S] -> [1,S]。
    return waveform.mean(dim=0, keepdim=True), sample_rate


def save_tensor(output: Path, name: str, value: torch.Tensor) -> str:
    """TensorをCPUで保存し、生成Parquet親からの相対pathを返す。

    shapeは保持。

    Args:
        output: 生成するcache、checkpoint、または保存先ディレクトリ。
        name: 設定名、または保存ファイルを識別する名前。
        value: 保存する数値Tensor。shapeは呼び出し元の特徴定義を保ち、CPUへ移して保存する。

    Returns:
        保存した.ptへの、出力ディレクトリからの相対パス。
    """
    relative = Path("tensors") / f"{name}.pt"
    torch.save(value.detach().cpu().contiguous(), output / relative)
    return relative.as_posix()


@torch.inference_mode()
def prepare(
    dataframe: pd.DataFrame,
    source_dir: Path,
    output: Path,
    model: FrontendAssets,
    language: str,
    stage: str,
    codec_checkpoint: Path | None,
    max_seconds: float = 30.0,
) -> None:
    """音声/text表から各stageのParquetを作る。

    話者ID/感情ラベルを要求しない。

    source_dirは入力音声の相対path基準。outputは新規ディレクトリ。 1発話ずつ抽出し、GPU上へ全datasetを保持しない。

    Args:
        dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
        source_dir: 入力表の相対音声パスを解決する基準ディレクトリ。
        output: 生成するcache、checkpoint、または保存先ディレクトリ。
        model: 固定特徴抽出器と前処理設定を保持したFrontendAssets。
        language: 読み上げ言語コードまたはモデル用言語ID。
        stage: codec、t2s、s2melのいずれか。
        codec_checkpoint: 学習済みsemantic codecのcheckpoint。
        max_seconds: 1発話として受け付ける音声長の上限、単位は秒。

    Returns:
        None。抽出Tensorを.pt、対応するメタデータをParquetへ保存する。
    """
    if not {"audio_path", "text"}.issubset(dataframe.columns):
        raise ValueError("Required columns: audio_path, text")
    if dataframe.empty or language not in LANGUAGES or max_seconds <= 0:
        raise ValueError(
            "Nonempty data, supported language and positive limit required")
    # 学習と合成で同じ特徴抽出・正規化を使う。固定frontendを推論modeにする。
    frontend = IndexTTSFrontend(model).eval()
    codec = None
    if stage != "codec":
        if codec_checkpoint is None:
            raise ValueError(
                "Train codec first; --codec-checkpoint is required")
        checkpoint = torch.load(codec_checkpoint,
                                map_location="cpu",
                                weights_only=True)
        if checkpoint["stage"] != "codec":
            raise ValueError("Expected a training_example.py codec checkpoint")
        codec = SemanticCodec(**checkpoint["config"])
        codec.load_state_dict(checkpoint["model"], strict=True)
        codec = codec.to(model.device).eval()
    output.mkdir(parents=True, exist_ok=False)
    (output / "tensors").mkdir()
    rows_codec, rows_t2s, rows_s2mel = [], [], []
    maximum_id = -1
    for index, row in enumerate(dataframe.to_dict("records")):
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Empty transcript at row {index}")
        path = Path(str(row["audio_path"]))
        path = path if path.is_absolute() else source_dir / path
        waveform, rate = read_audio(path, max_seconds)
        # mono化・resample・公式統計の正規化を合成時と共通のfrontendで行う。
        # 単一発話の正規化特徴[1,T50,1024]。T50は約50Hzのフレーム数。
        features = frontend.extract_semantic(waveform, rate)
        stem = f"{index:08d}"
        metadata = dict(row)
        metadata["audio_path"] = os.path.relpath(path.resolve(),
                                                 output.resolve())
        raw_path = save_tensor(output, stem + "_features", features[0].float())
        rows_codec.append({**metadata, "features_path": raw_path})
        if stage == "codec":
            continue
        assert codec is not None
        # 固定済みcodecでID[1,U]と復元特徴[1,2U,1024]を得る。U=25Hz長。
        _, codes = codec.encode(features.float())
        decoded = codec.decode(codes)
        codes_path = save_tensor(output, stem + "_codes", codes[0].long())
        decoded_path = save_tensor(output, stem + "_decoded",
                                   decoded[0].float())
        # 感情labelは不要。合成で感情参照を省略した場合と同じく音声から抽出。
        speaker, emotion = frontend.extract_embeddings(waveform, rate,
                                                       features)
        speaker_path = save_tensor(output, stem + "_speaker",
                                   speaker[0].float())
        emotion_path = save_tensor(output, stem + "_emotion",
                                   emotion[0].float())
        normalized = frontend.normalize_text(row["text"], language)
        ids = torch.tensor(model.tokenizer.encode(
            f"<|{language.lower()}|> " + normalized,
            allowed_special="all",
        ),
                           dtype=torch.long)  # [L]。L=言語prefix込み本文token数、境界なし。
        if ids.numel() == 0:
            raise ValueError(f"No text tokens at row {index}")
        maximum_id = max(maximum_id, int(ids.max()))
        text_path = save_tensor(output, stem + "_text", ids)
        # 単一発話のGT Mel[1,F,M]。FはMel bin数、MはMelフレーム数。
        mel = frontend.extract_mel(waveform, rate)
        mel_path = save_tensor(output, stem + "_mel", mel[0].float())
        rows_t2s.append({
            **metadata,
            "text_ids_path": text_path,
            "codes_path": codes_path,
            "speaker_path": speaker_path,
            "emotion_path": emotion_path,
            "language_id": language_id(language),
            "normalized_text": normalized,
        })
        rows_s2mel.append({
            **metadata,
            "features_path": decoded_path,
            "mel_path": mel_path,
            "speaker_path": speaker_path,
        })
    pd.DataFrame(rows_codec).to_parquet(output / "codec.parquet",
                                        index=False,
                                        engine="pyarrow",
                                        compression="zstd")
    if rows_t2s:
        pd.DataFrame(rows_t2s).to_parquet(output / "t2s.parquet",
                                          index=False,
                                          engine="pyarrow",
                                          compression="zstd")
        pd.DataFrame(rows_s2mel).to_parquet(output / "s2mel.parquet",
                                            index=False,
                                            engine="pyarrow",
                                            compression="zstd")
        # 公式の全設定を保存する。小型値を既定で上書きしない。
        from omegaconf import OmegaConf

        t2s_config = {
            "official": OmegaConf.to_container(model.cfg.gpt, resolve=True)
        }
        s2mel_config = {
            "official": OmegaConf.to_container(model.cfg.s2mel, resolve=True)
        }
        for name, config in (("config_t2s.json", t2s_config),
                             ("config_s2mel.json", s2mel_config)):
            (output / name).write_text(json.dumps(config, indent=2),
                                       encoding="utf-8")
    manifest = {
        "language_map": {
            name: language_id(name)
            for name in LANGUAGES
        },
        "stage":
        stage,
        "minimum_text_vocab_for_this_data":
        maximum_id + 1,
        "tokenizer_vocab_size":
        int(model.tokenizer.encoding.n_vocab),
        "codec_checkpoint":
        str(codec_checkpoint) if codec_checkpoint else None,
        "note":
        "Use full tokenizer vocab size for T2S, not only observed maximum.",
    }
    (output / "preparation.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_source_table(path: Path) -> pd.DataFrame:
    """raw表をCSV/Parquetから読む。

    数値配列を文字列へ変換しない。

    必須列はaudio_path/text。Parquetにはpyarrowが必要。CSVは利用者の元表の入力だけに対応し、中間データの保存には使わない。

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
    """CSV、公式資産、学習済みcodecをCLIで指定して前処理を開始する。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--stage", choices=("codec", "all"), default="codec")
    parser.add_argument("--codec-checkpoint", type=Path)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.repo:
        sys.path.insert(0, str(args.repo.resolve()))
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.stage == "all" and args.codec_checkpoint is None:
        parser.error("--stage all requires --codec-checkpoint")
    dataframe = read_source_table(args.metadata)
    model = load_pretrained_assets(args.model_dir.resolve(), args.device)
    prepare(dataframe,
            args.metadata.resolve().parent, args.output, model, args.language,
            args.stage, args.codec_checkpoint, args.max_seconds)


if __name__ == "__main__":
    main()
