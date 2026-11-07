"""音声pathと書き起こしの表から、VoxCPM2学習用latent/本文IDを作る。

入力CSV:
    必須はaudio_pathとtextだけ。1行は本文と対応する切り出し済み1発話。
    音声の相対pathはメタデータファイルの親基準。speaker_idや感情ラベルは不要。
    追加列は出力に保持するが、未知の感情・話者ラベルを補完したことにしない。

処理:
    音声→mono→16kHz→patch境界へ右padding→固定公式AudioVAE V2→
    float32[N,4,64]。N=patch数、4=patch内frame数、64=latentの特徴次元数。
    書き起こし→公式の中国語multi-character mask付きtokenizer→int64[L]。
    L=本文token数。本文に存在しないstyle instructionを自動で捏造しない。
    このモデルの通常CFM学習に感情カテゴリ/F0の追加教師は必要ない。

必要条件:
    公式VoxCPM repo/packageとその依存、ローカルVoxCPM2モデルディレクトリ。
    tokenizer/configとaudiovae.safetensorsまたはaudiovae.pthを使う。
    LM本体は構築/ロードしない。VAEとtokenizerだけを公式実体で読み込む。
    PyTorch、torchaudio、pandas、einopsと公式依存が必要。自動取得は無効。

使用例:
    python prepare_data.py --metadata data/raw.csv \
        --model-dir checkpoints/VoxCPM2 --output data/cache --device cuda:0
    python training_example.py --metadata data/cache/train.parquet \
        --config data/cache/config_scratch.json --output checkpoints/vox.pt

出力:
    tensors/*.pt、train.parquet、config_scratch.json、preparation.json。
    config_scratchは語彙/境界ID/patch特徴次元数を実資産へ揃えた小型構造実験用設定。
    精度の再現設定ではないので、LM容量は利用目的に応じて変更する。
    学習するのはbackboneで、固定VAEをゼロから学ぶスクリプトではない。
    outputは新規に作り、既存ファイルを上書きしない。

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
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torchaudio
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def load_frontend(
    model_dir: Path,
    device: str,
) -> tuple[nn.Module, Callable[[str], list[int]], int, int, int]:
    """公式VAE/tokenizerと、patch数・latentの特徴次元数・語彙数の設定をロードする。

    Args:
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。

    Returns:
        固定重みの音声VAE、本文token化関数、patch内フレーム数、latent特徴次元数、本文語彙数の順の5要素の組。
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import LlamaTokenizerFast
    from voxcpm.model.utils import mask_multichar_chinese_tokens
    from voxcpm.model.voxcpm2 import VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2

    values = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8"))
    if values.get("architecture", "").lower() != "voxcpm2":
        raise ValueError("Expected VoxCPM2 model assets")
    config = VoxCPMConfig.model_validate(values)
    vae = (AudioVAEV2(config=config.audio_vae_config)
           if config.audio_vae_config else AudioVAEV2())
    safetensors_path = model_dir / "audiovae.safetensors"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device="cpu")
    else:
        state = torch.load(model_dir / "audiovae.pth",
                           map_location="cpu",
                           weights_only=True)
        state = state.get("state_dict", state)
    vae.load_state_dict(state, strict=True)
    vae = vae.to(device=device, dtype=torch.float32).eval()
    tokenizer = LlamaTokenizerFast.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    # text_tokenizerは公式backboneと同じmask処理を使う。token ID空間は変更しない。
    tokenize = mask_multichar_chinese_tokens(tokenizer)
    return (vae, tokenize, config.patch_size, vae.latent_dim,
            config.lm_config.vocab_size)


@torch.inference_mode()
def prepare(
    dataframe: pd.DataFrame,
    source_dir: Path,
    output: Path,
    model_dir: Path,
    device: str = "cuda:0",
    max_seconds: float = 30.0,
) -> None:
    """音声/本文表から実latent cacheを抽出する。

    追加ラベルは要求しない。source_dirは入力音声pathの基準。Parquetの出力pathもそのメタデータファイルの親基準で保存する。
    書き起こしとの対応を壊すcrop/VADは行わず、長すぎる行は拒否する。

    Args:
        dataframe: 1行を1発話とする表。必要列は呼び出し元のstage schemaに従う。
        source_dir: 入力表の相対音声パスを解決する基準ディレクトリ。
        output: 生成するcache、checkpoint、または保存先ディレクトリ。
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        device: 処理先。cpu、cuda:0等のPyTorchデバイス指定。
        max_seconds: 1発話として受け付ける音声長の上限、単位は秒。

    Returns:
        None。抽出Tensorを.pt、対応するメタデータをParquetへ保存する。
    """
    if not {"audio_path", "text"}.issubset(
            dataframe.columns) or dataframe.empty:
        raise ValueError("Nonempty table with audio_path and text is required")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    vae, tokenize, patch_size, latent_dim, vocab = load_frontend(
        model_dir, device)
    if (patch_size, latent_dim, vae.sample_rate) != (4, 64, 16000):
        raise ValueError("This survey adapter requires V2 16kHz / 4x64 assets")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tensors").mkdir()
    rows = []
    for index, row in enumerate(dataframe.to_dict("records")):
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Missing transcript at row {index}")
        path = Path(str(row["audio_path"]))
        path = path if path.is_absolute() else source_dir / path
        waveform, rate = torchaudio.load(str(path))  # [C,S]。C=channel数。
        if (waveform.numel() == 0 or not torch.isfinite(waveform).all()
                or waveform.size(-1) / rate > max_seconds):
            raise ValueError(f"Empty/nonfinite/too long; split first: {path}")
        # sampleは削除せず、channel集約と16kHzへのsample rate変換だけ行う。
        waveform = torchaudio.functional.resample(
            waveform.mean(dim=0, keepdim=True),
            rate,
            vae.sample_rate,
        )  # 音声[C,S]→mono波形[1,S16]。Cは入力チャネル数、S16は16kHzでのサンプル数。
        patch_samples = vae.chunk_size * patch_size
        valid_samples = waveform.size(-1)
        waveform = F.pad(waveform, (0, (-valid_samples) % patch_samples))
        latent = vae.encode(waveform.to(device), vae.sample_rate)
        patches = rearrange(latent[0], "c (n p) -> n p c", p=patch_size)
        # [1,64,N*4] -> [N,4,64]。N=右padding込みpatch数。
        text = re.sub(r"\s+", " ", row["text"].replace("\n", " ")).strip()
        ids = torch.tensor(tokenize(text), dtype=torch.long)  # [L]、本文IDのみ。
        if ids.numel() == 0 or int(ids.max()) >= vocab:
            raise ValueError(f"Empty/out-of-vocabulary tokens at row {index}")
        text_path = Path("tensors") / f"{index:08d}_text.pt"
        latent_path = Path("tensors") / f"{index:08d}_latent.pt"
        torch.save(ids, output / text_path)
        torch.save(patches.cpu().float().contiguous(), output / latent_path)
        rows.append({
            **row,
            "audio_path":
            os.path.relpath(path.resolve(), output.resolve()),
            "text_ids_path":
            text_path.as_posix(),
            "latent_path":
            latent_path.as_posix(),
            "samples_16k":
            valid_samples,
            "normalized_text":
            text,
        })
    pd.DataFrame(rows).to_parquet(output / "train.parquet",
                                  index=False,
                                  engine="pyarrow",
                                  compression="zstd")
    # データとIDの互換性を保つ値だけ固定し、LM容量は小型の明示例にする。
    small = {
        "dim": 256,
        "intermediate": 768,
        "layers": 4,
        "heads": 4,
        "kv_heads": 2,
        "head_dim": 64
    }
    config = {
        "vocab_size": vocab,
        "patch_size": patch_size,
        "latent_dim": latent_dim,
        "base": small,
        "local": {
            **small, "layers": 2
        },
        "residual_layers": 2,
        "fsq_dim": 128,
        "audio_start": 101,
        "audio_end": 102,
        "ref_start": 103,
        "ref_end": 104,
    }
    (output / "config_scratch.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    (output / "preparation.json").write_text(json.dumps(
        {
            "model_dir":
            str(model_dir),
            "input_sample_rate":
            16000,
            "patch_size":
            patch_size,
            "latent_dim":
            latent_dim,
            "reference":
            "none; speaker_id/emotion annotations are not required",
            "padding":
            "right-zero; final partial patch retained as official encoding",
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
    """raw CSVと公式VAE/tokenizerの資産pathからcacheを生成するCLI。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.repo:
        sys.path.insert(0, str(args.repo.resolve() / "src"))
    if args.output.exists():
        raise FileExistsError(args.output)
    prepare(read_source_table(args.metadata),
            args.metadata.resolve().parent, args.output,
            args.model_dir.resolve(), args.device, args.max_seconds)


if __name__ == "__main__":
    main()
