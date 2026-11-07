"""IndexTTSの固定抽出器・vocoder・感情frontendの資産を準備する。

必要な学習済みモデルと由来:
    W2v-BERT: facebook/w2v-bert-2.0。音声のsemantic特徴を抽出する。
    CAMPPlus: funasr/campplusのcampplus_cn_common.bin。話者embeddingを抽出。
    BigVGAN: nvidia/bigvgan_v2_22khz_80band_256x。Melから波形を生成する。
    感情Conformer/Perceiver: IndexTeam/IndexTTS-2.5のGPT checkpoint内。
        独立した外部感情モデルではない。frontend.pyがこの部分だけ再利用する。
    正規化統計/tokenizer/モデル構成: 同じIndexTTS-2.5の配布資産。

使い方:
    python prepare_assets.py --model-dir checkpoints/IndexTTS-2.5
        ローカル資産だけ検査する。モデルを構築せず、不足pathを列挙する。
    python prepare_assets.py --model-dir checkpoints/IndexTTS-2.5 --download
        利用者が明示指定した場合だけ必要資産をHugging Faceから取得する。
        --index-revision等でcommitを指定できる。再現実験では固定すること。
        huggingface_hubが必要だが、このスクリプトは追加installしない。

前処理との関係:
    ここではdatasetも生成モデルも学習しない。音声/text表はprepare_data.pyへ
    渡す。training_example.py --raw-metadataはこの資産検査と前処理を呼び出す。
    前処理では固定特徴抽出器とvocoderを読み込む。感情encoderの重み抽出に
    GPT checkpointが必要だが、生成GPTや公式pipelineは構築しない。
    scratch各段の学習時には、公式codec/T2S/S2Melの重みで初期化しない。

検査範囲:
    configの参照先、重みファイル、外部モデルの設定・前処理設定を検査する。
    パラメータの数値互換や音質を検証する処理ではない。モデル実行はしない。
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json


def check_assets(model_dir: str | Path,
                 include_generation: bool = False) -> list[Path]:
    """固定モデルの読み込みに必要な資産を検査し、確認済みパスを返す。

    モデル構築・downloadは行わない。不足があれば全pathを1回で報告する。 OmegaConfは公式configの読み取りに使用する。

    Args:
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        include_generation: 生成用codec・S2Melの資産も検査するか。

    Returns:
        検査した設定・重みファイルのPathのリスト。不足があれば例外を送出する。
    """
    from omegaconf import OmegaConf

    root = Path(model_dir).resolve()
    config_path = root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}. "
                                "Run prepare_assets.py --download explicitly.")
    config = OmegaConf.load(config_path)
    paths = [config_path]
    paths += [
        root / str(config[key]) for key in ("gpt_checkpoint", "w2v_stat")
    ]
    if include_generation:
        paths += [root / str(config.s2mel_checkpoint), root / "codec.pth"]
    paths += [
        root / name for name in (
            "multilingual_zh_ja_yue_char_del.tiktoken",
            "hf_cache/campplus_cn_common.bin",
            "hf_cache/w2v-bert-2.0/config.json",
            "hf_cache/w2v-bert-2.0/preprocessor_config.json",
            "hf_cache/bigvgan/config.json",
        )
    ]
    missing = [path for path in paths if not path.is_file()]
    # W2v-BERTの単一重みと分割重みの両形式を検査する。
    w2v = root / "hf_cache/w2v-bert-2.0"
    weights = [
        w2v / name for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ) if (w2v / name).is_file()
    ]
    if not weights:
        missing.append(w2v / "model.safetensors (or pytorch/sharded weights)")
    for weight in weights:
        paths.append(weight)
        if weight.name.endswith(".index.json"):
            index = json.loads(weight.read_text(encoding="utf-8"))
            shards = {w2v / name for name in index["weight_map"].values()}
            paths.extend(sorted(shards))
            missing.extend(path for path in shards if not path.is_file())
    # 同梱BigVGANのloaderはbigvgan_generator.ptを読む。
    generator = root / "hf_cache/bigvgan/bigvgan_generator.pt"
    paths.append(generator)
    if not generator.is_file():
        missing.append(generator)
    if missing:
        raise FileNotFoundError("Missing pretrained assets:\n" +
                                "\n".join(str(path) for path in missing))
    return paths


def download_assets(
    model_dir: Path,
    index_revision: str,
    w2v_revision: str,
    campplus_revision: str,
    bigvgan_revision: str,
) -> None:
    """明示要求時にだけ公式配布資産を取得する。

    既存ファイルはHub側で照合する。

    Args:
        model_dir: 事前配置した公開重みと設定のディレクトリ。
        index_revision: IndexTTS公開重みの取得revision。
        w2v_revision: W2v-BERT公開重みの取得revision。
        campplus_revision: CAMPPlus公開重みの取得revision。
        bigvgan_revision: BigVGAN公開重みの取得revision。

    Returns:
        None。明示指定したrevisionの固定資産を保存する。
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    # 1. 感情frontendを含むGPT checkpointと共通設定/tokenizer/統計を取得。
    snapshot_download("IndexTeam/IndexTTS-2.5",
                      revision=index_revision,
                      local_dir=str(model_dir))
    # 2. 固定特徴抽出器とvocoderの重みを、ローカル資産読み込み処理の参照先へ置く。
    snapshot_download("facebook/w2v-bert-2.0",
                      revision=w2v_revision,
                      local_dir=str(model_dir / "hf_cache/w2v-bert-2.0"))
    snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x",
                      revision=bigvgan_revision,
                      local_dir=str(model_dir / "hf_cache/bigvgan"))
    # 3. CAMPPlusはrepo全体ではなく、必要な重みファイルを取得する。
    hf_hub_download("funasr/campplus",
                    "campplus_cn_common.bin",
                    revision=campplus_revision,
                    local_dir=str(model_dir / "hf_cache"))


def file_sha256(path: Path) -> str:
    """重みをTensorとして読まず、1MiBずつ読み取ってファイルhashを計算する。

    Args:
        path: 読み書きするファイルのパス。

    Returns:
        ファイル全体のSHA256を表す16進文字列。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """資産の検査、明示的取得、任意のhash記録を行うCLI。

    Returns:
        None。コマンドラインで指定した処理と保存を行う。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--index-revision", default="main")
    parser.add_argument("--w2v-revision", default="main")
    parser.add_argument("--campplus-revision", default="main")
    parser.add_argument("--bigvgan-revision", default="main")
    parser.add_argument("--hash-manifest", type=Path)
    parser.add_argument(
        "--normalization-models",
        type=Path,
        help="固定FST資産。zh/en/es以下のtagger.fst/verbalizer.fstをコピー")
    args = parser.parse_args()
    if args.download:
        download_assets(args.model_dir, args.index_revision, args.w2v_revision,
                        args.campplus_revision, args.bigvgan_revision)
    if args.normalization_models is not None:
        import shutil

        # 外部normalizerのコードは呼ばず、固定されたFSTモデルだけを配置する。
        for language in ("zh", "en", "es"):
            source = args.normalization_models / language
            if not source.is_dir():
                continue
            target = args.model_dir / "normalization" / language
            target.mkdir(parents=True, exist_ok=True)
            for name in ("tagger.fst", "verbalizer.fst"):
                if not (source / name).is_file():
                    raise FileNotFoundError(source / name)
                shutil.copy2(source / name, target / name)
    paths = check_assets(args.model_dir)
    if args.hash_manifest:
        # 大きな重みのhashには時間がかかるため、指定時だけ記録する。
        record = {
            str(path.relative_to(args.model_dir.resolve())): file_sha256(path)
            for path in paths
        }
        args.hash_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.hash_manifest.write_text(json.dumps(record, indent=2),
                                      encoding="utf-8")
    print(f"Required assets found: {len(paths)} files")


if __name__ == "__main__":
    main()
