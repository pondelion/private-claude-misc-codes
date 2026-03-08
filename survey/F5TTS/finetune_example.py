#!/usr/bin/env python3
"""
F5-TTS ファインチューンスクリプト
================================

pandas DataFrameで管理されたデータセットメタデータからF5-TTS (F5TTS_v1_Base) をファインチューニングする。

必要条件:
    pip install f5-tts pandas

DataFrameの必須カラム:
    - audio_path (str): 音声ファイルの絶対パス (wav, flac等)
    - text (str): 書き起こしテキスト (中国語/英語/日本語)
    - duration (float): 音声の長さ (秒)

使用例 (CLI):
    python finetune_example.py \
        --metadata_path /path/to/metadata.parquet \
        --output_dir /path/to/output \
        --epochs 50 \
        --learning_rate 1e-5

使用例 (Python):
    import pandas as pd
    from finetune_example import finetune_from_dataframe

    df = pd.read_parquet("metadata.parquet")
    finetune_from_dataframe(df, output_dir="./f5tts_finetune", epochs=50)

備考:
    - F5TTS_v1_Base はpinyinトークナイザで学習されているため、テキストは自動的にpinyin変換される
    - pinyinトークナイザは中国語→ピンイン変換、英語→文字単位分割を行う
    - 日本語漢字も中国語として扱われピンイン変換されるため、精度が落ちる可能性がある
    - batch_size_per_gpu はフレーム数指定 (例: 3200 ≈ 音声30秒分程度)
    - pretrained checkpoint と vocab.txt は未指定時にHuggingFaceから自動ダウンロードされる
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from cached_path import cached_path
from datasets import Dataset as HFDataset_
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import CustomDataset
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer


# ======================== 定数 (F5TTS_v1_Base) ========================

TARGET_SAMPLE_RATE = 24000
N_MEL_CHANNELS = 100
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
MEL_SPEC_TYPE = "vocos"

PRETRAINED_CKPT_HF = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
PRETRAINED_VOCAB_HF = "hf://SWivid/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"

F5TTS_V1_BASE_CONFIG = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

MEL_SPEC_KWARGS = dict(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mel_channels=N_MEL_CHANNELS,
    target_sample_rate=TARGET_SAMPLE_RATE,
    mel_spec_type=MEL_SPEC_TYPE,
)


# ======================== Vocab取得 ========================


def resolve_vocab_path(vocab_path: str | None = None) -> str:
    """
    F5TTS_v1_Baseのvocab.txtを取得する。

    検索順序:
        1. ユーザー指定パス
        2. F5-TTSパッケージのローカルデータディレクトリ (editableインストール時)
        3. HuggingFaceからダウンロード (cached_path)
    """
    if vocab_path and os.path.isfile(vocab_path):
        print(f"vocab.txt: {vocab_path} (ユーザー指定)")
        return vocab_path

    try:
        from importlib.resources import files as pkg_files

        local_vocab = str(pkg_files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt"))
        if os.path.isfile(local_vocab):
            print(f"vocab.txt: {local_vocab} (ローカル)")
            return local_vocab
    except Exception:
        pass

    print("vocab.txt をHuggingFaceからダウンロード中...")
    resolved = str(cached_path(PRETRAINED_VOCAB_HF))
    print(f"vocab.txt: {resolved} (HuggingFace)")
    return resolved


# ======================== データ準備 ========================


def prepare_dataset_from_dataframe(
    df: pd.DataFrame,
    dataset_dir: str,
    vocab_path: str,
) -> None:
    """
    pandas DataFrameからF5-TTSのCustomDataset形式のデータセットを準備する。

    出力ファイル:
        {dataset_dir}/raw.arrow    - 音声メタデータ (audio_path, text, duration)
        {dataset_dir}/duration.json - 全サンプルのdurationリスト
        {dataset_dir}/vocab.txt     - 語彙ファイル (pretrained model用)

    Args:
        df: audio_path, text, durationカラムを持つDataFrame
        dataset_dir: 出力ディレクトリ
        vocab_path: コピー元のvocab.txtパス
    """
    required = ["audio_path", "text", "duration"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrameに必須カラムがありません: {missing}")

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # テキストをpinyin形式に変換 (F5TTS_v1_Base pretrained modelに合わせる)
    # convert_char_to_pinyin: 中国語→ピンイントークン, 英語→文字単位, 記号→そのまま
    # 出力: list[list[str]] (各サンプルのトークンリスト)
    print("テキストをpinyin形式に変換中...")
    raw_texts = df["text"].tolist()
    converted_texts = []
    batch_size = 100
    for i in tqdm(range(0, len(raw_texts), batch_size), desc="Pinyin変換"):
        batch = raw_texts[i : i + batch_size]
        converted_texts.extend(convert_char_to_pinyin(batch, polyphone=True))

    # Arrow file 作成
    # CustomDataset.__getitem__が各行から audio_path, text, duration を読み出す
    print("raw.arrow を作成中...")
    arrow_path = dataset_dir / "raw.arrow"
    durations = []

    with ArrowWriter(path=str(arrow_path)) as writer:
        for idx in tqdm(range(len(df)), desc="Arrow書き込み"):
            row = df.iloc[idx]
            duration = float(row["duration"])
            audio_path = str(row["audio_path"])
            text = converted_texts[idx]  # list[str]: pinyinトークンリスト

            writer.write(
                {
                    "audio_path": audio_path,
                    "text": text,
                    "duration": duration,
                }
            )
            durations.append(duration)
        writer.finalize()

    # duration.json (DynamicBatchSamplerのフレーム長計算に使用)
    with open(dataset_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)

    # vocab.txt をコピー (pretrained modelと同じvocabを使用)
    shutil.copy2(vocab_path, str(dataset_dir / "vocab.txt"))

    total_hours = sum(durations) / 3600
    print(f"データ準備完了: {len(durations)} サンプル, {total_hours:.2f} 時間")


# ======================== メインエントリ ========================


def finetune_from_dataframe(
    df: pd.DataFrame,
    output_dir: str,
    epochs: int = 100,
    learning_rate: float = 1e-5,
    batch_size_per_gpu: int = 3200,
    max_samples: int = 64,
    grad_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    num_warmup_updates: int = 2000,
    save_per_updates: int = 500,
    last_per_updates: int = 100,
    keep_last_n_checkpoints: int = 5,
    pretrain_ckpt: str | None = None,
    vocab_path: str | None = None,
    logger: str | None = None,
    log_samples: bool = False,
    bnb_optimizer: bool = False,
    num_workers: int = 4,
) -> None:
    """
    pandas DataFrameからF5-TTS (F5TTS_v1_Base) をファインチューニングする。

    Args:
        df: 以下のカラムを持つDataFrame
            - audio_path (str): 音声ファイルの絶対パス
            - text (str): 書き起こしテキスト
            - duration (float): 音声の長さ (秒)
        output_dir: 出力先ディレクトリ (data/ と ckpts/ が作成される)
        epochs: エポック数
        learning_rate: 学習率 (finetune推奨: 1e-5 ~ 1e-4)
        batch_size_per_gpu: GPU毎バッチサイズ (フレーム数, frame-based batching)
        max_samples: 1バッチ内の最大サンプル数
        grad_accumulation_steps: 勾配蓄積ステップ数
        max_grad_norm: 勾配クリッピング閾値
        num_warmup_updates: ウォームアップ更新回数
        save_per_updates: チェックポイント保存間隔 (更新回数)
        last_per_updates: model_last.pt 保存間隔
        keep_last_n_checkpoints: 保持するチェックポイント数 (-1: 全保持)
        pretrain_ckpt: pretrained checkpointパス (None=HFから自動DL)
        vocab_path: vocab.txtパス (None=自動取得)
        logger: "wandb", "tensorboard", or None
        log_samples: チェックポイント保存時に推論サンプルを生成・保存
        bnb_optimizer: bitsandbytesの8bit AdamWを使用 (VRAM節約)
        num_workers: DataLoaderワーカー数
    """
    dataset_dir = os.path.join(output_dir, "data")
    checkpoint_dir = os.path.join(output_dir, "ckpts")

    # ---- Step 1: Vocab取得 ----
    print("=" * 60)
    print("Step 1: Vocab取得")
    print("=" * 60)
    vocab_path = resolve_vocab_path(vocab_path)

    # ---- Step 2: データ準備 ----
    print("\n" + "=" * 60)
    print("Step 2: データ準備")
    print("=" * 60)
    prepare_dataset_from_dataframe(df, dataset_dir, vocab_path)

    # ---- Step 3: モデル構築 ----
    print("\n" + "=" * 60)
    print("Step 3: モデル構築")
    print("=" * 60)

    # Tokenizer: データセットディレクトリのvocab.txtを使用
    dataset_vocab_path = os.path.join(dataset_dir, "vocab.txt")
    vocab_char_map, vocab_size = get_tokenizer(dataset_vocab_path, "custom")
    print(f"Vocab size: {vocab_size}")

    # F5TTS_v1_Base アーキテクチャでモデル構築
    model = CFM(
        transformer=DiT(
            **F5TTS_V1_BASE_CONFIG,
            text_num_embeds=vocab_size,
            mel_dim=N_MEL_CHANNELS,
        ),
        mel_spec_kwargs=MEL_SPEC_KWARGS,
        vocab_char_map=vocab_char_map,
    )

    # Pretrained checkpoint をチェックポイントディレクトリにコピー
    # Trainer.load_checkpoint() が "pretrained_" prefixで認識する
    os.makedirs(checkpoint_dir, exist_ok=True)
    if pretrain_ckpt is None:
        print("Pretrained checkpoint をHuggingFaceからダウンロード中...")
        pretrain_ckpt = str(cached_path(PRETRAINED_CKPT_HF))

    ckpt_filename = os.path.basename(pretrain_ckpt)
    if not ckpt_filename.startswith("pretrained_"):
        ckpt_filename = "pretrained_" + ckpt_filename
    dest_ckpt = os.path.join(checkpoint_dir, ckpt_filename)
    if not os.path.isfile(dest_ckpt):
        shutil.copy2(pretrain_ckpt, dest_ckpt)
        print(f"Pretrained checkpoint: {dest_ckpt}")

    # Trainer構築
    trainer = Trainer(
        model,
        epochs,
        learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        keep_last_n_checkpoints=keep_last_n_checkpoints,
        checkpoint_path=checkpoint_dir,
        batch_size_per_gpu=batch_size_per_gpu,
        batch_size_type="frame",
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        logger=logger,
        wandb_project="f5tts_finetune",
        wandb_run_name="finetune",
        log_samples=log_samples,
        last_per_updates=last_per_updates,
        bnb_optimizer=bnb_optimizer,
    )

    # ---- Step 4: データセット読み込み ----
    print("\n" + "=" * 60)
    print("Step 4: データセット読み込み")
    print("=" * 60)

    # Arrow fileからHuggingFace Datasetを作成し、CustomDatasetでラップ
    arrow_path = os.path.join(dataset_dir, "raw.arrow")
    hf_dataset = HFDataset_.from_file(arrow_path)

    with open(os.path.join(dataset_dir, "duration.json"), "r", encoding="utf-8") as f:
        durations = json.load(f)["duration"]

    train_dataset = CustomDataset(
        hf_dataset,
        durations=durations,
        preprocessed_mel=False,
        **MEL_SPEC_KWARGS,
    )
    print(f"データセットサイズ: {len(train_dataset)}")

    # ---- Step 5: トレーニング開始 ----
    print("\n" + "=" * 60)
    print("Step 5: トレーニング開始")
    print("=" * 60)

    # Trainer.train() 内部処理:
    #   1. DynamicBatchSampler でフレーム数ベースのバッチ構成
    #   2. load_checkpoint() で pretrained_ prefixのチェックポイントをロード
    #   3. LinearLR warmup → LinearLR decay スケジューラ
    #   4. batch["mel"].permute(0,2,1) して (B, T, mel_dim) に変換
    #   5. model(mel_spec, text=text_inputs, lens=mel_lengths) で Flow Matching loss計算
    #   6. EMA更新 (main processのみ)
    trainer.train(
        train_dataset,
        num_workers=num_workers,
        resumable_with_seed=666,
    )

    print("\nファインチューニング完了!")
    print(f"チェックポイント保存先: {checkpoint_dir}")


# ======================== CLI ========================


def load_dataframe(path: str) -> pd.DataFrame:
    """ファイル形式に基づいてDataFrameを読み込む。"""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        return pd.read_json(path)
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"未対応形式: {path} (parquet/csv/json/jsonl/tsv対応)")


def main():
    parser = argparse.ArgumentParser(
        description="F5-TTS Finetune from pandas DataFrame",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="DataFrameファイルパス (.parquet/.csv/.json/.jsonl/.tsv)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="出力ディレクトリ")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=3200,
        help="GPU毎バッチサイズ (フレーム数, 3200≈30秒分)",
    )
    parser.add_argument("--max_samples", type=int, default=64, help="1バッチ内最大サンプル数")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_warmup_updates", type=int, default=2000)
    parser.add_argument("--save_per_updates", type=int, default=500)
    parser.add_argument("--last_per_updates", type=int, default=100)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=5)
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=None,
        help="pretrainedチェックポイントパス (未指定=HFから自動DL)",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="vocab.txtパス (未指定=自動取得)",
    )
    parser.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard"])
    parser.add_argument("--log_samples", action="store_true", help="推論サンプルを保存")
    parser.add_argument("--bnb_optimizer", action="store_true", help="8bit AdamW (VRAM節約)")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    df = load_dataframe(args.metadata_path)
    print(f"メタデータ: {len(df)} 件")
    print(f"カラム: {list(df.columns)}")
    print(df.head())

    finetune_from_dataframe(
        df=df,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size_per_gpu=args.batch_size_per_gpu,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        last_per_updates=args.last_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        pretrain_ckpt=args.pretrain_ckpt,
        vocab_path=args.vocab_path,
        logger=args.logger,
        log_samples=args.log_samples,
        bnb_optimizer=args.bnb_optimizer,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
