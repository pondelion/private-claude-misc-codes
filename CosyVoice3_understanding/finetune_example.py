#!/usr/bin/env python3
"""
CosyVoice3 ファインチューンスクリプト
====================================

pandas DataFrameで管理されたデータセットメタデータからCosyVoice3をファインチューニングする。

必要条件:
    - CosyVoice リポジトリ (git clone https://github.com/FunAudioLLM/CosyVoice)
    - pip install -r requirements.txt (CosyVoiceリポジトリ内)
    - Pretrained model (Fun-CosyVoice3-0.5B):
        modelscope download --model FunAudioLLM/CosyVoice3-0.5B --local_dir pretrained_models/Fun-CosyVoice3-0.5B

DataFrameの必須カラム:
    - audio_path (str): 音声ファイルの絶対パス (wav推奨)
    - text (str): 書き起こしテキスト
    - speaker_id (str): 話者ID

DataFrameのオプションカラム:
    - instruct (str): 指示テキスト (デフォルト: "You are a helpful assistant.<|endofprompt|>")

使用例 (CLI):
    python finetune_example.py \
        --metadata_path /path/to/metadata.parquet \
        --cosyvoice_root /path/to/CosyVoice \
        --pretrained_model_dir /path/to/Fun-CosyVoice3-0.5B \
        --output_dir /path/to/output \
        --model llm \
        --num_gpus 1

使用例 (Python):
    import pandas as pd
    from finetune_example import finetune_from_dataframe

    df = pd.read_parquet("metadata.parquet")
    finetune_from_dataframe(
        df,
        cosyvoice_root="/path/to/CosyVoice",
        pretrained_model_dir="/path/to/Fun-CosyVoice3-0.5B",
        output_dir="./cosyvoice3_finetune",
        model="llm",
    )

備考:
    - LLM/Flow/HiFiGAN は別々にファインチューニングする (--model で指定)
    - 実務上は LLM のファインチューンが最も一般的
    - データ準備 (embedding/speech_token抽出) には ONNX Runtime が必要
    - トレーニングは torchrun (DDP) で起動される (1GPU時も)
    - dev_ratio で自動的に train/dev 分割される
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnxruntime
import pandas as pd
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from tqdm import tqdm


# ======================== 定数 ========================

DEFAULT_INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
NUM_UTTS_PER_PARQUET = 1000


# ======================== Step 1: Kaldi形式ファイル作成 ========================


def prepare_kaldi_files(
    df: pd.DataFrame,
    data_dir: str,
    instruct_text: str = DEFAULT_INSTRUCT,
) -> None:
    """
    DataFrameからKaldi形式のメタデータファイルを作成する。

    出力ファイル:
        {data_dir}/wav.scp   - utt_id → audio_path
        {data_dir}/text      - utt_id → text
        {data_dir}/utt2spk   - utt_id → speaker_id
        {data_dir}/spk2utt   - speaker_id → utt_id_list
        {data_dir}/instruct  - utt_id → instruct_text
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    utt2wav = {}
    utt2text = {}
    utt2spk = {}
    spk2utt = {}
    utt2instruct = {}

    for idx, row in df.iterrows():
        audio_path = str(row["audio_path"])
        text = str(row["text"])
        speaker_id = str(row["speaker_id"])

        # utt_id を生成 (speaker_id + index)
        utt_id = f"{speaker_id}_{idx:06d}"

        utt2wav[utt_id] = audio_path
        utt2text[utt_id] = text
        utt2spk[utt_id] = speaker_id
        if speaker_id not in spk2utt:
            spk2utt[speaker_id] = []
        spk2utt[speaker_id].append(utt_id)

        # instruct: DataFrameにカラムがあればそれを使用、なければデフォルト
        if "instruct" in row and pd.notna(row["instruct"]):
            utt2instruct[utt_id] = str(row["instruct"])
        else:
            utt2instruct[utt_id] = instruct_text

    with open(data_dir / "wav.scp", "w") as f:
        for k, v in utt2wav.items():
            f.write(f"{k} {v}\n")

    with open(data_dir / "text", "w") as f:
        for k, v in utt2text.items():
            f.write(f"{k} {v}\n")

    with open(data_dir / "utt2spk", "w") as f:
        for k, v in utt2spk.items():
            f.write(f"{k} {v}\n")

    with open(data_dir / "spk2utt", "w") as f:
        for k, v in spk2utt.items():
            f.write(f"{k} {' '.join(v)}\n")

    with open(data_dir / "instruct", "w") as f:
        for k, v in utt2instruct.items():
            f.write(f"{k} {v}\n")

    print(f"Kaldi files 作成完了: {len(utt2wav)} utterances, {len(spk2utt)} speakers")


# ======================== Step 2: Speaker Embedding 抽出 ========================


def extract_embeddings(
    data_dir: str,
    campplus_onnx_path: str,
    num_threads: int = 8,
) -> None:
    """
    CamPPlus ONNX モデルで話者埋め込みを抽出する。

    出力:
        {data_dir}/utt2embedding.pt  - {utt_id: list[float]}
        {data_dir}/spk2embedding.pt  - {spk_id: list[float]} (話者平均)
    """
    data_dir = Path(data_dir)

    utt2wav = {}
    with open(data_dir / "wav.scp") as f:
        for line in f:
            parts = line.strip().split()
            utt2wav[parts[0]] = parts[1]

    utt2spk = {}
    with open(data_dir / "utt2spk") as f:
        for line in f:
            parts = line.strip().split()
            utt2spk[parts[0]] = parts[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        campplus_onnx_path, sess_options=option, providers=["CPUExecutionProvider"]
    )

    def single_job(utt):
        audio, sample_rate = torchaudio.load(utt2wav[utt])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        # 10秒以上の場合はランダムクロップ
        max_len = 10 * 16000
        if audio.shape[1] > max_len:
            start = torch.randint(0, audio.shape[1] - max_len, (1,)).item()
            audio = audio[:, start : start + max_len]
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            ort_session.run(
                None,
                {ort_session.get_inputs()[0].name: feat.unsqueeze(0).cpu().numpy()},
            )[0]
            .flatten()
            .tolist()
        )
        return utt, embedding

    print("Speaker embedding 抽出中...")
    executor = ThreadPoolExecutor(max_workers=num_threads)
    all_tasks = [executor.submit(single_job, utt) for utt in utt2wav.keys()]

    utt2embedding = {}
    spk2embedding = {}
    for future in tqdm(as_completed(all_tasks), total=len(all_tasks)):
        utt, embedding = future.result()
        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    executor.shutdown()

    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    torch.save(utt2embedding, str(data_dir / "utt2embedding.pt"))
    torch.save(spk2embedding, str(data_dir / "spk2embedding.pt"))
    print(f"Embedding 抽出完了: {len(utt2embedding)} utts, {len(spk2embedding)} spks")


# ======================== Step 3: Speech Token 抽出 ========================


def extract_speech_tokens(
    data_dir: str,
    tokenizer_onnx_path: str,
    num_threads: int = 4,
) -> None:
    """
    Speech Tokenizer ONNX モデルで音声トークンを抽出する。

    出力:
        {data_dir}/utt2speech_token.pt - {utt_id: list[int]}
    """
    data_dir = Path(data_dir)

    utt2wav = {}
    with open(data_dir / "wav.scp") as f:
        for line in f:
            parts = line.strip().split()
            utt2wav[parts[0]] = parts[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        tokenizer_onnx_path,
        sess_options=option,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    def single_job(utt):
        audio, sample_rate = torchaudio.load(utt2wav[utt], backend="soundfile")
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[1] / 16000 > 30:
            print(f"Warning: {utt} is longer than 30s, skipping speech token extraction")
            return utt, []
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        speech_token = (
            ort_session.run(
                None,
                {
                    ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                    ort_session.get_inputs()[1].name: np.array(
                        [feat.shape[2]], dtype=np.int32
                    ),
                },
            )[0]
            .flatten()
            .tolist()
        )
        return utt, speech_token

    print("Speech token 抽出中...")
    executor = ThreadPoolExecutor(max_workers=num_threads)
    all_tasks = [executor.submit(single_job, utt) for utt in utt2wav.keys()]

    utt2speech_token = {}
    for future in tqdm(as_completed(all_tasks), total=len(all_tasks)):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token
    executor.shutdown()

    torch.save(utt2speech_token, str(data_dir / "utt2speech_token.pt"))
    print(f"Speech token 抽出完了: {len(utt2speech_token)} utts")


# ======================== Step 4: Parquet ファイル作成 ========================


def make_parquet_files(
    data_dir: str,
    parquet_dir: str,
    num_utts_per_parquet: int = NUM_UTTS_PER_PARQUET,
) -> None:
    """
    Kaldi形式ファイルと抽出済み特徴量からParquetファイルを作成する。

    出力:
        {parquet_dir}/parquet_XXXXXXXXX.tar  - Parquetファイル群
        {parquet_dir}/data.list              - Parquetファイルパスリスト
    """
    data_dir = Path(data_dir)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Load Kaldi files
    utt2wav = {}
    with open(data_dir / "wav.scp") as f:
        for line in f:
            parts = line.strip().split()
            utt2wav[parts[0]] = parts[1]

    utt2text = {}
    with open(data_dir / "text") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            utt2text[parts[0]] = parts[1] if len(parts) > 1 else ""

    utt2spk = {}
    with open(data_dir / "utt2spk") as f:
        for line in f:
            parts = line.strip().split()
            utt2spk[parts[0]] = parts[1]

    utt2instruct = None
    instruct_path = data_dir / "instruct"
    if instruct_path.exists():
        utt2instruct = {}
        with open(instruct_path) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                utt2instruct[parts[0]] = parts[1] if len(parts) > 1 else ""

    # Load pre-extracted features
    utt2embedding = None
    emb_path = data_dir / "utt2embedding.pt"
    if emb_path.exists():
        utt2embedding = torch.load(str(emb_path), weights_only=False)

    spk2embedding = None
    spk_emb_path = data_dir / "spk2embedding.pt"
    if spk_emb_path.exists():
        spk2embedding = torch.load(str(spk_emb_path), weights_only=False)

    utt2speech_token = None
    token_path = data_dir / "utt2speech_token.pt"
    if token_path.exists():
        utt2speech_token = torch.load(str(token_path), weights_only=False)

    utts = list(utt2wav.keys())

    print(f"Parquet ファイル作成中 ({len(utts)} utts)...")
    parquet_list = []

    for i in range(0, len(utts), num_utts_per_parquet):
        batch_utts = utts[i : i + num_utts_per_parquet]
        parquet_file = parquet_dir / f"parquet_{i // num_utts_per_parquet:09d}.tar"

        data_list = []
        for utt in batch_utts:
            audio_data = open(utt2wav[utt], "rb").read()
            data_list.append(audio_data)

        df = pd.DataFrame()
        df["utt"] = batch_utts
        df["audio_data"] = data_list
        df["wav"] = [utt2wav[utt] for utt in batch_utts]
        df["text"] = [utt2text[utt] for utt in batch_utts]
        df["spk"] = [utt2spk[utt] for utt in batch_utts]

        if utt2embedding is not None:
            df["utt_embedding"] = [utt2embedding[utt] for utt in batch_utts]
        if spk2embedding is not None:
            df["spk_embedding"] = [spk2embedding[utt2spk[utt]] for utt in batch_utts]
        if utt2speech_token is not None:
            df["speech_token"] = [utt2speech_token[utt] for utt in batch_utts]
        if utt2instruct is not None:
            df["instruct"] = [utt2instruct[utt] for utt in batch_utts]

        df.to_parquet(str(parquet_file))
        parquet_list.append(str(parquet_file))

    with open(parquet_dir / "data.list", "w") as f:
        for path in parquet_list:
            f.write(path + "\n")

    print(f"Parquet 作成完了: {len(parquet_list)} files")


# ======================== Step 5: Train/Dev 分割 ========================


def split_train_dev(
    df: pd.DataFrame,
    dev_ratio: float = 0.05,
    min_dev_samples: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DataFrameをtrain/devに分割する。話者単位ではなくランダム分割。
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_dev = max(min_dev_samples, int(len(df) * dev_ratio))
    n_dev = min(n_dev, len(df) // 2)  # devがtrainより大きくならないように

    dev_df = df.iloc[:n_dev]
    train_df = df.iloc[n_dev:]
    return train_df, dev_df


# ======================== Step 6: トレーニング起動 ========================


def launch_training(
    cosyvoice_root: str,
    pretrained_model_dir: str,
    output_dir: str,
    train_data_list: str,
    dev_data_list: str,
    model: str = "llm",
    config_path: str | None = None,
    num_gpus: int = 1,
    num_workers: int = 2,
    use_amp: bool = True,
    train_engine: str = "torch_ddp",
) -> None:
    """
    torchrun で CosyVoice3 のトレーニングを起動する。

    Args:
        cosyvoice_root: CosyVoice リポジトリのルートパス
        pretrained_model_dir: pretrained model ディレクトリパス
        output_dir: 出力ディレクトリ
        train_data_list: train data.list ファイルパス
        dev_data_list: dev data.list ファイルパス
        model: トレーニングするモデル ("llm", "flow", "hifigan")
        config_path: YAML config パス (None = デフォルト使用)
        num_gpus: 使用GPU数
        num_workers: DataLoader ワーカー数
        use_amp: AMP (混合精度) を使用
        train_engine: "torch_ddp" or "deepspeed"
    """
    cosyvoice_root = os.path.abspath(cosyvoice_root)
    pretrained_model_dir = os.path.abspath(pretrained_model_dir)

    # デフォルト config
    if config_path is None:
        config_path = os.path.join(
            cosyvoice_root,
            "examples/libritts/cosyvoice3/conf/cosyvoice3.yaml",
        )
    config_path = os.path.abspath(config_path)

    # checkpoint
    checkpoint_path = os.path.join(pretrained_model_dir, f"{model}.pt")
    if not os.path.isfile(checkpoint_path):
        print(f"Warning: checkpoint {checkpoint_path} not found, training from scratch")
        checkpoint_path = None

    # qwen pretrain path
    qwen_path = os.path.join(pretrained_model_dir, "CosyVoice-BlankEN")

    # model output dir
    model_dir = os.path.join(output_dir, "exp", model, train_engine)
    tensorboard_dir = os.path.join(output_dir, "tensorboard", model, train_engine)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # train script
    train_script = os.path.join(cosyvoice_root, "cosyvoice/bin/train.py")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "--rdzv_id=1986",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        train_script,
        "--train_engine",
        train_engine,
        "--config",
        config_path,
        "--train_data",
        os.path.abspath(train_data_list),
        "--cv_data",
        os.path.abspath(dev_data_list),
        "--qwen_pretrain_path",
        qwen_path,
        "--onnx_path",
        pretrained_model_dir,
        "--model",
        model,
        "--model_dir",
        model_dir,
        "--tensorboard_dir",
        tensorboard_dir,
        "--ddp.dist_backend",
        "nccl",
        "--num_workers",
        str(num_workers),
        "--prefetch",
        "100",
        "--pin_memory",
    ]

    if checkpoint_path is not None:
        cmd.extend(["--checkpoint", checkpoint_path])

    if use_amp:
        cmd.append("--use_amp")

    if train_engine == "deepspeed":
        ds_config = os.path.join(
            cosyvoice_root,
            "examples/libritts/cosyvoice3/conf/ds_stage2.json",
        )
        cmd.extend(["--deepspeed_config", ds_config])
        cmd.extend(["--deepspeed.save_states", "model+optimizer"])

    env = os.environ.copy()
    env["PYTHONPATH"] = cosyvoice_root + ":" + env.get("PYTHONPATH", "")
    # third_party を追加 (matcha等)
    third_party = os.path.join(cosyvoice_root, "third_party/Matcha-TTS")
    if os.path.isdir(third_party):
        env["PYTHONPATH"] = third_party + ":" + env["PYTHONPATH"]

    print("\n" + "=" * 60)
    print(f"Training command:")
    print(" ".join(cmd))
    print("=" * 60 + "\n")

    subprocess.run(cmd, env=env, check=True)


# ======================== メインエントリ ========================


def finetune_from_dataframe(
    df: pd.DataFrame,
    cosyvoice_root: str,
    pretrained_model_dir: str,
    output_dir: str,
    model: str = "llm",
    config_path: str | None = None,
    instruct_text: str = DEFAULT_INSTRUCT,
    dev_ratio: float = 0.05,
    num_gpus: int = 1,
    num_workers: int = 2,
    use_amp: bool = True,
    train_engine: str = "torch_ddp",
    embedding_threads: int = 8,
    token_threads: int = 4,
    skip_data_prep: bool = False,
) -> None:
    """
    pandas DataFrameからCosyVoice3をファインチューニングする。

    Args:
        df: 以下のカラムを持つDataFrame
            - audio_path (str): 音声ファイルの絶対パス
            - text (str): 書き起こしテキスト
            - speaker_id (str): 話者ID
            - instruct (str, optional): 指示テキスト
        cosyvoice_root: CosyVoice リポジトリのルートパス
        pretrained_model_dir: pretrained model ディレクトリ
            (Fun-CosyVoice3-0.5B: campplus.onnx, speech_tokenizer_v3.onnx,
             CosyVoice-BlankEN/, llm.pt, flow.pt 等を含む)
        output_dir: 出力先ディレクトリ
        model: トレーニング対象 ("llm", "flow", "hifigan")
        config_path: YAML config パス (None = examples/libritts/cosyvoice3/conf/cosyvoice3.yaml)
        instruct_text: デフォルト指示テキスト
        dev_ratio: dev データ割合
        num_gpus: 使用GPU数
        num_workers: DataLoader ワーカー数
        use_amp: AMP (bf16混合精度)
        train_engine: "torch_ddp" or "deepspeed"
        embedding_threads: embedding抽出スレッド数
        token_threads: speech token抽出スレッド数
        skip_data_prep: Trueの場合データ準備をスキップ (再開時用)
    """
    output_dir = os.path.abspath(output_dir)
    pretrained_model_dir = os.path.abspath(pretrained_model_dir)

    # 必須カラムチェック
    required = ["audio_path", "text", "speaker_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrameに必須カラムがありません: {missing}")

    # ONNX モデルパス確認
    campplus_path = os.path.join(pretrained_model_dir, "campplus.onnx")
    tokenizer_path = os.path.join(pretrained_model_dir, "speech_tokenizer_v3.onnx")
    if not os.path.isfile(campplus_path):
        raise FileNotFoundError(f"campplus.onnx が見つかりません: {campplus_path}")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            f"speech_tokenizer_v3.onnx が見つかりません: {tokenizer_path}"
        )

    if not skip_data_prep:
        # ---- Step 1: Train/Dev 分割 ----
        print("=" * 60)
        print("Step 1: Train/Dev 分割")
        print("=" * 60)
        train_df, dev_df = split_train_dev(df, dev_ratio=dev_ratio)
        print(f"Train: {len(train_df)} samples, Dev: {len(dev_df)} samples")

        # ---- Step 2: Kaldi files 作成 ----
        print("\n" + "=" * 60)
        print("Step 2: Kaldi files 作成")
        print("=" * 60)
        train_data_dir = os.path.join(output_dir, "data/train")
        dev_data_dir = os.path.join(output_dir, "data/dev")
        prepare_kaldi_files(train_df, train_data_dir, instruct_text)
        prepare_kaldi_files(dev_df, dev_data_dir, instruct_text)

        # ---- Step 3: Speaker Embedding 抽出 ----
        print("\n" + "=" * 60)
        print("Step 3: Speaker Embedding 抽出")
        print("=" * 60)
        extract_embeddings(train_data_dir, campplus_path, num_threads=embedding_threads)
        extract_embeddings(dev_data_dir, campplus_path, num_threads=embedding_threads)

        # ---- Step 4: Speech Token 抽出 ----
        print("\n" + "=" * 60)
        print("Step 4: Speech Token 抽出")
        print("=" * 60)
        extract_speech_tokens(
            train_data_dir, tokenizer_path, num_threads=token_threads
        )
        extract_speech_tokens(dev_data_dir, tokenizer_path, num_threads=token_threads)

        # ---- Step 5: Parquet 作成 ----
        print("\n" + "=" * 60)
        print("Step 5: Parquet ファイル作成")
        print("=" * 60)
        train_parquet_dir = os.path.join(output_dir, "data/train/parquet")
        dev_parquet_dir = os.path.join(output_dir, "data/dev/parquet")
        make_parquet_files(train_data_dir, train_parquet_dir)
        make_parquet_files(dev_data_dir, dev_parquet_dir)
    else:
        train_parquet_dir = os.path.join(output_dir, "data/train/parquet")
        dev_parquet_dir = os.path.join(output_dir, "data/dev/parquet")
        print("データ準備をスキップしました (--skip_data_prep)")

    # ---- Step 6: トレーニング起動 ----
    print("\n" + "=" * 60)
    print(f"Step 6: トレーニング起動 (model={model})")
    print("=" * 60)

    train_data_list = os.path.join(train_parquet_dir, "data.list")
    dev_data_list = os.path.join(dev_parquet_dir, "data.list")

    if not os.path.isfile(train_data_list):
        raise FileNotFoundError(f"train data.list が見つかりません: {train_data_list}")
    if not os.path.isfile(dev_data_list):
        raise FileNotFoundError(f"dev data.list が見つかりません: {dev_data_list}")

    launch_training(
        cosyvoice_root=cosyvoice_root,
        pretrained_model_dir=pretrained_model_dir,
        output_dir=output_dir,
        train_data_list=train_data_list,
        dev_data_list=dev_data_list,
        model=model,
        config_path=config_path,
        num_gpus=num_gpus,
        num_workers=num_workers,
        use_amp=use_amp,
        train_engine=train_engine,
    )

    print("\nファインチューニング完了!")
    print(f"モデル保存先: {os.path.join(output_dir, 'exp', model, train_engine)}")


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
        description="CosyVoice3 Finetune from pandas DataFrame",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="DataFrameファイルパス (.parquet/.csv/.json/.jsonl/.tsv)",
    )
    parser.add_argument(
        "--cosyvoice_root",
        type=str,
        required=True,
        help="CosyVoice リポジトリのルートパス",
    )
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        required=True,
        help="pretrained model ディレクトリ (Fun-CosyVoice3-0.5B)",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--model",
        type=str,
        default="llm",
        choices=["llm", "flow", "hifigan"],
        help="トレーニング対象モデル",
    )
    parser.add_argument("--config_path", type=str, default=None, help="YAML config パス")
    parser.add_argument(
        "--instruct_text",
        type=str,
        default=DEFAULT_INSTRUCT,
        help="デフォルト指示テキスト",
    )
    parser.add_argument("--dev_ratio", type=float, default=0.05, help="dev データ割合")
    parser.add_argument("--num_gpus", type=int, default=1, help="使用GPU数")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader ワーカー数")
    parser.add_argument(
        "--no_amp", action="store_true", help="AMP (混合精度) を無効化"
    )
    parser.add_argument(
        "--train_engine",
        type=str,
        default="torch_ddp",
        choices=["torch_ddp", "deepspeed"],
    )
    parser.add_argument(
        "--embedding_threads", type=int, default=8, help="embedding抽出スレッド数"
    )
    parser.add_argument(
        "--token_threads", type=int, default=4, help="speech token抽出スレッド数"
    )
    parser.add_argument(
        "--skip_data_prep",
        action="store_true",
        help="データ準備をスキップ (データ準備済みの場合)",
    )

    args = parser.parse_args()

    df = load_dataframe(args.metadata_path)
    print(f"メタデータ: {len(df)} 件")
    print(f"カラム: {list(df.columns)}")
    print(df.head())

    finetune_from_dataframe(
        df=df,
        cosyvoice_root=args.cosyvoice_root,
        pretrained_model_dir=args.pretrained_model_dir,
        output_dir=args.output_dir,
        model=args.model,
        config_path=args.config_path,
        instruct_text=args.instruct_text,
        dev_ratio=args.dev_ratio,
        num_gpus=args.num_gpus,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        train_engine=args.train_engine,
        embedding_threads=args.embedding_threads,
        token_threads=args.token_threads,
        skip_data_prep=args.skip_data_prep,
    )


if __name__ == "__main__":
    main()
