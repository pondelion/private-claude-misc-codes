#!/usr/bin/env python
"""
VibeVoice ASR LoRA Fine-tuning - pandas DataFrame ベース実行スクリプト

【前提】
  - pip install -e /home/ym/codes/paper_official/VibeVoice/VibeVoice
  - pip install peft transformers accelerate

【DataFrame のスキーマ】
  必須列:
    audio_path       str   音声ファイルの絶対/相対パス
    segments         any   [{speaker: int, text: str, start: float, end: float}, ...]
                           str (JSON文字列) でも list でも可

  任意列:
    audio_duration   float 音声長（秒）。なければ自動計算
    customized_context  any   ["ホットワード", "文章"] ホットワード等。なければ None
    split            str   "train" / "val" で分割。なければ全件 train 扱い

【実行例】
  # 最小限の実行
  python finetune_from_dataframe.py

  # torchrun でマルチGPU
  torchrun --nproc_per_node=2 finetune_from_dataframe.py \
      --model_path microsoft/VibeVoice-ASR \
      --output_dir ./output \
      --num_train_epochs 3 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1e-4 \
      --bf16
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import HfArgumentParser, Trainer, TrainingArguments

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# 引数クラス
# =============================================================================

@dataclass
class ModelArguments:
    model_path: str = field(
        default="microsoft/VibeVoice-ASR",
        metadata={"help": "HuggingFace モデルID またはローカルパス"},
    )
    lm_pretrained: str = field(
        default="Qwen/Qwen2.5-7B",
        metadata={"help": "プロセッサ初期化用の言語モデル名"},
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={"help": "flash_attention_2 / sdpa / eager"},
    )


@dataclass
class DataArguments:
    csv_path: Optional[str] = field(
        default=None,
        metadata={"help": "DataFrame CSV ファイルのパス。None なら build_sample_dataframe() を使用"},
    )
    audio_base_dir: Optional[str] = field(
        default=None,
        metadata={"help": "audio_path が相対パスの場合のベースディレクトリ"},
    )
    max_audio_length: Optional[float] = field(
        default=None,
        metadata={"help": "この秒数を超える音声をスキップ"},
    )
    val_split: float = field(
        default=0.1,
        metadata={"help": "DataFrame に split 列がない場合の validation 割合"},
    )


@dataclass
class LoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (通常 rank の 2倍)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


# =============================================================================
# サンプル DataFrame の構築（csv_path 未指定時のデモ用）
# =============================================================================

def build_sample_dataframe(toy_dataset_dir: str) -> pd.DataFrame:
    """
    公式 toy_dataset の JSON から DataFrame を構築するデモ。
    本番では CSVや DB クエリ結果から同じスキーマの DataFrame を渡せばよい。

    DataFrameのスキーマ:
        audio_path          str   音声ファイルパス
        audio_duration      float 音声長（秒）
        segments            str   JSON 文字列 [{speaker, text, start, end}, ...]
        customized_context  str   JSON 文字列 ["ホットワード", ...] or None
        split               str   "train" or "val"

    Args:
        toy_dataset_dir: 公式 toy_dataset のディレクトリ

    Returns:
        pd.DataFrame
    """
    records = []
    data_dir = Path(toy_dataset_dir)

    for json_path in sorted(data_dir.glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        audio_path = str(data_dir / data["audio_path"])
        if not Path(audio_path).exists():
            logger.warning(f"音声ファイルが見つかりません: {audio_path}")
            continue

        records.append({
            "audio_path": audio_path,
            "audio_duration": data.get("audio_duration"),
            # segments は JSON 文字列として保存
            "segments": json.dumps(data["segments"], ensure_ascii=False),
            # customized_context も JSON 文字列（なければ None）
            "customized_context": (
                json.dumps(data["customized_context"], ensure_ascii=False)
                if data.get("customized_context") else None
            ),
        })

    df = pd.DataFrame(records)

    # 8:2 で train/val に分割
    val_size = max(1, int(len(df) * 0.2))
    df["split"] = "train"
    df.iloc[-val_size:, df.columns.get_loc("split")] = "val"

    logger.info(f"サンプル DataFrame を構築: {len(df)} 件 "
                f"(train={len(df[df.split=='train'])}, val={len(df[df.split=='val'])})")
    return df


# =============================================================================
# DataFrame → Dataset
# =============================================================================

class VibeVoiceASRDataFrameDataset(Dataset):
    """
    pandas DataFrame を受け取る VibeVoice ASR Dataset。

    __getitem__ で 1 行を VibeVoiceASRProcessor に通し、
    学習に必要なテンソルを返す。

    Parameters
    ----------
    df : pd.DataFrame
        必須列: audio_path, segments
        任意列: audio_duration, customized_context, split
    processor : VibeVoiceASRProcessor
    max_audio_length : float | None
        秒単位。超えるサンプルをスキップ
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor: VibeVoiceASRProcessor,
        max_audio_length: Optional[float] = None,
    ):
        self.processor = processor
        self.max_audio_length = max_audio_length
        self.samples = self._validate_and_filter(df)
        logger.info(f"Dataset サイズ: {len(self.samples)} 件")

    # ------------------------------------------------------------------
    # 内部メソッド
    # ------------------------------------------------------------------

    def _validate_and_filter(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """DataFrame の各行を検証しリスト化する。"""
        samples = []

        for idx, row in df.iterrows():
            audio_path = row["audio_path"]

            # 音声ファイルの存在確認
            if not Path(audio_path).exists():
                logger.warning(f"[行 {idx}] 音声ファイルが存在しません: {audio_path}")
                continue

            # 音声長フィルタ
            duration = row.get("audio_duration") if "audio_duration" in row else None
            if duration is None:
                duration = self._get_audio_duration(audio_path)
            if self.max_audio_length and duration and duration > self.max_audio_length:
                logger.info(f"[行 {idx}] スキップ (duration={duration:.1f}s > {self.max_audio_length}s)")
                continue

            # segments のパース（str か list か両対応）
            segments_raw = row["segments"]
            if isinstance(segments_raw, str):
                try:
                    segments = json.loads(segments_raw)
                except json.JSONDecodeError:
                    logger.warning(f"[行 {idx}] segments を JSON パースできません")
                    continue
            elif isinstance(segments_raw, list):
                segments = segments_raw
            else:
                logger.warning(f"[行 {idx}] segments の型が不正です: {type(segments_raw)}")
                continue

            # customized_context（任意）
            context_raw = row.get("customized_context") if "customized_context" in row else None
            if isinstance(context_raw, str) and context_raw:
                try:
                    context_list = json.loads(context_raw)
                except json.JSONDecodeError:
                    # JSON でなければそのまま1要素リストとして扱う
                    context_list = [context_raw]
            elif isinstance(context_raw, list):
                context_list = context_raw
            else:
                context_list = None

            samples.append({
                "audio_path": audio_path,
                "duration": duration or 0.0,
                "segments": segments,
                "context_list": context_list,
            })

        return samples

    @staticmethod
    def _get_audio_duration(audio_path: str) -> Optional[float]:
        """音声ファイルの長さを取得（soundfile フォールバック）。"""
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return info.duration
        except Exception:
            return None

    @staticmethod
    def _format_segments_as_json(segments: List[Dict], duration: float) -> str:
        """
        segments リストをモデルの出力フォーマット（JSON文字列）に変換。

        出力形式:
            [{"Start":0.0,"End":38.68,"Speaker":0,"Content":"Hello..."},...]
        """
        formatted = []
        for seg in segments:
            formatted.append({
                "Start": round(seg["start"], 2),
                "End": round(seg["end"], 2),
                "Speaker": seg.get("speaker", 0),
                "Content": seg.get("text", ""),
            })
        # スペースなしのコンパクト JSON（公式実装に合わせる）
        return json.dumps(formatted, ensure_ascii=False, separators=(",", ":"))

    # ------------------------------------------------------------------
    # Dataset インターフェース
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        1 サンプルを返す。

        Returns
        -------
        dict with:
            input_ids          List[int]   system + user + generation_prompt のトークンID
            labels             List[int]   入力部は -100、応答部は実際のトークンID
            acoustic_input_mask List[int]  speech_pad 位置が 1
            speech             List[float] 24kHz モノラル波形（numpy → list）
            vae_tok_len        int         VAE トークン数

        入力トークン列の構造:
            <|im_start|>system
            You are a helpful assistant...
            <|im_end|>
            <|im_start|>user
            <speech_start><speech_pad>×N<speech_end>
            This is X.XX seconds audio, please transcribe...
            <|im_end|>
            <|im_start|>assistant\n
            ↑ここまでが input_ids、↓ここからが labels（target_tokens）

            [{"Start":0.0,"End":38.68,"Speaker":0,"Content":"..."},...]
            <|im_end|>
        """
        sample = self.samples[idx]

        # コンテキスト文字列の構築（"Term1\nTerm2\n..."）
        context_info = (
            "\n".join(sample["context_list"])
            if sample["context_list"] else None
        )

        # processor でトークン化 + 音声前処理
        # _process_single_audio: 音声ファイル → input_ids / acoustic_input_mask / speech / vae_tok_len
        encoding = self.processor._process_single_audio(
            sample["audio_path"],
            sampling_rate=None,
            add_generation_prompt=True,
            use_streaming=True,   # 60秒超を自動ストリーミング化
            context_info=context_info,
        )
        input_ids: List[int] = encoding["input_ids"]
        acoustic_input_mask: List[int] = encoding["acoustic_input_mask"]
        speech: np.ndarray = encoding["speech"]  # [T_samples]
        vae_tok_len: int = encoding["vae_tok_len"]

        # 正解テキストの構築（segments → JSON文字列）
        target_text = self._format_segments_as_json(
            sample["segments"], sample["duration"]
        )

        # assistant ターンのトークン化
        # apply_chat_template で <|im_start|>assistant\n ... <|im_end|>\n を付与
        target_tokens: List[int] = self.processor.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": target_text}],
            tokenize=True,
            add_generation_prompt=False,
        )

        # 入力 + 応答を結合
        full_input_ids = input_ids + target_tokens
        full_acoustic_mask = acoustic_input_mask + [0] * len(target_tokens)

        # labels: 入力部は -100（損失計算しない）、応答部は実トークン
        labels = [-100] * len(input_ids) + target_tokens

        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "acoustic_input_mask": full_acoustic_mask,
            "speech": speech.tolist() if isinstance(speech, np.ndarray) else list(speech),
            "vae_tok_len": vae_tok_len,
        }


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class VibeVoiceASRDataCollator:
    """
    可変長バッチをパディングして Tensor に変換。

    パディング方式: 右パディング（学習時推奨）
    　※ 推論時は左パディング（processor が自動対応）

    Batch テンソルの形状:
        input_ids          [B, S_max]        long
        attention_mask     [B, S_max]        long     (1=有効, 0=パッド)
        labels             [B, S_max]        long     (-100=無視)
        acoustic_input_mask [B, S_max]       bool
        speech_tensors     [B, T_audio_max]  float32  (ゼロパッド)
        speech_masks       [B, vae_max]      bool     (有効トークンが True)
    """
    processor: VibeVoiceASRProcessor
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(features)
        max_seq_len = max(len(f["input_ids"]) for f in features)
        max_speech_len = max(len(f["speech"]) for f in features)
        max_vae_len = max(f["vae_tok_len"] for f in features)

        # 初期化（ゼロ or pad_token_id で埋める）
        input_ids = torch.full((B, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_seq_len, dtype=torch.long)
        labels = torch.full((B, max_seq_len), self.label_pad_token_id, dtype=torch.long)
        acoustic_input_mask = torch.zeros(B, max_seq_len, dtype=torch.bool)
        speech_tensors = torch.zeros(B, max_speech_len, dtype=torch.float32)
        speech_masks = torch.zeros(B, max_vae_len, dtype=torch.bool)

        for i, feat in enumerate(features):
            s = len(feat["input_ids"])
            a = len(feat["speech"])
            v = feat["vae_tok_len"]

            # 右パディング（有効部分を先頭から埋める）
            input_ids[i, :s] = torch.tensor(feat["input_ids"], dtype=torch.long)
            attention_mask[i, :s] = 1
            labels[i, :s] = torch.tensor(feat["labels"], dtype=torch.long)
            acoustic_input_mask[i, :s] = torch.tensor(feat["acoustic_input_mask"], dtype=torch.bool)
            speech_tensors[i, :a] = torch.tensor(feat["speech"], dtype=torch.float32)
            speech_masks[i, :v] = True  # 有効な音声 VAE トークン位置

        return {
            "input_ids": input_ids,             # [B, S_max]
            "attention_mask": attention_mask,    # [B, S_max]
            "labels": labels,                    # [B, S_max]
            "acoustic_input_mask": acoustic_input_mask,  # [B, S_max]
            "speech_tensors": speech_tensors,    # [B, T_audio_max]
            "speech_masks": speech_masks,        # [B, vae_max]
        }


# =============================================================================
# モデルセットアップ
# =============================================================================

def setup_model(
    model_path: str,
    lm_pretrained: str,
    lora_args: LoraArguments,
    attn_impl: str = "flash_attention_2",
    gradient_checkpointing: bool = True,
) -> Tuple[nn.Module, VibeVoiceASRProcessor]:
    """
    モデルとプロセッサをロードし LoRA を適用する。

    凍結対象:
        acoustic_tokenizer (Enc + Dec) … 約340M パラメータ
        semantic_tokenizer (Enc)       … 約340M パラメータ

    LoRA 対象 (Qwen2.5 の Attention + MLP):
        q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

    Returns
    -------
    model : PEFT でラップされたモデル
    processor : VibeVoiceASRProcessor
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # --- プロセッサのロード ---
    logger.info(f"プロセッサをロード中: {model_path}")
    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path,
        language_model_pretrained_name=lm_pretrained,
    )

    # --- モデルのロード ---
    logger.info(f"モデルをロード中: {model_path}")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model = model.to(device)

    # --- 音声トークナイザを凍結 ---
    frozen_count = 0
    for name, param in model.named_parameters():
        if "acoustic_tokenizer" in name or "semantic_tokenizer" in name:
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"凍結パラメータ数: {frozen_count} 個 (acoustic + semantic tokenizer)")

    # --- LoRA 設定 ---
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
        ],
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 例: trainable params: 13,631,488 || all params: 8,043,647,936 || trainable%: 0.17

    # --- Gradient Checkpointing ---
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing を有効化")

    return model, processor


# =============================================================================
# メイン学習関数
# =============================================================================

def run_finetune(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
):
    """
    DataFrame ベースの LoRA ファインチューニングを実行。

    1. DataFrame の読み込み / 構築
    2. train / val に分割
    3. モデル + プロセッサのセットアップ
    4. Dataset / DataCollator の作成
    5. Trainer で学習
    6. モデルと設定の保存
    """
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    # ================================================================
    # Step 1: DataFrame の準備
    # ================================================================
    if data_args.csv_path:
        logger.info(f"CSV から DataFrame を読み込み: {data_args.csv_path}")
        df = pd.read_csv(data_args.csv_path)

        # audio_base_dir が指定されていれば audio_path を結合
        if data_args.audio_base_dir:
            df["audio_path"] = df["audio_path"].apply(
                lambda p: str(Path(data_args.audio_base_dir) / p)
            )
    else:
        # デモ用: 公式 toy_dataset から構築
        toy_dir = Path(__file__).parent.parent.parent / (
            "codes/paper_official/VibeVoice/VibeVoice/finetuning-asr/toy_dataset"
        )
        logger.info(f"デモ用 DataFrame を構築: {toy_dir}")
        df = build_sample_dataframe(str(toy_dir))

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"列名: {list(df.columns)}")

    # ================================================================
    # Step 2: train / val 分割
    # ================================================================
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
    else:
        # split 列がなければランダム分割
        val_size = max(1, int(len(df) * data_args.val_split))
        shuffled = df.sample(frac=1, random_state=training_args.seed).reset_index(drop=True)
        val_df = shuffled.iloc[:val_size].reset_index(drop=True)
        train_df = shuffled.iloc[val_size:].reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} 件 / Val: {len(val_df)} 件")

    # ================================================================
    # Step 3: モデル + プロセッサ
    # ================================================================
    model, processor = setup_model(
        model_path=model_args.model_path,
        lm_pretrained=model_args.lm_pretrained,
        lora_args=lora_args,
        attn_impl=model_args.attn_impl,
        gradient_checkpointing=training_args.gradient_checkpointing,
    )

    # ================================================================
    # Step 4: Dataset / DataCollator
    # ================================================================
    train_dataset = VibeVoiceASRDataFrameDataset(
        df=train_df,
        processor=processor,
        max_audio_length=data_args.max_audio_length,
    )
    eval_dataset = VibeVoiceASRDataFrameDataset(
        df=val_df,
        processor=processor,
        max_audio_length=data_args.max_audio_length,
    ) if len(val_df) > 0 else None

    data_collator = VibeVoiceASRDataCollator(
        processor=processor,
        pad_token_id=processor.pad_id,
    )

    # HuggingFace Trainer の注意事項
    training_args.remove_unused_columns = False   # speech_tensors 等を保持
    training_args.dataloader_num_workers = 0       # 音声ロードは multiprocessing と相性が悪い

    # ================================================================
    # Step 5: 学習
    # ================================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("=" * 60)
    logger.info("学習開始")
    logger.info(f"  Train サンプル数: {len(train_dataset)}")
    logger.info(f"  Val サンプル数:   {len(eval_dataset) if eval_dataset else 0}")
    logger.info(f"  Epochs:           {training_args.num_train_epochs}")
    logger.info(f"  Batch size/GPU:   {training_args.per_device_train_batch_size}")
    logger.info(f"  Grad accum steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate:    {training_args.learning_rate}")
    logger.info(f"  Output dir:       {training_args.output_dir}")
    logger.info("=" * 60)

    train_result = trainer.train()

    # ================================================================
    # Step 6: 保存
    # ================================================================
    logger.info(f"モデルを保存中: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if eval_dataset:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("学習完了！")
    return model, processor


# =============================================================================
# 推論関数（LoRA モデルを使用）
# =============================================================================

def run_inference(
    base_model_path: str,
    lora_path: str,
    audio_path: str,
    context_info: Optional[str] = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    device: Optional[str] = None,
    lm_pretrained: str = "Qwen/Qwen2.5-7B",
    merge_lora: bool = False,
) -> Dict[str, Any]:
    """
    LoRA ファインチューニング済みモデルで推論を実行。

    Parameters
    ----------
    base_model_path : str
        ベースモデルパス（HuggingFace ID またはローカル）
    lora_path : str
        LoRA アダプタの保存先ディレクトリ
    audio_path : str
        推論する音声ファイルのパス
    context_info : str | None
        ホットワード等のコンテキスト（例: "Tea Brew\nAiden Host"）
    max_new_tokens : int
        最大生成トークン数
    temperature : float
        サンプリング温度（0 = greedy）
    device : str | None
        None なら自動（CUDA があれば CUDA）
    merge_lora : bool
        True なら LoRA 重みをベースモデルにマージ（推論速度向上、但し保存サイズ増加）

    Returns
    -------
    dict:
        raw_text  str
            モデルの生テキスト出力
        segments  List[Dict]
            パース済みセグメントリスト
            [{start_time, end_time, speaker_id, text}, ...]
    """
    from peft import PeftModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # --- ロード ---
    logger.info(f"ベースモデルをロード: {base_model_path}")
    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model_path,
        language_model_pretrained_name=lm_pretrained,
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)

    logger.info(f"LoRA アダプタをロード: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    if merge_lora:
        logger.info("LoRA 重みをマージ（推論高速化）")
        model = model.merge_and_unload()

    model.eval()

    # --- 前処理 ---
    inputs = processor(
        audio=audio_path,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    # --- 生成 ---
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # --- デコード ---
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    # --- JSON パース ---
    try:
        segments = processor.post_process_transcription(raw_text)
    except Exception as e:
        logger.warning(f"構造化出力のパース失敗: {e}")
        segments = []

    return {"raw_text": raw_text, "segments": segments}


# =============================================================================
# エントリポイント
# =============================================================================

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, TrainingArguments)
    )

    # コマンドライン引数 or デフォルト値で実行
    if len(sys.argv) > 1:
        model_args, data_args, lora_args, training_args = (
            parser.parse_args_into_dataclasses()
        )
    else:
        # 引数なし実行時のデフォルト（デモ）
        logger.info("引数なし: デフォルト設定でデモ実行")
        model_args = ModelArguments()
        data_args = DataArguments()
        lora_args = LoraArguments()
        training_args = TrainingArguments(
            output_dir="./vibevoice_lora_output",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="none",
            seed=42,
        )

    run_finetune(model_args, data_args, lora_args, training_args)


if __name__ == "__main__":
    main()


# =============================================================================
# 使い方メモ: DataFrame を直接渡す場合
# =============================================================================

def example_usage_with_dataframe():
    """
    スクリプトをインポートして DataFrame を直接渡す使用例。

    想定する DataFrame のスキーマ:
    ┌─────────────────────┬─────────────────┬────────────────────┬────────────────────────┬───────┐
    │ audio_path          │ audio_duration  │ segments           │ customized_context     │ split │
    │ str                 │ float           │ str(JSON) or list  │ str(JSON) or list|None │ str   │
    ├─────────────────────┼─────────────────┼────────────────────┼────────────────────────┼───────┤
    │ /data/podcast_0.mp3 │ 351.73          │ '[{"speaker":0,    │ '["Tea Brew","Aiden"]' │ train │
    │                     │                 │  "text":"Hello",   │                        │       │
    │                     │                 │  "start":0.0,      │                        │       │
    │                     │                 │  "end":38.68}]'    │                        │       │
    │ /data/podcast_1.mp3 │ 210.50          │ [{"speaker":0,...}]│ None                   │ val   │
    └─────────────────────┴─────────────────┴────────────────────┴────────────────────────┴───────┘
    """
    import pandas as pd
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    # --- DataFrame の構築例 ---
    df = pd.DataFrame([
        {
            "audio_path": "/data/episode_01.mp3",
            "audio_duration": 351.73,
            "segments": json.dumps([
                {"speaker": 0, "text": "Welcome to the show.", "start": 0.0, "end": 5.2},
                {"speaker": 1, "text": "Thanks for having me.", "start": 5.5, "end": 10.1},
            ]),
            "customized_context": json.dumps(["ShowName", "GuestName"]),
            "split": "train",
        },
        {
            "audio_path": "/data/episode_02.mp3",
            "audio_duration": 120.0,
            "segments": json.dumps([
                {"speaker": 0, "text": "Let's talk about AI.", "start": 0.0, "end": 8.0},
            ]),
            "customized_context": None,
            "split": "val",
        },
    ])

    # --- プロセッサとデータセット ---
    processor = VibeVoiceASRProcessor.from_pretrained("microsoft/VibeVoice-ASR")

    train_dataset = VibeVoiceASRDataFrameDataset(
        df=df[df["split"] == "train"].reset_index(drop=True),
        processor=processor,
        max_audio_length=400.0,
    )

    # --- 1サンプルの確認 ---
    sample = train_dataset[0]
    print(f"input_ids length: {len(sample['input_ids'])}")
    print(f"labels: {[t for t in sample['labels'] if t != -100][:10]}...")  # 有効部分だけ表示
    print(f"vae_tok_len: {sample['vae_tok_len']}")
    print(f"speech length: {len(sample['speech'])} samples (~{len(sample['speech'])/24000:.1f}s)")
