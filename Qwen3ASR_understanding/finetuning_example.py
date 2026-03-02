"""
Qwen3-ASR ファインチューニング (Trainer不使用)
================================================

このファイルは音声-テキストペアのデータセットを使って
Qwen3-ASRをファインチューニングするサンプルコードです。
transformers の Trainer は使用せず、手動の訓練ループで実装しています。

公式実装参考: Qwen3-ASR/finetuning/qwen3_asr_sft.py

============================================================
前提条件
============================================================
- 音声ファイルとtranscript (書き起こし) のペアデータセットが
  pandas DataFrame (columns: audio, text, prompt(optional)) で渡される

============================================================
使い方
============================================================
# スクリプト単体実行
python finetuning_example.py \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --train_file /path/to/train.jsonl \
    --eval_file /path/to/eval.jsonl \
    --output_dir ./qwen3-asr-finetuned \
    --epochs 3 \
    --batch_size 4 \
    --grad_acc 8 \
    --lr 2e-5

# Python内からDataFrameで呼ぶ場合
import pandas as pd
df = pd.DataFrame({
    "audio": ["/path/to/a1.wav", "/path/to/a2.wav"],
    "text":  ["hello world", "goodbye"],
})
model, processor = load_model_for_finetuning("Qwen/Qwen3-ASR-1.7B")
train(model, processor, train_df=df, output_dir="./out")
"""

import argparse
import os
import shutil
import math
from typing import Any, Dict, List, Optional

import pandas as pd
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
import jiwer
from tqdm.auto import tqdm


# ============================================================
# 1. データセット定義
# ============================================================

class ASRFineTuneDataset(Dataset):
    """
    音声-テキストペアのデータセット (pandas DataFrame ベース)

    ========================================
    DataFrame columns
    ========================================
    - audio (str, 必須): 音声ファイルパス
    - text  (str, 必須): 書き起こしテキスト
    - prompt (str, オプション): System prompt (コンテキストバイアス用)

    ========================================
    前処理
    ========================================
    - 音声: librosa.load(path, sr=16000, mono=True) → float32 ndarray
    - テキスト: Chat Template + EOS付きの完全テキスト
    - ラベル: prefix (system + user + assistant開始) 部分を-100でマスク
    """

    def __init__(self, df: pd.DataFrame, processor, sampling_rate: int = 16000):
        """
        入力:
            df: pandas DataFrame (columns: audio, text, prompt(optional))
            processor: Qwen3ASRProcessor
            sampling_rate: サンプリングレート (デフォルト 16000)
        """
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.df = df.reset_index(drop=True)

        # prefix_text を事前計算
        # (各サンプルのsystem + user + "assistant\n" までのテキスト)
        self._precompute_prefix_texts()

    def _precompute_prefix_texts(self):
        """
        各サンプルのprefix_textを事前計算

        prefix_text = Chat Template の system + user + assistant開始 部分
        → この部分は損失計算から除外 (labels = -100)
        """
        prefix_texts = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            prompt = row.get("prompt", "") if "prompt" in self.df.columns else ""
            if pd.isna(prompt):
                prompt = ""

            # ダミー音声でChat Templateを生成 (prefix部分のみ必要)
            prefix_messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "audio", "audio": None}]},
            ]

            prefix_text = self.processor.apply_chat_template(
                [prefix_messages],
                add_generation_prompt=True,  # "assistant\n" を追加
                tokenize=False,
            )[0]

            prefix_texts.append(prefix_text)

        self.df = self.df.copy()
        self.df["prefix_text"] = prefix_texts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        1サンプル取得

        出力:
            {
                "audio": "/path/to/audio.wav",
                "prefix_text": "<prefix chat template>",
                "target": "transcript text",
            }
        """
        row = self.df.iloc[idx]
        return {
            "audio": row["audio"],
            "prefix_text": row["prefix_text"],
            "target": row["text"],
        }


# ============================================================
# 2. Data Collator (バッチ化)
# ============================================================

class ASRCollator:
    """
    バッチ化 + ラベル生成

    ========================================
    処理フロー
    ========================================
    1. 音声ファイルをロード → 16kHz mono float32
    2. full_text = prefix_text + target + EOS
    3. Processor: full_text + audio → input_ids, input_features, masks
    4. labels 生成: prefix部分を -100 でマスク

    ========================================
    Shape (バッチサイズ B の場合)
    ========================================
    出力:
        input_ids:              (B, T_max) int64      - パディング済みトークンID
        attention_mask:         (B, T_max) int64      - パディングマスク
        input_features:         (B, T_mel_max, 128)   - メルスペクトログラム
        feature_attention_mask: (B, T_mel_max) int64  - mel特徴マスク
        labels:                 (B, T_max) int64      - -100 or トークンID
    """

    def __init__(self, processor, sampling_rate: int = 16000):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        バッチ化

        ========================================
        Labels生成の仕組み
        ========================================
        full_text = prefix_text + target + EOS

        例:
            prefix_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<audio><|im_end|>\n<|im_start|>assistant\n"
            target      = "language English<asr_text>Hello world"
            EOS         = "<|im_end|>"

            full_text = prefix_text + target + EOS

        Tokenize:
            full_ids  = [tok_1, tok_2, ..., tok_P, tok_P+1, ..., tok_N]
                         ↑ prefix 部分 (P tokens) ↑  target 部分

        Labels:
            labels    = [-100, -100, ..., -100, tok_P+1, ..., tok_N]
                         ↑ prefix は無視            ↑ target のみ損失計算
        """
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        # EOS トークン
        eos = self.processor.tokenizer.eos_token or ""

        # Full text = prefix + target + EOS
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        # 音声ロード
        audios = []
        for path in audio_paths:
            wav, _ = librosa.load(path, sr=self.sampling_rate, mono=True)
            audios.append(wav)
        # audios: List[ndarray], 各 (num_samples_i,) float32

        # ========================================
        # Full text + Audio → Processor
        # ========================================
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        # full_inputs:
        # {
        #   "input_ids":             (B, T_max) int64
        #   "attention_mask":        (B, T_max) int64
        #   "input_features":        (B, T_mel_max, 128) float
        #   "feature_attention_mask": (B, T_mel_max) int64
        # }

        # ========================================
        # Prefix text のみ → Processor (prefix長を取得)
        # ========================================
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # 各サンプルのprefix長を取得
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        # prefix_lens: List[int], 各サンプルのprefix トークン数

        # ========================================
        # Labels生成
        # ========================================
        labels = full_inputs["input_ids"].clone()  # (B, T_max)

        for i, pl in enumerate(prefix_lens):
            # prefix部分を-100でマスク (損失計算から除外)
            labels[i, :pl] = -100

        # パディングトークンも-100でマスク
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        # labels: (B, T_max) int64
        #   prefix部分: -100 (損失除外)
        #   target部分: トークンID (損失計算対象)
        #   padding部分: -100 (損失除外)

        return full_inputs


# ============================================================
# 3. モデルの準備
# ============================================================

def load_model_for_finetuning(model_path: str, compute_dtype=None):
    """
    ファインチューニング用のモデルロード

    ========================================
    重要なポイント
    ========================================
    1. Qwen3ASRForConditionalGeneration の forward() は
       thinker.forward() に委譲するようパッチが必要
    2. Audio Encoderは凍結 (freeze) 推奨
    3. モデルは float32 でロード (Mixed Precisionの正しいパターン)
       → autocast が forward pass のみ compute_dtype で計算
       → 勾配は fp32 → GradScaler 正常動作

    ========================================
    出力
    ========================================
    model: Qwen3ASRForConditionalGeneration (パッチ済み, float32)
    processor: Qwen3ASRProcessor
    compute_dtype: autocast に渡す計算dtype (bf16 or fp16)
    """
    from qwen_asr import Qwen3ASRModel

    # autocast 用の計算dtype (forward pass で使用)
    if compute_dtype is None:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            compute_dtype = torch.bfloat16  # Ampere以降
        else:
            compute_dtype = torch.float16

    # float32でロード (Mixed Precisionのマスター重みパターン)
    # autocast が forward pass のみ compute_dtype (fp16/bf16) で計算する
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    # ========================================
    # forward() パッチ
    # ========================================
    # Qwen3ASRForConditionalGeneration.forward() が
    # thinker.forward() を呼ぶようにパッチ
    # (デフォルトでは forward() が generate() 用のロジックのため)
    _patch_forward(model)

    # ========================================
    # Audio Encoder を凍結 (オプション)
    # ========================================
    # AuT Encoderは40M時間のデータで事前学習済みなので、
    # ファインチューニングでは凍結するのが一般的
    for param in model.thinker.audio_tower.parameters():
        param.requires_grad = False

    return model, processor, compute_dtype


def _patch_forward(model):
    """
    model.forward() が thinker.forward() を呼ぶようにパッチ

    理由: デフォルトの forward() は generate() 用の前処理を含み、
    学習時に必要な labels 引数をサポートしないため。
    """
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ============================================================
# 4. 学習率スケジューラ
# ============================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """
    Cosine学習率スケジューラ with Warmup

    ========================================
    スケジュール
    ========================================
    step < warmup: lr = base_lr * step / warmup
    step >= warmup: lr = base_lr * 0.5 * (1 + cos(π * (step - warmup) / (total - warmup)))
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 5. AMP設定ヘルパー
# ============================================================

def setup_amp(compute_dtype: torch.dtype, device: torch.device):
    """
    Mixed Precision (AMP) の設定

    ========================================
    前提
    ========================================
    モデルは float32 でロード済み (マスター重み)
    autocast が forward pass のみ compute_dtype で計算
    → 勾配は fp32 で返る → GradScaler 正常動作

    ========================================
    戦略
    ========================================
    - bfloat16: torch.autocast(enabled=True, dtype=bf16), GradScaler無効
    - float16:  torch.autocast(enabled=True, dtype=fp16), GradScaler有効
    - float32:  torch.autocast(enabled=False), GradScaler無効

    torch.autocast の enabled パラメータで統一的に制御し、
    呼び出し側で if 分岐を不要にする。

    ========================================
    出力
    ========================================
    amp_ctx_kwargs: dict - torch.autocast に渡す kwargs
    scaler: GradScaler - fp16時のみ有効
    """
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = compute_dtype in (torch.bfloat16, torch.float16)

    amp_ctx_kwargs = dict(
        device_type=device_type,
        dtype=compute_dtype if use_amp else torch.float32,
        enabled=use_amp,
    )

    # GradScaler: fp16のみ有効 (bf16では不要、fp32では不要)
    scaler = GradScaler(device_type, enabled=(compute_dtype == torch.float16))

    return amp_ctx_kwargs, scaler


# ============================================================
# 6. メイン訓練ループ
# ============================================================

def train(
    model: torch.nn.Module,
    processor,
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    compute_dtype: Optional[torch.dtype] = None,
    output_dir: str = "./qwen3-asr-finetuned",
    epochs: int = 3,
    batch_size: int = 4,
    grad_acc: int = 8,
    lr: float = 2e-5,
    warmup_ratio: float = 0.02,
    save_steps: int = 200,
    log_steps: int = 10,
    max_grad_norm: float = 1.0,
    num_workers: int = 4,
):
    """
    手動訓練ループ (Trainerなし)

    ========================================
    Shape (訓練中)
    ========================================
    各バッチ:
        input_ids:              (B, T_max) int64
        attention_mask:         (B, T_max) int64
        input_features:         (B, T_mel_max, 128) float
        feature_attention_mask: (B, T_mel_max) int64
        labels:                 (B, T_max) int64

    Forward出力:
        logits: (B, T_max, 151936)
        loss:   scalar

    ========================================
    訓練ハイパーパラメータ (公式推奨)
    ========================================
    - 学習率: 2e-5
    - バッチサイズ: 32 (per-GPU) × 4 (grad_acc) = 128 effective
    - Warmup: 2%
    - スケジューラ: Linear or Cosine
    - Precision: bfloat16 (Ampere+) or float16
    - Audio Encoder: 凍結推奨
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================================
    # 0. Mixed Precision 用 dtype 管理
    # ========================================
    # モデルは float32 でロード済みが前提 (マスター重みパターン)
    # compute_dtype: autocast の forward pass で使用する計算dtype
    if compute_dtype is None:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

    # ========================================
    # 1. データセット + DataLoader
    # ========================================
    train_dataset = ASRFineTuneDataset(train_df, processor)
    train_collator = ASRCollator(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collator,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    eval_loader = None
    if eval_df is not None:
        eval_dataset = ASRFineTuneDataset(eval_df, processor)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_collator,
            pin_memory=True,
        )

    # ========================================
    # 2. Optimizer + Scheduler
    # ========================================
    # 学習可能パラメータのみ (Audio Encoderは凍結済み)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # 総ステップ数の計算
    steps_per_epoch = math.ceil(len(train_loader) / grad_acc)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ========================================
    # 3. Mixed Precision (AMP)
    # ========================================
    # torch.autocast の enabled で統一制御 (if分岐不要)
    amp_ctx_kwargs, scaler = setup_amp(compute_dtype, device)

    # ========================================
    # 4. モデルをGPUに移動
    # ========================================
    model = model.to(device)
    model.train()

    # ========================================
    # 5. 訓練ループ
    # ========================================
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    accumulated_loss = 0.0

    precision_name = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(compute_dtype, "float32")
    print(f"Training config:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_acc}")
    print(f"  Effective batch size: {batch_size * grad_acc}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {lr}")
    print(f"  Precision: {precision_name}")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"  Frozen params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print()

    # ========================================
    # 開始前の評価
    # ========================================
    if eval_loader is not None:
        eval_result = evaluate(
            model, eval_loader, device, amp_ctx_kwargs,
            processor=processor, eval_df=eval_df,
        )
        msg = f"  Epoch 0 (initial zero shot) eval loss: {eval_result['loss']:.4f}"
        if eval_result["wer"] is not None:
            msg += f" | WER: {eval_result['wer']:.4f}"
        print(msg)

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1}/{epochs} ===")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            # ========================================
            # 5a. バッチをGPUに移動
            # ========================================
            # dtype変換は不要: モデルはfp32、autocastがforward内で自動キャスト
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # ========================================
            # 5b. Forward Pass (autocast で統一)
            # ========================================
            # enabled=True/False で自動切替、if分岐不要
            with torch.autocast(**amp_ctx_kwargs):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            # Gradient accumulation: 損失をgrad_accで割る
            loss = loss / grad_acc

            # ========================================
            # 5c. Backward Pass (scaler.scale は enabled=False なら no-op)
            # ========================================
            scaler.scale(loss).backward()

            accumulated_loss += loss.item()

            # ========================================
            # 5d. Gradient Step (grad_accステップごと)
            # ========================================
            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1) == len(train_loader):
                # scaler.unscale_: enabled=False なら no-op
                scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                # scaler.step / scaler.update: enabled=False なら通常のoptimizer.step()相当
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # ========================================
                # 5e. ロギング
                # ========================================
                if global_step % log_steps == 0:
                    avg_loss = accumulated_loss / log_steps
                    current_lr = scheduler.get_last_lr()[0]
                    print(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )
                    accumulated_loss = 0.0

                # ========================================
                # 5f. チェックポイント保存
                # ========================================
                if global_step % save_steps == 0:
                    save_checkpoint(
                        model, processor, output_dir,
                        global_step, model_path=None,
                    )

        # ========================================
        # 5g. Epoch終了時の評価
        # ========================================
        if eval_loader is not None:
            eval_result = evaluate(
                model, eval_loader, device, amp_ctx_kwargs,
                processor=processor, eval_df=eval_df,
            )
            msg = f"  Epoch {epoch + 1} eval loss: {eval_result['loss']:.4f}"
            if eval_result["wer"] is not None:
                msg += f" | WER: {eval_result['wer']:.4f}"
            print(msg)

    # ========================================
    # 6. 最終モデル保存
    # ========================================
    save_checkpoint(model, processor, output_dir, global_step, model_path=None, is_final=True)
    print(f"\nTraining complete! Model saved to {output_dir}")


# ============================================================
# 7. 評価関数
# ============================================================

def evaluate(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    amp_ctx_kwargs: dict,
    processor=None,
    eval_df: Optional[pd.DataFrame] = None,
    max_new_tokens: int = 256,
) -> dict:
    """
    評価 (損失 + WER)

    ========================================
    出力
    ========================================
    {
        "loss": float,           - 平均損失
        "wer":  float or None,   - Word Error Rate (processor + eval_df がある場合)
    }

    ========================================
    WER計算
    ========================================
    eval_df の各サンプルに対して model.generate() でテキスト生成し、
    jiwer.wer(reference, hypothesis) で WER を算出する。

    generate用の入力は prefix (system + user + audio + assistant開始) のみ。
    eval_loader のバッチ (prefix + target) とは別に構築する。
    """
    model.eval()

    # ========================================
    # 1. Loss 計算 (eval_loader ベース)
    # ========================================
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval loss"):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            with torch.autocast(**amp_ctx_kwargs):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # ========================================
    # 2. WER 計算 (eval_df + generate ベース)
    # ========================================
    wer_score = None

    if processor is not None and eval_df is not None:
        refs, hyps = [], []
        tokenizer = processor.tokenizer
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        with torch.no_grad():
            for idx in tqdm(range(len(eval_df)), desc="Eval WER"):
                row = eval_df.iloc[idx]
                ref = row["text"]
                prompt = row.get("prompt", "") if "prompt" in eval_df.columns else ""
                if pd.isna(prompt):
                    prompt = ""

                # 音声ロード
                wav, _ = librosa.load(row["audio"], sr=16000, mono=True)

                # Generation入力: prefix のみ (target なし)
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [{"type": "audio", "audio": None}]},
                ]
                gen_text = processor.apply_chat_template(
                    [messages], add_generation_prompt=True, tokenize=False
                )[0]
                gen_inputs = processor(
                    text=gen_text, audio=[wav], return_tensors="pt"
                )
                gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}

                input_len = gen_inputs["input_ids"].shape[1]

                with torch.autocast(**amp_ctx_kwargs):
                    gen_output = model.generate(
                        **gen_inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=pad_token_id,
                    )

                # GenerateOutput → tensor 取り出し
                generated_ids = gen_output.sequences if hasattr(gen_output, "sequences") else gen_output

                # input prefix を除いた生成部分のみデコード
                hyp = tokenizer.decode(
                    generated_ids[0, input_len:], skip_special_tokens=True,
                )

                refs.append(ref)
                hyps.append(hyp)

        wer_score = jiwer.wer(refs, hyps)

    model.train()
    return {"loss": avg_loss, "wer": wer_score}


# ============================================================
# 8. チェックポイント保存
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    processor,
    output_dir: str,
    global_step: int,
    model_path: Optional[str] = None,
    is_final: bool = False,
):
    """
    チェックポイント保存

    ========================================
    保存内容
    ========================================
    - model weights (safetensors)
    - config.json
    - tokenizer files
    - processor files
    """
    if is_final:
        save_dir = os.path.join(output_dir, "final")
    else:
        save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")

    os.makedirs(save_dir, exist_ok=True)

    # generation_config の矛盾を修正 (do_sample=False なのに temperature が設定されている場合)
    if hasattr(model, "generation_config"):
        gc = model.generation_config
        if not getattr(gc, "do_sample", True):
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(gc, attr) and getattr(gc, attr) is not None:
                    setattr(gc, attr, None)

    # モデル重み保存
    model.save_pretrained(save_dir, safe_serialization=True)

    # Processor保存 (tokenizer + feature_extractor)
    processor.save_pretrained(save_dir)

    # 推論に必要な追加ファイルをコピー (元モデルから)
    if model_path:
        required_files = [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "chat_template.json",
            "merges.txt",
            "vocab.json",
        ]
        for fn in required_files:
            src = os.path.join(model_path, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(save_dir, fn))

    print(f"  Checkpoint saved: {save_dir}")


# ============================================================
# 9. エントリポイント
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Fine-tuning (No Trainer)")

    # パス
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B",
                    help="HuggingFace model path or local path")
    p.add_argument("--train_file", type=str, required=True,
                    help="JSONL file: {audio, text, prompt(optional)} → pd.read_json(lines=True)")
    p.add_argument("--eval_file", type=str, default="",
                    help="JSONL file for evaluation (optional)")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuned",
                    help="Output directory for checkpoints")

    # 訓練ハイパーパラメータ
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4,
                    help="Per-GPU batch size")
    p.add_argument("--grad_acc", type=int, default=8,
                    help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate")
    p.add_argument("--warmup_ratio", type=float, default=0.02,
                    help="Warmup ratio (fraction of total steps)")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="Max gradient norm for clipping")

    # 保存・ログ
    p.add_argument("--save_steps", type=int, default=200,
                    help="Save checkpoint every N steps")
    p.add_argument("--log_steps", type=int, default=10,
                    help="Log every N steps")

    # データローダー
    p.add_argument("--num_workers", type=int, default=4)

    # Audio Encoder凍結制御
    p.add_argument("--freeze_audio_encoder", type=int, default=1,
                    help="1=freeze audio encoder, 0=train all")

    return p.parse_args()


def main():
    args = parse_args()

    # ========================================
    # モデルロード
    # ========================================
    print(f"Loading model: {args.model_path}")
    model, processor, compute_dtype = load_model_for_finetuning(args.model_path)

    # Audio Encoder凍結の制御
    if not args.freeze_audio_encoder:
        print("Warning: Unfreezing audio encoder (not recommended)")
        for param in model.thinker.audio_tower.parameters():
            param.requires_grad = True

    # ========================================
    # データ読み込み (JSONL → pandas DataFrame)
    # ========================================
    train_df = pd.read_json(args.train_file, lines=True)
    eval_df = pd.read_json(args.eval_file, lines=True) if args.eval_file else None

    # ========================================
    # 訓練開始
    # ========================================
    train(
        model=model,
        processor=processor,
        train_df=train_df,
        eval_df=eval_df,
        compute_dtype=compute_dtype,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_acc=args.grad_acc,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
        max_grad_norm=args.max_grad_norm,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()


# ============================================================
# 入出力shape一覧表
# ============================================================
"""
========================================
ファインチューニング時のShape遷移
========================================

データ: 音声10秒, テキスト "Hello, this is Qwen"

| 段階                     | テンソル名              | Shape                    | 説明                         |
|--------------------------|------------------------|--------------------------|------------------------------|
| 生音声                   | raw_audio              | (160000,)                | 16kHz × 10s                  |
| メルスペクトログラム      | input_features         | (B, 1000, 128)           | Processor出力                |
| mel attention mask       | feature_attention_mask | (B, 1000)                | 有効フレームのマスク           |
| Token IDs (full)         | input_ids              | (B, T_max)               | prefix + target + EOS        |
| Attention Mask           | attention_mask         | (B, T_max)               | パディングマスク              |
| Labels                   | labels                 | (B, T_max)               | -100 (prefix) / ID (target)  |
| Audio Encoder出力        | audio_features         | (B, 125, 3584)           | 12.5Hz音声表現               |
| 統合Embeddings           | inputs_embeds          | (B, T_combined, 4096)    | Audio + Text統合             |
| LM出力                  | logits                 | (B, T_combined, 151936)  | 語彙分布                     |
| Loss                     | loss                   | scalar                   | CE loss (target部分のみ)     |

========================================
Labels の具体例
========================================
Token列:     [SYS, ..., <aud>, <aud>, ..., USER_END, ASST, "lang", "Eng", "<asr", ...target..., EOS, PAD, PAD]
Labels:      [-100, ..., -100, -100, ..., -100,      -100, -100,  -100,  -100, ...target..., EOS, -100, -100]
                                                                                    ↑ ここからのみ損失計算

prefix部分はすべて -100 → 損失計算から除外
target部分のみトークンIDをラベルとして使用
パディング部分も -100

========================================
推奨ハイパーパラメータ
========================================
| パラメータ          | 推奨値           | 備考                           |
|--------------------|-----------------|--------------------------------|
| learning_rate      | 2e-5            | 公式推奨値                      |
| batch_size         | 4-32            | GPU VRAM に依存                |
| grad_acc           | 4-8             | effective BS=128 程度が目安     |
| epochs             | 1-5             | データ量に依存                  |
| warmup_ratio       | 0.02            | 全ステップの2%                  |
| scheduler          | cosine/linear   | 公式はlinear                   |
| Audio Encoder      | frozen          | 事前学習済みEncoderは凍結推奨   |
| precision          | bfloat16        | Ampere以降のGPU推奨            |
| max_grad_norm      | 1.0             | 勾配クリッピング                |

========================================
DataFrame入力の使用例
========================================
import pandas as pd

# CSVから読み込み
df = pd.read_csv("dataset.csv")  # columns: audio, text, prompt(optional)

# 直接構築
df = pd.DataFrame({
    "audio": ["audio/001.wav", "audio/002.wav", "audio/003.wav"],
    "text":  ["hello world", "good morning", "how are you"],
})

# HuggingFace datasets から変換
from datasets import load_dataset
ds = load_dataset("my_asr_dataset", split="train")
df = ds.to_pandas()

model, processor = load_model_for_finetuning("Qwen/Qwen3-ASR-1.7B")
train(model, processor, train_df=df, output_dir="./out")
"""
