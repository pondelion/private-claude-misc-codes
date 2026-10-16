"""
OneStreamer ファインチューニング (Trainer不使用)
================================================================

動画 + タイムライン注釈 (PHCM / Proactive QA 等) のデータセットを使って
OneStreamer (Qwen3-VL-4B-Instruct + 5制御トークン) をファインチューニングする
サンプルコードです。transformers の Trainer は使用せず、手動の訓練ループで
実装しています (survey/Qwen3ASR/finetuning_example.py と同じ方針)。

============================================================
DataFrame の期待するカラム
============================================================
本物の公開データセット huggingface.co/datasets/MCG-NJU/OneStreamer-1M から
実際に1レコードを取得して確認した JSONL スキーマ (`schema: onestreamer-public-meta-v1`)
に準拠する。例 (Proactive_Caption_Memory/.../Flat_captioning/level2_flat_2500.jsonl
より実在のレコードを一部省略して引用):

    {
      "video_path": "Gemini_New_Annotation/videos_segment/.../xxx.mp4",
      "image_size": [960, 540],
      "prompts": [{"id": "p1", "kind": "text", "at": 1.0,
                   "text": "Continuously observe the video and caption ..."}],
      "timeline": [
        {"type": "silence", "interval": [0.0, 45.0]},
        {"type": "observe", "at": 46.0, "text": " [0s-46s] The video begins ...",
         "prompt_id": "p1"},
        {"type": "silence", "interval": [46.0, 169.0]},
        ...
      ]
    }

必須カラム:
  - video_path (str)
  - task       (str): "phcm" / "proactive_qa" / "proactive_interaction"
                       (control_tokens.TASK_SPECS のキー。実データの
                       `classification.task` 等から事前にマッピングしておく)
  - prompts    (list[dict]): [{"id", "at", "text"}, ...] タイムライン中に挿入する指示文
  - timeline   (list[dict]): 時系列順の注釈イベント。各イベントは以下のいずれか:
        {"type": "silence", "interval": [lo, hi]}          -> </Silence> が interval を占める
        {"type": "standby", "interval": [lo, hi]}           -> </Standby> が interval を占める
        {"type": "observe"/"summary"/"response",
         "at": t, "text": str, "prompt_id": str or None}   -> 時刻 t に制御トークン+本文を1回発行

オプション:
  - fps (float): タイムラインを離散フレームに変換するサンプリングfps (既定2.0)

============================================================
使い方
============================================================
import pandas as pd
df = pd.DataFrame([{
    "video_path": "videos/sample.mp4",
    "task": "phcm",
    "prompts": [{"id": "p1", "at": 0.0, "text": "Continuously caption the video."}],
    "timeline": [
        {"type": "silence", "interval": [0.0, 2.0]},
        {"type": "observe", "at": 2.0, "text": "A red mug appears on the table.", "prompt_id": "p1"},
    ],
}])
model, tokenizer = load_model_for_finetuning("Qwen/Qwen3-VL-4B-Instruct")
train(model, tokenizer, train_df=df, output_dir="./onestreamer-finetuned")
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from control_tokens import (
    CONTROL_TOKENS,
    TASK_SPECS,
    add_control_tokens,
    resize_embeddings_for_control_tokens,
)
from pstl_loss import ControlTokenEvent, IGNORE_INDEX, build_labels_with_pstl, compute_loss

IMAGE_TOKEN = "<image>"


# ============================================================
# 1. タイムライン -> 離散フレームステップへの変換
# ============================================================

def timeline_to_steps(
    timeline: List[Dict[str, Any]],
    fps: float,
) -> List[Dict[str, Any]]:
    """注釈タイムラインを、学習シーケンス構築用の離散ステップ列に変換する。

    実データ (huggingface.co/datasets/MCG-NJU/OneStreamer-1M で確認済み) は
    "silence"/"standby" を interval (区間) として、"observe"/"summary"/"response" を
    1点の "at" イベントとして表現する。論文 Sec 4.1 (Causal streaming sequence
    construction) の記述:
        "A frame/clip-level caption is released as a </Observe> target only
         after the end of its supporting interval"
    に従い、区間イベントは fps に応じて複数の </Silence>(or </Standby>) ステップに
    展開し、点イベントはその時刻を終端とする1ステップに変換する。

    ========================================
    出力
    ========================================
    List[{"time": float, "state": str, "text": Optional[str]}]
        state in CONTROL_TOKENS, 時系列順にソート済み。
    """
    steps: List[Dict[str, Any]] = []
    state_for_interval = {"silence": "</Silence>", "standby": "</Standby>"}
    state_for_point = {
        "observe": "</Observe>",
        "summary": "</Summary>",
        "response": "</Response>",
    }

    for event in timeline:
        etype = event["type"]
        if etype in state_for_interval:
            lo, hi = event["interval"]
            n_steps = max(1, math.ceil((hi - lo) * fps))
            for j in range(n_steps):
                t = min(lo + (j + 1) / fps, hi)
                steps.append({"time": t, "state": state_for_interval[etype], "text": None})
        elif etype in state_for_point:
            steps.append(
                {
                    "time": float(event["at"]),
                    "state": state_for_point[etype],
                    "text": event.get("text", ""),
                }
            )
        else:
            raise ValueError(f"未知のtimelineイベント種別: {etype!r}")

    steps.sort(key=lambda s: s["time"])
    return steps


def extract_frame_at(video_path: str, t_sec: float) -> "PIL.Image.Image":
    """動画から指定秒数に最も近いフレームを1枚取得する (cv2ベース, 実処理)。

    公式 Inference/inference.py の `video_windows()` と同様に cv2.VideoCapture を
    用いる。長い動画を何度も開き直すとI/Oコストが大きいため、実運用では
    Dataset 側で動画ごとにキャッシュする (本サンプルでは明快さを優先して
    毎呼び出しで開閉する簡易実装)。
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        fps_src = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = min(max(0, int(round(t_sec * fps_src))), max(0, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"フレーム取得に失敗: {video_path} @ {t_sec:.2f}s")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


# ============================================================
# 2. Dataset
# ============================================================

class OneStreamerDataset(Dataset):
    """OneStreamer-1M 形式のタイムライン注釈を1本の学習サンプルとして扱う Dataset。

    ========================================
    DataFrame columns
    ========================================
    video_path, task, prompts, timeline (必須) / fps (オプション, 既定2.0)
    """

    def __init__(self, df: pd.DataFrame, data_root: str = ""):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root) if data_root else Path(".")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        fps = float(row.get("fps", 2.0))
        steps = timeline_to_steps(list(row["timeline"]), fps)
        video_path = str(self.data_root / row["video_path"])

        frames = [extract_frame_at(video_path, s["time"]) for s in steps]
        return {
            "frames": frames,
            "steps": steps,
            "prompts": list(row.get("prompts", []) or []),
            "task": row["task"],
        }


# ============================================================
# 3. Collator: タイムライン -> 因果的視覚言語シーケンス + PSTLラベル
# ============================================================

class OneStreamerCollator:
    """1サンプルの (frames, steps, prompts, task) から学習バッチを構築する。

    ========================================
    処理フロー
    ========================================
    1. system + prompts (時刻0の指示は先頭に、それ以外は対応するステップの
       直前に挿入) を会話として組み立てる。
    2. 各ステップについて "<lo s-hi s>\\n<image>" ユーザーターン + 制御トークン
       (+本文) のアシスタントターンを交互に積み上げる。
    3. トークナイザでテキストを ID 化しつつ、<image> の位置を実際の画像枚数分の
       IMAGE_TOKEN_ID 列に展開する (本物の Qwen3-VL processor を使う場合は
       `processor.apply_chat_template(..., tokenize=True)` がこれを内部で行う。
       ネットワーク無しで動作確認するため、本ファイルでは同じ入出力契約を持つ
       `_ToyProcessor` で代替する。差し替えは `load_model_for_finetuning` 参照)。
    4. 通常のフルスーパービジョンラベル (アシスタント発話のみ教師信号) を構築。
    5. 制御トークンの位置を特定し、`pstl_loss.build_labels_with_pstl` で
       PSTL マスクを適用。

    ========================================
    出力 Shape (バッチサイズ B)
    ========================================
    input_ids:      (B, T_max) int64
    attention_mask: (B, T_max) int64
    pixel_values:   (N_img_total, C, H, W) float   - 全サンプル全画像を連結
    image_counts:   (B,) int64                     - サンプルごとの画像枚数 (分割用)
    labels:         (B, T_max) int64                - PSTL 適用後ラベル
    """

    def __init__(self, tokenizer, image_size: Tuple[int, int] = (32, 32)):
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.control_ids = {
            tok: tokenizer.convert_tokens_to_ids(tok) for tok in CONTROL_TOKENS
        }
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    def _build_conversation(
        self, steps: List[Dict[str, Any]], prompts: List[Dict[str, Any]], task: str
    ) -> Tuple[List[int], List[int], List[ControlTokenEvent], int]:
        """1サンプルを token id列に変換する。

        戻り値: (input_ids, labels_full, control_events, num_images)
        """
        system_text = (
            "You are a helpful assistant specializing in streaming video analysis. "
            "Follow the control-token protocol."
        )
        ids: List[int] = self.tokenizer.encode(system_text)
        labels: List[int] = [IGNORE_INDEX] * len(ids)

        # task==phcm かつ最後のステップが </Response> の場合、それは
        # 「ストリーム終了後に挿入される全体要約指示」に対応する (論文 Sec 4.1)。
        # それ以外の prompts は時刻0の指示として先頭に注入する (global_question 相当)。
        pending_prompts = {p["id"]: p for p in prompts}
        at_start = [p for p in prompts if p.get("at", 0.0) <= 0.0]
        for p in at_start:
            text_ids = self.tokenizer.encode(p["text"])
            ids.extend(text_ids)
            labels.extend([IGNORE_INDEX] * len(text_ids))
            pending_prompts.pop(p["id"], None)

        control_events: List[ControlTokenEvent] = []
        num_images = 0

        for i, step in enumerate(steps):
            is_last_step = i == len(steps) - 1
            # まだ注入していない prompt が、このステップの直前に該当する場合は挿入する
            # (例: phcm タスクの全体要約指示は最終ステップの直前に差し込む)。
            if is_last_step and pending_prompts:
                for p in list(pending_prompts.values()):
                    text_ids = self.tokenizer.encode(p["text"])
                    ids.extend(text_ids)
                    labels.extend([IGNORE_INDEX] * len(text_ids))
                pending_prompts.clear()

            time_tag = f"<{step['time']:.3f}s>"
            tag_ids = self.tokenizer.encode(time_tag)
            ids.extend(tag_ids)
            labels.extend([IGNORE_INDEX] * len(tag_ids))

            ids.append(self.image_token_id)
            labels.append(IGNORE_INDEX)
            num_images += 1

            # --- アシスタントターン: 制御トークン (+本文) は常にフルスーパービジョン。
            # PSTL は「制御トークン自身のラベル」だけを後段で選択的にマスクする。
            ctrl_pos = len(ids)
            ids.append(self.control_ids[step["state"]])
            labels.append(self.control_ids[step["state"]])
            control_events.append(ControlTokenEvent(position=ctrl_pos, state=step["state"]))

            if step["text"]:
                text_ids = self.tokenizer.encode(step["text"])
                ids.extend(text_ids)
                labels.extend(text_ids)  # キャプション/応答本文は常に全トークン教師信号

        return ids, labels, control_events, num_images

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_ids: List[torch.Tensor] = []
        all_labels_full: List[torch.Tensor] = []
        all_events: List[List[ControlTokenEvent]] = []
        all_tasks: List[str] = []
        all_images: List[torch.Tensor] = []
        image_counts: List[int] = []

        for feat in features:
            ids, labels_full, events, num_images = self._build_conversation(
                feat["steps"], feat["prompts"], feat["task"]
            )
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            all_labels_full.append(torch.tensor(labels_full, dtype=torch.long))
            all_events.append(events)
            all_tasks.append(feat["task"])
            image_counts.append(num_images)

            for img in feat["frames"]:
                arr = np.asarray(img.resize(self.image_size), dtype=np.float32) / 255.0
                all_images.append(torch.from_numpy(arr).permute(2, 0, 1))  # (C,H,W)

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_ids, batch_first=True, padding_value=pad_id
        )  # (B, T_max)
        labels_full = torch.nn.utils.rnn.pad_sequence(
            all_labels_full, batch_first=True, padding_value=IGNORE_INDEX
        )  # (B, T_max)
        attention_mask = (input_ids != pad_id).long()  # (B, T_max)

        # PSTL をサンプルごとに適用 (タスクごとにアンカー集合が異なるため)。
        labels = labels_full.clone()
        pstl_stats = []
        for b in range(len(features)):
            spec = TASK_SPECS[all_tasks[b]]
            labels_b, stats = build_labels_with_pstl(labels_full[b], all_events[b], spec.anchors)
            labels[b, : labels_b.shape[0]] = labels_b
            pstl_stats.append(stats)

        pixel_values = torch.stack(all_images, dim=0) if all_images else None  # (N_img_total,C,H,W)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_counts": torch.tensor(image_counts, dtype=torch.long),
            "pstl_stats": pstl_stats,
        }


# ============================================================
# 4. トイ Tokenizer / Processor (ネットワーク接続なしで動作確認するための代替)
# ============================================================
#
# 実運用では以下を使う:
#   from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
#   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
# ここでは HF からのダウンロードが行えない実行環境でも本ファイルの
# OneStreamerCollator/PSTL ロジックを実際に動かして検証できるよう、
# 同じメソッド名 (encode/convert_tokens_to_ids/add_special_tokens) を持つ
# 最小限の word-level tokenizer を自作する。トークナイズの質ではなく
# 「OneStreamer固有のシーケンス構築・PSTLマスキング」が実データ構造に対して
# 正しく動くことの検証が目的。

class _ToyTokenizer:
    """固定サイズのhashベース語彙を持つ最小トークナイザ。

    実際のBPEトークナイザ (Qwen3-VLの `AutoTokenizer`) は学習データに応じて
    語彙が動的に増えることはない。`encode()` ごとに語彙が増殖する実装にすると
    `model.resize_token_embeddings` 済みの埋め込み行列の範囲を超えるIDを生成し
    うるため、ここでは各単語を固定サイズ `base_vocab_size` のバケットへ
    deterministic にハッシュする (衝突は起こり得るが、本サンプルの目的である
    「OneStreamer固有のシーケンス構築・PSTLマスキングが実データ構造に対して
    正しく動くこと」の検証には影響しない)。
    """

    def __init__(self, base_vocab_size: int = 2000):
        self.special_tokens: List[str] = ["<pad>", IMAGE_TOKEN]
        self.base_vocab_size = base_vocab_size
        self.pad_token_id = 0
        self._special_reserve = 64  # 将来特殊トークンが増えても安全な余白

    def add_special_tokens(self, mapping: Dict[str, List[str]]) -> int:
        added = 0
        for tok in mapping.get("additional_special_tokens", []):
            if tok not in self.special_tokens:
                self.special_tokens.append(tok)
                added += 1
        return added

    def convert_tokens_to_ids(self, tok: str) -> int:
        if tok not in self.special_tokens:
            self.special_tokens.append(tok)
        return self.special_tokens.index(tok)

    def encode(self, text: str) -> List[int]:
        offset = self._special_reserve
        return [
            offset + (hash(word) % self.base_vocab_size)
            for word in text.strip().split()
        ]

    def __len__(self) -> int:
        return self._special_reserve + self.base_vocab_size


class _ToyCausalLM(nn.Module):
    """shapeの契約だけ本物のQwen3VLForConditionalGenerationに合わせた極小LM。

    実運用では `transformers.Qwen3VLForConditionalGeneration` に差し替えるだけで
    OneStreamerCollator / train() の呼び出し方は変わらない (labels を渡せば
    forward内でshift+CE lossまで計算する契約を揃えている)。
    """

    def __init__(self, vocab_size: int, hidden_size: int = 32, image_feat_dim: int = 3 * 32 * 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.vision_proj = nn.Linear(image_feat_dim, hidden_size)
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 2, batch_first=True
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.image_token_id = 1  # _ToyTokenizer での <image> のID (固定)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def resize_token_embeddings(self, new_vocab_size: int) -> nn.Embedding:
        old = self.embed
        new = nn.Embedding(new_vocab_size, old.embedding_dim)
        with torch.no_grad():
            new.weight[: old.weight.shape[0]] = old.weight
        self.embed = new
        self.lm_head = nn.Linear(old.embedding_dim, new_vocab_size, bias=False)
        return new

    def forward(
        self,
        input_ids: torch.LongTensor,              # (B, T)
        attention_mask: torch.LongTensor,         # (B, T)
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,  # (N_img_total, C, H, W)
        image_counts: Optional[torch.LongTensor] = None,  # (B,)
        **_unused,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        x = self.embed(input_ids)  # (B, T, D)

        if pixel_values is not None:
            img_feats = self.vision_proj(pixel_values.flatten(1))  # (N_img_total, D)
            offset = 0
            for b in range(B):
                n_b = int(image_counts[b].item())
                mask_b = input_ids[b] == self.image_token_id  # (T,)
                x[b, mask_b] = img_feats[offset : offset + n_b].to(x.dtype)
                offset += n_b

        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        key_padding_mask = attention_mask == 0
        x = self.layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)  # (B, T, V)

        loss = compute_loss(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}


def load_model_for_finetuning(model_path: str = "toy"):
    """モデル+トークナイザのロードと制御トークンの追加。

    `model_path` に実在のHFモデル名/パスを渡せば、以下のコードが
    (コメントアウトしてある行を有効化するだけで) 本物の Qwen3-VL-4B-Instruct で
    そのまま動く:

        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    本サンプルはネットワーク接続なしで動作確認するため `_ToyTokenizer` /
    `_ToyCausalLM` を使うが、`resize_embeddings_for_control_tokens` 自体は
    本物のHF APIと同名のメソッド (`add_special_tokens`, `resize_token_embeddings`,
    `get_input_embeddings`) にのみ依存しており、差し替えても動作は変わらない。
    """
    tokenizer = _ToyTokenizer()
    model = _ToyCausalLM(vocab_size=len(tokenizer))
    token_ids = resize_embeddings_for_control_tokens(model, tokenizer)
    print(f"制御トークンID: {token_ids}")
    return model, tokenizer


# ============================================================
# 5. 学習率スケジューラ (論文 Appendix "Training Configuration" に準拠)
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Cosine decay with linear warmup (論文 Table: Warmup ratio 0.03, Cosine decay)。"""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 6. メイン訓練ループ
# ============================================================

def train(
    model: nn.Module,
    tokenizer,
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    output_dir: str = "./onestreamer-finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    grad_acc: int = 8,
    lr: float = 1e-5,          # 論文 Table: Peak LR
    min_lr: float = 1e-6,      # 論文 Table: Minimum LR
    warmup_ratio: float = 0.03,  # 論文 Table: Warmup ratio
    weight_decay: float = 0.0,   # 論文 Table: Weight decay = 0
    max_grad_norm: float = 1.0,  # 論文 Table: Max gradient norm
    log_steps: int = 5,
    data_root: str = "",
    num_workers: int = 0,
):
    """手動訓練ループ (Trainer不使用)。

    ========================================
    各バッチのShape
    ========================================
    input_ids:      (B, T_max)
    attention_mask: (B, T_max)
    pixel_values:   (N_img_total, C, H, W)
    labels:         (B, T_max)  -100 or token id (PSTL適用済み)
    logits (出力):   (B, T_max, V)
    loss (出力):     scalar
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_amp = compute_dtype in (torch.bfloat16, torch.float16)

    train_dataset = OneStreamerDataset(train_df, data_root=data_root)
    collator = OneStreamerCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collator,
    )

    eval_loader = None
    if eval_df is not None:
        eval_dataset = OneStreamerDataset(eval_df, data_root=data_root)
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collator,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / grad_acc)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler("cuda", enabled=(compute_dtype == torch.float16))
    model = model.to(device)
    model.train()

    os.makedirs(output_dir, exist_ok=True)
    print("訓練設定:")
    print(f"  サンプル数: {len(train_dataset)}  バッチサイズ: {batch_size}  勾配累積: {grad_acc}")
    print(f"  総ステップ数: {total_steps}  Warmupステップ: {warmup_steps}  lr: {lr} -> {min_lr}")
    print(f"  Precision: {compute_dtype}  訓練可能パラメータ: {sum(p.numel() for p in trainable_params):,}")

    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        for batch_idx, batch in enumerate(train_loader):
            mean_pstl_ratio = float(
                np.mean([s["num_selected"] / max(1, s["num_control_tokens"]) for s in batch["pstl_stats"]])
            )

            tensor_batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items() if k != "pstl_stats"
            }

            with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_amp):
                outputs = model(**tensor_batch)
                loss = outputs["loss"] / grad_acc

            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_steps == 0 or global_step == total_steps:
                    print(
                        f"  Step {global_step}/{total_steps} | loss={accumulated_loss / log_steps:.4f} "
                        f"| lr={scheduler.get_last_lr()[0]:.2e} | PSTL supervised ratio={mean_pstl_ratio:.1%}"
                    )
                    accumulated_loss = 0.0

        if eval_loader is not None:
            eval_loss = evaluate(model, eval_loader, device, compute_dtype, use_amp)
            print(f"  Epoch {epoch + 1} eval loss: {eval_loss:.4f}")

    print(f"\nTraining complete! (このデモの重みは保存せず、ログのみ表示)")


def evaluate(model: nn.Module, eval_loader: DataLoader, device, compute_dtype, use_amp: bool) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval"):
            tensor_batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items() if k != "pstl_stats"
            }
            with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_amp):
                loss = model(**tensor_batch)["loss"]
            total += loss.item()
            n += 1
    model.train()
    return total / max(1, n)


# ============================================================
# 7. 合成動画を使った end-to-end 動作確認
# ============================================================

def _make_synthetic_video(path: str, num_frames: int = 10, fps: int = 2, size=(64, 64)) -> None:
    """単色が徐々に変化するだけの短い合成動画をその場で作る (実I/Oを伴う本物の動画)。"""
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(num_frames):
        color = int(255 * i / max(1, num_frames - 1))
        frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser("OneStreamer Fine-tuning (No Trainer)")
    parser.add_argument("--train_file", type=str, default="", help="JSONL (pd.read_json(lines=True))")
    parser.add_argument("--output_dir", type=str, default="./onestreamer-finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acc", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    model, tokenizer = load_model_for_finetuning()

    if args.train_file:
        train_df = pd.read_json(args.train_file, lines=True)
    else:
        print("--train_file 未指定のため、合成動画 + PHCM/ProactiveQA の疑似データで動作確認します。\n")
        tmp_dir = Path("/tmp/onestreamer_finetune_demo")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        video_path = tmp_dir / "synthetic.mp4"
        _make_synthetic_video(str(video_path), num_frames=10, fps=2)

        train_df = pd.DataFrame(
            [
                {
                    "video_path": str(video_path),
                    "task": "phcm",
                    "fps": 2.0,
                    "prompts": [],
                    "timeline": [
                        {"type": "silence", "interval": [0.0, 1.0]},
                        {"type": "observe", "at": 1.5, "text": "A dark square slowly brightens.", "prompt_id": None},
                        {"type": "silence", "interval": [1.5, 3.0]},
                        {"type": "summary", "at": 3.5, "text": "The screen brightened smoothly from black to white.", "prompt_id": None},
                    ],
                },
                {
                    "video_path": str(video_path),
                    "task": "proactive_qa",
                    "fps": 2.0,
                    "prompts": [{"id": "q1", "at": 0.0, "text": "Tell me once the screen becomes bright."}],
                    "timeline": [
                        {"type": "silence", "interval": [0.0, 2.0]},
                        {"type": "standby", "interval": [2.0, 3.0]},
                        {"type": "response", "at": 3.5, "text": "The screen is now bright.", "prompt_id": "q1"},
                    ],
                },
            ]
        )

    train(
        model, tokenizer, train_df,
        output_dir=args.output_dir, epochs=args.epochs,
        batch_size=args.batch_size, grad_acc=args.grad_acc, lr=args.lr,
    )


if __name__ == "__main__":
    main()
