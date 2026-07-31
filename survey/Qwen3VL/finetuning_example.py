"""
Qwen3-VL ファインチューニング (Trainer不使用)
=============================================

このファイルは画像/動画+テキストペアのデータセットを使って
Qwen3-VLをファインチューニングするサンプルコードです。
transformers の Trainer は使用せず、手動の訓練ループで実装しています。

公式実装参考: Qwen3-VL/qwen-vl-finetune/qwenvl/

============================================================
前提条件
============================================================
- 画像ファイルパスとQ&A (または会話) のペアデータセットが
  pandas DataFrame または JSONL ファイルとして渡される

============================================================
DataFrameの期待するカラム
============================================================
必須:
  - conversations: list of dicts [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]
  - image: str (画像ファイルパス) or list of str

オプション:
  - video: str (動画ファイルパス)

============================================================
使い方
============================================================
# スクリプト単体実行
python finetuning_example.py \
    --model_path Qwen/Qwen3-VL-7B-Instruct \
    --train_file /path/to/train.jsonl \
    --eval_file /path/to/eval.jsonl \
    --output_dir ./qwen3vl-finetuned \
    --epochs 3 \
    --batch_size 2 \
    --grad_acc 16 \
    --lr 2e-5 \
    --lora

# Python内からDataFrameで呼ぶ場合
import pandas as pd
df = pd.DataFrame({
    "conversations": [[
        {"from": "human", "value": "<image>\nこの画像を説明してください"},
        {"from": "gpt",   "value": "画像には...が写っています。"},
    ]],
    "image": ["path/to/image.jpg"],
})
model, processor = load_model_for_finetuning("Qwen/Qwen3-VL-7B-Instruct")
train(model, processor, train_df=df, output_dir="./out")
"""

import argparse
import math
import os
import re
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from tqdm.auto import tqdm
from PIL import Image

# ============================================================
# 公式を直接 import (精度に直結するため簡略版は使わない)
# ============================================================
# https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
from official.vision_process import process_vision_info
# https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/data/rope2d.py
from official.rope2d import get_rope_index_3  # noqa: E402


# ============================================================
# 定数
# ============================================================

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655   # <image>
VIDEO_TOKEN_INDEX = 151656   # <video>
ASSISTANT_TOKEN_ID = 77091   # "assistant"
IM_END_TOKEN_ID = 151645     # <|im_end|>


# ============================================================
# 1. データセット定義
# ============================================================

class Qwen3VLDataset(Dataset):
    """
    Qwen3-VL ファインチューニング用データセット (pandas DataFrame ベース)

    ========================================
    DataFrame columns
    ========================================
    必須:
        conversations (list): 会話リスト
            [{"from": "human", "value": "<image>\\n質問"}, {"from": "gpt", "value": "回答"}]
            - human ターン: <image> プレースホルダーを含む場合がある
            - gpt ターン: アシスタントの応答テキスト

    オプション:
        image (str or list): 画像ファイルパス
            - str: 1枚の画像
            - list: 複数枚の画像 (conversations内の<image>数と一致必要)
        video (str or list): 動画ファイルパス

    ========================================
    JSONL フォーマット例
    ========================================
    {"conversations": [{"from": "human", "value": "<image>\\n説明して"},
                        {"from": "gpt", "value": "犬が..."}],
     "image": "images/dog.jpg"}
    {"conversations": [{"from": "human", "value": "<image>\\nOCRして"},
                        {"from": "gpt", "value": "テキスト内容: ..."}],
     "image": "images/doc.png"}
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str = "",
    ):
        """
        入力:
            df: pandas DataFrame (conversations, image 等のカラムを含む)
            data_root: 画像パスのベースディレクトリ
        """
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root) if data_root else Path(".")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        1サンプル取得

        ========================================
        出力
        ========================================
        {
            "conversations": list of {"role": "user"/"assistant", "content": ...},
            "image_paths": list of str,
            "video_paths": list of str,
        }
        """
        row = self.df.iloc[idx]

        # conversations を取得 (from/value 形式 → role/content 形式に変換)
        raw_convs = row.get("conversations", row.get("conversation", []))
        if isinstance(raw_convs, str):
            raw_convs = json.loads(raw_convs)

        # 画像パスを取得
        image_paths = []
        if "image" in row and pd.notna(row["image"]):
            imgs = row["image"]
            if isinstance(imgs, str):
                imgs = [imgs]
            elif isinstance(imgs, list):
                pass
            for img in imgs:
                p = self.data_root / img
                image_paths.append(str(p))

        # 動画パスを取得
        video_paths = []
        if "video" in row and pd.notna(row["video"]):
            vids = row["video"]
            if isinstance(vids, str):
                vids = [vids]
            for vid in vids:
                p = self.data_root / vid
                video_paths.append(str(p))

        return {
            "raw_conversations": raw_convs,
            "image_paths": image_paths,
            "video_paths": video_paths,
        }


# ============================================================
# 2. Collator (バッチ化)
# ============================================================

class Qwen3VLCollator:
    """
    Qwen3-VL バッチ化コレーター

    DataLoader が Dataset.__getitem__() の返り値リストをバッチに変換する際に
    呼ばれる。Qwen3VL 特有の複雑な前処理（位置エンコーディング、ラベルマスク等）
    をここで一括して行う。

    ========================================
    処理フロー
    ========================================
    1. 各サンプルの raw_conversations を ChatML messages 形式に変換
       {"from":"human","value":"<image>\n質問"} → {"role":"user","content":[{"type":"image",...},{"type":"text",...}]}

    2. processor() でトークン化 + 画像パッチ化
       テキスト → input_ids (1, T_seq)
       画像 → pixel_values (N_patches, 588), image_grid_thw (num_images, 3)

    3. labels 生成: アシスタント応答部分のみを有効に設定
       input_ids と同形状 (1, T_seq) で、assistant 応答部分のみが実トークンID、
       それ以外は IGNORE_INDEX(-100) → Cross-Entropy で自動的に無視される

    4. position_ids 計算 (Interleaved MRoPE)
       トークンごとに3次元の位置ID (t, h, w) を割り当てる → (3, 1, T_seq)
       テキスト: (pos, pos, pos) — 通常の1D位置と同等
       画像パッチ: (0, row, col) — 空間位置を明示、時間軸は常に0

    5. パディングしてバッチ化
       サンプルごとに T_seq が異なる → 右パディングで T_max に揃える

    ========================================
    出力 Shape (バッチサイズ B)
    ========================================
    input_ids:              (B, T_max) int64
    attention_mask:         (B, T_max) int64     — パディング位置=0, 有効位置=1
    pixel_values:           (N_patches_total, 588) float  or None
        N_patches_total = Σ(各画像のパッチ数) — 画像ごとにパッチ数が異なるため (B,N,588) にはできない
        image_grid_thw でどこまでが何番目の画像かをモデルが判断する
    image_grid_thw:         (num_images_total, 3) int64   or None
        各行が [T, H_patches, W_patches] → pixel_values の分割キー
    pixel_values_videos:    (N_patches_video, 588) float   or None
    video_grid_thw:         (num_videos_total, 3) int64    or None
    position_ids:           (3, B, T_max) int64
        axis0=3: [temporal軸の位置, height軸の位置, width軸の位置]
        axis1=B: バッチ
        axis2=T_max: シーケンス位置
    labels:                 (B, T_max) int64
        - アシスタント応答 + <|im_end|>: トークンID (損失計算対象)
        - システムプロンプト/ユーザー発話/視覚トークン/パディング: -100 (損失除外)
    """

    def __init__(self, processor, model_type: str = "qwen3vl"):
        self.processor = processor
        self.model_type = model_type
        # spatial_merge_size はモデルの vision_config から取得 (通常=2)
        self.spatial_merge_size = getattr(
            getattr(processor, "image_processor", None), "merge_size", 2
        )

    def _convert_conversations(
        self,
        raw_conversations: List[Dict],
        image_paths: List[str],
        video_paths: List[str],
    ) -> List[Dict]:
        """
        raw_conversations (from/value形式) を
        processor 用の messages (role/content形式) に変換

        ========================================
        入力例 (from/value形式)
        ========================================
        [
            {"from": "human", "value": "<image>\\nこの画像に何が写っていますか？"},
            {"from": "gpt", "value": "猫が座っています。"},
        ]

        ========================================
        出力例 (role/content形式) — 公式 _build_messages と同一形式
        ========================================
        [
            {"role": "user", "content": [
                {"type": "image", "image": "/path/to/img.jpg"},
                {"type": "text",  "text": "この画像に何が写っていますか？"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "猫が座っています。"}]},
        ]

        【注意】アシスタントの content は文字列ではなくリスト形式にすること。
        公式 _build_messages (data_processor.py) に合わせた形式。
        文字列形式だと apply_chat_template の挙動が変わり出力が異なる場合がある。
        """
        messages = []
        image_pool = [{"type": "image", "image": p} for p in image_paths]
        video_pool = [{"type": "video", "video": p} for p in video_paths]

        for turn in raw_conversations:
            role_raw = turn.get("from", turn.get("role", "human"))
            role = "user" if role_raw in ("human", "user") else "assistant"
            text = turn.get("value", turn.get("content", ""))

            if role == "user":
                content = []
                # <image>/<video> プレースホルダーを実ファイルパスのメディアdictに置換
                # re.split で区切り文字自体も保持しながら分割
                parts = re.split(r"(<image>|<video>)", text)

                for part in parts:
                    if part == "<image>":
                        if not image_pool:
                            raise ValueError("<image>プレースホルダー数が画像ファイル数を超えています")
                        content.append(image_pool.pop(0))
                    elif part == "<video>":
                        if not video_pool:
                            raise ValueError("<video>プレースホルダー数が動画ファイル数を超えています")
                        content.append(video_pool.pop(0))
                    elif part.strip():
                        content.append({"type": "text", "text": part.strip()})

                messages.append({"role": "user", "content": content})
            else:
                # 公式 _build_messages に合わせてアシスタントも content をリスト形式にする
                # (文字列 "猫が..." ではなく [{"type":"text","text":"猫が..."}] 形式)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

        # 未消費のメディアがあればエラー (公式と同じチェック)
        if image_pool:
            raise ValueError(f"画像ファイルが{len(image_pool)}枚余っています (<image>プレースホルダー不足)")
        if video_pool:
            raise ValueError(f"動画ファイルが{len(video_pool)}本余っています (<video>プレースホルダー不足)")

        return messages

    def _create_labels(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        input_ids からラベルを生成する

        【なぜラベルマスクが必要か】
        Qwen3VLは自己回帰言語モデルなので、全トークンに対して「次トークン予測」の
        Cross-Entropy Lossを計算できる。しかし訓練目標はアシスタントの応答生成のみ
        なので、ユーザーの発話や視覚トークンに対する損失は除外する必要がある。
        PyTorchのF.cross_entropy(ignore_index=-100)がIDが-100のトークンを無視する
        仕様を利用して、不要な位置を -100 で埋める。

        【ChatMLのトークン列構造】
        ChatMLフォーマットは以下の構造を持つ:
            <|im_start|>system\n{system_prompt}<|im_end|>\n
            <|im_start|>user\n{question}<|im_end|>\n
            <|im_start|>assistant\n{response}<|im_end|>\n  ← ここだけ損失計算
            <|im_start|>user\n...                           (マルチターンの場合繰り返し)
            <|im_start|>assistant\n...

        【アシスタント応答の検出方法】
        トークン列を先頭から舐めて ASSISTANT_TOKEN_ID (=77091, "assistant"に相当) を探す。
        ChatML形式では "assistant" の直後に必ず "\n" が来るため、
        応答の開始位置は ASSISTANT_TOKEN_ID の2つ後ろ (pos + 2) になる。
            pos+0: ASSISTANT_TOKEN_ID  ("assistant")
            pos+1: \n のトークン
            pos+2: 応答の最初のトークン  ← ans_start はここ

        応答の終了位置は IM_END_TOKEN_ID (=151645, <|im_end|>) を探して特定する。
        <|im_end|> 自体も損失計算対象に含め、その後の \n も含めるため +2 する。

        ========================================
        Shape
        ========================================
        入力: input_ids (1, T_seq)
        出力: labels    (1, T_seq)
            - アシスタント応答 + <|im_end|> + \n: 元のトークンID (損失計算対象)
            - system/user発話/視覚トークン/パディング: -100 (損失除外)
        """
        # 全位置を -100 で初期化 → デフォルトは全部「損失除外」
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        ids_list = input_ids[0].tolist()
        L = len(ids_list)
        pos = 0

        while pos < L:
            if ids_list[pos] == ASSISTANT_TOKEN_ID:
                # ── アシスタントターン検出 ──
                # pos:   "assistant" トークン
                # pos+1: "\n" トークン (ChatMLの区切り)
                # pos+2: 応答本文の先頭  ← ここから損失計算したい
                ans_start = pos + 2
                ans_end = ans_start

                # 応答末尾の <|im_end|> を探す (マルチターンでも1つずつ対応)
                while ans_end < L and ids_list[ans_end] != IM_END_TOKEN_ID:
                    ans_end += 1
                # ans_end: <|im_end|> の位置

                if ans_end < L:
                    # ans_start ~ ans_end+1 (<|im_end|>を含む) を損失計算対象に設定
                    # +2 は <|im_end|>(ans_end) と その後の \n(ans_end+1) を含めるため
                    end_idx = min(ans_end + 2, L)
                    labels[0, ans_start:end_idx] = input_ids[0, ans_start:end_idx]
                    # 次のターン検索のため <|im_end|> の位置から再開
                    pos = ans_end

            pos += 1

        return labels

    def _compute_position_ids(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
        video_grid_thw: Optional[torch.LongTensor],
        attention_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interleaved MRoPE 用 position_ids を計算する

        公式実装 rope2d.py::get_rope_index_3 をそのまま呼ぶ。
        精度に直結するため、簡略版 (rope_and_position.py) は絶対に使わない。

        【position_ids の具体的な中身】
        shape (3, B, T_seq) で、T_seq 軸の各トークン位置に [temporal, height, width] が入る。

        例: テキスト3トークン → 画像2×2パッチ(merge後=4token) → テキスト1トークン
            T_seq = 3 + 4 + 1 = 8

            T_seq位置:   0     1     2     3          4          5          6       7
            モダリティ: text  text  text  vision     vision     vision     vision  text
                                         (r=0,c=0)  (r=0,c=1)  (r=1,c=0)  (r=1,c=1)

            temporal行:  0     1     2     0          0          0          0       4
            height行:    0     1     2     0          0          1          1       4
            width行:     0     1     2     0          1          0          1       4

        テキストトークン: temporal=height=width=同じ通し番号
            → 3軸全て同値なので通常の1D RoPEと等価。
              textにh/wという概念はないが、3軸を同値にすることで事実上無効化している。
        visionトークン: temporal=0固定, height=行index, width=列index
            → 2D空間RoPEとして機能。

        【Qwen3VL の Interleaved MRoPE の特徴】
        動画の temporal 軸は常に 0 固定。時間情報は "<t0.00>","<t0.50>" のような
        テキストタイムスタンプトークンで表現する (Qwen2.5VLとの最大の違い)。
        rope2d.py 内コメント: "we use timestamps rather than absolute time position ids"

        ========================================
        Shape
        ========================================
        入力:
            input_ids:      (1, T_seq)
            image_grid_thw: (num_images, 3) or None — 各画像 [T=1, H_patches, W_patches]
            video_grid_thw: (num_videos, 3) or None — 各動画フレーム [T=1, H, W] (分割済み)
            attention_mask: (1, T_seq)
        出力:
            position_ids:   (3, 1, T_seq)
            mrope_deltas:   (1, 1)
        """
        # 公式 rope2d.py の get_rope_index_3 を直接呼ぶ
        # (ファイル冒頭でimport済み)
        return get_rope_index_3(
            spatial_merge_size=self.spatial_merge_size,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        バッチ化: サンプルリストをモデルに渡せるバッチに変換する

        DataLoader が `collate_fn=Qwen3VLCollator(...)` として呼ぶ。
        features は Qwen3VLDataset.__getitem__() の返り値のリスト (長さ=batch_size)。

        ========================================
        処理フロー
        ========================================
        サンプルごとに個別処理 (Step 1〜5) → バッチ結合 (Step 6)

        Step 1: conversations → messages 変換
            Dataset は raw 形式 {"from":"human","value":"..."} で保持
            processor.apply_chat_template() が要求する形式
            {"role":"user","content":[{"type":"image",...},{"type":"text",...}]} に変換

        Step 2: apply_chat_template(tokenize=True) でトークン化 + 画像パッチ化を一括実行
            公式 preprocess_qwen_visual と同一の呼び出し方式:
            processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
            ・テキスト → input_ids: <image> トークン位置に IMAGE_TOKEN_ID=151655 が入る
            ・画像 → pixel_values: (N_patches, 3×14²=588) に変換
                     image_grid_thw: その画像の [T, H_patches, W_patches]
            ・pixel_values は (B, N_max, 588) ではなく (N_patches_total, 588) のフラット形式
              → 画像ごとにパッチ数が異なるためパディングなしで全画像を1次元に並べる
              → モデルは image_grid_thw を参照して各画像の範囲を判断する

        Step 3: labels 生成 (_create_labels)
            input_ids と同形状で、損失計算したいトークンにのみ実IDを入れ、
            残りは IGNORE_INDEX(-100) にする。
            → アシスタント応答部分のみを損失計算対象とする
            （ユーザー発話・システムプロンプト・視覚トークンはモデルが生成するものではない）

        Step 4: Interleaved MRoPE 用 position_ids 計算 (_compute_position_ids)
            公式 rope2d.py::get_rope_index_3 を直接呼ぶ。
            通常の1D位置IDではなく3軸 (temporal, height, width) の位置IDを計算:
            ・テキストトークン位置p: position_ids[:, 0, p] = [p, p, p]  (3軸同じ値)
            ・画像パッチ (行r, 列c): position_ids[:, 0, p] = [0, r, c]  (t=0固定, h=行, w=列)
            これによりアテンションが「どのパッチがどの空間位置か」を正確に把握できる

        Step 5: パディング & バッチ結合
            サンプルごとに T_seq が異なる → 最大長 T_max に右パディング
            input_ids: pad_token_id でパディング
            labels:    IGNORE_INDEX(-100) でパディング (損失計算されない)
            position_ids: 1 でパディング (0は有効な位置IDなので1を使う)
            pixel_values: 全サンプルの全画像パッチを dim=0 方向に連結
                         → (Σ N_patches_i, 588)
        """
        # サンプルごとの中間結果を格納するリスト
        all_input_ids = []          # List[(T_i,)]
        all_labels = []             # List[(T_i,)]
        all_pixel_values = []       # List[(N_patches_i, 588)] — 画像があるサンプルのみ
        all_image_grid_thw = []     # List[(num_images_i, 3)]
        all_pixel_values_videos = []
        all_video_grid_thw = []
        all_position_ids = []       # List[(3, 1, T_i)]

        for item in features:
            raw_convs = item["raw_conversations"]
            image_paths = item["image_paths"]
            video_paths = item["video_paths"]

            # ──────────────────────────────────────────────────
            # Step 1: conversations → messages 変換
            # ──────────────────────────────────────────────────
            # 入力: [{"from":"human","value":"<image>\n質問"},{"from":"gpt","value":"回答"}]
            # 出力: [{"role":"user","content":[{"type":"image","image":"/path"},{"type":"text","text":"質問"}]},
            #        {"role":"assistant","content":[{"type":"text","text":"回答"}]}]
            # ※アシスタントもリスト形式 — 公式 _build_messages に準拠
            messages = self._convert_conversations(raw_convs, image_paths, video_paths)

            # ──────────────────────────────────────────────────
            # Step 2: apply_chat_template(tokenize=True) でトークン化 + 画像パッチ化を一括実行
            # ──────────────────────────────────────────────────
            # 公式 preprocess_qwen_visual と同一の呼び出し:
            #   processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
            # tokenize=True にすることで、テキスト展開→トークン化→画像パッチ化を1回で行う。
            # ・tokenize=False + processor() の二段階にしないこと (公式と挙動が変わる可能性)
            # ・add_generation_prompt=False: 訓練時はアシスタント応答まで含める
            #   (推論時は True にして "<|im_start|>assistant\n" を末尾に付与)
            result = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            # result の主要なフィールド:
            #   input_ids:           (1, T_seq) int64
            #       text → token IDs, <image>位置は IMAGE_TOKEN_ID(151655) に置換済み
            #   attention_mask:      (1, T_seq) int64  全て1 (パディングなし)
            #   pixel_values:        (N_patches, 588) float  ← 画像がある場合のみ
            #       画像をsmart_resizeして14×14パッチに分割し flatten した形状
            #       N_patches = H_patches × W_patches (画像ごとに異なる)
            #       588 = C × P² = 3 × 14² (チャンネル × パッチサイズ²)
            #   image_grid_thw:      (num_images, 3) int64  ← 画像がある場合のみ
            #       各行が [T=1, H_patches, W_patches]
            #       これによりモデルが pixel_values をどこで区切るかを判断する
            #   pixel_values_videos: (N_patches_v, 588) float  ← 動画がある場合のみ
            #   video_grid_thw:      (num_videos, 3) int64

            input_ids = result["input_ids"]  # (1, T_seq)
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids).unsqueeze(0)

            # ──────────────────────────────────────────────────
            # Step 3: labels 生成
            # ──────────────────────────────────────────────────
            # input_ids と同形状で、アシスタント応答部分のみに実トークンIDを入れ
            # それ以外 (system/user/視覚トークン/パディング) は -100 にする
            # Cross-Entropy の ignore_index=-100 で -100 位置は自動的に損失ゼロになる
            labels = self._create_labels(input_ids)
            # labels: (1, T_seq)

            # ──────────────────────────────────────────────────
            # Step 4: Interleaved MRoPE 用 position_ids 計算
            # ──────────────────────────────────────────────────
            # 通常のRoPEは 1D の位置ID (0, 1, 2, ...) を使うが、
            # Qwen3VL は 3D の位置ID (temporal, height, width) を各トークンに割り当てる
            #
            # image_grid_thw が必要な理由:
            #   input_ids 内の IMAGE_TOKEN_ID の位置が「どの画像の何行何列目か」を
            #   grid_thw から逆引きして h/w の位置IDを計算するため
            image_grid_thw = result.get("image_grid_thw", None)
            video_grid_thw = result.get("video_grid_thw", None)

            position_ids, _ = self._compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=result.get("attention_mask"),
            )
            # position_ids: (3, 1, T_seq)
            #   [0, :, :] = temporal 軸の位置ID
            #   [1, :, :] = height   軸の位置ID
            #   [2, :, :] = width    軸の位置ID

            # バッチ次元を除いて格納
            all_input_ids.append(input_ids.squeeze(0))        # (T_seq,)
            all_labels.append(labels.squeeze(0))              # (T_seq,)
            all_position_ids.append(position_ids)             # (3, 1, T_seq) ← バッチ次元はそのまま保持

            # 画像/動画パッチデータは None でない場合のみ収集
            if "pixel_values" in result and result["pixel_values"] is not None:
                all_pixel_values.append(result["pixel_values"])     # (N_patches_i, 588)
                all_image_grid_thw.append(result["image_grid_thw"]) # (num_images_i, 3)

            if "pixel_values_videos" in result and result["pixel_values_videos"] is not None:
                all_pixel_values_videos.append(result["pixel_values_videos"])
                all_video_grid_thw.append(result["video_grid_thw"])

        # ──────────────────────────────────────────────────
        # Step 6: パディング & バッチ結合
        # ──────────────────────────────────────────────────
        # サンプルごとに T_seq が異なるため、最大長 T_max に合わせて右パディングする
        pad_token_id = self.processor.tokenizer.pad_token_id

        # input_ids: 短いサンプルの末尾を pad_token_id で埋める
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(
            all_input_ids,       # List[(T_i,)]
            batch_first=True,
            padding_value=pad_token_id,
        )
        # input_ids_batch: (B, T_max)

        # labels: パディング位置は IGNORE_INDEX(-100) → 損失計算に影響しない
        labels_batch = torch.nn.utils.rnn.pad_sequence(
            all_labels,          # List[(T_i,)]
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        # labels_batch: (B, T_max)

        # attention_mask: パディング位置=0, 有効位置=1
        # (pad_sequence後の input_ids から逆算するのが最も確実)
        attention_mask = (input_ids_batch != pad_token_id).long()
        # attention_mask: (B, T_max)

        # position_ids のパディング
        # ・各サンプルの position_ids は (3, 1, T_i) → バッチ結合して (3, B, T_max) にする
        # ・パディング位置には 1 を使う (0は有効な位置IDとして使われているため 0 は使えない)
        # ・公式の pad_and_cat() と同等の処理
        T_max = input_ids_batch.shape[1]
        B = len(features)
        position_ids_padded = torch.ones(3, B, T_max, dtype=torch.long)
        for i, pid in enumerate(all_position_ids):
            # pid: (3, 1, T_i)
            T_i = pid.shape[-1]
            position_ids_padded[:, i, :T_i] = pid[:, 0, :]
            # 先頭 T_i 位置に実際の position_ids を書き込み、残り(T_i..T_max)は1のまま
        # position_ids_padded: (3, B, T_max)

        # ──────────────────────────────────────────────────
        # model_max_length で切り詰め (公式と同一)
        # ──────────────────────────────────────────────────
        # 公式 DataCollatorForSupervisedDataset と同じ処理:
        #   input_ids = input_ids[:, : self.tokenizer.model_max_length]
        #   labels = labels[:, : self.tokenizer.model_max_length]
        #   position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        max_len = self.processor.tokenizer.model_max_length
        input_ids_batch   = input_ids_batch[:, :max_len]           # (B, T_max) → (B, min(T_max,max_len))
        labels_batch      = labels_batch[:, :max_len]
        attention_mask    = attention_mask[:, :max_len]
        position_ids_padded = position_ids_padded[:, :, :max_len]  # (3, B, T_max) → (3, B, min(...))

        batch = {
            "input_ids":      input_ids_batch,      # (B, T_clip)
            "attention_mask": attention_mask,        # (B, T_clip)
            "labels":         labels_batch,          # (B, T_clip)
            "position_ids":   position_ids_padded,  # (3, B, T_clip)
        }

        # 画像データ: バッチ内の全画像パッチを dim=0 方向に連結
        # (B, N_max, 588) にしない理由: 画像ごとにパッチ数が異なるためパディングが難しく
        # モデル側も (N_total, 588) のフラット形式を前提として image_grid_thw で分割している
        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0)
            # (Σ N_patches_i, 588) — 全バッチ・全画像のパッチを連結
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0)
            # (Σ num_images_i, 3) — 各画像の [T, H_patches, W_patches]

        # 動画データも同様に連結
        if all_pixel_values_videos:
            batch["pixel_values_videos"] = torch.cat(all_pixel_values_videos, dim=0)
            batch["video_grid_thw"] = torch.cat(all_video_grid_thw, dim=0)

        return batch


# ============================================================
# 3. モデルの準備
# ============================================================

def load_model_for_finetuning(
    model_path: str,
    use_lora: bool = False,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    freeze_vision: bool = True,
    freeze_merger: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    quantization: Optional[str] = None,
):
    """
    ファインチューニング用モデルロード

    ========================================
    モデル構成の選択肢
    ========================================
    FullFT (全パラメータ訓練):
        tune_mm_vision=True, tune_mm_mlp=True, tune_mm_llm=True

    LoRA (推奨, メモリ効率):
        LLMのq_proj, k_proj, v_proj, o_projのみLoRA
        Vision Encoderはデフォルトで凍結

    VisonFT (Visionのみ):
        tune_mm_vision=True, tune_mm_llm=False

    ========================================
    出力
    ========================================
    model: Qwen3VLForConditionalGeneration
    processor: AutoProcessor
    compute_dtype: torch.bfloat16 or torch.float16
    """
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
        ModelClass = Qwen3VLForConditionalGeneration
    except ImportError:
        # フォールバック: Qwen2.5-VL (transformers が古い場合)
        from transformers import Qwen2_5_VLForConditionalGeneration
        ModelClass = Qwen2_5_VLForConditionalGeneration
        print("警告: Qwen3VLForConditionalGeneration が見つからないため Qwen2_5_VL を使用")

    # compute_dtype の決定
    if compute_dtype is None:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            compute_dtype = torch.bfloat16  # Ampere以降: bf16
        else:
            compute_dtype = torch.float16

    print(f"モデルロード: {model_path}")
    print(f"compute_dtype: {compute_dtype}")

    # 量子化設定 (QLoRA)
    _bnb_cfg = None
    if quantization is not None:
        from transformers import BitsAndBytesConfig
        if quantization == '4bit':
            _bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        elif quantization == '8bit':
            _bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"quantization は '4bit' / '8bit' / None のみ有効: {quantization!r}")
        print(f"量子化: {quantization} (QLoRA)")

    # モデルロード
    # 量子化時は device_map="auto" が必須 (.to(device) 不可)
    _device_map = "auto" if quantization is not None else None
    model = ModelClass.from_pretrained(
        model_path,
        torch_dtype=compute_dtype,
        device_map=_device_map,
        attn_implementation="flash_attention_2",  # FlashAttention2推奨
        quantization_config=_bnb_cfg,
    )

    processor = AutoProcessor.from_pretrained(model_path)

    # ========================================
    # パラメータの訓練設定
    # ========================================
    # 量子化後処理: kbit 学習の準備 (LoRA 前に必須)
    if quantization is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("prepare_model_for_kbit_training 完了")

    if use_lora:
        # LoRA 設定
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # 全パラメータを凍結
            for p in model.parameters():
                p.requires_grad = False

            lora_config = LoraConfig(
                r=lora_r,           # LoRAランク (デフォルト: 64)
                lora_alpha=lora_alpha,  # スケーリング係数 (デフォルト: 128)
                lora_dropout=lora_dropout,  # ドロップアウト (デフォルト: 0.05)
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    # LLMのアテンション投影層 (公式設定に準拠)
                ],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        except ImportError:
            print("警告: peft がインストールされていません。LoRAなしで続行します。")
            use_lora = False

    if not use_lora:
        # Vision Encoder の凍結 (デフォルト)
        if freeze_vision:
            for n, p in model.visual.named_parameters():
                p.requires_grad = False
            print("Vision Encoder を凍結しました")

        # MLP Merger の設定
        if freeze_merger:
            for n, p in model.visual.merger.named_parameters():
                p.requires_grad = False
            print("MLP Merger を凍結しました")
        else:
            # Merger のみ unfreeze (Vision凍結の場合でも)
            for n, p in model.visual.merger.named_parameters():
                p.requires_grad = True
            print("MLP Merger を学習対象にしました")

        # LLM は全部学習
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.weight.requires_grad = True

    # 訓練可能パラメータ数を表示
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"訓練可能パラメータ: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model.config.use_cache = False  # 訓練時はKVキャッシュを無効化

    return model, processor, compute_dtype


# ============================================================
# 4. 学習率スケジューラ
# ============================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine 学習率スケジューラ with Warmup

    ========================================
    スケジュール
    ========================================
    step < warmup:
        lr = base_lr × step / warmup_steps

    step >= warmup:
        progress = (step - warmup) / (total - warmup)
        lr = base_lr × (min_ratio + (1 - min_ratio) × 0.5 × (1 + cos(π × progress)))

    典型的な値:
        warmup_ratio = 0.03 (全ステップの3%)
        min_lr_ratio = 0.0 (最終的に学習率→0)
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 5. AMP 設定
# ============================================================

def setup_amp(
    compute_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Dict, GradScaler]:
    """
    Mixed Precision (AMP) の設定

    ========================================
    戦略
    ========================================
    bfloat16: torch.autocast(dtype=bf16), GradScaler無効 (bf16は安定)
    float16:  torch.autocast(dtype=fp16), GradScaler有効 (スケーリング必要)
    float32:  torch.autocast(enabled=False), GradScaler無効

    ========================================
    出力
    ========================================
    amp_ctx_kwargs: dict - torch.autocast に渡す kwargs
    scaler: GradScaler
    """
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = compute_dtype in (torch.bfloat16, torch.float16)

    amp_ctx_kwargs = dict(
        device_type=device_type,
        dtype=compute_dtype if use_amp else torch.float32,
        enabled=use_amp,
    )

    # bf16 では GradScaler 不要 (enabled=False にしてno-opに)
    scaler = GradScaler(
        device_type,
        enabled=(compute_dtype == torch.float16),
    )

    return amp_ctx_kwargs, scaler


# ============================================================
# 6. チェックポイント保存
# ============================================================

def save_checkpoint(
    model,
    processor,
    output_dir: str,
    global_step: int,
    is_final: bool = False,
):
    """
    チェックポイント保存

    ========================================
    保存内容
    ========================================
    - モデル重み (safetensors)
    - config.json
    - tokenizer / processor ファイル
    - LoRA の場合はアダプター重みのみ
    """
    if is_final:
        save_dir = os.path.join(output_dir, "final")
    else:
        save_dir = os.path.join(output_dir, f"checkpoint-{global_step}")

    os.makedirs(save_dir, exist_ok=True)

    # generation_config の矛盾を修正
    if hasattr(model, "generation_config"):
        gc = model.generation_config
        if not getattr(gc, "do_sample", True):
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(gc, attr):
                    setattr(gc, attr, None)

    # モデル保存 (LoRAの場合はアダプターのみ)
    model.save_pretrained(save_dir, safe_serialization=True)

    # Processor 保存
    processor.save_pretrained(save_dir)

    print(f"  チェックポイント保存: {save_dir}")


# ============================================================
# 7. 評価関数
# ============================================================

def evaluate(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    amp_ctx_kwargs: dict,
) -> Dict:
    """
    評価 (損失計算)

    ========================================
    Shape
    ========================================
    各バッチ:
        input_ids:          (B, T_max) int64
        labels:             (B, T_max) int64
        pixel_values:       (N_patches, 588) float
        logits:             (B, T_max, 151936) float
        loss:               scalar

    出力:
        {"loss": float}
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.autocast(**amp_ctx_kwargs):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return {"loss": total_loss / max(num_batches, 1)}


# ============================================================
# 8. 推論関数
# ============================================================

def generate_response(
    model,
    processor,
    image_path: Optional[str] = None,
    question: str = "",
    max_new_tokens: int = 512,
    device: Optional[torch.device] = None,
) -> str:
    """
    単一サンプルの推論

    ========================================
    Shape
    ========================================
    入力:
        image_path: str (画像パス)
        question:   str (質問テキスト)

    内部:
        pixel_values:   (N_patches, 588)
        input_ids:      (1, T_seq)
        position_ids:   (3, 1, T_seq)
        generated_ids:  (1, T_seq + max_new_tokens)

    出力:
        response: str - 生成されたテキスト
    """

    if device is None:
        device = next(model.parameters()).device

    # 入力メッセージ構成
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]

    # テキストのトークン化
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 推論時: アシスタント応答の開始を促す
    )

    # 画像ロード
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    # inputs:
    #   input_ids:      (1, T_seq)
    #   pixel_values:   (N_patches, 588)  [画像がある場合]
    #   image_grid_thw: (1, 3)

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        # generated_ids: (1, T_seq + 生成トークン数)

    # 入力部分を除いた生成テキストをデコード
    response = processor.decode(
        generated_ids[0, input_len:],
        skip_special_tokens=True,
    )

    return response


# ============================================================
# 9. 学習中サンプル生成表示
# ============================================================

def log_sample_predictions(
    model,
    processor,
    sample_df: pd.DataFrame,
    device: torch.device,
    global_step: int,
    max_new_tokens: int = 128,
    display_size: tuple = (300, 200),
    display_image: bool = True,
):
    """
    log_steps ごとに val サンプルを generate して画像・質問・生成文を表示する

    呼び出し元で train_df/eval_df から固定サンプルを切り出して渡すことで、
    ステップをまたいで同じサンプルに対する生成変化を追える。

        # 呼び出し側での準備例
        log_samples = eval_df.sample(n=2, random_state=42).reset_index(drop=True)
        # log_steps ごとに
        log_sample_predictions(model, processor, log_samples, device, global_step)

    Jupyter 環境では画像を display_size にリサイズして inline 表示し、
    それ以外の環境では print のみにフォールバックする。

    引数:
        sample_df:     表示対象の固定サンプル DataFrame (呼び出し元で固定しておく)
        global_step:   現在のステップ数 (表示用ラベル)
        max_new_tokens: 生成の最大トークン数 (ログ用なので短めでよい)
        display_size:  Jupyter 表示時のリサイズ後サイズ (width, height)
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"[Step {global_step}] val サンプル生成プレビュー")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(sample_df.iterrows()):
        # 画像パスを取得 (リストの場合は先頭1枚)
        image_path = row.get("image", None)
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else None

        # conversations から human/gpt 発話を取得
        convs = row.get("conversations", [])
        if isinstance(convs, str):
            convs = json.loads(convs)
        human_turn = next((c for c in convs if c.get("from") in ("human", "user")), None)
        gpt_turn   = next((c for c in convs if c.get("from") in ("gpt", "assistant")), None)

        # "<image>\n" プレースホルダーを除いた質問テキストを抽出
        question = ""
        if human_turn:
            question = human_turn.get("value", "").replace("<image>", "").strip()

        ground_truth = gpt_turn.get("value", "") if gpt_turn else ""

        print(f"\n--- サンプル {i + 1} ---")
        print(f"質問: {question[:120]}{'...' if len(question) > 120 else ''}")
        print(f"正解: {ground_truth[:120]}{'...' if len(ground_truth) > 120 else ''}")

        # 画像表示 (Jupyter のみ、失敗時はパスを print してフォールバック)
        if image_path and Path(image_path).exists() and display_image:
            try:
                from IPython.display import display as ipy_display
                pil_img = Image.open(image_path).convert("RGB")
                pil_img_small = pil_img.resize(display_size, Image.LANCZOS)
                ipy_display(pil_img_small)
            except Exception:
                print(f"[画像: {image_path}]")

        # generate して生成文を表示
        try:
            response = generate_response(
                model=model,
                processor=processor,
                image_path=image_path if (image_path and Path(image_path).exists()) else None,
                question=question,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            print(f"生成: {response[:200]}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"生成エラー: {e}")

    print(f"{'='*60}\n")
    model.train()


# ============================================================
# 10. メイン訓練ループ
# ============================================================

def train(
    model,
    processor,
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    log_preview_df: Optional[pd.DataFrame] = None,
    compute_dtype: Optional[torch.dtype] = None,
    output_dir: str = "./qwen3vl-finetuned",
    epochs: int = 3,
    batch_size: int = 2,
    grad_acc: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.03,
    save_steps: int = 200,
    save_checkpoint: bool = False,
    log_steps: int = 10,
    max_grad_norm: float = 1.0,
    num_workers: int = 4,
    data_root: str = "",
):
    """
    手動訓練ループ (Trainerなし)

    引数:
        log_preview_df: log_steps ごとに generate 表示するサンプルの固定 DataFrame。
            None の場合は生成プレビューをスキップする。
            eval_df から事前にサンプリングして渡すことで、ステップをまたいで
            同じサンプルへの生成変化を追える。

                log_preview_df = eval_df.sample(n=2, random_state=42).reset_index(drop=True)

    ========================================
    訓練ハイパーパラメータ (公式推奨)
    ========================================
    学習率:         2e-5 (LLM), 2e-6 (Vision Encoder)
    バッチサイズ:    2 × 16 (grad_acc) = 32 effective
    Warmup:         3% (warmup_ratio=0.03)
    スケジューラ:    Cosine
    Precision:      bfloat16 (Ampere+) or float16
    Vision Encoder: 凍結推奨 (事前学習済みSigLIP-2)
    MLP Merger:     学習推奨 (タスク適応に有効)

    ========================================
    各バッチのShape遷移
    ========================================
    input_ids:          (B, T_seq)
    pixel_values:       (N_patches, 588)   # 全バッチの画像パッチ
    position_ids:       (3, B, T_seq)      # Interleaved MRoPE
    labels:             (B, T_seq)         # -100 または トークンID
    logits (出力):       (B, T_seq, 151936)
    loss (出力):         scalar
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if compute_dtype is None:
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

    # ========================================
    # 1. データセット + DataLoader
    # ========================================
    train_dataset = Qwen3VLDataset(train_df, data_root=data_root)
    train_collator = Qwen3VLCollator(processor, model_type="qwen3vl")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collator,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    eval_loader = None
    if eval_df is not None:
        eval_dataset = Qwen3VLDataset(eval_df, data_root=data_root)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_collator,
            pin_memory=(device.type == "cuda"),
        )

    # ========================================
    # 2. Optimizer + Scheduler
    # ========================================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )

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
    amp_ctx_kwargs, scaler = setup_amp(compute_dtype, device)

    # ========================================
    # 4. モデルを GPU に移動
    # ========================================
    # 量子化モデル (device_map="auto") はすでに GPU に配置されているため .to() 不可
    _is_quantized = getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False)
    if not _is_quantized:
        model = model.to(device)
    model.train()

    # ========================================
    # 5. 訓練ループ
    # ========================================
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0
    accumulated_loss = 0.0

    precision_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }.get(compute_dtype, "float32")

    print(f"\n訓練設定:")
    print(f"  サンプル数:          {len(train_dataset)}")
    print(f"  バッチサイズ:        {batch_size}")
    print(f"  勾配累積:            {grad_acc}")
    print(f"  実効バッチサイズ:    {batch_size * grad_acc}")
    print(f"  エポック数:          {epochs}")
    print(f"  総ステップ数:        {total_steps}")
    print(f"  Warmupステップ:      {warmup_steps}")
    print(f"  学習率:              {lr}")
    print(f"  Precision:           {precision_name}")
    print(f"  訓練可能パラメータ:  {sum(p.numel() for p in trainable_params):,}")
    print()

    # 初期評価
    if eval_loader is not None:
        eval_result = evaluate(model, eval_loader, device, amp_ctx_kwargs)
        print(f"初期評価 loss: {eval_result['loss']:.4f}")
    if log_preview_df is not None:
        log_sample_predictions(
            model=model,
            processor=processor,
            sample_df=log_preview_df,
            device=device,
            global_step=global_step,
        )

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            # ========================================
            # 5a. バッチを GPU に移動
            # ========================================
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # ========================================
            # 5b. Forward Pass
            # ========================================
            # autocast で bf16/fp16 計算 (enabled=True) or fp32 (enabled=False)
            with torch.autocast(**amp_ctx_kwargs):
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                # loss: scalar

            # 勾配累積: grad_acc で割って平均化
            loss = loss / grad_acc

            # ========================================
            # 5c. Backward
            # ========================================
            scaler.scale(loss).backward()
            # enabled=False の場合は通常の backward() と等価

            accumulated_loss += loss.item()

            # ========================================
            # 5d. Gradient Step (grad_acc ごと)
            # ========================================
            is_accumulation_complete = (
                (batch_idx + 1) % grad_acc == 0
                or (batch_idx + 1) == len(train_loader)
            )

            if is_accumulation_complete:
                # Gradient Unscale (fp16 の場合のみ実効)
                scaler.unscale_(optimizer)

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                # Optimizer Step
                scaler.step(optimizer)
                scaler.update()

                # Scheduler Step
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

                    # val サンプルへの生成プレビュー (log_preview_df が渡されている場合のみ)
                    if log_preview_df is not None:
                        log_sample_predictions(
                            model=model,
                            processor=processor,
                            sample_df=log_preview_df,
                            device=device,
                            global_step=global_step,
                        )

                # ========================================
                # 5f. チェックポイント保存
                # ========================================
                if save_checkpoint and global_step % save_steps == 0:
                    save_checkpoint(model, processor, output_dir, global_step)

        # ========================================
        # Epoch 終了時の評価
        # ========================================
        if eval_loader is not None:
            eval_result = evaluate(model, eval_loader, device, amp_ctx_kwargs)
            print(f"  Epoch {epoch + 1} 評価 loss: {eval_result['loss']:.4f}")

    # ========================================
    # 6. 最終モデル保存
    # ========================================
    if save_checkpoint:
        save_checkpoint(model, processor, output_dir, global_step, is_final=True)
        print(f"\n訓練完了! モデルを保存: {output_dir}/final")


# ============================================================
# 10. エントリポイント
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Qwen3-VL Fine-tuning (No Trainer)")

    # パス
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-7B-Instruct",
                   help="HuggingFace モデルパスまたはローカルパス")
    p.add_argument("--train_file", type=str, required=True,
                   help="訓練データ JSONL ファイル (conversations, image カラムを持つ)")
    p.add_argument("--eval_file", type=str, default="",
                   help="評価データ JSONL ファイル (オプション)")
    p.add_argument("--data_root", type=str, default="",
                   help="画像ファイルのベースディレクトリ")
    p.add_argument("--output_dir", type=str, default="./qwen3vl-finetuned",
                   help="チェックポイント出力ディレクトリ")

    # 訓練ハイパーパラメータ
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-GPU バッチサイズ")
    p.add_argument("--grad_acc", type=int, default=16,
                   help="勾配累積ステップ数")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="学習率")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="Warmup比率 (全ステップに対する割合)")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="勾配クリッピングの最大ノルム")

    # LoRA 設定
    p.add_argument("--lora", action="store_true",
                   help="LoRAを使用する")
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRAランク")
    p.add_argument("--lora_alpha", type=int, default=128,
                   help="LoRA alpha (スケーリング係数)")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRAドロップアウト率")
    p.add_argument("--quantization", type=str, default=None,
                   choices=["4bit", "8bit"],
                   help="量子化モード (QLoRA): 4bit / 8bit")

    # Vision 設定
    p.add_argument("--freeze_vision", action="store_true", default=True,
                   help="Vision Encoderを凍結する (デフォルト: True)")
    p.add_argument("--unfreeze_vision", action="store_true",
                   help="Vision Encoderを学習対象にする")

    # 保存・ログ
    p.add_argument("--save_steps", type=int, default=200,
                   help="N ステップごとにチェックポイントを保存")
    p.add_argument("--log_steps", type=int, default=10,
                   help="N ステップごとにログを出力")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader のワーカー数")

    return p.parse_args()


def main():
    args = parse_args()

    # ========================================
    # モデルロード
    # ========================================
    freeze_vision = args.freeze_vision and not args.unfreeze_vision

    model, processor, compute_dtype = load_model_for_finetuning(
        model_path=args.model_path,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_vision=freeze_vision,
        quantization=args.quantization,
    )

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
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()


# ============================================================
# データセット変換ユーティリティ
# ============================================================

def convert_scienceqa_to_df(
    hf_dataset,
    image_save_dir: str,
    split: str = "train",
    skip_no_image: bool = True,
) -> pd.DataFrame:
    """
    HuggingFace の ScienceQA データセットを Qwen3VLDataset 用 DataFrame に変換する

    ========================================
    入力
    ========================================
    hf_dataset:
        load_dataset("derek-thomas/ScienceQA", split="train[:500]") の返り値
        features:
            image    : PIL.Image or None  (画像なしサンプルは None)
            question : str               (問題文)
            choices  : list[str]         (選択肢 例: ["A", "B", "C", "D"])
            answer   : int               (正解の選択肢インデックス)
            hint     : str or None       (ヒント文)
            lecture  : str or None       (背景知識)
            solution : str or None       (解説)

    image_save_dir : str
        PIL画像をJPEGとして保存するディレクトリ
        (Qwen3VLDatasetは画像をファイルパスで受け取るため)

    skip_no_image : bool
        image=None のサンプル(テキストのみ問題)をスキップするか
        Trueにするとテキストのみ問題を除外してVLMのfinetuneに適したデータになる

    ========================================
    出力 DataFrame columns
    ========================================
    conversations : list[dict]
        [
            {"from": "human", "value": "<image>\n{hint}\n{question}\nA. ...\nB. ...\n"},
            {"from": "gpt",   "value": "答えは{choice_label}です。\n{solution}"},
        ]
        ※ hint が空の場合は hint 行を省略
        ※ image がある場合のみ "<image>\n" を先頭に付与
    image : str or None
        保存した画像ファイルの絶対パス (image=None のサンプルは None)

    ========================================
    使用例
    ========================================
    from datasets import load_dataset
    import pandas as pd

    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train[:500]")
    df = convert_scienceqa_to_df(hf_ds, image_save_dir="./scienceqa_images")

    # Qwen3VLDataset に渡す
    dataset = Qwen3VLDataset(df)
    """
    import os
    from pathlib import Path

    save_dir = Path(image_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 選択肢インデックス → ラベル文字列 (0→"A", 1→"B", ...)
    CHOICE_LABELS = "ABCDEFGH"

    rows = []
    for i, sample in enumerate(hf_dataset):
        pil_image = sample["image"]   # PIL.Image or None
        has_image = pil_image is not None

        if skip_no_image and not has_image:
            continue

        # ── 画像を保存 ──
        image_path = None
        if has_image:
            image_path = str(save_dir / f"{split}_{i:06d}.jpg")
            pil_image.convert("RGB").save(image_path, format="JPEG", quality=95)

        # ── human ターンのテキスト組み立て ──
        # "<image>\n" は画像がある場合のみ先頭に付与
        parts = []
        if has_image:
            parts.append("<image>")

        # hint があれば文脈として追加
        hint = (sample.get("hint") or "").strip()
        if hint:
            parts.append(f"ヒント: {hint}")

        # 問題文
        parts.append(sample["question"])

        # 選択肢を "A. xxx\nB. yyy\n..." 形式で追加
        choices_text = "\n".join(
            f"{CHOICE_LABELS[j]}. {choice}"
            for j, choice in enumerate(sample["choices"])
        )
        parts.append(choices_text)

        human_value = "\n".join(parts)

        # ── gpt ターンのテキスト組み立て ──
        answer_idx = sample["answer"]                    # int: 正解の選択肢インデックス
        answer_label = CHOICE_LABELS[answer_idx]         # "A" / "B" / ...
        answer_text = sample["choices"][answer_idx]      # 正解の選択肢テキスト

        solution = (sample.get("solution") or "").strip()
        gpt_value = f"答えは{answer_label}です。{answer_text}"
        if solution:
            gpt_value += f"\n{solution}"

        rows.append({
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt",   "value": gpt_value},
            ],
            "image": image_path,   # str or None
        })

    df = pd.DataFrame(rows)
    print(f"変換完了: {len(df)} サンプル "
          f"(画像あり: {df['image'].notna().sum()}, "
          f"画像なしスキップ: {len(hf_dataset) - len(df)})")
    return df


# ============================================================
# 入出力 Shape 一覧表
# ============================================================
"""
========================================
ファインチューニング時の Shape 遷移
========================================

シナリオ: 448×448 画像1枚 + 質問20トークン + 回答50トークン

| 段階                   | テンソル名              | Shape                    | 説明                              |
|------------------------|------------------------|--------------------------|-----------------------------------|
| 画像入力               | PIL.Image              | (448, 448, 3)            | 元画像                            |
| スマートリサイズ        | PIL.Image              | (448, 448, 3)            | factor=28で割り切れるサイズ        |
| パッチ化               | pixel_values           | (1024, 588)              | N_patches=32×32, C×P²=3×196      |
| グリッド情報           | image_grid_thw         | (1, 3) = [[1,32,32]]     | T=1, H=32, W=32                  |
| テキストトークン化      | input_ids              | (1, T_seq)               | ~80 tokens (text) + 256 IMG_TOK  |
| MRoPE 位置ID          | position_ids           | (3, 1, T_seq)            | [t, h, w] 各次元                  |
| ラベル                 | labels                 | (1, T_seq)               | -100 (img/sys/usr) / ID (asst)   |
| [バッチ化後 B=2]       |                        |                          |                                   |
| バッチ入力             | input_ids              | (2, T_max)               | パディング済み                    |
| バッチラベル           | labels                 | (2, T_max)               | -100 でパディング                 |
| バッチ画像             | pixel_values           | (2048, 588)              | 2枚分の合計パッチ                 |
| バッチ位置ID           | position_ids           | (3, 2, T_max)            | Interleaved MRoPE                 |
| Vision Encoder出力     | visual_features        | (512, 3584)              | N_v=2048/4=512, D_llm=3584        |
| LLM 入力埋め込み       | inputs_embeds          | (2, T_max, 3584)         | テキスト+視覚トークン混合         |
| LLM 出力              | logits                 | (2, T_max, 151936)       | 語彙分布                          |
| 損失                   | loss                   | scalar                   | アシスタント応答トークンのCE損失   |

========================================
ラベルの具体例
========================================
トークン列:   [SYS, user, \\n, IMG×256, ユーザーテキスト, IM_END, \\n, ASST, \\n, 回答, ..., IM_END]
ラベル:       [-100,-100,-100,-100×256,-100×20,          -100,  -100, -100,-100, tok1,...,tok50, 1645 ]
                                                                               ↑ここから損失計算

========================================
推奨ハイパーパラメータ
========================================
| パラメータ          | 推奨値            | 備考                              |
|--------------------|------------------|-----------------------------------|
| learning_rate      | 2e-5             | 公式推奨値 (LLM部分)              |
| batch_size         | 2-4              | GPU VRAMに依存 (80GB GPU推奨)     |
| grad_acc           | 8-16             | 実効BS=16-64程度                  |
| epochs             | 1-3              | データ量に依存                    |
| warmup_ratio       | 0.03             | 全ステップの3%                    |
| scheduler          | cosine           | Cosine + Warmup                  |
| Vision Encoder     | frozen           | SigLIP-2は大量事前学習済み        |
| MLP Merger         | unfrozen         | タスク適応に有効                  |
| LoRA r             | 64               | 公式設定                          |
| LoRA alpha         | 128              | 公式設定                          |
| precision          | bfloat16         | Ampere以降のGPU推奨              |
| max_grad_norm      | 1.0              | 勾配クリッピング                  |

========================================
JSONL データフォーマット例
========================================
{"conversations": [{"from": "human", "value": "<image>\\nこの画像を説明してください"},
                   {"from": "gpt",   "value": "画像には犬が..."}],
 "image": "images/dog.jpg"}

{"conversations": [{"from": "human", "value": "<image>\\nOCRしてください"},
                   {"from": "gpt",   "value": "テキスト: Hello World"}],
 "image": "images/document.png"}

{"conversations": [{"from": "human", "value": "日本語で挨拶してください"},
                   {"from": "gpt",   "value": "こんにちは！"}]}
"""
