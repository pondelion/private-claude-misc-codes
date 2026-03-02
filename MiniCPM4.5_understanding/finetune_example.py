"""
MiniCPM-V 4.5 - 微調整サンプルスクリプト（実動作用）
================================================

pandas DataFrameでデータセットメタデータを受け取り、
transformers Trainerを使わない素のPyTorchループで微調整する。

公式実装:
    - finetune/finetune.py: train()
    - finetune/dataset.py: SupervisedDataset, preprocess(), data_collator()
    - finetune/trainer.py: CPMTrainer.compute_loss()

必要パッケージ:
    pip install torch transformers pillow pandas peft torchvision

使い方:
    python finetune_example.py
"""

"""
============================================================
Shape Convention (形状表記規則)
============================================================
B       : バッチサイズ
L       : シーケンス長 (max_length でクリップ)
L_max   : バッチ内最大シーケンス長
V       : 語彙サイズ
Q       : Resamplerクエリ数 (64)
N_img   : サンプル内の画像スライス数
P       : パッチサイズ (14)
============================================================
"""

import copy
import math
import os
import random
import re
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


# ========================================
# 1. 設定
# ========================================
class FinetuneConfig:
    """微調整の設定"""

    # --- モデル ---
    model_name_or_path: str = "openbmb/MiniCPM-V-2_6"
    # llm_type: "minicpm" / "llama3" / "qwen"
    # MiniCPM-V 2.6 以降は Qwen2 ベース → "qwen"
    llm_type: str = "qwen"

    # --- 学習 ---
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 1  # VLLM系はバッチサイズ1が基本（画像サイズ可変のため）
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 2048
    max_slice_nums: int = 9

    # --- 学習対象 ---
    tune_vision: bool = False  # ViTを学習するか
    tune_llm: bool = True      # LLMを学習するか

    # --- LoRA (オプション) ---
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"

    # --- 保存 ---
    output_dir: str = "./minicpmv_finetuned"
    save_steps: int = 500
    logging_steps: int = 10

    # --- デバイス ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ========================================
# 2. DataFrame → 学習データ変換
# ========================================
def dataframe_to_raw_data(
    df: pd.DataFrame,
    image_col: str = "image_path",
    question_col: str = "question",
    answer_col: str = "answer",
) -> List[Dict]:
    """
    pandas DataFrameを公式フォーマットの学習データに変換する

    ========================================
    入力:
        df: pandas DataFrame
            必須カラム:
                - image_path: str - 画像ファイルパス
                - question: str - ユーザー質問
                - answer: str - アシスタント回答
            オプションカラム:
                - multi_turn: List[Dict] - マルチターン会話
                  (指定時はquestion/answerより優先)

    出力:
        raw_data: List[Dict]
            [
                {
                    "image": "path/to/image.jpg",
                    "conversations": [
                        {"role": "user", "content": "<image>\n質問"},
                        {"role": "assistant", "content": "回答"},
                    ]
                },
                ...
            ]
    ========================================
    """
    raw_data = []
    for _, row in df.iterrows():
        image_path = row[image_col]

        if "multi_turn" in df.columns and pd.notna(row.get("multi_turn")):
            # マルチターン会話がある場合はそのまま使用
            conversations = row["multi_turn"]
        else:
            # シングルターン: question + answer → 会話形式に変換
            question = row[question_col]
            answer = row[answer_col]
            conversations = [
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer},
            ]

        raw_data.append({
            "image": image_path,
            "conversations": conversations,
        })
    return raw_data


# ========================================
# 3. 画像スライス処理
# ========================================
def _ensure_divide(length: int, patch_size: int) -> int:
    """patch_sizeで割り切れるように丸める"""
    return max(round(length / patch_size) * patch_size, patch_size)


def _find_best_resize(
    original_size: Tuple[int, int],
    scale_resolution: int,
    patch_size: int,
    allow_upscale: bool = False,
) -> Tuple[int, int]:
    """
    最適なリサイズ先を計算する

    ========================================
    入力:
        original_size: (W, H)
        scale_resolution: 基準解像度 (448)
        patch_size: ViTパッチサイズ (14)

    出力:
        (best_W, best_H) - patch_sizeで割り切れるサイズ
    ========================================
    """
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    return (_ensure_divide(width, patch_size), _ensure_divide(height, patch_size))


def slice_image(
    image: Image.Image,
    max_slice_nums: int = 9,
    scale_resolution: int = 448,
    patch_size: int = 14,
) -> Tuple[Image.Image, List[List[Image.Image]], Optional[List[int]]]:
    """
    LLaVA-UHD方式で画像をスライス分割する

    公式実装: finetune/dataset.py: slice_image()

    ========================================
    入力:
        image: PIL Image - 元画像
        max_slice_nums: 最大スライス数 (9)
        scale_resolution: 基準解像度 (448)
        patch_size: ViTパッチサイズ (14)

    出力:
        source_image: PIL Image - ダウンスケールされたソース画像
        patches: List[List[PIL Image]] - rows x cols のスライスグリッド
        best_grid: [cols, rows] or None
    ========================================
    """
    original_width, original_height = image.size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    if multiple <= 1:
        best_size = _find_best_resize(image.size, scale_resolution, patch_size, allow_upscale=True)
        return image.resize(best_size, Image.Resampling.BICUBIC), [], None

    # 候補グリッドの列挙
    candidate_grids = []
    for n in [multiple - 1, multiple, multiple + 1]:
        if n <= 1 or n > max_slice_nums:
            continue
        m = 1
        while m <= n:
            if n % m == 0:
                candidate_grids.append([m, n // m])
            m += 1

    # アスペクト比に最も近いグリッドを選択
    best_grid = [1, 1]
    min_error = float("inf")
    for grid in candidate_grids:
        error = abs(log_ratio - math.log(grid[0] / grid[1]))
        if error < min_error:
            best_grid = grid
            min_error = error

    # ソース画像
    best_resize = _find_best_resize(image.size, scale_resolution, patch_size)
    source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)

    # スライス用リサイズ
    grid_x, grid_y = best_grid
    refine_width = _ensure_divide(original_width, grid_x)
    refine_height = _ensure_divide(original_height, grid_y)
    grid_w = refine_width / grid_x
    grid_h = refine_height / grid_y
    best_grid_size = _find_best_resize(
        (grid_w, grid_h), scale_resolution, patch_size, allow_upscale=True
    )
    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
    refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)

    # グリッドに分割
    patches = []
    pw = int(refine_size[0] / grid_x)
    ph = int(refine_size[1] / grid_y)
    for i in range(0, refine_size[1], ph):
        row = []
        for j in range(0, refine_size[0], pw):
            row.append(refine_image.crop((j, i, j + pw, i + ph)))
        patches.append(row)

    return source_image, patches, best_grid


# ========================================
# 4. 会話 → トークン + ラベル変換
# ========================================
def _get_grid_placeholder(tokenizer, grid, query_num: int, new_schema: bool = False) -> str:
    """
    スライスグリッドのプレースホルダ文字列を構築

    ========================================
    入力:
        tokenizer: トークナイザ
        grid: [cols, rows]
        query_num: Q (64)

    出力:
        placeholder: str
            例 (grid=[2,3], new_schema=False):
            <slice_start>
            <im_start><unk>*64<im_end><im_start><unk>*64<im_end>
            <im_start><unk>*64<im_end><im_start><unk>*64<im_end>
            <im_start><unk>*64<im_end><im_start><unk>*64<im_end>
            <slice_end>
    ========================================
    """
    if new_schema:
        image_placeholder = tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
    else:
        image_placeholder = tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end

    cols, rows = grid[0], grid[1]
    slices = []
    for _ in range(rows):
        slices.append(image_placeholder * cols)
    if new_schema:
        return "\n".join(slices)
    else:
        return tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end


def conversation_to_ids(
    conversation: List[Dict],
    tokenizer,
    llm_type: str = "qwen",
    new_schema: bool = False,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    会話をトークン化し、入力ID・ラベル・image_boundを構築する

    公式実装: finetune/dataset.py: conversation_to_ids()

    ========================================
    入力:
        conversation: [{"role": "user", "content": "..."}, ...]
        tokenizer: HuggingFaceトークナイザ
        llm_type: "minicpm" / "qwen"

    出力:
        {
            "input_ids": (L,) - トークンID
            "target": (L,) - ラベル (-100=マスク, token_id=損失対象)
            "image_bound": (N_bound, 2) or [] - 画像トークン位置
            "position_ids": (L,)
        }
    ========================================
    """
    if llm_type == "qwen":
        input_ids, context, raw_msg = _conversation_to_ids_qwen2(conversation, tokenizer)
    else:
        input_ids_list, context_list, raw_msg = _conversation_to_ids_minicpm(conversation, tokenizer)
        input_ids = np.hstack(input_ids_list).astype(np.int32)
        context = np.hstack(context_list).astype(np.int8)

    ids = torch.from_numpy(np.array(input_ids, dtype=np.int32) if not isinstance(input_ids, np.ndarray) else input_ids)
    ctx = torch.from_numpy(np.array(context, dtype=np.int8) if not isinstance(context, np.ndarray) else context)

    if len(ids) > max_length:
        ids = ids[:max_length]
        ctx = ctx[:max_length]

    if torch.all(ctx == 1):
        raise ValueError("No assistant tokens found — cannot compute loss.")

    # ラベル構築: context==0 のトークンが損失計算対象
    target = torch.full_like(ids, -100, dtype=torch.int32)
    for i in range(1, len(ids)):
        if ctx[i] == 0:
            target[i - 1] = ids[i]
        if ctx[i] == 1 and ctx[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_token_id

    # image_bound: <im_start> + 1 ~ <im_end> の位置
    if new_schema:
        start_cond = (ids == tokenizer.im_start_id) | (ids == tokenizer.slice_start_id)
        end_cond = (ids == tokenizer.im_end_id) | (ids == tokenizer.slice_end_id)
    else:
        start_cond = ids == tokenizer.im_start_id
        end_cond = ids == tokenizer.im_end_id

    image_start_tokens = torch.where(start_cond)[0] + 1
    image_end_tokens = torch.where(end_cond)[0]

    if len(image_start_tokens) != len(image_end_tokens):
        raise ValueError(
            f"Mismatched image tokens: {len(image_start_tokens)} starts vs {len(image_end_tokens)} ends"
        )

    if len(image_start_tokens) > 0:
        image_bound = torch.hstack([
            image_start_tokens.unsqueeze(-1),
            image_end_tokens.unsqueeze(-1),
        ])
        # image_bound: (N_bound, 2)
    else:
        image_bound = []

    position_ids = torch.arange(ids.size(0)).long()

    return {
        "input_ids": ids,          # (L,)
        "target": target,          # (L,)
        "image_bound": image_bound,  # (N_bound, 2) or []
        "position_ids": position_ids,  # (L,)
    }


def _conversation_to_ids_minicpm(conversation, tokenizer):
    """MiniCPM形式: <用户>user_msg<AI>assistant_msg"""
    raw_msg = ""
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        prefix = "<用户>" if role == "user" else "<AI>"
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix)[1:]
        message_ids = tokenizer.encode(message)[1:]
        input_ids.append(prefix_ids)
        input_ids.append(message_ids)
        context.append(np.ones(len(prefix_ids), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros(len(message_ids), dtype=np.int8))
        else:
            context.append(np.ones(len(message_ids), dtype=np.int8))
        raw_msg += prefix + message
    return input_ids, context, raw_msg


def _conversation_to_ids_qwen2(conversation, tokenizer):
    """Qwen2形式: <|im_start|>role\nメッセージ<|im_end|>"""
    chat = []
    for msg in conversation:
        role = msg["role"]
        assert role in ["user", "assistant"]
        chat.append({"role": role, "content": msg["content"]})

    enable_thinking = False
    if "<think>" in chat[-1]["content"] and "</think>" in chat[-1]["content"]:
        enable_thinking = True

    raw_msg = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking,
    )
    input_ids = np.array(tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False, enable_thinking=enable_thinking,
    ))

    if "<think>\n\n</think>\n\n" in raw_msg:
        offset = 4
    else:
        offset = 0

    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>"))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("assistant"))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids("<|im_end|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)
    for i, assistant_idx in enumerate(assistant_idxs):
        if assistant_idx - 1 in set(start_idxs):
            if i == len(assistant_idxs) - 1:
                st = assistant_idx + 2 + offset
            else:
                st = assistant_idx + 2
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st:end_idx + 1] = 0
                    break

    return input_ids, context, raw_msg


# ========================================
# 5. データセット
# ========================================
class MiniCPMVDataset(Dataset):
    """
    微調整用データセット

    ========================================
    __getitem__ 出力:
        {
            "input_ids": (L,) - トークンID
            "position_ids": (L,) - 位置ID
            "labels": (L,) - ラベル (-100=マスク)
            "attention_mask": (L,) - 全てTrue
            "pixel_values": List[Tensor(3, H_i, W_i)] - スライスごとの画像テンソル
            "tgt_sizes": Tensor(N_img, 2) or [] - パッチグリッドサイズ
            "image_bound": Tensor(N_bound, 2) or [] - 画像トークン位置
        }
    ========================================
    """

    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer,
        transform,
        slice_config: Dict,
        llm_type: str = "qwen",
        patch_size: int = 14,
        query_nums: int = 64,
        batch_vision: bool = False,
        max_length: int = 2048,
    ):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.llm_type = llm_type
        self.patch_size = patch_size
        self.query_nums = query_nums
        self.batch_vision = batch_vision
        self.max_length = max_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            sample = self.raw_data[idx]

            # --- 画像読み込み ---
            if isinstance(sample["image"], str):
                images_dict = {"<image>": Image.open(sample["image"]).convert("RGB")}
            elif isinstance(sample["image"], dict):
                images_dict = {
                    name: Image.open(path).convert("RGB")
                    for name, path in sample["image"].items()
                }
            else:
                raise ValueError(f"Unsupported image type: {type(sample['image'])}")

            # --- 前処理 ---
            ret = self._preprocess(images_dict, sample["conversations"])
            return {
                "input_ids": ret["input_ids"],
                "position_ids": ret["position_ids"],
                "labels": ret["target"],
                "attention_mask": torch.ones_like(ret["input_ids"], dtype=torch.bool),
                "pixel_values": ret["pixel_values"],
                "tgt_sizes": ret["tgt_sizes"],
                "image_bound": ret["image_bound"],
            }
        except Exception as e:
            print(f"[Warning] Error at index {idx}: {e}. Sampling random item.")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _preprocess(
        self,
        images_dict: Dict[str, Image.Image],
        conversations: List[Dict],
    ) -> Dict:
        """
        画像と会話を前処理してモデル入力を構築する

        ========================================
        処理の流れ:
            1. 各画像をスライス分割
            2. プレースホルダ構築 (<im_start><unk>*Q<im_end>)
            3. 会話テキスト中の <image> をプレースホルダに置換
            4. トークン化 + ラベル構築
            5. batch_vision時: reshape_by_patch + tgt_sizes計算
        ========================================
        """
        conversations = copy.deepcopy(conversations)

        new_schema = (self.llm_type == "qwen")
        use_image_id = (self.llm_type == "qwen")
        default_image_placeholder = (
            self.tokenizer.im_start + self.tokenizer.unk_token * self.query_nums + self.tokenizer.im_end
        )

        # --- スライス分割 + プレースホルダ構築 ---
        image_placeholder_dict = {}
        images = []
        image_id_cnt = 0
        for img_name, image in images_dict.items():
            source_image, patches, best_grid = slice_image(
                image,
                self.slice_config["max_slice_nums"],
                self.slice_config["scale_resolution"],
                self.slice_config["patch_size"],
            )
            images.append(source_image)
            placeholder = default_image_placeholder

            if use_image_id:
                placeholder = f"{self.tokenizer.im_id_start}{image_id_cnt}{self.tokenizer.im_id_end}" + placeholder
                image_id_cnt += 1

            if patches:
                for row in patches:
                    for patch in row:
                        images.append(patch)
                placeholder += _get_grid_placeholder(
                    self.tokenizer, best_grid, self.query_nums, new_schema=new_schema
                )
            image_placeholder_dict[img_name] = placeholder

        # --- 画像変換 ---
        images = [self.transform(img) for img in images]
        # images: List[Tensor(3, H_i, W_i)]

        # --- プレースホルダの置換 ---
        if len(images_dict) == 1 and "<image>" in images_dict:
            if "<image>" in conversations[0]["content"]:
                conversations[0]["content"] = conversations[0]["content"].replace(
                    "<image>", image_placeholder_dict["<image>"]
                )
            else:
                conversations[0]["content"] = (
                    image_placeholder_dict["<image>"] + "\n" + conversations[0]["content"]
                )
        else:
            pattern = r"<image_\d+>"
            for conv in conversations:
                parts = re.split(f"({pattern})", conv["content"])
                for i, part in enumerate(parts):
                    if re.match(pattern, part) and part in image_placeholder_dict:
                        parts[i] = image_placeholder_dict[part]
                conv["content"] = "\n".join([p for p in parts if p.strip()])

        # --- トークン化 ---
        input_dict = conversation_to_ids(
            conversations, self.tokenizer, self.llm_type,
            new_schema=new_schema, max_length=self.max_length,
        )

        # --- batch_vision: パッチ再配置 ---
        if self.batch_vision:
            tgt_sizes = []
            reshape_images = []
            for image_tensor in images:
                H, W = image_tensor.shape[1:]
                # (3, H, W) → unfold → (3, P, H*W/P)
                reshape_img = torch.nn.functional.unfold(
                    image_tensor.unsqueeze(0),
                    (self.patch_size, self.patch_size),
                    stride=(self.patch_size, self.patch_size),
                )[0]
                reshape_img = reshape_img.reshape(3, self.patch_size, self.patch_size, -1)
                reshape_img = reshape_img.permute(0, 1, 3, 2).reshape(3, self.patch_size, -1)
                # reshape_img: (3, P, H*W/P^2)
                reshape_images.append(reshape_img)
                tgt_sizes.append([H // self.patch_size, W // self.patch_size])
            input_dict["pixel_values"] = reshape_images
            input_dict["tgt_sizes"] = torch.tensor(tgt_sizes, dtype=torch.int32) if tgt_sizes else []
        else:
            input_dict["pixel_values"] = images
            input_dict["tgt_sizes"] = []

        return input_dict


# ========================================
# 6. Data Collator
# ========================================
def collate_fn(
    examples: List[Dict],
    padding_value: int = 0,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    バッチのパディングと照合

    ========================================
    入力:
        examples: List[Dict] - MiniCPMVDataset.__getitem__の出力リスト

    出力:
        {
            "input_ids": (B, L_max)
            "position_ids": (B, L_max)
            "labels": (B, L_max) - パディング位置は-100
            "attention_mask": (B, L_max) - パディング位置は0
            "pixel_values": List[List[Tensor(3, H_i, W_i)]]
            "image_bound": List[Tensor(N_bound, 2)]
            "tgt_sizes": List[Tensor or []]
        }
    ========================================
    """
    def trim_and_pad(seqs, pad_val):
        return pad_sequence(
            [s[:max_length] for s in seqs], batch_first=True, padding_value=pad_val
        )

    return {
        "input_ids": trim_and_pad([e["input_ids"] for e in examples], padding_value),
        "position_ids": trim_and_pad([e["position_ids"] for e in examples], padding_value),
        "labels": trim_and_pad([e["labels"] for e in examples], -100),
        "attention_mask": trim_and_pad([e["attention_mask"] for e in examples], 0),
        "pixel_values": [e["pixel_values"] for e in examples],
        "image_bound": [e["image_bound"] for e in examples],
        "tgt_sizes": [e["tgt_sizes"] for e in examples],
    }


# ========================================
# 7. 画像変換
# ========================================
def build_transform():
    """
    学習用画像変換パイプラインを構築する

    公式実装: finetune/finetune.py: build_transform()

    ========================================
    処理:
        1. ToTensor: PIL Image → Tensor (0~1)
        2. Normalize: ImageNet inception統計量で正規化
           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)

    出力:
        transform: Callable[PIL Image → Tensor(3, H, W)]
    ========================================
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


# ========================================
# 8. 損失計算
# ========================================
def compute_loss(
    model,
    batch: Dict[str, torch.Tensor],
    vocab_size: int,
) -> torch.Tensor:
    """
    モデルのフォワードパス + CrossEntropyLoss

    公式実装: finetune/trainer.py: CPMTrainer.compute_loss()

    ========================================
    入力:
        model: MiniCPM-V モデル
        batch: collate_fnの出力 (labelsキーを含む)
        vocab_size: 語彙サイズ

    出力:
        loss: スカラー - クロスエントロピー損失

    処理:
        1. labelsをbatchから分離（モデルのforward内で使わないため）
        2. model(data=batch_without_labels) でlogits取得
        3. logitsとlabelsからCrossEntropyLoss計算
           - labels=-100 の位置は損失から自動除外

    形状:
        logits: (B, L_max, V) → flatten → (B*L_max, V)
        labels: (B, L_max) → flatten → (B*L_max,)
        loss: スカラー
    ========================================
    """
    labels = batch.pop("labels")
    # labels: (B, L_max)

    outputs = model(data=batch, use_cache=False)
    # outputs.logits: (B, L_max, V)

    logits = outputs.logits
    logits = logits.view(-1, vocab_size).contiguous()
    # logits: (B*L_max, V)

    labels = labels.view(-1).long().contiguous().to(logits.device)
    # labels: (B*L_max,)

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels)
    # loss: スカラー (labels=-100 の位置はignore_index=-100で自動除外)

    return loss


# ========================================
# 9. 学習率スケジューラ（cosine with warmup）
# ========================================
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    ========================================
    Warmup + Cosine Decay スケジューラ

    lr(step) =
        step < warmup: lr_base * step / warmup
        step >= warmup: lr_base * 0.5 * (1 + cos(pi * (step - warmup) / (total - warmup)))
    ========================================
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ========================================
# 10. メイン学習ループ
# ========================================
def train(
    df: pd.DataFrame,
    config: FinetuneConfig = None,
    image_col: str = "image_path",
    question_col: str = "question",
    answer_col: str = "answer",
):
    """
    MiniCPM-V 4.5 の微調整を実行する

    ========================================
    引数:
        df: pandas DataFrame - 学習データのメタデータ
            必須カラム: image_path, question, answer
        config: FinetuneConfig - 学習設定
        image_col, question_col, answer_col: DataFrameのカラム名

    処理の流れ:
        1. モデル・トークナイザの読み込み
        2. 学習対象パラメータの設定 (LoRA or full)
        3. DataFrame → Dataset → DataLoader
        4. Optimizerとスケジューラの設定
        5. 学習ループ (forward → loss → backward → step)
        6. チェックポイント保存
    ========================================
    """
    if config is None:
        config = FinetuneConfig()

    print(f"[Config] model={config.model_name_or_path}")
    print(f"[Config] device={config.device}, dtype={config.dtype}")
    print(f"[Config] lr={config.learning_rate}, epochs={config.num_epochs}")
    print(f"[Config] batch_size={config.batch_size}, grad_accum={config.gradient_accumulation_steps}")

    # ========================================
    # 10.1 モデル読み込み
    # ========================================
    print("[1/6] Loading model...")
    model = AutoModel.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=config.dtype,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True
    )

    # ========================================
    # 10.2 学習対象の設定
    # ========================================
    print("[2/6] Configuring trainable parameters...")
    if not config.tune_vision:
        model.vpm.requires_grad_(False)
    if not config.tune_llm:
        model.llm.requires_grad_(False)

    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        if config.tune_llm:
            raise ValueError("Cannot use LoRA and tune_llm=True simultaneously")

        for param in model.llm.parameters():
            param.requires_grad = False

        modules_to_save = ["embed_tokens", "resampler"]
        if config.tune_vision:
            modules_to_save.append("vpm")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)

    model.to(config.device)
    model.train()

    # パラメータ数の表示
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.2f}%)")

    # ========================================
    # 10.3 データセット構築
    # ========================================
    print("[3/6] Building dataset...")
    raw_data = dataframe_to_raw_data(df, image_col, question_col, answer_col)
    print(f"  Samples: {len(raw_data)}")

    # slice_config: モデルのconfigから取得
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = config.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        slice_config = {
            "max_slice_nums": config.max_slice_nums,
            "scale_resolution": 448,
            "patch_size": 14,
        }

    batch_vision = getattr(model.config, "batch_vision_input", False)

    dataset = MiniCPMVDataset(
        raw_data=raw_data,
        tokenizer=tokenizer,
        transform=build_transform(),
        slice_config=slice_config,
        llm_type=config.llm_type,
        patch_size=getattr(model.config, "patch_size", 14),
        query_nums=getattr(model.config, "query_num", 64),
        batch_vision=batch_vision,
        max_length=config.max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, max_length=config.max_length),
        num_workers=2,
        pin_memory=True,
    )

    # ========================================
    # 10.4 Optimizer + Scheduler
    # ========================================
    print("[4/6] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    num_training_steps = (
        len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    print(f"  Total steps: {num_training_steps}, Warmup: {num_warmup_steps}")

    vocab_size = model.config.vocab_size

    # ========================================
    # 10.5 学習ループ
    # ========================================
    print("[5/6] Training...")
    os.makedirs(config.output_dir, exist_ok=True)
    global_step = 0
    accumulation_loss = 0.0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            # --- バッチをデバイスへ転送 ---
            batch["input_ids"] = batch["input_ids"].to(config.device)
            batch["position_ids"] = batch["position_ids"].to(config.device)
            batch["labels"] = batch["labels"].to(config.device)
            batch["attention_mask"] = batch["attention_mask"].to(config.device)
            # pixel_values, image_bound, tgt_sizes はリストのまま
            # (モデル内部でデバイスに転送される)

            # --- Forward + Loss ---
            loss = compute_loss(model, batch, vocab_size)
            # loss: スカラー

            # --- Gradient accumulation ---
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            accumulation_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                # --- Gradient clipping ---
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )

                # --- Optimizer step ---
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg_loss = accumulation_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"  [Epoch {epoch+1}/{config.num_epochs}] "
                        f"Step {global_step}/{num_training_steps} "
                        f"Loss={avg_loss:.4f} LR={lr:.2e}"
                    )
                    accumulation_loss = 0.0

                # --- チェックポイント保存 ---
                if config.save_steps > 0 and global_step % config.save_steps == 0:
                    save_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    _save_checkpoint(model, tokenizer, save_dir, config.use_lora)

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_batches += 1

            # メモリ解放
            del loss
            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} completed. Avg Loss={avg_epoch_loss:.4f}")

    # ========================================
    # 10.6 最終モデル保存
    # ========================================
    print("[6/6] Saving final model...")
    _save_checkpoint(model, tokenizer, config.output_dir, config.use_lora)
    print(f"  Model saved to {config.output_dir}")
    print("Done!")


def _save_checkpoint(model, tokenizer, output_dir: str, use_lora: bool = False):
    """
    モデルとトークナイザをチェックポイントとして保存する

    ========================================
    LoRA使用時: adapter_model のみ保存
    Full fine-tune時: モデル全体を保存
    ========================================
    """
    os.makedirs(output_dir, exist_ok=True)
    if use_lora:
        model.save_pretrained(output_dir)
    else:
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    print(f"  Checkpoint saved to {output_dir}")


# ========================================
# 使用例
# ========================================
def example_usage():
    """
    微調整のデモ

    実行方法:
        python finetune_example.py
    """
    print("=== MiniCPM-V 4.5 微調整サンプル ===\n")

    # --- 1. 学習データをDataFrameで準備 ---
    # 実際には CSVやParquet から読み込む:
    #   df = pd.read_csv("train_data.csv")
    #   df = pd.read_parquet("train_data.parquet")

    df = pd.DataFrame([
        {
            "image_path": "/path/to/image1.jpg",
            "question": "この画像に何が写っていますか？",
            "answer": "赤い車が道路を走っている場面です。",
        },
        {
            "image_path": "/path/to/image2.jpg",
            "question": "この建物の特徴を教えてください。",
            "answer": "ゴシック様式の大聖堂で、尖塔とステンドグラスが特徴的です。",
        },
        {
            "image_path": "/path/to/image3.jpg",
            "question": "グラフの傾向を分析してください。",
            "answer": "2020年から2023年にかけて売上が約30%増加しています。",
        },
    ])

    print("学習データ:")
    print(df.to_string(index=False))
    print()

    # --- 2. 設定 ---
    config = FinetuneConfig()
    config.model_name_or_path = "openbmb/MiniCPM-V-2_6"
    config.num_epochs = 3
    config.learning_rate = 1e-5
    config.batch_size = 1
    config.gradient_accumulation_steps = 8
    config.max_length = 2048
    config.max_slice_nums = 9
    config.tune_vision = False
    config.tune_llm = True
    config.use_lora = False
    config.output_dir = "./minicpmv_finetuned"

    # --- LoRAを使う場合 ---
    # config.use_lora = True
    # config.tune_llm = False  # LoRA使用時はtune_llm=Falseにする
    # config.lora_r = 64
    # config.lora_alpha = 64

    print("設定:")
    print(f"  Model: {config.model_name_or_path}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LR: {config.learning_rate}")
    print(f"  LoRA: {config.use_lora}")
    print()

    # --- 3. 学習実行 ---
    # 実際に実行するには画像ファイルが必要
    # train(df, config)

    print("train(df, config) を呼び出すと学習が開始されます。")
    print("（実行には画像ファイルとGPUが必要）")


if __name__ == "__main__":
    example_usage()
