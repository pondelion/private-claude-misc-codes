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
    pip install torch transformers pillow pandas peft torchvision tqdm

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# ========================================
# 1. 設定
# ========================================
class FinetuneConfig:
    """微調整の設定"""

    # --- モデル ---
    model_name_or_path: str = "openbmb/MiniCPM-V-4_5-AWQ"
    # llm_type: "minicpm" / "llama3" / "qwen"
    # MiniCPM-V 4.5 は Qwen2.5 ベース → "qwen"
    # ※ -AWQ 版 (openbmb/MiniCPM-V-4_5-AWQ) は LoRA finetune 可能 (autoawq 要)
    #   ただし quantization= オプションとの併用不可 (AWQ+BnB の二重量子化はクラッシュ)
    #   VRAM が不足する場合: 本モデル + quantization="4bit" で QLoRA を推奨
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
    tune_llm: bool = False      # LLMを学習するか

    # --- LoRA (オプション) ---
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"

    # --- 保存・ログ ---
    output_dir: str = "./minicpmv_finetuned"
    save_steps: int = 500    # チェックポイント保存間隔 (optimizer step 単位)
    eval_steps: int = 200    # 評価間隔 (optimizer step 単位)
    logging_steps: int = 10  # ロス表示 + 生成プレビュー間隔 (optimizer step 単位)

    # --- 量子化 (QLoRA) ---
    quantization: Optional[str] = None  # '4bit' / '8bit' / None
    #
    # ============================================================
    # 量子化の仕組み: 何が量子化されるか
    # ============================================================
    #
    # │ 設定                        │ 元モデル重み        │ LoRA重み   │ 計算精度 │
    # │-----------------------------│---------------------│-----------│---------|
    # │ AWQモデル + quantization=None│ 4bit INT (AWQ)      │ bf16 (全精度) │ bf16   │
    # │ ベース + quantization="4bit" │ 4bit NF4 (BnB)      │ bf16 (全精度) │ bf16   │
    # │ ベース + quantization="8bit" │ 8bit INT (BnB)      │ bf16 (全精度) │ bf16   │
    # │ ベース + quantization=None   │ bf16 (量子化なし)   │ bf16 (全精度) │ bf16   │
    # │ AWQモデル + quantization設定 │ → 自動で None に上書き (二重量子化防止)     │
    # ============================================================
    # ポイント:
    #   - 量子化されるのは「凍結された元モデルの重み」のみ
    #   - LoRAパラメータは常に bf16 全精度 (勾配計算が必要なため)
    #   - 計算時は 4bit重み → bf16 に dequantize してから演算
    #   - AWQ は calibration データを使った高品質な量子化 (BnB NF4 より僅かに高精度)
    #   - メモリ: AWQ ≈ BnB 4bit < BnB 8bit << 量子化なし
    # ============================================================

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
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    ========================================
    Warmup + Cosine Decay スケジューラ

    lr(step) =
        step < warmup: lr_base * step / warmup
        step >= warmup: lr_base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + cos(pi * progress)))

    min_lr_ratio=0.0 で最終的に lr → 0
    ========================================
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


# ========================================
# 10. AMP (混合精度) 設定
# ========================================
def setup_amp(
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Dict, GradScaler]:
    """
    Mixed Precision (AMP) の設定

    ========================================
    戦略
    ========================================
    bfloat16: torch.autocast(dtype=bf16), GradScaler 無効 (bf16 は安定)
    float16:  torch.autocast(dtype=fp16), GradScaler 有効 (スケーリング必要)
    float32:  torch.autocast(enabled=False), GradScaler 無効

    ========================================
    出力
    ========================================
    amp_ctx_kwargs : dict      - torch.autocast に渡す kwargs
    scaler         : GradScaler - fp16 時のみ有効、それ以外は no-op
    """
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = dtype in (torch.bfloat16, torch.float16)
    use_fp16 = dtype == torch.float16

    amp_ctx_kwargs = dict(
        device_type=device_type,
        dtype=dtype if use_amp else torch.float32,
        enabled=use_amp,
    )

    # bf16 は GradScaler 不要 (enabled=False で no-op)
    scaler = GradScaler(device_type, enabled=use_fp16)

    return amp_ctx_kwargs, scaler


# ========================================
# 11. モデルロード
# ========================================
def load_model_for_finetuning(config: FinetuneConfig = None):
    """
    FinetuneConfig からモデルとトークナイザをロードし、学習対象パラメータを設定する。

    ========================================
    出力
    ========================================
    model     : ファインチューニング可能な状態のモデル (CPUまたはGPU)
    tokenizer : AutoTokenizer

    ========================================
    モデル構成の選択肢
    ========================================
    FullFT (デフォルト):
        tune_llm=True, tune_vision=False
        → LLM 全体を学習, ViT は凍結

    LoRA:
        use_lora=True, tune_llm=False
        → LLM の attention 投影層のみ LoRA, ViT は凍結

    VisionFT:
        tune_vision=True, tune_llm=False
        → ViT のみ学習
    """
    if config is None:
        config = FinetuneConfig()

    print(f"モデルロード: {config.model_name_or_path}")
    print(f"dtype: {config.dtype}")

    # AWQ モデルの自動検出
    # AWQ は事前量子化済みのため BitsAndBytes と併用不可 (二重量子化)
    # モデル名に "awq" が含まれる場合は quantization を強制的に None にする
    _is_awq = 'awq' in config.model_name_or_path.lower()
    _quantization = config.quantization
    if _is_awq and _quantization is not None:
        print(f"  ⚠ AWQ モデルが検出されました。quantization={_quantization!r} を無視して None に上書きします。")
        _quantization = None

    # 量子化設定 (QLoRA / BitsAndBytes)
    _bnb_cfg = None
    if _quantization is not None:
        from transformers import BitsAndBytesConfig
        if _quantization == '4bit':
            _bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=config.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        elif _quantization == '8bit':
            _bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"quantization は '4bit' / '8bit' / None のみ有効: {_quantization!r}")
        print(f"量子化: {_quantization} (QLoRA / BitsAndBytes)")
    elif _is_awq:
        print("量子化: AWQ (事前量子化済みモデル、autoawq 要)")

    # AWQ または BnB 量子化時は device_map="auto" が必須 (.to(device) 不可)
    _device_map = "auto" if (_is_awq or _quantization is not None) else None

    # AWQ + trust_remote_code 互換性パッチ:
    # MiniCPMVConfig がクラス属性 quantization_config=None を持つため
    # transformers 5.x の hasattr() 検出 → .get() 呼び出しでクラッシュする。
    # config を事前ロードして None をパッチすることで回避する。
    _pretrained_cfg = None
    if _is_awq:
        from transformers import AutoConfig
        _pretrained_cfg = AutoConfig.from_pretrained(
            config.model_name_or_path, trust_remote_code=True
        )
        if getattr(_pretrained_cfg, 'quantization_config', 'sentinel') is None:
            _pretrained_cfg.quantization_config = {}

    model = AutoModel.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        dtype=config.dtype,
        device_map=_device_map,
        quantization_config=_bnb_cfg,
        **({"config": _pretrained_cfg} if _pretrained_cfg is not None else {}),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, trust_remote_code=True
    )

    # ---- パラメータ凍結設定 ----
    if not config.tune_vision:
        model.vpm.requires_grad_(False)
    if not config.tune_llm:
        model.llm.requires_grad_(False)

    # 量子化後処理: kbit 学習の準備 (LoRA 前に必須)
    # BnB 量子化モデルと AWQ モデルの両方に適用
    if _quantization is not None or _is_awq:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("prepare_model_for_kbit_training 完了")

    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        if config.tune_llm:
            raise ValueError("LoRA 使用時は tune_llm=False にしてください")

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

    model.config.use_cache = False  # 訓練時はKVキャッシュを無効化

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"訓練可能パラメータ: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    return model, tokenizer


# ========================================
# 12. 評価関数
# ========================================
@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    eval_loader: DataLoader,
    device: torch.device,
    amp_ctx_kwargs: Dict,
    vocab_size: int,
) -> Dict:
    """
    評価データセットでの平均損失を計算する。

    ========================================
    出力
    ========================================
    {"loss": float}

    ========================================
    注意
    ========================================
    compute_loss() が batch.pop("labels") でバッチを破壊するため、
    各バッチを dict() でシャローコピーしてから渡す。
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(eval_loader, desc="Eval", leave=False):
        # テンソルをデバイスへ転送
        batch = {
            "input_ids":      batch["input_ids"].to(device),
            "position_ids":   batch["position_ids"].to(device),
            "labels":         batch["labels"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pixel_values":   batch["pixel_values"],   # List のまま (モデル内部で転送)
            "image_bound":    batch["image_bound"],
            "tgt_sizes":      batch["tgt_sizes"],
        }

        with torch.autocast(**amp_ctx_kwargs):
            # compute_loss は batch.pop("labels") するため shallow copy で渡す
            loss = compute_loss(model, dict(batch), vocab_size)

        total_loss += loss.item()
        num_batches += 1

    model.train()
    return {"loss": total_loss / max(num_batches, 1)}


# ========================================
# 13. 単一サンプル推論
# ========================================
def generate_response(
    model,
    tokenizer,
    image_path: Optional[str] = None,
    question: str = "",
    max_new_tokens: int = 512,
    device: Optional[torch.device] = None,
) -> str:
    """
    単一サンプルの推論 (ロギング用)

    MiniCPM-V の model.chat() API を使用する。

    ========================================
    Shape
    ========================================
    入力:
        image_path: str (画像パス)
        question:   str (質問テキスト)

    内部:
        msgs[0]["content"]: [PIL.Image, str]
        → model.chat() が内部でスライス分割・Resampler処理

    出力:
        response : str - 生成テキスト
    """
    if device is None:
        device = next(model.parameters()).device

    # ---- メッセージ構築 ----
    content = []
    if image_path and Path(image_path).exists():
        pil_img = Image.open(image_path).convert("RGB")
        content.append(pil_img)
    content.append(question)

    msgs = [{"role": "user", "content": content}]

    model.eval()
    with torch.no_grad():
        response = model.chat(
            image=None,  # content に PIL.Image を直接入れる場合は None
            msgs=msgs,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            sampling=False,  # greedy decoding
        )

    return response


# ========================================
# 14. 学習中サンプル生成表示
# ========================================
def log_sample_predictions(
    model,
    tokenizer,
    sample_df: pd.DataFrame,
    device: torch.device,
    global_step: int,
    image_col: str = "image_path",
    question_col: str = "question",
    answer_col: str = "answer",
    max_new_tokens: int = 128,
    display_size: tuple = (300, 200),
):
    """
    logging_steps ごとに val サンプルを generate して画像・質問・生成文を表示する。

    呼び出し元で eval_df から固定サンプルを切り出して渡すことで、
    ステップをまたいで同じサンプルに対する生成変化を追える。

        # 呼び出し側での準備例
        log_preview_df = eval_df.sample(n=2, random_state=42).reset_index(drop=True)
        # logging_steps ごとに
        log_sample_predictions(model, tokenizer, log_preview_df, device, global_step)

    DataFrame カラム:
        image_path : str  画像ファイルパス
        question   : str  質問テキスト
        answer     : str  正解テキスト

    引数:
        sample_df     : 表示対象の固定サンプル DataFrame (呼び出し元で固定しておく)
        global_step   : 現在のステップ数 (表示用ラベル)
        image_col     : 画像パスカラム名
        question_col  : 質問カラム名
        answer_col    : 正解カラム名
        max_new_tokens: 生成の最大トークン数 (ログ用なので短めでよい)
        display_size  : Jupyter 表示時のリサイズ後サイズ (width, height)
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"[Step {global_step}] val サンプル生成プレビュー")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(sample_df.iterrows()):
        image_path   = row.get(image_col, None)
        question_raw = row.get(question_col, "")
        ground_truth = row.get(answer_col, "")

        print(f"\n--- サンプル {i + 1} ---")
        print(f"質問: {str(question_raw)[:120]}{'...' if len(str(question_raw)) > 120 else ''}")
        print(f"正解: {str(ground_truth)[:120]}{'...' if len(str(ground_truth)) > 120 else ''}")

        # 画像表示 (Jupyter のみ、失敗時はパスを print してフォールバック)
        if image_path and Path(str(image_path)).exists():
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
                tokenizer=tokenizer,
                image_path=image_path if (image_path and Path(str(image_path)).exists()) else None,
                question=str(question_raw),
                max_new_tokens=max_new_tokens,
                device=device,
            )
            print(f"生成: {response[:200]}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"生成エラー: {e}")

    print(f"{'='*60}\n")
    model.train()


# ========================================
# 15. メイン学習ループ
# ========================================
def train(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    log_preview_df: Optional[pd.DataFrame] = None,
    config: FinetuneConfig = None,
    image_col: str = "image_path",
    question_col: str = "question",
    answer_col: str = "answer",
    save_checkpoint: bool = False,
):
    """
    MiniCPM-V 4.5 の微調整学習ループ。

    引数:
        model          : load_model_for_finetuning() でロードしたモデル
        tokenizer      : 対応するトークナイザ
        train_df       : pd.DataFrame  学習データ (image_path, question, answer 必須)
        eval_df        : pd.DataFrame  評価データ (省略可)
        log_preview_df : pd.DataFrame  logging_steps ごとに generate 表示する固定サンプル。
                         None の場合はスキップ。eval_df から事前にサンプリングして渡す:
                             log_preview_df = eval_df.sample(n=2, random_state=42)
        config         : FinetuneConfig  学習設定
        image_col      : 画像パスカラム名
        question_col   : 質問カラム名
        answer_col     : 正解カラム名

    ========================================
    処理の流れ
    ========================================
        1. DataFrame → Dataset → DataLoader
        2. Optimizer + Cosine Warmup Scheduler
        3. AMP (GradScaler) 設定
        4. 初期評価 + 生成プレビュー
        5. 学習ループ (forward → loss → backward → clip → step)
        6. logging_steps ごとにロス表示 + 生成プレビュー
        7. eval_steps ごとに評価
        8. save_steps ごとにチェックポイント保存
        9. エポック末評価
       10. 最終モデル保存
    """
    if config is None:
        config = FinetuneConfig()

    device = torch.device(config.device)

    # ========================================
    # 1. データセット構築
    # ========================================
    raw_train = dataframe_to_raw_data(train_df, image_col, question_col, answer_col)

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
    patch_size   = getattr(model.config, "patch_size", 14)
    query_nums   = getattr(model.config, "query_num", 64)
    vocab_size   = model.config.vocab_size

    _collate = partial(collate_fn, max_length=config.max_length)

    train_dataset = MiniCPMVDataset(
        raw_data=raw_train,
        tokenizer=tokenizer,
        transform=build_transform(),
        slice_config=slice_config,
        llm_type=config.llm_type,
        patch_size=patch_size,
        query_nums=query_nums,
        batch_vision=batch_vision,
        max_length=config.max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    eval_loader = None
    if eval_df is not None:
        raw_eval = dataframe_to_raw_data(eval_df, image_col, question_col, answer_col)
        eval_dataset = MiniCPMVDataset(
            raw_data=raw_eval,
            tokenizer=tokenizer,
            transform=build_transform(),
            slice_config=slice_config,
            llm_type=config.llm_type,
            patch_size=patch_size,
            query_nums=query_nums,
            batch_vision=batch_vision,
            max_length=config.max_length,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

    # ========================================
    # 2. Optimizer + Scheduler
    # ========================================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    num_training_steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # ========================================
    # 3. AMP (GradScaler)
    # ========================================
    amp_ctx_kwargs, scaler = setup_amp(config.dtype, device)

    # ========================================
    # 設定サマリー表示
    # ========================================
    precision_name = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(config.dtype, "float32")
    print(f"\n訓練設定:")
    print(f"  サンプル数:          train={len(train_dataset)}, eval={len(eval_df) if eval_df is not None else 0}")
    print(f"  バッチサイズ:        {config.batch_size}  (実効: {config.batch_size * config.gradient_accumulation_steps})")
    print(f"  エポック数:          {config.num_epochs}")
    print(f"  総ステップ数:        {num_training_steps}  (warmup: {num_warmup_steps})")
    print(f"  学習率:              {config.learning_rate}")
    print(f"  Precision:           {precision_name}")
    print(f"  訓練可能パラメータ:  {sum(p.numel() for p in trainable_params):,}")
    print(f"  デバイス:            {device}")
    print("-" * 60)

    # 量子化モデル (device_map="auto") はすでに GPU に配置されているため .to() 不可
    # BnB: is_loaded_in_4bit / is_loaded_in_8bit、AWQ: モデル名で判定
    _is_quantized = (
        getattr(model, 'is_loaded_in_4bit', False)
        or getattr(model, 'is_loaded_in_8bit', False)
        or 'awq' in getattr(config, 'model_name_or_path', '').lower()
    )
    if not _is_quantized:
        model = model.to(device)
    model.train()
    os.makedirs(config.output_dir, exist_ok=True)

    # ========================================
    # 4. 初期評価
    # ========================================
    optimizer_step = 0
    if eval_loader is not None:
        init_eval = evaluate(model, tokenizer, eval_loader, device, amp_ctx_kwargs, vocab_size)
        print(f"初期評価  loss: {init_eval['loss']:.4f}")
    if log_preview_df is not None:
        log_sample_predictions(
            model, tokenizer, log_preview_df, device, optimizer_step,
            image_col=image_col, question_col=question_col, answer_col=answer_col,
        )

    # ========================================
    # 5. 学習ループ
    # ========================================
    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(config.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            # ---- バッチをデバイスへ転送 ----
            batch = {
                "input_ids":      batch["input_ids"].to(device),
                "position_ids":   batch["position_ids"].to(device),
                "labels":         batch["labels"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "pixel_values":   batch["pixel_values"],   # List のまま
                "image_bound":    batch["image_bound"],
                "tgt_sizes":      batch["tgt_sizes"],
            }

            # ---- フォワードパス ----
            with torch.autocast(**amp_ctx_kwargs):
                loss = compute_loss(model, dict(batch), vocab_size)
                # compute_loss は batch.pop("labels") するため dict() でコピー

            # 勾配累積: grad_acc で割って平均化
            loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

            global_step += 1

            is_update_step = (
                global_step % config.gradient_accumulation_steps == 0
                or global_step == len(train_loader)
            )
            if is_update_step:
                # ---- 勾配クリップ + Optimizer Step ----
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(
                    loss=f"{accumulated_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    step=optimizer_step,
                )

                # ---- ロギング + 生成プレビュー ----
                if optimizer_step % config.logging_steps == 0:
                    avg_loss = accumulated_loss / config.logging_steps
                    print(
                        f"  [Step {optimizer_step}/{num_training_steps}] "
                        f"loss={avg_loss:.4f}, lr={current_lr:.2e}"
                    )
                    accumulated_loss = 0.0

                    if log_preview_df is not None:
                        log_sample_predictions(
                            model, tokenizer, log_preview_df, device, optimizer_step,
                            image_col=image_col, question_col=question_col, answer_col=answer_col,
                        )

                # ---- 中間評価 ----
                if eval_loader is not None and optimizer_step % config.eval_steps == 0:
                    eval_result = evaluate(model, tokenizer, eval_loader, device, amp_ctx_kwargs, vocab_size)
                    print(f"  [Eval Step {optimizer_step}] eval_loss={eval_result['loss']:.4f}")
                    model.train()

                # ---- チェックポイント保存 ----
                if save_checkpoint and config.save_steps > 0 and optimizer_step % config.save_steps == 0:
                    ckpt_path = os.path.join(config.output_dir, f"checkpoint-{optimizer_step}")
                    _save_checkpoint(model, tokenizer, ckpt_path, config.use_lora)

            # メモリ解放
            torch.cuda.empty_cache()

        # ---- エポック末評価 ----
        if eval_loader is not None:
            epoch_eval = evaluate(model, tokenizer, eval_loader, device, amp_ctx_kwargs, vocab_size)
            print(f"  Epoch {epoch + 1} 評価 loss: {epoch_eval['loss']:.4f}")

    # ========================================
    # 6. 最終モデル保存
    # ========================================
    if save_checkpoint:
        _save_checkpoint(model, tokenizer, config.output_dir, config.use_lora)
        print(f"\n学習完了。最終モデル保存: {config.output_dir}")


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
    print(f"  チェックポイント保存: {output_dir}")


# ========================================
# 16. ScienceQA データセット変換
# ========================================

def convert_scienceqa_to_df(
    hf_dataset,
    image_save_dir: str,
    split: str = "train",
    skip_no_image: bool = True,
) -> pd.DataFrame:
    """
    HuggingFace の ScienceQA データセットを MiniCPMVDataset 用 DataFrame に変換する。

    ========================================
    入力
    ========================================
    hf_dataset:
        load_dataset("derek-thomas/ScienceQA", split="train[:500]") の返り値
        features:
            image    : PIL.Image or None  (画像なしサンプルは None)
            question : str               (問題文)
            choices  : list[str]         (選択肢 例: ["cat", "dog", "fish"])
            answer   : int               (正解の選択肢インデックス, 0始まり)
            hint     : str or None       (ヒント文)
            solution : str or None       (解説)

    image_save_dir : str
        PIL画像をJPEGとして保存するディレクトリ

    split : str
        保存ファイル名のプレフィックス用 (例: "train", "validation", "test")

    skip_no_image : bool
        image=None のサンプル (テキストのみ問題) をスキップするか。

    ========================================
    出力 DataFrame columns
    ========================================
    image_path : str
        保存した画像ファイルの絶対パス
    question   : str
        "<image>\\nヒント: {hint}\\n{問題文}\\nA. ...\\nB. ..." 形式
        ※ hint が空の場合は hint 行を省略
    answer     : str
        "答えはAです。{正解テキスト}\\n{solution}" 形式
        ※ solution が空の場合は solution 行を省略

    ========================================
    使用例
    ========================================
    from datasets import load_dataset

    hf_ds = load_dataset("derek-thomas/ScienceQA", split="train[:500]")
    train_df = convert_scienceqa_to_df(hf_ds, image_save_dir="./scienceqa_images")

    hf_val = load_dataset("derek-thomas/ScienceQA", split="validation[:100]")
    eval_df = convert_scienceqa_to_df(hf_val, image_save_dir="./scienceqa_images",
                                      split="validation")

    config = FinetuneConfig(use_lora=True)
    model, tokenizer = load_model_for_finetuning(config)
    log_preview_df = eval_df.sample(n=2, random_state=42).reset_index(drop=True)
    train(model, tokenizer, train_df=train_df, eval_df=eval_df,
          log_preview_df=log_preview_df, config=config)
    """
    save_dir = Path(image_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    CHOICE_LABELS = "ABCDEFGH"

    rows = []
    skipped = 0
    for i, sample in enumerate(hf_dataset):
        pil_image = sample["image"]   # PIL.Image or None
        has_image = pil_image is not None

        if skip_no_image and not has_image:
            skipped += 1
            continue

        # ── 画像を保存 ──
        image_path = None
        if has_image:
            image_path = str((save_dir / f"{split}_{i:06d}.jpg").resolve())
            pil_image.convert("RGB").save(image_path, format="JPEG", quality=95)

        # ── question カラム: "<image>\n{hint}\n{問題文}\nA. ...\nB. ..." ──
        parts = []
        if has_image:
            parts.append("<image>")

        hint = (sample.get("hint") or "").strip()
        if hint:
            parts.append(f"ヒント: {hint}")

        parts.append(sample["question"])

        choices_text = "\n".join(
            f"{CHOICE_LABELS[j]}. {choice}"
            for j, choice in enumerate(sample["choices"])
        )
        parts.append(choices_text)

        question = "\n".join(parts)

        # ── answer カラム: "答えはAです。{正解テキスト}\n{solution}" ──
        answer_idx   = sample["answer"]
        answer_label = CHOICE_LABELS[answer_idx]
        answer_text  = sample["choices"][answer_idx]

        solution = (sample.get("solution") or "").strip()
        answer = f"答えは{answer_label}です。{answer_text}"
        if solution:
            answer += f"\n{solution}"

        rows.append({
            "image_path": image_path,
            "question":   question,
            "answer":     answer,
        })

    df = pd.DataFrame(rows)
    print(f"ScienceQA 変換完了 [{split}]: {len(df)} サンプル "
          f"(画像あり: {df['image_path'].notna().sum()}, "
          f"画像なしスキップ: {skipped})")
    return df


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
    #   train_df = pd.read_csv("train_data.csv")
    #   eval_df  = pd.read_csv("eval_data.csv")

    train_df = pd.DataFrame([
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

    eval_df = pd.DataFrame([
        {
            "image_path": "/path/to/eval1.jpg",
            "question": "画像の内容を説明してください。",
            "answer": "画像には建物が写っています。",
        },
    ])

    print("学習データ:")
    print(train_df.to_string(index=False))
    print()

    # --- 2. 設定 ---
    config = FinetuneConfig()
    config.model_name_or_path = "openbmb/MiniCPM-V-4_5-AWQ"
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
    config.logging_steps = 10
    config.eval_steps = 50
    config.save_steps = 200

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

    # --- 3. モデルロード ---
    # model, tokenizer = load_model_for_finetuning(config)

    # --- 4. 生成プレビュー用サンプル ---
    # eval_df が存在する場合は事前にサンプリング
    # log_preview_df = eval_df.sample(n=min(2, len(eval_df)), random_state=42).reset_index(drop=True)

    # --- 5. 学習実行 ---
    # 実際に実行するには画像ファイルとGPUが必要
    # train(model, tokenizer, train_df, eval_df, log_preview_df, config)

    print("load_model_for_finetuning(config) → train(model, tokenizer, train_df, eval_df, log_preview_df, config)")
    print("を呼び出すと学習が開始されます。")
    print("（実行には画像ファイルとGPUが必要）")


if __name__ == "__main__":
    example_usage()
