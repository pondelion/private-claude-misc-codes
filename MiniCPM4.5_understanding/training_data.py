"""
MiniCPM-V 4.5 - データセット・前処理
================================================

教師あり学習データセット、会話前処理、画像スライス処理、
および統一文書学習パラダイム（動的ビジュアルコラプション）の実装。

論文: MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipes
公式実装:
    - finetune/dataset.py: SupervisedDataset, preprocess(), data_collator()
    - omnilmm/train/train_utils.py: omni_preprocess()
    - finetune/dataset.py: slice_image(), conversation_to_ids_*()

処理の流れ:
1. 画像の読み込みとスライス分割
2. 会話のトークン化とラベル構築
3. 視覚トークンプレースホルダの挿入
4. データの照合（パディング・バッチ化）
"""

"""
============================================================
Shape Convention (形状表記規則)
============================================================
B       : バッチサイズ
L       : シーケンス長 (max_length でクリップ)
Q       : Resamplerクエリ数 (64)
N_img   : サンプル内の画像/スライス数
P       : パッチサイズ (14)
============================================================
"""

import copy
import math
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# ========================================
# 定数
# ========================================
IGNORE_INDEX = -100

# トークナイザのチャットテンプレート区切り文字
RESPONSE_TEMPLATE = "\n<|assistant|>\n"
INSTRUCTION_TEMPLATE = "\n<|user|>\n"

# デフォルトシステムメッセージ
DEFAULT_SYSTEM = (
    "You are an artificial intelligence assistant, which gives helpful, "
    "detailed, and polite answers to the human's questions."
)


# ========================================
# 1. 会話前処理 (omni_preprocess)
# ========================================
def omni_preprocess(
    sources: List[List[Dict]],
    tokenizer,
    generation: bool = False,
) -> Dict[str, List[torch.Tensor]]:
    """
    会話データをトークン化し、ラベル（損失マスク）を構築する

    公式実装: omnilmm/train/train_utils.py: omni_preprocess()

    ========================================
    入力:
        sources: List[List[Dict]]
            バッチ分の会話データ。各会話は辞書のリスト:
            [
                [
                    {"role": "user", "content": "画像を説明して"},
                    {"role": "assistant", "content": "これは猫の画像です"},
                    {"role": "user", "content": "色は？"},
                    {"role": "assistant", "content": "茶色と白です"},
                ],
                ...
            ]
        tokenizer: HuggingFace トークナイザ
        generation: True=生成モード (最後にassistantプロンプト追加)

    出力:
        {
            "input_ids": List[Tensor(L,)]    - トークンID列
            "labels": List[Tensor(L,)]       - ラベル (-100=マスク)
        }

    ラベルマスキングルール:
        - system メッセージ: -100 (損失から除外)
        - user メッセージ: -100 (損失から除外)
        - assistant メッセージ: トークンID (損失計算対象)

    処理の流れ:
        1. ロール名の正規化 (human→user, gpt→assistant)
        2. システムメッセージの追加
        3. tokenizer.apply_chat_template() でテキスト全体をフォーマット
        4. トークン化
        5. <|assistant|> と <|user|> の区切りを検出
        6. assistant応答部分以外をマスク (-100)
    ========================================
    """
    batch_input_ids = []
    batch_labels = []

    # レスポンステンプレートのトークンID化（区切り検出用）
    response_token_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    instruction_token_ids = tokenizer.encode(INSTRUCTION_TEMPLATE, add_special_tokens=False)

    for i in range(len(sources)):
        # --- 1. ロール名の正規化 ---
        new_source = []
        for conv_turn in sources[i]:
            role = conv_turn.get("from", conv_turn.get("role"))
            content = conv_turn.get("value", conv_turn.get("content"))
            role = "user" if role == "human" else role
            role = "assistant" if role == "gpt" else role
            assert role in ["user", "assistant"]
            new_source.append({"role": role, "content": content})

        # --- 2. システムメッセージの追加 ---
        if new_source[0]["role"] != "system":
            new_source.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})

        # --- 3. チャットテンプレートの適用 ---
        res_text = tokenizer.apply_chat_template(
            new_source, tokenize=False, add_generation_prompt=generation
        )
        if not generation:
            res_text = res_text.strip()

        # --- 4. トークン化 ---
        tokenized = tokenizer(
            res_text, return_tensors="pt",
            padding="longest", max_length=tokenizer.model_max_length,
            truncation=True,
        )
        res_input_ids = tokenized.input_ids[0]
        # res_input_ids: (L,) - トークンID列

        res_labels = res_input_ids.clone()
        # res_labels: (L,) - 初期状態はinput_idsのコピー

        # --- 5. 区切り位置の検出 ---
        # <|assistant|> の開始位置を検出
        response_token_ids_idxs = []
        for idx in torch.where(res_labels == response_token_ids[0])[0]:
            if (response_token_ids == res_labels[idx:idx + len(response_token_ids)].tolist()):
                response_token_ids_idxs.append(idx + len(response_token_ids))

        # <|user|> の開始位置を検出
        human_token_ids_idxs = []
        for idx in torch.where(res_labels == instruction_token_ids[0])[0]:
            if (instruction_token_ids == res_labels[idx:idx + len(instruction_token_ids)].tolist()):
                human_token_ids_idxs.append(idx)

        # --- 6. assistant応答部分以外をマスク ---
        if len(response_token_ids_idxs) == 0:
            # assistant応答が見つからない → 全体をマスク
            res_labels[:] = IGNORE_INDEX
        else:
            for idx, (start, end) in enumerate(
                zip(human_token_ids_idxs, response_token_ids_idxs)
            ):
                if idx == 0:
                    # 最初のuser: 先頭からassistant応答開始まで全てマスク
                    res_labels[:end] = IGNORE_INDEX
                else:
                    # 2番目以降のuser: userメッセージ部分をマスク
                    res_labels[start:end] = IGNORE_INDEX

            # 最後のuser（応答なし）があればマスク
            if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                res_labels[human_token_ids_idxs[-1]:] = IGNORE_INDEX

        batch_input_ids.append(res_input_ids)
        batch_labels.append(res_labels)

    return {"input_ids": batch_input_ids, "labels": batch_labels}


# ========================================
# 2. 会話 → トークン + ラベル変換 (MiniCPM / Llama3 / Qwen2)
# ========================================
def conversation_to_ids_minicpm(
    conversation: List[Dict],
    tokenizer,
) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
    """
    MiniCPM形式の会話をトークン化する

    公式実装: finetune/dataset.py: conversation_to_ids_minicpm()

    ========================================
    フォーマット:
        <用户>ユーザーメッセージ<AI>アシスタント応答[EOS]

    入力:
        conversation: [
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "お元気ですか"},
        ]

    出力:
        input_ids: List[np.ndarray] - 各ターンのトークンID
        context: List[np.ndarray] - 各トークンのコンテキストフラグ
            1: コンテキスト（損失から除外）
            0: 応答（損失計算対象）
        raw_msg: str - 結合されたテキスト

    ラベル構築:
        context[i] == 0 のトークンが損失計算対象
        target[i-1] = input_ids[i] (context[i] == 0 の場合)
    ========================================
    """
    raw_msg = ""
    input_ids = []
    context = []

    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]

        prefix = "<用户>" if role == "user" else "<AI>"

        # 最後のターンにEOSを追加
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token

        prefix_ids = tokenizer.encode(prefix)[1:]  # BOS除去
        message_ids = tokenizer.encode(message)[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        # コンテキストフラグ: prefixは常に1（マスク）
        context.append(np.ones(len(prefix_ids), dtype=np.int8))
        if role == "assistant":
            # assistant応答: 0（損失計算対象）
            context.append(np.zeros(len(message_ids), dtype=np.int8))
        else:
            # userメッセージ: 1（マスク）
            context.append(np.ones(len(message_ids), dtype=np.int8))

        raw_msg += prefix + message

    return input_ids, context, raw_msg


def conversation_to_ids_qwen2(
    conversation: List[Dict],
    tokenizer,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Qwen2形式の会話をトークン化する

    公式実装: finetune/dataset.py: conversation_to_ids_qwen2()

    ========================================
    フォーマット (Qwen2 chat template):
        <|im_start|>user\nメッセージ<|im_end|>
        <|im_start|>assistant\n応答<|im_end|>

    長い推論モード対応:
        応答に <think>...</think> が含まれる場合、
        enable_thinking=True でチャットテンプレートを適用

    処理:
        1. apply_chat_template() でフォーマット
        2. <|im_start|>assistant の位置を検出
        3. assistant応答部分のcontext=0、他は1
    ========================================
    """
    chat = []
    for msg in conversation:
        role = msg["role"]
        assert role in ["user", "assistant"]
        chat.append({"role": role, "content": msg["content"]})

    # 長い推論モードの検出
    enable_thinking = False
    if "<think>" in chat[-1]["content"] and "</think>" in chat[-1]["content"]:
        enable_thinking = True

    raw_msg = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    input_ids = np.array(tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=False,
        enable_thinking=enable_thinking,
    ))

    # assistant応答部分の検出
    start_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>")
    )[0]
    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    end_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|im_end|>")
    )[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for i, assistant_idx in enumerate(assistant_idxs):
        if assistant_idx - 1 in set(start_idxs):
            # <think>\n\n</think>\n\n の offset 対応
            offset = 4 if (i == len(assistant_idxs) - 1 and enable_thinking) else 0
            st = assistant_idx + 2 + offset  # assistant\n の後

            for end_idx in end_idxs:
                if end_idx > st:
                    context[st:end_idx + 1] = 0
                    break

    return input_ids, context, raw_msg


# ========================================
# 3. 画像前処理 (preprocess)
# ========================================
def preprocess(
    images_dict: Dict[str, Image.Image],
    conversations: List[Dict],
    tokenizer,
    transform,
    query_nums: int = 64,
    slice_config: Optional[Dict] = None,
    llm_type: str = "minicpm",
    patch_size: int = 14,
    batch_vision: bool = False,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    画像と会話を統合して学習用データを構築する

    公式実装: finetune/dataset.py: preprocess()

    ========================================
    入力:
        images_dict: {"<image>": PIL Image} or
                     {"<image_00>": PIL Image, "<image_01>": PIL Image}
        conversations: 会話データ
        tokenizer: トークナイザ
        transform: 画像変換関数
        query_nums: Resamplerクエリ数 (64)
        slice_config: {"max_slice_nums": 9, "scale_resolution": 448, "patch_size": 14}
        llm_type: "minicpm" / "llama3" / "qwen"
        patch_size: ViTパッチサイズ (14)
        batch_vision: バッチビジョンモード
        max_length: 最大シーケンス長

    出力:
        {
            "input_ids": (L,) - トークンID列
            "target": (L,) - ラベル (-100=マスク)
            "image_bound": (N_img, 2) or [] - 画像トークンの開始/終了位置
            "pixel_values": List[Tensor(3, H, W)] - 変換済み画像テンソル
            "tgt_sizes": Tensor(N_img, 2) or [] - 各画像のパッチグリッドサイズ
            "position_ids": (L,) - 位置ID
        }

    処理の流れ:
        1. 各画像をスライス分割 (slice_image)
        2. プレースホルダを構築
           ソース画像: <im_start> + <unk>*Q + <im_end>
           各スライス: <im_start> + <unk>*Q + <im_end> (グリッド配置)
        3. 会話テキスト中の <image> をプレースホルダに置換
        4. トークン化 + ラベル構築
        5. image_bound (画像トークンの開始/終了位置) を計算
    ========================================
    """
    conversations = copy.deepcopy(conversations)
    assert len(conversations) > 1, "会話は2ターン以上必要"
    assert conversations[0]["role"] == "user", "最初のロールはuser"

    # デフォルトの画像プレースホルダ
    default_image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )

    # --- 1. 各画像をスライス分割 ---
    image_placeholder_dict = {}
    images = []
    for img_name, image in images_dict.items():
        if slice_config:
            from vision_encoder import slice_image
            source_image, patches, best_grid = slice_image(
                image,
                slice_config["max_slice_nums"],
                slice_config["scale_resolution"],
                slice_config["patch_size"],
            )
            images.append(source_image)
            image_placeholder = default_image_placeholder

            if len(patches) > 0:
                for row in patches:
                    for patch in row:
                        images.append(patch)
                # グリッドプレースホルダの構築
                image_placeholder += _get_grid_placeholder(
                    tokenizer, best_grid, query_nums
                )
            image_placeholder_dict[img_name] = image_placeholder
        else:
            images.append(image)
            image_placeholder_dict[img_name] = default_image_placeholder

    # --- 2. 画像変換 ---
    images = [transform(img) for img in images]
    # images: List[Tensor(3, H, W)]

    # --- 3. プレースホルダの置換 ---
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
        # マルチイメージ: <image_XX> パターンを置換
        pattern = r"<image_\d+>"
        for conv in conversations:
            parts = re.split(f"({pattern})", conv["content"])
            for i, part in enumerate(parts):
                if re.match(pattern, part) and part in image_placeholder_dict:
                    parts[i] = image_placeholder_dict[part]
            conv["content"] = "\n".join([p for p in parts if p.strip()])

    # --- 4. トークン化 + ラベル構築 ---
    if llm_type == "qwen":
        input_ids, context, raw_msg = conversation_to_ids_qwen2(conversations, tokenizer)
    else:
        input_ids_list, context_list, raw_msg = conversation_to_ids_minicpm(conversations, tokenizer)
        input_ids = np.hstack(input_ids_list).astype(np.int32)
        context = np.hstack(context_list).astype(np.int8)

    ids = torch.from_numpy(input_ids)
    context_tensor = torch.from_numpy(context)
    # ids: (L,), context_tensor: (L,)

    if len(ids) > max_length:
        ids = ids[:max_length]
        context_tensor = context_tensor[:max_length]

    # ターゲットラベルの構築
    target = torch.full_like(ids, IGNORE_INDEX, dtype=torch.int32)
    for i in range(1, len(ids)):
        if context_tensor[i] == 0:
            target[i - 1] = ids[i]
        if context_tensor[i] == 1 and context_tensor[i - 1] == 0:
            target[i - 1] = tokenizer.eos_token_id
    # target: (L,)
    #   -100: マスク位置
    #   token_id: 予測対象位置

    # --- 5. image_bound の計算 ---
    image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0] + 1
    image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]
    assert len(image_start_tokens) == len(image_end_tokens)

    if len(image_start_tokens) > 0:
        image_bound = torch.hstack([
            image_start_tokens.unsqueeze(-1),
            image_end_tokens.unsqueeze(-1),
        ])
        # image_bound: (N_img, 2) - 各画像の [start, end] 位置
    else:
        image_bound = []

    position_ids = torch.arange(ids.size(0)).long()
    # position_ids: (L,)

    result = {
        "input_ids": ids,          # (L,)
        "target": target,          # (L,)
        "image_bound": image_bound,  # (N_img, 2) or []
        "pixel_values": images,    # List[Tensor(3, H, W)]
        "tgt_sizes": [],           # batch_vision時のみ使用
        "position_ids": position_ids,  # (L,)
    }

    return result


def _get_grid_placeholder(tokenizer, grid, query_num):
    """
    グリッドスライスのプレースホルダ文字列を構築する

    公式実装: finetune/dataset.py: get_grid_placeholder()

    ========================================
    入力:
        tokenizer: トークナイザ (im_start, im_end, unk_token, slice_start, slice_end)
        grid: [grid_x, grid_y] - 例: [2, 3]
        query_num: Q (64)

    出力:
        placeholder: str - グリッド配置されたプレースホルダ文字列

    例 (grid=[2, 3]):
        <slice_start>
        <im_start><unk>*64<im_end><im_start><unk>*64<im_end>
        <im_start><unk>*64<im_end><im_start><unk>*64<im_end>
        <im_start><unk>*64<im_end><im_start><unk>*64<im_end>
        <slice_end>
    ========================================
    """
    image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols, rows = grid[0], grid[1]
    slices = []
    for i in range(rows):
        line = ""
        for j in range(cols):
            line += image_placeholder
        slices.append(line)

    slice_placeholder = (
        tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end
    )
    return slice_placeholder


# ========================================
# 4. SupervisedDataset
# ========================================
class SupervisedDataset(Dataset):
    """
    教師あり微調整用データセット

    公式実装: finetune/dataset.py: SupervisedDataset

    ========================================
    入力データ形式 (JSON):
        [
            {
                "image": "path/to/image.jpg",  # or Dict for multi-image
                "conversations": [
                    {"role": "user", "content": "<image>\n質問"},
                    {"role": "assistant", "content": "回答"},
                ]
            },
            ...
        ]

    __getitem__ の出力:
        {
            "input_ids": (L,),
            "position_ids": (L,),
            "labels": (L,),
            "attention_mask": (L,) - all True
            "pixel_values": List[Tensor(3, H, W)],
            "tgt_sizes": Tensor or [],
            "image_bound": Tensor(N_img, 2) or [],
        }
    ========================================
    """

    def __init__(
        self,
        raw_data: List[Dict],
        transform,
        tokenizer,
        slice_config: Optional[Dict] = None,
        llm_type: str = "minicpm",
        patch_size: int = 14,
        query_nums: int = 64,
        batch_vision: bool = False,
        max_length: int = 2048,
    ):
        super().__init__()
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

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        1サンプルの取得

        ========================================
        処理:
            1. 画像の読み込み
            2. preprocess() でトークン化 + ラベル構築
            3. エラー時はランダムに別サンプルを返す
        ========================================
        """
        try:
            sample = self.raw_data[i]

            # --- 画像の読み込み ---
            if isinstance(sample["image"], str):
                images_dict = {"<image>": Image.open(sample["image"]).convert("RGB")}
            elif isinstance(sample["image"], dict):
                images_dict = {
                    name: Image.open(path).convert("RGB")
                    for name, path in sample["image"].items()
                }
            else:
                raise ValueError(f"Unsupported image format: {type(sample['image'])}")

            # --- 前処理 ---
            ret = preprocess(
                images_dict,
                sample["conversations"],
                self.tokenizer,
                self.transform,
                query_nums=self.query_nums,
                slice_config=self.slice_config,
                llm_type=self.llm_type,
                patch_size=self.patch_size,
                batch_vision=self.batch_vision,
                max_length=self.max_length,
            )

            return {
                "input_ids": ret["input_ids"],           # (L,)
                "position_ids": ret["position_ids"],     # (L,)
                "labels": ret["target"],                 # (L,)
                "attention_mask": torch.ones_like(ret["input_ids"], dtype=torch.bool),
                "pixel_values": ret["pixel_values"],     # List[Tensor]
                "tgt_sizes": ret["tgt_sizes"],
                "image_bound": ret["image_bound"],       # (N_img, 2) or []
            }

        except Exception:
            # エラー時: ランダムな別サンプルを返す
            return self.__getitem__(random.randint(0, len(self) - 1))


# ========================================
# 5. Data Collator
# ========================================
def data_collator(
    examples: List[Dict],
    padding_value: int = 0,
    max_length: int = 2048,
) -> Dict[str, torch.Tensor]:
    """
    バッチのパディングと照合

    公式実装: finetune/dataset.py: data_collator()

    ========================================
    入力:
        examples: List[Dict] - SupervisedDatasetの出力リスト
        padding_value: パディング値 (input_ids用)
        max_length: 最大シーケンス長

    出力:
        {
            "input_ids": (B, L_max) - パディング済みトークンID
            "position_ids": (B, L_max)
            "labels": (B, L_max) - パディング値=-100
            "attention_mask": (B, L_max)
            "pixel_values": List[List[Tensor]] - 各サンプルの画像リスト
            "image_bound": List[Tensor] - 各サンプルのimage_bound
            "tgt_sizes": List[Tensor]
        }

    パディングルール:
        - input_ids: padding_value (0)
        - labels: -100 (損失から除外)
        - attention_mask: 0 (パディング位置)
        - pixel_values: パディングなし（リストのまま）
    ========================================
    """
    def trim_and_pad(seq_list, padding_value):
        trimmed = [s[:max_length] for s in seq_list]
        return pad_sequence(trimmed, batch_first=True, padding_value=padding_value)

    input_ids = trim_and_pad(
        [ex["input_ids"] for ex in examples], padding_value
    )
    # input_ids: (B, L_max)

    position_ids = trim_and_pad(
        [ex["position_ids"] for ex in examples], padding_value
    )
    # position_ids: (B, L_max)

    labels = trim_and_pad(
        [ex["labels"] for ex in examples], IGNORE_INDEX
    )
    # labels: (B, L_max) - パディング位置は-100

    attention_mask = trim_and_pad(
        [ex["attention_mask"] for ex in examples], 0
    )
    # attention_mask: (B, L_max) - パディング位置は0

    # pixel_values と image_bound はサンプルごとにサイズが異なるため
    # リストのまま保持
    pixel_values = [ex["pixel_values"] for ex in examples]
    image_bound = [ex["image_bound"] for ex in examples]
    tgt_sizes = [ex["tgt_sizes"] for ex in examples]

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "pixel_values": pixel_values,
    }


# ========================================
# 6. 統一文書学習パラダイム (Dynamic Visual Corruption)
# ========================================
class DocumentCorruptionAugment:
    """
    文書画像に動的レベルのノイズを付与する

    論文 Section 2.2.3:
        「We unify both capabilities into a single learning objective:
         predicting original text from corrupted document images.」

    3段階のコラプションレベル:
        1. Low: テキストが視認可能 → OCR認識力を強化
        2. Moderate: テキストが曖昧 → 視覚的手がかりと文脈の統合
        3. High: テキストが完全マスク → 文書レイアウト・知識からの推論

    ========================================
    入力:
        image: PIL Image - 文書画像
        text_regions: List[Tuple[int,int,int,int]] - テキスト領域のバウンディングボックス
            [(x1, y1, x2, y2), ...]

    出力:
        corrupted_image: PIL Image - コラプション適用後の画像
        corruption_level: str - 適用されたレベル ("low"/"moderate"/"high")
    ========================================
    """

    def __init__(
        self,
        low_prob: float = 0.33,
        moderate_prob: float = 0.33,
        high_prob: float = 0.34,
    ):
        """
        Args:
            low_prob: 低コラプションの確率
            moderate_prob: 中コラプションの確率
            high_prob: 高コラプションの確率
        """
        self.probs = [low_prob, moderate_prob, high_prob]
        self.levels = ["low", "moderate", "high"]

    def __call__(
        self,
        image: Image.Image,
        text_regions: List[Tuple[int, int, int, int]],
    ) -> Tuple[Image.Image, str]:
        """
        ========================================
        処理:
            各テキスト領域に対して:
            1. ランダムにコラプションレベルを選択
            2. レベルに応じたノイズを適用
               - Low: 軽いガウシアンノイズ (σ=5-15)
               - Moderate: 中程度のノイズ (σ=30-60) + ブラー
               - High: 領域全体をマスク（灰色で塗りつぶし）

        入力:
            image: PIL Image (H, W, 3)
            text_regions: [(x1, y1, x2, y2), ...]

        出力:
            corrupted_image: PIL Image
            corruption_level: 選択されたレベル名
        ========================================
        """
        corrupted = image.copy()
        level = np.random.choice(self.levels, p=self.probs)

        for region in text_regions:
            x1, y1, x2, y2 = region

            if level == "low":
                # --- 低コラプション: 軽いガウシアンノイズ ---
                # テキストは視認可能、OCR認識力を強化
                region_crop = corrupted.crop((x1, y1, x2, y2))
                region_array = np.array(region_crop).astype(np.float32)
                noise = np.random.normal(0, np.random.uniform(5, 15), region_array.shape)
                noisy = np.clip(region_array + noise, 0, 255).astype(np.uint8)
                corrupted.paste(Image.fromarray(noisy), (x1, y1))

            elif level == "moderate":
                # --- 中コラプション: ノイズ + ブラー ---
                # テキストが曖昧、視覚的手がかりと文脈の統合推論
                region_crop = corrupted.crop((x1, y1, x2, y2))
                region_array = np.array(region_crop).astype(np.float32)
                noise = np.random.normal(0, np.random.uniform(30, 60), region_array.shape)
                noisy = np.clip(region_array + noise, 0, 255).astype(np.uint8)
                blurred = Image.fromarray(noisy).filter(
                    ImageFilter.GaussianBlur(radius=np.random.uniform(2, 5))
                )
                corrupted.paste(blurred, (x1, y1))

            elif level == "high":
                # --- 高コラプション: 完全マスク ---
                # テキストが見えない、文書レイアウト・知識からの推論
                mask_color = tuple(np.random.randint(180, 220, 3).tolist())
                region_img = Image.new("RGB", (x2 - x1, y2 - y1), mask_color)
                corrupted.paste(region_img, (x1, y1))

        return corrupted, level


# ========================================
# 使用例
# ========================================
def example_usage():
    """
    データセット・前処理のデモ
    """
    print("=== 会話前処理のデモ ===")
    # 会話データ
    sources = [[
        {"role": "user", "content": "<image>\nこの画像の内容を説明してください。"},
        {"role": "assistant", "content": "この画像は猫が窓辺で寝ている場面です。"},
        {"role": "user", "content": "猫の色は何色ですか？"},
        {"role": "assistant", "content": "茶色と白のまだら模様です。"},
    ]]
    print(f"会話ターン数: {len(sources[0])}")
    print(f"ラベルマスキング: user/systemは-100, assistantのみ損失対象")

    print("\n=== 文書コラプションのデモ ===")
    corruption = DocumentCorruptionAugment()
    print(f"コラプションレベル確率: low={corruption.probs[0]}, "
          f"moderate={corruption.probs[1]}, high={corruption.probs[2]}")

    # ダミー画像でテスト
    dummy_img = Image.new("RGB", (800, 600), "white")
    text_regions = [(50, 50, 200, 80), (50, 100, 300, 130)]
    corrupted_img, level = corruption(dummy_img, text_regions)
    print(f"選択されたコラプションレベル: {level}")
    print(f"出力画像サイズ: {corrupted_img.size}")

    print("\n=== データ照合のデモ ===")
    print("data_collator():")
    print("  - input_ids: (B, L_max) - パディング値=0")
    print("  - labels: (B, L_max) - パディング値=-100")
    print("  - pixel_values: List[List[Tensor]] - パディングなし")


if __name__ == "__main__":
    example_usage()
