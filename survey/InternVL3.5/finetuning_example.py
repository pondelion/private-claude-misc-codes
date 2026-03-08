"""
InternVL3.5 ファインチューニングサンプル (Trainer不使用)
=========================================================

このファイルは pandas DataFrame 形式のデータセットを使って
InternVL3.5 をファインチューニングするサンプルコードです。
transformers の Trainer は使用せず、PyTorch 標準機能のみで実装しています。

============================================================
前提条件
============================================================
- 画像ファイルパスとQ&A (または会話) のペアデータセットが
  pandas DataFrame として渡される

============================================================
DataFrame の期待するカラム
============================================================
必須:
  - image      : str  画像ファイルパス (絶対パスまたは相対パス)
  - question   : str  ユーザーの質問 (テキストのみ, <image> タグなし可)
  - answer     : str  アシスタントの回答 (正解テキスト)

オプション:
  - num_tiles  : int  最大タイル数 (デフォルト 6)
                      細かい視覚理解が必要なサンプルは増やせる
  - system_msg : str  カスタムシステムメッセージ

============================================================
使い方
============================================================
# スクリプト単体実行
python finetuning_example.py \\
    --model_path OpenGVLab/InternVL3_5-8B \\
    --train_csv /path/to/train.csv \\
    --eval_csv /path/to/eval.csv \\
    --output_dir ./internvl35-finetuned \\
    --epochs 3 \\
    --batch_size 1 \\
    --grad_acc 16 \\
    --lr 2e-5 \\
    --lora \\
    --lora_r 128

# Python 内から DataFrame で使用
import pandas as pd
df = pd.DataFrame({
    "image":    ["path/to/img.jpg"],
    "question": ["この画像を説明してください。"],
    "answer":   ["この画像には...が写っています。"],
})
model, tokenizer = load_model_for_finetuning("OpenGVLab/InternVL3_5-8B")
train(model, tokenizer, train_df=df, output_dir="./out")

============================================================
モデルサイズの目安 (LoRA r=128)
============================================================
  InternVL3.5-1B : VRAM ~5GB  (LoRA)
  InternVL3.5-2B : VRAM ~8GB  (LoRA)
  InternVL3.5-4B : VRAM ~12GB (LoRA)
  InternVL3.5-8B : VRAM ~18GB (LoRA)
  InternVL3.5-8B : VRAM ~60GB (フルファインチューニング)
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# 定数
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_START_TOKEN   = '<img>'
IMG_END_TOKEN     = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_CONTEXT_ID    = None   # tokenizer ロード後に設定

IGNORE_INDEX = -100


# ============================================================
# 画像前処理
# ============================================================

def preprocess_tile(tile: Image.Image, size: int = 448) -> torch.Tensor:
    """
    PIL タイル画像をモデル入力テンソルに変換。

    入力:  PIL.Image (size, size) RGB
    出力:  (3, size, size) float32, ImageNet正規化済み
    """
    tile = tile.resize((size, size), Image.LANCZOS).convert('RGB')
    arr = np.array(tile, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    arr  = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))


def get_target_ratios(min_tiles: int = 1, max_tiles: int = 6) -> List[Tuple[int, int]]:
    """候補タイルグリッド (n_w, n_h) のリストを返す。"""
    ratios = set()
    for n_w in range(1, max_tiles + 1):
        for n_h in range(1, max_tiles + 1):
            if min_tiles <= n_w * n_h <= max_tiles:
                ratios.add((n_w, n_h))
    return sorted(ratios, key=lambda x: x[0] * x[1])


def dynamic_preprocess(
    image: Image.Image,
    tile_size: int = 448,
    max_tiles: int = 6,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """
    高解像度画像を tile_size×tile_size のタイルに分割する
    Dynamic High Resolution 前処理。

    入力:  PIL.Image  (任意サイズ)
    出力:  List[PIL.Image]  各 (tile_size, tile_size)
           長さ = n_tiles + 1 (サムネイル込み, use_thumbnail=True の場合)
    """
    W, H = image.size
    aspect = W / H

    ratios = get_target_ratios(1, max_tiles)
    best_ratio = min(ratios, key=lambda r: abs(aspect - r[0] / r[1]))
    n_w, n_h = best_ratio
    n_tiles = n_w * n_h

    resized = image.resize((tile_size * n_w, tile_size * n_h), Image.LANCZOS)
    tiles = []
    for row in range(n_h):
        for col in range(n_w):
            tile = resized.crop((col * tile_size, row * tile_size,
                                  (col + 1) * tile_size, (row + 1) * tile_size))
            tiles.append(tile)

    if use_thumbnail and n_tiles > 1:
        thumb = image.resize((tile_size, tile_size), Image.LANCZOS)
        tiles = [thumb] + tiles

    return tiles


def load_image_as_tensor(
    image_path: str,
    max_tiles: int = 6,
    tile_size: int = 448,
) -> Tuple[torch.Tensor, int]:
    """
    画像ファイルを読み込んでモデル入力テンソルに変換する。

    入力:
      image_path : str  画像ファイルパス
      max_tiles  : int  最大タイル数
      tile_size  : int  タイルサイズ (デフォルト 448)

    出力:
      pixel_values : (P, 3, tile_size, tile_size)  float32 テンソル
      num_patches  : P (パッチ数)
    """
    image = Image.open(image_path).convert('RGB')
    tiles = dynamic_preprocess(image, tile_size=tile_size, max_tiles=max_tiles)
    tensors = [preprocess_tile(t, size=tile_size) for t in tiles]
    pixel_values = torch.stack(tensors, dim=0)   # (P, 3, tile_size, tile_size)
    return pixel_values, len(tiles)


# ============================================================
# データセット
# ============================================================

class InternVL35Dataset(Dataset):
    """
    pandas DataFrame からロードする InternVL3.5 学習用データセット。

    DataFrame の必須カラム:
      - image    : str  画像ファイルパス
      - question : str  ユーザーの質問
      - answer   : str  アシスタントの回答

    オプションカラム:
      - num_tiles : int  最大タイル数 (デフォルト 6)
      - system_msg: str  カスタムシステムメッセージ
    """
    DEFAULT_SYSTEM = (
        "You are InternVL3.5, created by Shanghai AI Laboratory. "
        "You are a helpful assistant."
    )

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        num_image_token: int = 256,
        tile_size: int = 448,
        default_max_tiles: int = 6,
        max_seq_len: int = 4096,
        image_base_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.tile_size = tile_size
        self.default_max_tiles = default_max_tiles
        self.max_seq_len = max_seq_len
        self.image_base_dir = image_base_dir

        # 必須カラムの確認
        required = ['image', 'question', 'answer']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"DataFrame に必須カラム '{col}' がありません。")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, path: str) -> str:
        """画像パスを解決する (base_dir が指定されている場合は結合)。"""
        if self.image_base_dir:
            return os.path.join(self.image_base_dir, path)
        return path

    def _build_prompt(
        self,
        question: str,
        num_patches: int,
        system_msg: str,
    ) -> str:
        """
        Qwen3 形式のチャットプロンプトを構築する。

        フォーマット:
          <|im_start|>system
          {system_msg}<|im_end|>
          <|im_start|>user
          <img><IMG_CONTEXT>×(256×P)</img>
          {question}<|im_end|>
          <|im_start|>assistant

        返値: プロンプト文字列
        """
        # IMG_CONTEXT トークン文字列
        img_tokens = (IMG_START_TOKEN
                      + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                      + IMG_END_TOKEN)

        if '<image>' in question:
            user_msg = question.replace('<image>', img_tokens, 1)
        else:
            user_msg = img_tokens + '\n' + question

        prompt = (f'<|im_start|>system\n{system_msg}<|im_end|>\n'
                  f'<|im_start|>user\n{user_msg}<|im_end|>\n'
                  f'<|im_start|>assistant\n')
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        1サンプルを返す。

        返値の形状:
          pixel_values  : (P, 3, 448, 448)
          input_ids     : (N,)                N = プロンプト + 回答 + EOS
          labels        : (N,)                プロンプト部分は -100 でマスク
          image_flags   : (P,)                全て 1 (有効)
          loss_weight   : (N,)                Square Averaging 重み
        """
        row = self.df.iloc[idx]

        # ---- 画像ロード & 前処理 ----
        image_path = self._resolve_image_path(str(row['image']))
        max_tiles = int(row.get('num_tiles', self.default_max_tiles))

        pixel_values, num_patches = load_image_as_tensor(
            image_path, max_tiles=max_tiles, tile_size=self.tile_size
        )

        # ---- テキスト準備 ----
        question   = str(row['question'])
        answer     = str(row['answer'])
        system_msg = str(row.get('system_msg', self.DEFAULT_SYSTEM))

        # プロンプト構築
        prompt = self._build_prompt(
            question=question,
            num_patches=num_patches,
            system_msg=system_msg,
        )
        full_text = prompt + answer + '<|im_end|>'

        # ---- トークナイズ ----
        # max_seq_len 以内に切り詰め
        full_enc = self.tokenizer(
            full_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
        )
        input_ids = full_enc['input_ids'].squeeze(0)  # (N,)

        # プロンプト部分の長さを計算 (マスク範囲の決定)
        prompt_enc = self.tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=False,
        )
        n_prompt = min(prompt_enc['input_ids'].shape[1], len(input_ids))

        # ---- ラベル作成 ----
        labels = input_ids.clone()          # (N,)
        labels[:n_prompt] = IGNORE_INDEX    # プロンプトはマスク

        # ---- Square Averaging 重み ----
        # 回答トークン数で 1/N^0.6 重み付け
        n_response = max(1, len(input_ids) - n_prompt)
        weight = 1.0 / (n_response ** 0.6)
        loss_weight = torch.zeros(len(input_ids))
        loss_weight[n_prompt:] = weight

        # ---- image_flags ----
        image_flags = torch.ones(num_patches, dtype=torch.long)  # (P,)

        return {
            'pixel_values': pixel_values,            # (P, 3, 448, 448)
            'input_ids'   : input_ids,               # (N,)
            'labels'      : labels,                  # (N,)
            'image_flags' : image_flags,             # (P,)
            'loss_weight' : loss_weight,             # (N,)
        }


# ============================================================
# カスタムコレーター
# ============================================================

class InternVL35Collator:
    """
    InternVL35Dataset の可変長サンプルをバッチに整形するコレーター。

    処理:
      - pixel_values: 全サンプルのパッチを結合 → (Σ P_i, 3, 448, 448)
      - input_ids:    右パディング → (B, N_max)
      - labels:       右パディング (IGNORE_INDEX) → (B, N_max)
      - image_flags:  全サンプルを結合 → (Σ P_i, 1)
      - loss_weight:  右パディング (0.0) → (B, N_max)
      - attention_mask: パディング位置を 0 に → (B, N_max)
    """
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        # --- pixel_values: (Σ P_i, 3, 448, 448) ---
        pixel_values = torch.cat(
            [s['pixel_values'] for s in samples], dim=0
        )

        # --- image_flags: (Σ P_i, 1) ---
        image_flags = torch.cat(
            [s['image_flags'] for s in samples], dim=0
        ).unsqueeze(-1)

        # --- テキスト系列のパディング ---
        max_len = max(s['input_ids'].shape[0] for s in samples)

        input_ids_list    = []
        labels_list       = []
        loss_weight_list  = []
        attention_mask_list = []

        for s in samples:
            n = s['input_ids'].shape[0]
            pad_len = max_len - n

            # 右パディング
            ids = torch.cat([s['input_ids'],
                             torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            lbl = torch.cat([s['labels'],
                             torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
            wt  = torch.cat([s['loss_weight'],
                             torch.zeros(pad_len)])
            attn = torch.cat([torch.ones(n, dtype=torch.long),
                              torch.zeros(pad_len, dtype=torch.long)])

            input_ids_list.append(ids)
            labels_list.append(lbl)
            loss_weight_list.append(wt)
            attention_mask_list.append(attn)

        return {
            'pixel_values'  : pixel_values,                      # (Σ P_i, 3, 448, 448)
            'input_ids'     : torch.stack(input_ids_list),       # (B, N_max)
            'labels'        : torch.stack(labels_list),          # (B, N_max)
            'image_flags'   : image_flags,                       # (Σ P_i, 1)
            'loss_weight'   : torch.stack(loss_weight_list),     # (B, N_max)
            'attention_mask': torch.stack(attention_mask_list),  # (B, N_max)
        }


# ============================================================
# モデルのロード & LoRA 適用
# ============================================================

def load_model_for_finetuning(
    model_path: str,
    use_lora: bool = True,
    lora_r: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
    freeze_vit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = 'auto',
):
    """
    InternVL3.5 モデルとトークナイザーをロードして学習用に設定する。

    引数:
      model_path   : str  モデルパスまたはHugging Face モデルID
                          例: 'OpenGVLab/InternVL3_5-8B'
      use_lora     : bool LoRA を使用するか (True: VRAM 節約)
      lora_r       : int  LoRA ランク (デフォルト 128)
      lora_alpha   : int  LoRA alpha (通常 2×r)
      lora_dropout : float LoRA dropout
      freeze_vit   : bool ViT を凍結するか (通常 True)
      dtype        : torch.dtype (デフォルト bfloat16)
      device_map   : str  'auto' で複数GPU自動分散

    返値:
      model      : InternVLChatModel  学習準備済みモデル
      tokenizer  : AutoTokenizer
    """
    print(f"モデルをロード中: {model_path}")

    # トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # IMG_CONTEXT トークン ID を設定
    global IMG_CONTEXT_ID
    IMG_CONTEXT_ID = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    print(f"  IMG_CONTEXT_TOKEN ID: {IMG_CONTEXT_ID}")

    # モデルをロード
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.img_context_token_id = IMG_CONTEXT_ID

    # ViT を凍結
    if freeze_vit:
        print("  ViT を凍結中...")
        for name, param in model.named_parameters():
            if 'vision_model' in name:
                param.requires_grad = False
        vit_params = sum(p.numel() for p in model.vision_model.parameters())
        print(f"  ViT パラメータ数 (凍結): {vit_params:,}")

    # LoRA を適用
    if use_lora:
        print(f"  LoRA を適用中 (r={lora_r}, alpha={lora_alpha})...")

        # LLM のアーキテクチャに応じてターゲットモジュールを設定
        llm_arch = model.config.llm_config.architectures[0]
        if llm_arch in ('Qwen2ForCausalLM', 'LlamaForCausalLM'):
            target_modules = [
                'self_attn.q_proj', 'self_attn.k_proj',
                'self_attn.v_proj', 'self_attn.o_proj',
                'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj',
            ]
        elif llm_arch == 'InternLM2ForCausalLM':
            target_modules = [
                'attention.wqkv', 'attention.wo',
                'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3',
            ]
        else:
            # デフォルト: Qwen 互換
            target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'down_proj', 'up_proj',
            ]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.enable_input_require_grads()
        model.language_model.print_trainable_parameters()

    # MLP Projector は常に学習可能
    for param in model.mlp1.parameters():
        param.requires_grad = True
    mlp_params = sum(p.numel() for p in model.mlp1.parameters() if p.requires_grad)
    print(f"  MLP Projector 学習可能パラメータ: {mlp_params:,}")

    # 総学習可能パラメータ
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  総学習可能パラメータ: {total_trainable:,}")

    return model, tokenizer


# ============================================================
# 学習ループ
# ============================================================

def train(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    eval_df: Optional[pd.DataFrame] = None,
    output_dir: str = './internvl35-finetuned',
    epochs: int = 3,
    batch_size: int = 1,
    grad_acc: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    max_seq_len: int = 4096,
    num_workers: int = 2,
    max_tiles: int = 6,
    num_image_token: int = 256,
    save_every_n_steps: int = 500,
    eval_every_n_steps: int = 200,
    fp16: bool = False,
    bf16: bool = True,
) -> None:
    """
    InternVL3.5 のファインチューニング学習ループ。

    引数:
      model        : load_model_for_finetuning でロードしたモデル
      tokenizer    : 対応するトークナイザー
      train_df     : pd.DataFrame  学習データ
                     必須カラム: image, question, answer
      eval_df      : pd.DataFrame  評価データ (省略可)
      output_dir   : str  チェックポイント保存先
      epochs       : int  エポック数
      batch_size   : int  ミニバッチサイズ (GPUメモリに応じて調整)
      grad_acc     : int  勾配累積ステップ数
                     実効バッチ = batch_size × grad_acc
      lr           : float  学習率
      weight_decay : float  AdamW 重み減衰
      warmup_ratio : float  ウォームアップ比率 (全ステップ数に対する比)
      max_seq_len  : int  最大系列長
      num_workers  : int  DataLoader のワーカー数
      max_tiles    : int  画像の最大タイル数 (通常 6)
      num_image_token: int  1パッチあたりの IMG_CONTEXT 数 (通常 256)
      save_every_n_steps: int  チェックポイント保存間隔
      eval_every_n_steps: int  評価実行間隔
      fp16         : bool  FP16 混合精度 (V100等)
      bf16         : bool  BF16 混合精度 (A100/H100等, 推奨)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- デバイス設定 ----
    device = next(model.parameters()).device
    print(f"  学習デバイス: {device}")

    # ---- データセット & DataLoader ----
    pad_id = tokenizer.pad_token_id or 0
    collator = InternVL35Collator(pad_token_id=pad_id)

    train_dataset = InternVL35Dataset(
        df=train_df,
        tokenizer=tokenizer,
        num_image_token=num_image_token,
        default_max_tiles=max_tiles,
        max_seq_len=max_seq_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = None
    if eval_df is not None:
        eval_dataset = InternVL35Dataset(
            df=eval_df,
            tokenizer=tokenizer,
            num_image_token=num_image_token,
            default_max_tiles=max_tiles,
            max_seq_len=max_seq_len,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

    # ---- オプティマイザー ----
    # ViT はすでに凍結されているので model.parameters() で学習可能パラメータのみ
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ---- スケジューラー (コサインアニーリング + ウォームアップ) ----
    total_steps = len(train_loader) // grad_acc * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"  総ステップ数: {total_steps}, ウォームアップ: {warmup_steps}")

    def lr_lambda(current_step: int) -> float:
        """ウォームアップ + コサインアニーリング スケジュール。"""
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- 混合精度スケーラー ----
    scaler = torch.cuda.amp.GradScaler() if (fp16 and not bf16) else None
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)

    # ---- 学習ループ ----
    global_step = 0
    optimizer_step = 0
    running_loss = 0.0
    model.train()

    print(f"\n学習開始: {epochs}エポック, 実効バッチ={batch_size * grad_acc}")
    print(f"  train: {len(train_dataset)}サンプル / eval: {len(eval_dataset) if eval_df is not None else 0}サンプル")
    print("-" * 60)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        model.train()

        for step, batch in enumerate(train_loader):
            # バッチをデバイスに転送
            pixel_values   = batch['pixel_values'].to(device, dtype=torch.bfloat16 if bf16 else torch.float32)
            input_ids      = batch['input_ids'].to(device)
            labels         = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_flags    = batch['image_flags'].to(device)
            loss_weight    = batch['loss_weight'].to(device, dtype=torch.float32)

            # フォワードパス & 損失計算
            ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype else torch.no_grad.__class__()
            if amp_dtype:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_flags=image_flags,
                        labels=labels,
                        loss_weight=loss_weight.tolist(),
                    )
                    loss = outputs.loss / grad_acc
            else:
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    labels=labels,
                    loss_weight=loss_weight.tolist(),
                )
                loss = outputs.loss / grad_acc

            running_loss += loss.item() * grad_acc

            # バックプロパゲーション
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1

            # 勾配累積: grad_acc ステップごとに更新
            if global_step % grad_acc == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                    )
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                avg_loss = running_loss / grad_acc
                running_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]

                if optimizer_step % 10 == 0:
                    print(f"  [Epoch {epoch+1}, Step {optimizer_step}] "
                          f"loss={avg_loss:.4f}, lr={current_lr:.2e}")

                # ---- 評価 ----
                if eval_loader is not None and optimizer_step % eval_every_n_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, amp_dtype, bf16)
                    print(f"  [Eval Step {optimizer_step}] eval_loss={eval_loss:.4f}")
                    model.train()

                # ---- チェックポイント保存 ----
                if optimizer_step % save_every_n_steps == 0:
                    ckpt_path = os.path.join(output_dir, f'checkpoint-{optimizer_step}')
                    save_model(model, tokenizer, ckpt_path)
                    print(f"  チェックポイント保存: {ckpt_path}")

    # ---- 最終モデル保存 ----
    final_path = os.path.join(output_dir, 'final')
    save_model(model, tokenizer, final_path)
    print(f"\n学習完了。最終モデル保存: {final_path}")


# ============================================================
# 評価
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    bf16: bool,
) -> float:
    """
    評価データセットでの平均損失を計算する。

    入力:
      model       : 学習中のモデル
      eval_loader : DataLoader  評価用
      device      : torch.device
      amp_dtype   : 混合精度の dtype (None=無効)
      bf16        : BF16 を使うか
    出力:
      eval_loss : float  平均評価損失
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in eval_loader:
        pixel_values   = batch['pixel_values'].to(device, dtype=torch.bfloat16 if bf16 else torch.float32)
        input_ids      = batch['input_ids'].to(device)
        labels         = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_flags    = batch['image_flags'].to(device)

        if amp_dtype:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    labels=labels,
                )
        else:
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                labels=labels,
            )

        if outputs.loss is not None:
            total_loss += outputs.loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# モデル保存
# ============================================================

def save_model(model, tokenizer, output_dir: str) -> None:
    """
    モデルとトークナイザーを保存する。
    LoRA モデルの場合は LoRA アダプターのみを保存。

    引数:
      model      : 保存するモデル
      tokenizer  : 保存するトークナイザー
      output_dir : str  保存先ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)

    # LoRA モデルかどうか確認
    from peft import PeftModel
    if isinstance(model.language_model, PeftModel):
        # LoRA アダプターのみ保存 (サイズ小)
        model.language_model.save_pretrained(output_dir)
        print(f"    LoRA アダプター保存: {output_dir}")
        # MLP Projector の重みも保存
        mlp_path = os.path.join(output_dir, 'mlp1.pt')
        torch.save(model.mlp1.state_dict(), mlp_path)
        print(f"    MLP Projector 保存: {mlp_path}")
    else:
        # フルファインチューニングの場合: 全体を保存
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)


# ============================================================
# コマンドライン引数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='InternVL3.5 ファインチューニング')
    parser.add_argument('--model_path',   type=str, required=True,
                        help='モデルパスまたはHugging Face ID (例: OpenGVLab/InternVL3_5-8B)')
    parser.add_argument('--train_csv',    type=str, default=None,
                        help='学習CSVファイルパス (image, question, answer カラム必須)')
    parser.add_argument('--eval_csv',     type=str, default=None,
                        help='評価CSVファイルパス (省略可)')
    parser.add_argument('--output_dir',   type=str, default='./internvl35-finetuned')
    parser.add_argument('--image_dir',    type=str, default=None,
                        help='画像ベースディレクトリ (CSVのパスが相対パスの場合)')
    parser.add_argument('--epochs',       type=int,   default=3)
    parser.add_argument('--batch_size',   type=int,   default=1)
    parser.add_argument('--grad_acc',     type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len',  type=int,   default=4096)
    parser.add_argument('--max_tiles',    type=int,   default=6)
    parser.add_argument('--lora',         action='store_true', help='LoRA を使用')
    parser.add_argument('--lora_r',       type=int,   default=128)
    parser.add_argument('--lora_alpha',   type=int,   default=256)
    parser.add_argument('--freeze_vit',   action='store_true', default=True,
                        help='ViT を凍結する (デフォルト: True)')
    parser.add_argument('--no_freeze_vit', dest='freeze_vit', action='store_false')
    parser.add_argument('--bf16',         action='store_true', default=True)
    parser.add_argument('--fp16',         action='store_true')
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--save_steps',   type=int, default=500)
    parser.add_argument('--eval_steps',   type=int, default=200)
    return parser.parse_args()


# ============================================================
# エントリポイント
# ============================================================

if __name__ == '__main__':
    args = parse_args()

    # ---- DataFrame の準備 ----
    if args.train_csv:
        train_df = pd.read_csv(args.train_csv)
        print(f"学習データ: {len(train_df)}サンプル")
    else:
        # デモ用ダミー DataFrame
        print("警告: --train_csv が指定されていないためデモ用データを使用します。")
        print("      実際の学習には --train_csv で CSV ファイルを指定してください。")

        # ダミーデータのセットアップ (実際には存在しないパスを使用)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # ダミー画像を作成
            dummy_img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))
            img_path = os.path.join(tmpdir, 'dummy.jpg')
            dummy_img.save(img_path)

            train_df = pd.DataFrame({
                'image':    [img_path] * 4,
                'question': [
                    'この画像を説明してください。',
                    'What objects are in this image?',
                    '画像の中心にあるものは何ですか？',
                    'Describe the color scheme of this image.',
                ],
                'answer': [
                    'この画像には様々な色彩のランダムなパターンが含まれています。',
                    'The image contains random colorful patterns.',
                    '中心部には色彩豊かなランダムなパターンがあります。',
                    'The image has a diverse color scheme with various hues.',
                ],
            })

            eval_df = pd.DataFrame({
                'image':    [img_path] * 2,
                'question': [
                    'この画像を簡単に説明してください。',
                    'What do you see in this image?',
                ],
                'answer': [
                    'ランダムなカラーパターンの画像です。',
                    'I see a random colorful pattern.',
                ],
            })

            print(f"\n[デモモード] ダミーデータセット情報:")
            print(f"  学習データ: {len(train_df)}サンプル")
            print(f"  評価データ: {len(eval_df)}サンプル")
            print(f"  DataFrame カラム: {list(train_df.columns)}")
            print(f"\n  サンプル確認:")
            print(train_df.head(2).to_string(max_colwidth=50))

            # InternVL35Dataset の動作確認 (モデルロードなし)
            print(f"\n[InternVL35Dataset 動作確認]")
            print("  ※ 実際のモデルロードはスキップ (デモモード)")

            # ダミートークナイザーで Dataset の形状確認
            class _DummyTokenizer:
                pad_token_id = 0
                def __call__(self, text, return_tensors='pt', truncation=False,
                             max_length=None, add_special_tokens=True):
                    # <IMG_CONTEXT> の数を数えてシーケンス長に反映
                    n_ctx = text.count('<IMG_CONTEXT>')
                    n_words = len(text.split())
                    n = min(n_ctx + n_words + 5, max_length or 4096)
                    return {'input_ids': torch.randint(1, 1000, (1, n))}
                def convert_tokens_to_ids(self, token):
                    return 999

            dummy_tok = _DummyTokenizer()

            dataset = InternVL35Dataset(
                df=train_df,
                tokenizer=dummy_tok,
                num_image_token=256,
                default_max_tiles=6,
                max_seq_len=1024,
            )

            print(f"\n  データセット長: {len(dataset)}")
            sample = dataset[0]
            print(f"\n  サンプル[0] の形状:")
            for k, v in sample.items():
                print(f"    {k:20s}: {v.shape}")

            # コレーターのテスト
            collator = InternVL35Collator(pad_token_id=0)
            batch = collator([dataset[0], dataset[1]])
            print(f"\n  バッチ (B=2) の形状:")
            for k, v in batch.items():
                print(f"    {k:20s}: {v.shape}")

            print("\n[コレーター 形状確認]")
            print(f"  pixel_values:   (P_total, 3, 448, 448) "
                  f"※ P_total={batch['pixel_values'].shape[0]} (全パッチ結合)")
            print(f"  input_ids:      (B=2, N_max={batch['input_ids'].shape[1]}) ※ 右パディング済み")
            print(f"  labels:         (B=2, N_max) ※ プロンプト部分 IGNORE_INDEX(-100)")
            print(f"  image_flags:    (P_total, 1) ※ 全て1 (有効パッチ)")
            print(f"  attention_mask: (B=2, N_max) ※ パディング部分0")
            print(f"  loss_weight:    (B=2, N_max) ※ 回答部分に 1/N^0.6 重み")

            print("\n" + "=" * 60)
            print("全テスト完了! 実際の学習を行う場合:")
            print("  python finetuning_example.py \\")
            print("    --model_path OpenGVLab/InternVL3_5-8B \\")
            print("    --train_csv /path/to/train.csv \\")
            print("    --eval_csv /path/to/eval.csv \\")
            print("    --output_dir ./internvl35-finetuned \\")
            print("    --lora --lora_r 128 \\")
            print("    --epochs 3 --batch_size 1 --grad_acc 16")
            print("=" * 60)
            sys.exit(0)

    # ---- 評価データ ----
    eval_df = None
    if args.eval_csv:
        eval_df = pd.read_csv(args.eval_csv)
        print(f"評価データ: {len(eval_df)}サンプル")

    # ---- モデルロード ----
    model, tokenizer = load_model_for_finetuning(
        model_path=args.model_path,
        use_lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_vit=args.freeze_vit,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    # ---- 学習実行 ----
    train(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        eval_df=eval_df,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_acc=args.grad_acc,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        max_tiles=args.max_tiles,
        save_every_n_steps=args.save_steps,
        eval_every_n_steps=args.eval_steps,
        bf16=args.bf16,
        fp16=args.fp16,
    )
