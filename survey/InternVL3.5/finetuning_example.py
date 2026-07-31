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
from tqdm.auto import tqdm


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

def _find_submodule(model, attr_names):
    """
    モデルから指定属性名のサブモジュールを探す (深さ 1 および 2 に対応)。

    trust_remote_code 版 (InternVLChatModel):
        model.language_model, model.mlp1, model.vision_model  ← 深さ 1
    HF ネイティブ版 (InternVLForConditionalGeneration):
        model.model.language_model, model.model.multi_modal_projector ← 深さ 2

    返値: (parent_module, attr_name) または (None, None)
    """
    for a in attr_names:
        if hasattr(model, a):
            return model, a
    # 深さ 2: model.model.*
    inner = getattr(model, 'model', None)
    if inner is not None:
        for a in attr_names:
            if hasattr(inner, a):
                return inner, a
    return None, None

def load_model_for_finetuning(
    model_path: str,
    use_lora: bool = True,
    lora_r: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
    freeze_vit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = 'auto',
    quantization: Optional[str] = None,
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
      quantization : str | None  '4bit' / '8bit' / None (QLoRA 用)

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

    # 量子化設定 (QLoRA)
    _bnb_cfg = None
    if quantization is not None:
        from transformers import BitsAndBytesConfig
        if quantization == '4bit':
            _bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        elif quantization == '8bit':
            _bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"quantization は '4bit' / '8bit' / None のみ有効: {quantization!r}")
        print(f"  量子化: {quantization} (QLoRA)")

    # モデルをロード
    # ----------------------------------------------------------------
    # ロード戦略 (優先順位順):
    #
    # [1] HF ネイティブ LM ヘッド付きクラス (InternVLForConditionalGeneration など)
    #     → -HF 版に対応。outputs.loss / logits が得られる
    #       InternVLConfig は AutoModelForCausalLM に登録されていないため
    #       直接クラスをインポートして試みる
    #
    # [2] AutoModel + trust_remote_code=True
    #     → 非-HF 版 (InternVLChatModel) に対応。outputs.loss が得られる
    #       -HF 版でこれを使うと InternVLModel (LM ヘッドなし) がロードされ
    #       loss も logits も返らないため fine-tune 不可
    # ----------------------------------------------------------------
    import transformers as _tf

    _HF_NATIVE_CLASSES = ('InternVLForConditionalGeneration', 'InternVLForCausalLM')
    _model = None
    _is_hf_native = False

    for _cls_name in _HF_NATIVE_CLASSES:
        # transformers のトップレベル or submodule から探す
        _cls = getattr(_tf, _cls_name, None)
        if _cls is None:
            try:
                from transformers.models.internvl import modeling_internvl as _m
                _cls = getattr(_m, _cls_name, None)
            except Exception:
                pass
        if _cls is None:
            continue
        try:
            _model = _cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                quantization_config=_bnb_cfg,
            )
            _is_hf_native = True
            print(f"  (HF ネイティブ {_cls_name}: {type(_model).__name__})")
            break
        except Exception as _e:
            print(f"  {_cls_name} 失敗: {type(_e).__name__}: {_e}")

    if _model is None:
        # フォールバック: trust_remote_code (非-HF 版の InternVLChatModel など)
        _model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=_bnb_cfg,
        )
        _cls_name = type(_model).__name__
        print(f"  (trust_remote_code: {_cls_name})")
        # -HF モデルを trust_remote_code でロードすると InternVLModel になる
        # → LM ヘッドがなく loss/logits が返らないため fine-tune 不可
        if 'Chat' not in _cls_name and 'ForCausal' not in _cls_name and 'ForConditional' not in _cls_name:
            print(f"\n  ⚠️  警告: {_cls_name} は LM ヘッドを持たないベースモデルです。")
            print(f"     fine-tune には outputs.loss / logits が必要ですが、このクラスは返しません。")
            print(f"     → 非-HF 版 (例: OpenGVLab/InternVL3_5-1B) の使用を推奨します。")

    model = _model

    # 量子化後処理: kbit 学習の準備 (LoRA 前に必須)
    if quantization is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("  prepare_model_for_kbit_training 完了")

    # モデルにタイプフラグを付与 (forward / loss 抽出で参照)
    model._is_hf_native = _is_hf_native
    # img_context_token_id は trust_remote_code 版 (InternVLChatModel) のみ使用
    if not _is_hf_native:
        model.img_context_token_id = IMG_CONTEXT_ID

    # ---- ViT を凍結 ----
    # trust_remote_code 版 (InternVLChatModel): model.vision_model
    # HF ネイティブ版 (InternVLModel):          model.vision_model が直接属性にない場合あり
    _VIT_NAMES = ('vision_model', 'visual_encoder', 'vit', 'vision_tower', 'visual')
    if freeze_vit:
        print("  ViT を凍結中...")
        _vit_module = next(
            (getattr(model, a) for a in _VIT_NAMES if hasattr(model, a)),
            None,
        )
        if _vit_module is not None:
            _vit_module.requires_grad_(False)
            vit_params = sum(p.numel() for p in _vit_module.parameters())
        else:
            # フォールバック: 名前パターンで凍結 (HF 版など直接属性がない場合)
            vit_params = 0
            for name, param in model.named_parameters():
                if any(k in name for k in _VIT_NAMES):
                    param.requires_grad = False
                    vit_params += param.numel()
        print(f"  ViT パラメータ数 (凍結): {vit_params:,}")

    # ---- LoRA を適用 ----
    if use_lora:
        print(f"  LoRA を適用中 (r={lora_r}, alpha={lora_alpha})...")

        # LLM のアーキテクチャを取得
        # trust_remote_code 版: model.config.llm_config.architectures
        # HF ネイティブ版:       model.config.text_config.architectures
        _cfg = model.config
        llm_arch = ''
        for _sub in ('llm_config', 'text_config', 'language_config'):
            _sub_cfg = getattr(_cfg, _sub, None)
            if _sub_cfg is not None and hasattr(_sub_cfg, 'architectures'):
                llm_arch = (_sub_cfg.architectures or [''])[0]
                break
        if not llm_arch:
            llm_arch = (getattr(_cfg, 'architectures', None) or [''])[0]

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

        # LLM サブモジュールを探す
        # trust_remote_code 版: model.language_model               (深さ 1)
        # HF ネイティブ版:       model.model.language_model          (深さ 2)
        _LLM_NAMES = ('language_model', 'llm', 'text_model', 'decoder')
        _llm_parent, _llm_attr = _find_submodule(model, _LLM_NAMES)
        if _llm_parent is None:
            raise RuntimeError(
                f"LLM サブモジュールが見つかりません。試行した属性: {_LLM_NAMES}\n"
                f"model の属性一覧: {[n for n, _ in model.named_children()]}"
            )
        _llm_module = getattr(_llm_parent, _llm_attr)

        # task_type=CAUSAL_LM は prepare_inputs_for_generation が必要。
        # HF 版 Qwen3Model (ベースモデル) はそれを持たないため None にフォールバック。
        _task_type = TaskType.CAUSAL_LM if hasattr(_llm_module, 'prepare_inputs_for_generation') else None

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias='none',
            task_type=_task_type,
        )
        if _task_type is None:
            print(f"  (task_type=None: {type(_llm_module).__name__} は CausalLM ではなくベースモデル)")

        _llm_module = get_peft_model(_llm_module, lora_config)
        _llm_module.enable_input_require_grads()
        _llm_module.print_trainable_parameters()
        setattr(_llm_parent, _llm_attr, _llm_module)

    # ---- MLP Projector は常に学習可能 ----
    # trust_remote_code 版: model.mlp1                         (深さ 1)
    # HF ネイティブ版:       model.model.multi_modal_projector  (深さ 2)
    _MLP_NAMES = ('mlp1', 'multi_modal_projector', 'projector', 'connector', 'vision_proj')
    _mlp_parent, _mlp_attr = _find_submodule(model, _MLP_NAMES)
    if _mlp_parent is not None:
        _mlp_mod = getattr(_mlp_parent, _mlp_attr)
        _mlp_mod.requires_grad_(True)
        mlp_params = sum(p.numel() for p in _mlp_mod.parameters() if p.requires_grad)
        print(f"  MLP Projector ({_mlp_attr}) 学習可能パラメータ: {mlp_params:,}")
    else:
        print("  MLP Projector: 対応属性が見つかりません (スキップ)")

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
    log_preview_df: Optional[pd.DataFrame] = None,
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
    save_checkpoint: bool = False,
    eval_every_n_steps: int = 200,
    log_steps: int = 10,
    max_grad_norm: float = 1.0,
    fp16: bool = False,
    bf16: bool = True,
) -> None:
    """
    InternVL3.5 のファインチューニング学習ループ。

    引数:
      model            : load_model_for_finetuning でロードしたモデル
      tokenizer        : 対応するトークナイザー
      train_df         : pd.DataFrame  学習データ (image, question, answer 必須)
      eval_df          : pd.DataFrame  評価データ (省略可)
      log_preview_df   : pd.DataFrame  log_steps ごとに generate 表示する固定サンプル
                         (None の場合スキップ。eval_df から事前に数件サンプリングして渡す)
                             log_preview_df = eval_df.sample(n=2, random_state=42)
      output_dir       : str  チェックポイント保存先
      epochs           : int  エポック数
      batch_size       : int  ミニバッチサイズ
      grad_acc         : int  勾配累積ステップ数 (実効バッチ = batch_size × grad_acc)
      lr               : float  学習率
      weight_decay     : float  AdamW 重み減衰
      warmup_ratio     : float  ウォームアップ比率
      max_seq_len      : int  最大系列長
      num_workers      : int  DataLoader ワーカー数
      max_tiles        : int  最大タイル数
      num_image_token  : int  1パッチあたりの IMG_CONTEXT 数
      save_every_n_steps: int  チェックポイント保存間隔 (optimizer step 単位)
      eval_every_n_steps: int  中間評価間隔 (optimizer step 単位)
      log_steps        : int  ロス表示 + 生成プレビュー間隔 (optimizer step 単位)
      max_grad_norm    : float  勾配クリッピング閾値
      fp16             : bool  FP16 混合精度
      bf16             : bool  BF16 混合精度 (A100/H100 推奨)
    """

    os.makedirs(output_dir, exist_ok=True)

    # ---- デバイス設定 ----
    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]

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
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ---- スケジューラー (コサインアニーリング + ウォームアップ) ----
    total_steps = len(train_loader) // grad_acc * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- 混合精度スケーラー ----
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)
    scaler = torch.cuda.amp.GradScaler() if (fp16 and not bf16) else None

    # ---- 設定サマリー ----
    precision_name = {torch.bfloat16: "bfloat16", torch.float16: "float16"}.get(amp_dtype, "float32")
    print(f"\n訓練設定:")
    print(f"  サンプル数:          train={len(train_dataset)}, eval={len(eval_df) if eval_df is not None else 0}")
    print(f"  バッチサイズ:        {batch_size}  (実効: {batch_size * grad_acc})")
    print(f"  エポック数:          {epochs}")
    print(f"  総ステップ数:        {total_steps}  (warmup: {warmup_steps})")
    print(f"  学習率:              {lr}")
    print(f"  Precision:           {precision_name}")
    print(f"  訓練可能パラメータ:  {sum(p.numel() for p in trainable_params):,}")
    print(f"  デバイス:            {device}")
    print("-" * 60)

    # ---- 初期評価 ----
    optimizer_step = 0
    if eval_loader is not None:
        init_eval = evaluate(model, eval_loader, device, amp_dtype, bf16)
        print(f"初期評価  loss: {init_eval['loss']:.4f}")
    if log_preview_df is not None:
        log_sample_predictions(model, tokenizer, log_preview_df, device, optimizer_step,
                                amp_dtype=amp_dtype, bf16=bf16)

    # ---- 学習ループ ----
    global_step = 0
    running_loss = 0.0
    model.train()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        for batch in pbar:
            # バッチをデバイスに転送
            pixel_values   = batch['pixel_values'].to(device, dtype=torch.bfloat16 if bf16 else torch.float32)
            input_ids      = batch['input_ids'].to(device)
            labels         = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_flags    = batch['image_flags'].to(device)
            loss_weight    = batch['loss_weight'].to(device, dtype=torch.float32)

            # フォワードパス
            outputs = _model_forward(
                model, pixel_values, input_ids, attention_mask, image_flags,
                labels, loss_weight, amp_dtype=amp_dtype, bf16=bf16,
            )
            loss = _extract_loss(outputs, labels) / grad_acc

            running_loss += loss.item() * grad_acc

            # バックプロパゲーション
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1

            # 勾配累積: grad_acc ステップごとに更新
            is_update_step = (
                global_step % grad_acc == 0
                or global_step == len(train_loader)
            )
            if is_update_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                avg_loss = running_loss / grad_acc
                running_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]

                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}",
                                 step=optimizer_step)

                # ---- ロギング + 生成プレビュー ----
                if optimizer_step % log_steps == 0:
                    print(f"  [Step {optimizer_step}/{total_steps}] "
                          f"loss={avg_loss:.4f}, lr={current_lr:.2e}")
                    if log_preview_df is not None:
                        log_sample_predictions(model, tokenizer, log_preview_df, device,
                                               optimizer_step, amp_dtype=amp_dtype, bf16=bf16,
                                               display_image=False)

                # ---- 中間評価 ----
                if eval_loader is not None and optimizer_step % eval_every_n_steps == 0:
                    eval_result = evaluate(model, eval_loader, device, amp_dtype, bf16)
                    print(f"  [Eval Step {optimizer_step}] eval_loss={eval_result['loss']:.4f}")
                    model.train()

                # ---- チェックポイント保存 ----
                if save_checkpoint and optimizer_step % save_every_n_steps == 0:
                    ckpt_path = os.path.join(output_dir, f'checkpoint-{optimizer_step}')
                    save_model(model, tokenizer, ckpt_path)
                    print(f"  チェックポイント保存: {ckpt_path}")

        # ---- エポック末評価 ----
        if eval_loader is not None:
            epoch_eval = evaluate(model, eval_loader, device, amp_dtype, bf16)
            print(f"  Epoch {epoch + 1} 評価 loss: {epoch_eval['loss']:.4f}")

    # ---- 最終モデル保存 ----
    if save_checkpoint:
        final_path = os.path.join(output_dir, 'final')
        save_model(model, tokenizer, final_path)
        print(f"\n学習完了。最終モデル保存: {final_path}")


# ============================================================
# 評価
# ============================================================

def _model_forward(model, pixel_values, input_ids, attention_mask, image_flags,
                   labels, loss_weight, amp_dtype, bf16):
    """
    trust_remote_code 版と HF ネイティブ版で forward の kwargs が異なるため吸収する。

    InternVLChatModel (trust_remote_code, 非-HF 版):
        image_flags, loss_weight を受け付ける独自 forward → outputs.loss あり

    InternVLForConditionalGeneration など (HF ネイティブ):
        pixel_values / input_ids / attention_mask / labels のみ
        image_flags・loss_weight は不要 (渡すと TypeError)
        → outputs.loss および outputs.logits あり

    InternVLModel (HF ネイティブ, ベースクラス):
        loss も logits も返さない → fine-tune 不可
    """
    is_hf = getattr(model, '_is_hf_native', False)
    ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if (amp_dtype and torch.cuda.is_available()) else _null_ctx()

    with ctx:
        if is_hf:
            # HF ネイティブ系: image_flags / loss_weight は渡さない
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            # trust_remote_code (InternVLChatModel): image_flags / loss_weight が必要
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                labels=labels,
                loss_weight=loss_weight.tolist() if loss_weight is not None else None,
            )
    return outputs


def _extract_loss(outputs, labels):
    """
    outputs から loss を取り出す。loss 属性がない場合は logits から手動計算する。

    trust_remote_code 版: outputs.loss が直接得られる
    HF ネイティブ版:      outputs.loss があれば使用、なければ logits から CE 損失を計算
    """
    loss = getattr(outputs, 'loss', None)
    if loss is not None:
        return loss

    logits = getattr(outputs, 'logits', None)
    if logits is not None:
        import torch.nn.functional as F
        # logits: (B, L, V) → (B*L, V)
        # labels: (B, L)    → (B*L,)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1).to(logits.device),
            ignore_index=IGNORE_INDEX,
        )

    raise RuntimeError(
        f"損失計算不可: {type(outputs).__name__} に 'loss' も 'logits' もありません。\n"
        f"AutoModelForCausalLM でロードされているか確認してください。"
    )


class _null_ctx:
    """torch.autocast の代わりに使う no-op コンテキストマネージャ"""
    def __enter__(self): return self
    def __exit__(self, *_): pass


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

    for batch in tqdm(eval_loader):
        pixel_values   = batch['pixel_values'].to(device, dtype=torch.bfloat16 if bf16 else torch.float32)
        input_ids      = batch['input_ids'].to(device)
        labels         = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_flags    = batch['image_flags'].to(device)

        outputs = _model_forward(
            model, pixel_values, input_ids, attention_mask, image_flags,
            labels, loss_weight=None, amp_dtype=amp_dtype, bf16=bf16,
        )
        loss = _extract_loss(outputs, labels)
        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


# ============================================================
# 単一サンプル推論
# ============================================================

def generate_response(
    model,
    tokenizer,
    image_path: Optional[str] = None,
    question: str = "",
    max_new_tokens: int = 512,
    device: Optional[torch.device] = None,
    bf16: bool = True,
) -> str:
    """
    単一サンプルの推論 (ロギング用)

    ========================================
    Shape
    ========================================
    入力:
        image_path: str (画像パス)
        question:   str (質問テキスト, <image> タグ含んでもよい)

    内部:
        pixel_values : (N_tiles, 3, 448, 448)  タイル画像
        input_ids    : (1, T_seq)               テキストトークン列
        output_ids   : (1, T_seq + T_gen)       生成トークン列

    出力:
        response : str  生成テキスト
    """
    if device is None:
        device = next(model.parameters()).device

    # ---- 画像の前処理 ----
    pixel_values = None
    num_patches_list = None
    if image_path and Path(image_path).exists():
        from PIL import Image as PILImage
        pil_img = PILImage.open(image_path).convert("RGB")
        tiles = dynamic_preprocess(pil_img, tile_size=448, max_tiles=6, use_thumbnail=True)
        # tiles: List[PIL.Image]  各タイルを前処理してスタック
        pixel_values = torch.stack([preprocess_tile(t) for t in tiles])
        # pixel_values: (N_tiles, 3, 448, 448)
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16 if bf16 else torch.float32)
        num_patches_list = [len(tiles)]

    # ---- テキストプロンプト構築 ----
    # <image> タグが含まれていない場合は先頭に付加
    if pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'chat'):
            # trust_remote_code 版 (InternVLChatModel): model.chat() が利用可能
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                ),
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        else:
            # HF ネイティブ版: 学習時と同じチャットテンプレートでプロンプトを構築する
            # 素のテキストを渡すと <|im_start|>assistant\n が欠けてループ生成になる
            DEFAULT_SYSTEM = (
                "You are InternVL3.5, created by Shanghai AI Laboratory. "
                "You are a helpful assistant."
            )
            if pixel_values is not None:
                n_tiles = pixel_values.shape[0]  # (N_tiles, 3, 448, 448)
                img_context = IMG_CONTEXT_TOKEN * (256 * n_tiles)
                img_tag = f'{IMG_START_TOKEN}{img_context}{IMG_END_TOKEN}'
                user_content = question.replace('<image>', img_tag) if '<image>' in question else img_tag + '\n' + question
            else:
                user_content = question

            prompt = (
                f'<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n'
                f'<|im_start|>user\n{user_content}<|im_end|>\n'
                f'<|im_start|>assistant\n'
            )

            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
            # input_ids: (1, T_seq)
            if pixel_values is not None:
                inputs['pixel_values'] = pixel_values
                # pixel_values: (N_tiles, 3, 448, 448)  ← バッチ次元は不要

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            # output_ids: (1, T_seq + T_gen)
            input_len = inputs['input_ids'].shape[1]
            response = tokenizer.decode(
                output_ids[0, input_len:], skip_special_tokens=True
            )

    return response


# ============================================================
# 学習中サンプル生成表示
# ============================================================

def log_sample_predictions(
    model,
    tokenizer,
    sample_df: pd.DataFrame,
    device: torch.device,
    global_step: int,
    max_new_tokens: int = 128,
    display_size: tuple = (300, 200),
    display_image: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
    bf16: bool = True,
):
    """
    log_steps ごとに val サンプルを generate して画像・質問・生成文を表示する。

    呼び出し元で eval_df から固定サンプルを切り出して渡すことで、
    ステップをまたいで同じサンプルに対する生成変化を追える。

        # 呼び出し側での準備例
        log_preview_df = eval_df.sample(n=2, random_state=42).reset_index(drop=True)
        # log_steps ごとに
        log_sample_predictions(model, tokenizer, log_preview_df, device, global_step)

    DataFrame カラム:
        image    : str  画像ファイルパス
        question : str  質問テキスト
        answer   : str  正解テキスト

    引数:
        sample_df     : 表示対象の固定サンプル DataFrame (呼び出し元で固定しておく)
        global_step   : 現在のステップ数 (表示用ラベル)
        max_new_tokens: 生成の最大トークン数 (ログ用なので短めでよい)
        display_size  : Jupyter 表示時のリサイズ後サイズ (width, height)
        amp_dtype     : 混合精度 dtype (generate_response には未使用、シグネチャ統一のため)
        bf16          : BF16 を使うか
    """
    model.eval()

    print(f"\n{'='*60}")
    print(f"[Step {global_step}] val サンプル生成プレビュー")
    print(f"{'='*60}")

    for i, (_, row) in enumerate(sample_df.iterrows()):
        image_path   = row.get("image", None)
        question_raw = row.get("question", "")
        ground_truth = row.get("answer", "")

        # <image> タグを除いた表示用質問テキスト
        question_display = question_raw.replace("<image>", "").strip()

        print(f"\n--- サンプル {i + 1} ---")
        print(f"質問: {question_display[:120]}{'...' if len(question_display) > 120 else ''}")
        print(f"正解: {str(ground_truth)[:120]}{'...' if len(str(ground_truth)) > 120 else ''}")

        # 画像表示 (Jupyter のみ、失敗時はパスを print してフォールバック)
        if display_image and image_path and Path(str(image_path)).exists():
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
                question=question_raw,
                max_new_tokens=max_new_tokens,
                device=device,
                bf16=bf16,
            )
            print(f"生成: {response[:200]}{'...' if len(response) > 200 else ''}")
        except Exception as e:
            print(f"生成エラー: {e}")

    print(f"{'='*60}\n")
    model.train()


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

    from peft import PeftModel

    # LLM サブモジュールを探す (trust_remote_code 版は深さ 1、HF 版は深さ 2)
    _LLM_NAMES = ('language_model', 'llm', 'text_model', 'decoder')
    _llm_parent, _llm_attr = _find_submodule(model, _LLM_NAMES)
    _llm_module = getattr(_llm_parent, _llm_attr) if _llm_parent is not None else None

    if _llm_module is not None and isinstance(_llm_module, PeftModel):
        # LoRA アダプターのみ保存 (サイズ小)
        _llm_module.save_pretrained(output_dir)
        print(f"    LoRA アダプター保存: {output_dir}")
        # MLP Projector の重みも保存
        _MLP_NAMES = ('mlp1', 'multi_modal_projector', 'projector', 'connector', 'vision_proj')
        _mlp_parent2, _mlp_attr2 = _find_submodule(model, _MLP_NAMES)
        if _mlp_parent2 is not None:
            mlp_path = os.path.join(output_dir, 'mlp_projector.pt')
            torch.save(getattr(_mlp_parent2, _mlp_attr2).state_dict(), mlp_path)
            print(f"    MLP Projector ({_mlp_attr2}) 保存: {mlp_path}")
    else:
        # フルファインチューニングの場合: 全体を保存
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)


# ============================================================
# ScienceQA データセット変換
# ============================================================

def convert_scienceqa_to_df(
    hf_dataset,
    image_save_dir: str,
    split: str = "train",
    skip_no_image: bool = True,
) -> pd.DataFrame:
    """
    HuggingFace の ScienceQA データセットを InternVL35Dataset 用 DataFrame に変換する。

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
        (InternVL35Dataset は画像をファイルパスで受け取るため)

    split : str
        保存ファイル名のプレフィックス用 (例: "train", "validation", "test")

    skip_no_image : bool
        image=None のサンプル (テキストのみ問題) をスキップするか。
        True にすると画像が必須の VLM ファインチューニングに適したデータになる。

    ========================================
    出力 DataFrame columns
    ========================================
    image    : str
        保存した画像ファイルの絶対パス
    question : str
        "<image>\\nヒント: {hint}\\n{問題文}\\nA. ...\\nB. ..." 形式
        ※ hint が空の場合は hint 行を省略
        ※ <image> タグは InternVL35Dataset._build_prompt が img_tokens に置換する
    answer   : str
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

    model, tokenizer = load_model_for_finetuning("OpenGVLab/InternVL3_5-8B")
    train(model, tokenizer, train_df=train_df, eval_df=eval_df,
          output_dir="./scienceqa-finetuned")
    """
    from pathlib import Path

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
        answer_idx   = sample["answer"]               # int (0始まり)
        answer_label = CHOICE_LABELS[answer_idx]      # "A", "B", ...
        answer_text  = sample["choices"][answer_idx]  # 正解の選択肢テキスト

        solution = (sample.get("solution") or "").strip()
        answer = f"答えは{answer_label}です。{answer_text}"
        if solution:
            answer += f"\n{solution}"

        rows.append({
            "image":    image_path,   # str (画像あり) or None (画像なし, skip_no_image=False時)
            "question": question,
            "answer":   answer,
        })

    df = pd.DataFrame(rows)
    print(f"ScienceQA 変換完了 [{split}]: {len(df)} サンプル "
          f"(画像あり: {df['image'].notna().sum()}, "
          f"画像なしスキップ: {skipped})")
    return df


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
    parser.add_argument('--quantization', type=str,   default=None,
                        choices=['4bit', '8bit'],
                        help='量子化モード (QLoRA): 4bit / 8bit')
    parser.add_argument('--freeze_vit',   action='store_true', default=True,
                        help='ViT を凍結する (デフォルト: True)')
    parser.add_argument('--no_freeze_vit', dest='freeze_vit', action='store_false')
    parser.add_argument('--bf16',         action='store_true', default=True)
    parser.add_argument('--fp16',         action='store_true')
    parser.add_argument('--num_workers',  type=int, default=2)
    parser.add_argument('--save_steps',    type=int,   default=500)
    parser.add_argument('--eval_steps',    type=int,   default=200)
    parser.add_argument('--log_steps',     type=int,   default=10,
                        help='ロス表示 + 生成プレビュー間隔 (optimizer step 単位)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='勾配クリッピング閾値')
    # ScienceQA
    parser.add_argument('--scienceqa',          action='store_true',
                        help='HuggingFace の ScienceQA で学習する (--train_csv より優先)')
    parser.add_argument('--scienceqa_image_dir', type=str, default='./scienceqa_images',
                        help='ScienceQA 画像の保存先ディレクトリ')
    parser.add_argument('--scienceqa_train_split', type=str, default='train',
                        help='ScienceQA 学習スプリット (例: "train" / "train[:500]")')
    parser.add_argument('--scienceqa_eval_split',  type=str, default='validation[:200]',
                        help='ScienceQA 評価スプリット (例: "validation[:200]")')
    return parser.parse_args()


# ============================================================
# エントリポイント
# ============================================================

if __name__ == '__main__':
    args = parse_args()

    # ---- DataFrame の準備 ----
    if args.scienceqa:
        # ScienceQA (HuggingFace) から DataFrame を作成
        from datasets import load_dataset
        print(f"ScienceQA をロード中 (train: {args.scienceqa_train_split})")
        hf_train = load_dataset("derek-thomas/ScienceQA", split=args.scienceqa_train_split)
        train_df = convert_scienceqa_to_df(
            hf_train,
            image_save_dir=args.scienceqa_image_dir,
            split="train",
        )
        eval_df = None
        if args.scienceqa_eval_split:
            print(f"ScienceQA をロード中 (eval: {args.scienceqa_eval_split})")
            hf_eval = load_dataset("derek-thomas/ScienceQA", split=args.scienceqa_eval_split)
            eval_df = convert_scienceqa_to_df(
                hf_eval,
                image_save_dir=args.scienceqa_image_dir,
                split="validation",
            )
        # モデルロード & 学習実行
        model, tokenizer = load_model_for_finetuning(
            model_path=args.model_path,
            use_lora=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            freeze_vit=args.freeze_vit,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
            quantization=args.quantization,
        )
        log_preview_df = eval_df.sample(n=min(2, len(eval_df)), random_state=42).reset_index(drop=True) \
            if eval_df is not None and len(eval_df) > 0 else None
        train(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            eval_df=eval_df,
            log_preview_df=log_preview_df,
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
            log_steps=args.log_steps,
            max_grad_norm=args.max_grad_norm,
            bf16=args.bf16,
            fp16=args.fp16,
        )
    elif args.train_csv:
        train_df = pd.read_csv(args.train_csv)
        print(f"学習データ: {len(train_df)}サンプル")
        eval_df = None
        if args.eval_csv:
            eval_df = pd.read_csv(args.eval_csv)
            print(f"評価データ: {len(eval_df)}サンプル")
        model, tokenizer = load_model_for_finetuning(
            model_path=args.model_path,
            use_lora=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            freeze_vit=args.freeze_vit,
            dtype=torch.bfloat16 if args.bf16 else torch.float32,
            quantization=args.quantization,
        )
        log_preview_df = eval_df.sample(n=min(2, len(eval_df)), random_state=42).reset_index(drop=True) \
            if eval_df is not None and len(eval_df) > 0 else None
        train(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            eval_df=eval_df,
            log_preview_df=log_preview_df,
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
            log_steps=args.log_steps,
            max_grad_norm=args.max_grad_norm,
            bf16=args.bf16,
            fp16=args.fp16,
        )
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
            print("  ScienceQA で学習する場合:")
            print("  python finetuning_example.py \\")
            print("    --model_path OpenGVLab/InternVL3_5-8B \\")
            print("    --scienceqa \\")
            print("    --scienceqa_train_split 'train[:500]' \\")
            print("    --scienceqa_eval_split  'validation[:100]' \\")
            print("    --scienceqa_image_dir   ./scienceqa_images \\")
            print("    --output_dir ./scienceqa-finetuned \\")
            print("    --lora --lora_r 128 \\")
            print("    --epochs 3 --batch_size 1 --grad_acc 16")
            print("=" * 60)
            sys.exit(0)
