"""
InternVL3.5 メインフロー - 推論・学習の全体パイプライン
=========================================================

このファイルは InternVL3.5 の画像→テキスト生成における
全体処理フローを一つのファイルにまとめたものです。

以下の2つのモードを実装:
  1. 推論フロー (generate): 画像 + テキストプロンプト → テキスト生成
  2. 学習フロー (training_step): 画像 + テキスト → 損失計算

処理の大きな流れ:
  [前処理]  元画像 → Dynamic Tiling → ViT 入力テンソル
  [ViT]     pixel_values → ViT 特徴 (1025 tokens/patch)
  [圧縮]    CLS 除去 → Pixel Shuffle → 256 tokens/patch
  [投影]    MLP Projector → LLM 次元特徴
  [結合]    IMG_CONTEXT トークンを視覚特徴で置換
  [LLM]     完全な入力埋め込みで次トークン予測

InternVL3.5-Flash の場合:
  [ViR]     各パッチのCLSトークンでルーティング → 256 or 64 tokens
  [圧縮]    ルーティング結果に応じて異なる Pixel Shuffle を適用

公式実装参考:
  modeling_internvl_chat.py (forward, generate, chat)
  dataset.py (preprocess, build_transform)

============================================================
テンソル形状記法
============================================================
  B       : バッチサイズ
  P_i     : i 番目サンプルのパッチ数
  P_total : 全バッチの合計パッチ数 = Σ P_i
  S_v     : ViT 系列長 = (image_size/patch_size)^2 + 1 = 1025
  S'_v    : CLS 除去後 = 1024
  N_tok   : 1パッチあたりのビジュアルトークン = 256 (標準) or 64 (高圧縮)
  N_text  : テキスト系列長 (IMG_CONTEXT を含む)
  D_v     : ViT hidden size (InternViT-6B: 3200)
  D_l     : LLM hidden size (モデルによって異なる)
  V       : 語彙サイズ
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# 本 understanding ディレクトリの他ファイルを参照
# (実際の実行では PYTHONPATH の設定が必要)
from model_architecture import (
    InternVisionModel, InternVLChatModel, build_mlp_projector
)
from dynamic_resolution import (
    prepare_image_patches, batch_prepare_images, build_image_token_string
)


# ============================================================
# 1. チャットテンプレート
# ============================================================

class InternVLChatTemplate:
    """
    InternVL3.5 の会話テンプレート。

    フォーマット (Qwen3 ベース):
      <|im_start|>system
      {system_message}<|im_end|>
      <|im_start|>user
      {user_message}<|im_end|>
      <|im_start|>assistant
      {assistant_message}<|im_end|>

    特殊トークン:
      <img>          : 画像開始
      </img>         : 画像終了
      <IMG_CONTEXT>  : 視覚特徴のプレースホルダー (256枚/パッチ)
    """
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

    SYSTEM_MESSAGE = (
        "You are InternVL3.5, created by Shanghai AI Laboratory. "
        "You are a helpful assistant."
    )

    @classmethod
    def build_prompt(
        cls,
        question: str,
        num_patches: int,
        num_image_token: int = 256,
        history: Optional[List[Tuple[str, str]]] = None,
        system_message: Optional[str] = None,
    ) -> str:
        """
        会話プロンプトを構築する。

        引数:
          question       : ユーザーの質問 (画像参照は <image> プレースホルダー)
          num_patches    : パッチ数 (サムネイル + タイル)
          num_image_token: 1パッチあたりの IMG_CONTEXT 数 (デフォルト 256)
          history        : 過去の (question, answer) ペアのリスト
          system_message : システムメッセージ (None でデフォルト使用)
        返値:
          prompt : str  完全なチャットプロンプト

        例 (画像1枚, 3パッチ, history なし):
          <|im_start|>system
          You are InternVL3.5...
          <|im_end|>
          <|im_start|>user
          <img><IMG_CONTEXT>×768</img>
          この画像を説明してください。<|im_end|>
          <|im_start|>assistant
        """
        system_msg = system_message or cls.SYSTEM_MESSAGE

        # 画像トークン文字列を構築 (256×num_patches 個の IMG_CONTEXT)
        image_token_str = build_image_token_string(
            num_patches=num_patches,
            num_image_token=num_image_token,
            img_start=cls.IMG_START_TOKEN,
            img_end=cls.IMG_END_TOKEN,
            img_context=cls.IMG_CONTEXT_TOKEN,
        )

        # <image> プレースホルダーを実際のトークン列に置換
        if '<image>' in question:
            processed_question = question.replace('<image>', image_token_str, 1)
        else:
            # <image> がない場合は先頭に追加
            processed_question = image_token_str + '\n' + question

        # プロンプト組み立て
        prompt = f'<|im_start|>system\n{system_msg}<|im_end|>\n'

        # 過去の会話を追加
        if history:
            for old_q, old_a in history:
                prompt += f'<|im_start|>user\n{old_q}<|im_end|>\n'
                prompt += f'<|im_start|>assistant\n{old_a}<|im_end|>\n'

        # 現在の質問を追加
        prompt += f'<|im_start|>user\n{processed_question}<|im_end|>\n'
        prompt += '<|im_start|>assistant\n'

        return prompt


# ============================================================
# 2. 推論パイプライン
# ============================================================

class InternVL35InferencePipeline:
    """
    InternVL3.5 の推論パイプライン。

    使い方:
      pipeline = InternVL35InferencePipeline(model, tokenizer)
      response = pipeline.generate(image, "この画像を説明してください。")
    """
    def __init__(
        self,
        model: InternVLChatModel,
        tokenizer,
        num_image_token: int = 256,
        max_tiles: int = 6,
        tile_size: int = 448,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.max_tiles = max_tiles
        self.tile_size = tile_size

        # IMG_CONTEXT トークン ID を設定
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(
            InternVLChatTemplate.IMG_CONTEXT_TOKEN
        )
        model.img_context_token_id = self.img_context_token_id

    @torch.no_grad()
    def generate(
        self,
        image,                         # PIL.Image
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ) -> str:
        """
        1枚の画像に対してテキストを生成する推論フロー。

        処理フロー:
          1. 画像を Dynamic Tiling → テンソル変換
             入力: PIL.Image (任意サイズ)
             出力: pixel_values (P, 3, 448, 448)

          2. プロンプト構築 + トークナイズ
             出力: input_ids (1, N_text)

          3. モデル forward
             ViT: (P, 3, 448, 448) → (P, 1025, D_v)
             Pixel Shuffle: (P, 1025, D_v) → (P, 256, D_v*4)
             MLP: (P, 256, D_v*4) → (P, 256, D_l)
             IMG_CONTEXT 置換: input_embeds (1, N_text, D_l)
             LLM decode: (1, N_text, D_l) → (1, N_gen)

          4. デコード: トークン列 → テキスト

        入力:
          image    : PIL.Image
          question : str (例: "この画像を説明してください。")
        出力:
          response : str  モデルの生成テキスト
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        # ステップ1: 画像前処理
        # pixel_values: (P, 3, 448, 448)
        pixel_values, num_patches = prepare_image_patches(
            image,
            tile_size=self.tile_size,
            max_tiles=self.max_tiles,
            use_thumbnail=True,
        )
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        # image_flags: (P, 1)  全て有効
        image_flags = torch.ones(num_patches, 1, dtype=torch.long, device=device)

        # ステップ2: プロンプト構築
        prompt = InternVLChatTemplate.build_prompt(
            question=question,
            num_patches=num_patches,
            num_image_token=self.num_image_token,
            history=history,
        )

        # ステップ3: トークナイズ
        # input_ids: (1, N_text)
        model_inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)

        # ステップ4: 生成
        # 出力: (1, N_gen)
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=eos_token_id,
        )

        # ステップ5: デコード (プロンプト部分を除いた生成トークンを取り出す)
        input_len = input_ids.shape[1]
        response_ids = generation_output[:, input_len:]
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        response = response.strip()

        return response

    @torch.no_grad()
    def batch_generate(
        self,
        images: List,
        questions: List[str],
        max_new_tokens: int = 512,
        do_sample: bool = False,
    ) -> List[str]:
        """
        複数画像を一括処理するバッチ推論フロー。

        入力:
          images    : List[PIL.Image]  長さ B
          questions : List[str]        長さ B
        出力:
          responses : List[str]        長さ B
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        # ステップ1: 全画像を一括前処理
        # pixel_values: (Σ P_i, 3, 448, 448)
        pixel_values, num_patches_list = batch_prepare_images(
            images,
            tile_size=self.tile_size,
            max_tiles=self.max_tiles,
        )
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

        # ステップ2: 各サンプルのプロンプトを構築 + トークナイズ
        prompts = []
        for question, n_patches in zip(questions, num_patches_list):
            prompt = InternVLChatTemplate.build_prompt(
                question=question,
                num_patches=n_patches,
                num_image_token=self.num_image_token,
            )
            prompts.append(prompt)

        # バッチトークナイズ (左パディング)
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True
        )
        input_ids = model_inputs['input_ids'].to(device)              # (B, N_max)
        attention_mask = model_inputs['attention_mask'].to(device)    # (B, N_max)

        # image_flags: (Σ P_i, 1)  全て有効
        total_patches = sum(num_patches_list)
        image_flags = torch.ones(total_patches, 1, dtype=torch.long, device=device)

        # ステップ3: バッチ生成
        eos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        )

        # ステップ4: デコード
        input_len = input_ids.shape[1]
        responses = []
        for i in range(len(images)):
            response_ids = generation_output[i, input_len:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            # <|im_end|> より後を切り捨て
            response = response.split('<|im_end|>')[0].strip()
            responses.append(response)

        return responses


# ============================================================
# 3. 学習用フォワードパス
# ============================================================

class InternVL35TrainingStep:
    """
    InternVL3.5 の1学習ステップを管理するクラス。

    SFT (Supervised Fine-Tuning) の学習ループを実装。
    損失計算:
      - NTP (Next Token Prediction) 損失
      - Square Averaging による重み付け (1/N^0.6)
      - アシスタントの回答部分のみに損失を計算 (-100 でマスク)
    """
    def __init__(
        self,
        model: InternVLChatModel,
        num_image_token: int = 256,
    ):
        self.model = model
        self.num_image_token = num_image_token

    def build_training_sample(
        self,
        pixel_values: torch.Tensor,
        num_patches: int,
        question: str,
        answer: str,
        tokenizer,
    ) -> Dict[str, torch.Tensor]:
        """
        1サンプルの学習データを構築する。

        入力:
          pixel_values : (P, 3, 448, 448)
          num_patches  : P
          question     : str  ユーザーの質問
          answer       : str  アシスタントの回答
          tokenizer    : トークナイザー
        出力:
          {
            'pixel_values' : (P, 3, 448, 448)
            'input_ids'    : (1, N)
            'attention_mask': (1, N)
            'labels'       : (1, N)  ※ 質問部分は -100 でマスク
            'image_flags'  : (P, 1)
            'loss_weight'  : (1, N)  ※ Square Averaging 用
          }
        """
        device = pixel_values.device

        # ---- 入力プロンプト構築 ----
        prompt = InternVLChatTemplate.build_prompt(
            question=question,
            num_patches=num_patches,
            num_image_token=self.num_image_token,
        )
        full_text = prompt + answer + '<|im_end|>'

        # ---- トークナイズ ----
        full_ids = tokenizer(full_text, return_tensors='pt')['input_ids']  # (1, N)
        prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']   # (1, N_prompt)

        N = full_ids.shape[1]
        N_prompt = prompt_ids.shape[1]

        # ---- ラベル作成: 質問部分をマスク ----
        labels = full_ids.clone()
        labels[:, :N_prompt] = -100  # プロンプト部分はマスク

        # ---- Square Averaging 重み ----
        # 回答トークン数で重み付け
        n_response_tokens = N - N_prompt
        weight = 1.0 / (n_response_tokens ** 0.6)
        loss_weight = torch.zeros(1, N)
        loss_weight[:, N_prompt:] = weight

        return {
            'pixel_values': pixel_values,
            'input_ids': full_ids.to(device),
            'attention_mask': torch.ones(1, N, dtype=torch.long, device=device),
            'labels': labels.to(device),
            'image_flags': torch.ones(num_patches, 1, dtype=torch.long, device=device),
            'loss_weight': loss_weight.to(device),
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        学習用フォワードパス。

        入力 batch:
          pixel_values  : (P_total, 3, 448, 448)
          input_ids     : (B, N)
          attention_mask: (B, N)
          labels        : (B, N)   -100=無視
          image_flags   : (P_total, 1)
          loss_weight   : (B, N)   Square Averaging 用重み

        処理フロー:
          Step1: ViT 特徴抽出
            (P_total, 3, 448, 448) → (P_total, 256, D_l)
          Step2: IMG_CONTEXT 置換
            input_embeds (B, N, D_l) の IMG_CONTEXT 位置に vit_embeds を挿入
          Step3: LLM フォワード
            (B, N, D_l) → logits (B, N, V)
          Step4: 損失計算 (Square Averaging)
            shift_logits × shift_labels × loss_weight → スカラー損失

        出力:
          {'loss': スカラー, 'logits': (B, N, V)}
        """
        self.model.train()

        outputs = self.model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            image_flags=batch['image_flags'],
            labels=batch['labels'],
            loss_weight=batch.get('loss_weight', None),
        )

        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }


# ============================================================
# 4. テスト時スケーリング (Test-Time Scaling)
# ============================================================

class BestOfNSampler:
    """
    Best-of-N (BoN) テスト時スケーリング。

    N 個の回答を生成して報酬モデル (VisualPRM-v1.1) で
    最良の回答を選択する推論戦略。

    適用場面: 複雑な多段推論問題 (数学, 科学等)
    """
    def __init__(
        self,
        pipeline: InternVL35InferencePipeline,
        reward_model: Optional[nn.Module] = None,
        n_samples: int = 8,
    ):
        self.pipeline = pipeline
        self.reward_model = reward_model
        self.n_samples = n_samples

    def generate_best_of_n(
        self,
        image,
        question: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Tuple[str, List[str]]:
        """
        N 個の回答を生成して最良のものを返す。

        入力:
          image    : PIL.Image
          question : str
        出力:
          best_response : str  最良の回答
          all_responses : List[str]  全 N 個の回答
        """
        # N 個の回答を生成 (do_sample=True で多様性)
        all_responses = []
        for i in range(self.n_samples):
            response = self.pipeline.generate(
                image=image,
                question=question,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            all_responses.append(response)

        # 報酬モデルで最良を選択
        if self.reward_model is not None:
            # VisualPRM-v1.1 でスコアリング (詳細は省略)
            best_idx = self._select_best(image, question, all_responses)
        else:
            # 報酬モデルなし: 最初の回答を返す
            best_idx = 0

        return all_responses[best_idx], all_responses

    def _select_best(self, image, question: str, responses: List[str]) -> int:
        """
        VisualPRM-v1.1 報酬モデルで最良の回答を選択。
        ※ 実装は報酬モデルの詳細に依存するため省略。
        返値: best_idx (int)
        """
        # 実装省略 (VisualPRM-v1.1 は別途公開)
        return 0


# ============================================================
# 5. 完全な推論フローのデモ
# ============================================================

class InternVL35ForwardPassDemo:
    """
    InternVL3.5 のフォワードパスをステップごとに可視化するデモクラス。

    実際のモデルをロードせず、ダミーテンソルで動作を確認する。
    """
    @staticmethod
    def demonstrate_forward_pass(
        batch_size: int = 2,
        patches_per_sample: List[int] = [3, 5],  # 各サンプルのパッチ数
        max_text_len: int = 128,
        vit_hidden: int = 1024,  # InternViT-300M (テスト用)
        llm_hidden: int = 2048,
        vocab_size: int = 1000,
        num_image_token: int = 256,
    ) -> None:
        """
        フォワードパスの各ステップの入出力形状を表示する。

        引数:
          batch_size          : バッチサイズ B
          patches_per_sample  : 各サンプルのパッチ数 [P_0, P_1, ...]
          max_text_len        : テキスト系列長
          vit_hidden          : ViT hidden size D_v
          llm_hidden          : LLM hidden size D_l
          vocab_size          : 語彙サイズ V
          num_image_token     : 1パッチあたりのビジュアルトークン数
        """
        assert len(patches_per_sample) == batch_size

        total_patches = sum(patches_per_sample)
        print(f"\n{'='*70}")
        print(f"InternVL3.5 フォワードパス デモ")
        print(f"  B={batch_size}, patches={patches_per_sample}, total_P={total_patches}")
        print(f"  ViT hidden={vit_hidden}, LLM hidden={llm_hidden}")
        print(f"{'='*70}")

        # ----------------------------------------------------------------
        # Step 0: 入力
        # ----------------------------------------------------------------
        print("\n[Step 0] 入力テンソル")
        pixel_values = torch.randn(total_patches, 3, 448, 448)
        # テキスト系列: IMG_CONTEXT × num_image_token × P_i + テキスト部
        n_visual_tokens_per_sample = [p * num_image_token for p in patches_per_sample]
        # バッチの最大系列長
        N = max(n_visual_tokens_per_sample) + max_text_len
        input_ids = torch.randint(0, vocab_size, (batch_size, N))
        attention_mask = torch.ones(batch_size, N, dtype=torch.long)
        image_flags = torch.ones(total_patches, 1, dtype=torch.long)

        print(f"  pixel_values : {pixel_values.shape}")
        print(f"    ↑ ({total_patches}パッチ, 3ch, 448px, 448px)")
        print(f"  input_ids    : {input_ids.shape}")
        print(f"    ↑ (B={batch_size}, N={N}=ビジュアルトークン+テキスト)")
        print(f"  image_flags  : {image_flags.shape}")

        # ----------------------------------------------------------------
        # Step 1: ViT 特徴抽出
        # ----------------------------------------------------------------
        print("\n[Step 1] InternViT による特徴抽出")
        # ViT 内部: patch embedding → transformer layers → CLS+patches
        S_v = (448 // 14) ** 2 + 1  # = 1025
        vit_output_full = torch.randn(total_patches, S_v, vit_hidden)
        print(f"  入力:  pixel_values        {pixel_values.shape}")
        print(f"    ↓ Conv2d patch embedding (kernel=14, stride=14)")
        print(f"    ↓ → (P_total, 1024, D_v) patch tokens")
        print(f"    ↓ + CLS token → (P_total, 1025, D_v)")
        print(f"    ↓ 48層 Transformer (QK Norm付き)")
        print(f"  出力:  vit_last_hidden     {vit_output_full.shape}")
        print(f"    ↑ (P_total={total_patches}, S_v=1025, D_v={vit_hidden})")

        # ----------------------------------------------------------------
        # Step 2: CLS 除去
        # ----------------------------------------------------------------
        print("\n[Step 2] CLS トークン除去")
        vit_patch_only = vit_output_full[:, 1:, :]  # (P_total, 1024, D_v)
        print(f"  入力:  vit_last_hidden     {vit_output_full.shape}")
        print(f"    ↓ vit_embeds[:, 1:, :]  ← CLS (index=0) を除去")
        print(f"  出力:                      {vit_patch_only.shape}")

        # ----------------------------------------------------------------
        # Step 3: 2D 空間変換 → Pixel Shuffle
        # ----------------------------------------------------------------
        print("\n[Step 3] Pixel Shuffle (トークン圧縮)")
        H_t = W_t = 32   # 448 / 14 = 32
        vit_2d = vit_patch_only.reshape(total_patches, H_t, W_t, vit_hidden)
        # scale_factor=0.5 → 16×16=256 tokens, チャンネル 4倍
        compressed_tokens = int(H_t * 0.5) ** 2   # = 256
        compressed_channels = vit_hidden * 4       # = vit_hidden * (1/0.5)^2
        vit_shuffled = torch.randn(total_patches, compressed_tokens, compressed_channels)

        print(f"  入力:  vit_patch_only      {vit_patch_only.shape}")
        print(f"    ↓ reshape → (P_total, H_t=32, W_t=32, D_v)")
        print(f"    ↓ pixel_shuffle (scale=0.5)")
        print(f"    ↓   → (P_total, 16, 16, D_v*4)")
        print(f"    ↓ reshape → (P_total, 256, D_v*4)")
        print(f"  出力:  vit_shuffled        {vit_shuffled.shape}")
        print(f"    ↑ (P_total={total_patches}, {compressed_tokens}tok, {compressed_channels}ch)")
        print(f"  圧縮率: 1024 → {compressed_tokens} トークン (4倍圧縮)")

        # ----------------------------------------------------------------
        # Step 4: MLP Projector
        # ----------------------------------------------------------------
        print("\n[Step 4] MLP Projector (ViT→LLM 次元変換)")
        vit_projected = torch.randn(total_patches, num_image_token, llm_hidden)

        print(f"  入力:  vit_shuffled        {vit_shuffled.shape}")
        print(f"    ↓ LayerNorm({compressed_channels})")
        print(f"    ↓ Linear({compressed_channels} → {llm_hidden})")
        print(f"    ↓ GELU")
        print(f"    ↓ Linear({llm_hidden} → {llm_hidden})")
        print(f"  出力:  vit_projected       {vit_projected.shape}")
        print(f"    ↑ (P_total={total_patches}, {num_image_token}tok, D_l={llm_hidden})")

        # ----------------------------------------------------------------
        # Step 5: テキスト埋め込み + IMG_CONTEXT 置換
        # ----------------------------------------------------------------
        print("\n[Step 5] テキスト埋め込み + IMG_CONTEXT 置換")
        D_l = llm_hidden
        input_embeds = torch.randn(batch_size, N, D_l)

        # IMG_CONTEXT トークンの位置に vit_projected を挿入
        # (P_total, 256, D_l) → (P_total*256, D_l) → selected 位置に代入
        n_visual_positions = total_patches * num_image_token
        print(f"  テキスト埋め込み input_embeds: {input_embeds.shape}")
        print(f"  視覚特徴 vit_projected:         {vit_projected.shape}")
        print(f"    ↓ IMG_CONTEXT 位置 (合計 {n_visual_positions} トークン) に視覚特徴を挿入")
        print(f"  置換後 input_embeds:            {input_embeds.shape}  (形状は変わらない)")
        print(f"    ↑ IMG_CONTEXT 位置には視覚特徴, 他はテキスト埋め込み")

        # ----------------------------------------------------------------
        # Step 6: LLM フォワード
        # ----------------------------------------------------------------
        print("\n[Step 6] Language Model フォワード (Qwen3 / GPT-OSS)")
        logits = torch.randn(batch_size, N, vocab_size)

        print(f"  入力:  input_embeds        {input_embeds.shape}")
        print(f"    ↓ Qwen3 Transformer (例: 32層, D_l={llm_hidden})")
        print(f"    ↓ FlashAttention-2")
        print(f"  出力:  logits              {logits.shape}")
        print(f"    ↑ (B={batch_size}, N={N}, V={vocab_size})")

        # ----------------------------------------------------------------
        # Step 7: 損失計算 (学習時)
        # ----------------------------------------------------------------
        print("\n[Step 7] 損失計算 (学習時のみ)")
        labels = torch.randint(-100, vocab_size, (batch_size, N))
        # シフト
        shift_logits = logits[:, :-1, :].contiguous()   # (B, N-1, V)
        shift_labels = labels[:, 1:].contiguous()         # (B, N-1)
        loss_weight = torch.ones(batch_size, N - 1) * 0.1
        mask = (shift_labels != -100)

        print(f"  labels       : {labels.shape}  (-100: 質問部分はマスク)")
        print(f"  shift_logits : {shift_logits.shape}  ← logits[:-1]")
        print(f"  shift_labels : {shift_labels.shape}  ← labels[1:]")
        print(f"  loss_weight  : {loss_weight.shape}  ← 1/N^0.6 (Square Averaging)")
        print(f"  有効トークン数: {mask.sum().item()}")

        # ロスの計算
        loss_fct = CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1).clamp(min=0)
        )  # (B*(N-1),)
        per_token_loss = per_token_loss * mask.view(-1).float()
        loss = per_token_loss.sum() / mask.sum().clamp(min=1)
        print(f"\n  最終損失: {loss.item():.4f} (スカラー)")

        # ----------------------------------------------------------------
        # サマリー
        # ----------------------------------------------------------------
        print(f"\n{'='*70}")
        print("フォワードパス サマリー")
        print(f"{'='*70}")
        print(f"  画像入力:    ({total_patches}, 3, 448, 448)")
        print(f"  ViT 出力:   ({total_patches}, 1025, {vit_hidden})")
        print(f"  圧縮後:      ({total_patches}, {num_image_token}, {compressed_channels})")
        print(f"  投影後:      ({total_patches}, {num_image_token}, {llm_hidden})")
        print(f"  LLM 入力:   ({batch_size}, {N}, {llm_hidden})")
        print(f"  LLM 出力:   ({batch_size}, {N}, {vocab_size})")
        print(f"  総ビジュアルトークン: {total_patches * num_image_token}")
        print(f"    内訳: {[f'サンプル{i}: {p}パッチ×{num_image_token}tok={p*num_image_token}' for i, p in enumerate(patches_per_sample)]}")


# ============================================================
# 使用例
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("InternVL3.5 メインフロー 動作確認")
    print("=" * 70)

    # --- 1. チャットテンプレートテスト ---
    print("\n[1] チャットテンプレートテスト")
    prompt = InternVLChatTemplate.build_prompt(
        question="<image>\nこの画像を日本語で詳しく説明してください。",
        num_patches=3,
        num_image_token=256,
        history=None,
    )
    # IMG_CONTEXT 数を確認
    n_context = prompt.count(InternVLChatTemplate.IMG_CONTEXT_TOKEN)
    n_img_start = prompt.count(InternVLChatTemplate.IMG_START_TOKEN)
    n_img_end = prompt.count(InternVLChatTemplate.IMG_END_TOKEN)

    print(f"  <IMG_CONTEXT> トークン数: {n_context} (期待: 768 = 3パッチ×256)")
    print(f"  <img> / </img> 数: {n_img_start} / {n_img_end}")

    # プロンプトの最初と最後を表示
    preview = prompt[:200].replace(InternVLChatTemplate.IMG_CONTEXT_TOKEN, '.')
    print(f"\n  プロンプト冒頭 (IMG_CONTEXT='.'に置換):")
    print(f"  {preview}...")
    assert n_context == 768, f"期待: 768, 実際: {n_context}"
    print("  OK: IMG_CONTEXT 数が正しい")

    # --- 2. 学習サンプル構築テスト ---
    print("\n[2] 学習サンプル構築テスト (build_training_sample)")
    training_step = InternVL35TrainingStep(
        model=None,  # モデル不要 (形状確認のみ)
        num_image_token=256,
    )

    # ダミートークナイザー
    class DummyTokenizer:
        def __call__(self, text, return_tensors='pt'):
            n = max(10, len(text.split()) + text.count('<IMG_CONTEXT>'))
            input_ids = torch.randint(0, 1000, (1, n))
            return {'input_ids': input_ids}
        def convert_tokens_to_ids(self, token):
            return 1

    dummy_tok = DummyTokenizer()

    # ダミー pixel_values (P=3パッチ)
    pv = torch.randn(3, 3, 448, 448)
    sample = training_step.build_training_sample(
        pixel_values=pv,
        num_patches=3,
        question="<image>\n説明してください。",
        answer="この画像には...",
        tokenizer=dummy_tok,
    )
    print(f"  pixel_values  : {sample['pixel_values'].shape}")
    print(f"  input_ids     : {sample['input_ids'].shape}")
    print(f"  labels        : {sample['labels'].shape}")
    print(f"  image_flags   : {sample['image_flags'].shape}")
    print(f"  loss_weight   : {sample['loss_weight'].shape}")
    # ラベルのマスク確認
    n_masked = (sample['labels'] == -100).sum().item()
    print(f"  マスク済みトークン数 (-100): {n_masked}")
    print("  OK")

    # --- 3. フォワードパス全体デモ ---
    print("\n[3] フォワードパス全体デモ (ダミーテンソル)")
    InternVL35ForwardPassDemo.demonstrate_forward_pass(
        batch_size=2,
        patches_per_sample=[3, 5],
        max_text_len=64,
        vit_hidden=1024,    # InternViT-300M 相当 (テスト用)
        llm_hidden=2048,
        vocab_size=500,
        num_image_token=256,
    )

    # --- 4. InternVL3.5-Flash ビジュアルトークン削減効果 ---
    print("\n[4] InternVL3.5-Flash トークン削減効果")
    scenarios = [
        ("通常モデル (全256tok)", [256, 256, 256, 256], 0.0),
        ("Flash (50%高圧縮)",   [256, 64, 256, 64],  0.5),
        ("Flash (75%高圧縮)",   [64, 64, 64, 256],   0.75),
    ]
    for name, tok_per_patch, high_ratio in scenarios:
        total_tok = sum(tok_per_patch)
        baseline = 4 * 256
        reduction = (1 - total_tok / baseline) * 100
        print(f"  {name}: {tok_per_patch} → 計{total_tok}tok "
              f"({reduction:.0f}%削減)")

    print("\n全テスト完了!")
