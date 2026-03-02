"""
MiniCPM-V 4.5 - マルチモーダル融合
================================================

視覚トークンとテキスト埋め込みのインターリーブ統合ロジック。
<im_start>/<im_end> による特殊トークンプレースホルダの置換処理。

論文: MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipes
公式実装:
    - omnilmm/model/omnilmm.py: OmniLMMModel.forward(), get_vllm_embedding()
    - chat.py: expand_question_into_multimodal(), wrap_question_for_omni_lmm()

処理の流れ:
1. テキストをトークン化し、画像プレースホルダを挿入
2. テキスト埋め込みを取得
3. <im_patch> プレースホルダを実際の視覚埋め込みで置換
4. 統合された埋め込みをLLMに入力
"""

"""
============================================================
Shape Convention (形状表記規則)
============================================================
B       : バッチサイズ
L_text  : テキストトークン長 (プレースホルダ含む)
L_total : テキスト + 視覚トークンの合計長 (置換後はL_textと同一)
Q       : Resamplerクエリ数 (64)
D_llm   : LLM埋め込み次元 (4096)
V       : 語彙サイズ
============================================================
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# 特殊トークン文字列
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


# ========================================
# テキストプロンプト展開
# ========================================
def expand_question_into_multimodal(
    question_text: List[Dict],
    image_token_len: int,
    im_st_token: str = DEFAULT_IM_START_TOKEN,
    im_ed_token: str = DEFAULT_IM_END_TOKEN,
    im_patch_token: str = DEFAULT_IMAGE_PATCH_TOKEN,
) -> List[Dict]:
    """
    ユーザーの質問テキスト中の <image> プレースホルダを
    画像トークン列に展開する

    公式実装: chat.py: expand_question_into_multimodal()

    ========================================
    入力:
        question_text: 会話メッセージのリスト
            [{"role": "user", "content": "<image>\nこの画像は何ですか？"}]
        image_token_len: 画像トークン数 = Q (64)
        im_st_token: 画像開始トークン文字列
        im_ed_token: 画像終了トークン文字列
        im_patch_token: パッチトークン文字列

    出力:
        question_text: 展開されたメッセージリスト
            [{"role": "user", "content": "<im_start><im_patch>*64<im_end>\nこの画像は何ですか？"}]

    トークン構造:
        <im_start> [<im_patch> × Q] <im_end>
        │          │               │
        │          │               └── 画像終了マーカー
        │          └── Q個のプレースホルダ (後で視覚埋め込みに置換)
        └── 画像開始マーカー
    ========================================
    """
    # 画像トークン列を構築
    image_token_str = im_st_token + im_patch_token * image_token_len + im_ed_token
    # image_token_str: "<im_start><im_patch><im_patch>...(64個)...<im_end>"

    if "<image>" in question_text[0]["content"]:
        # <image> プレースホルダがある場合は置換
        question_text[0]["content"] = question_text[0]["content"].replace(
            "<image>", image_token_str
        )
    else:
        # プレースホルダがない場合は先頭に挿入
        question_text[0]["content"] = (
            image_token_str + "\n" + question_text[0]["content"]
        )

    return question_text


def expand_multiimage_question(
    question_text: List[Dict],
    image_placeholder_dict: Dict[str, str],
) -> List[Dict]:
    """
    マルチイメージ用のプレースホルダ展開

    公式実装: finetune/dataset.py: preprocess() のマルチイメージ部分

    ========================================
    入力:
        question_text: 会話メッセージリスト
            content内に <image_00>, <image_01> 等のプレースホルダ
        image_placeholder_dict: {
            "<image_00>": "<im_start><im_patch>*64<im_end>...",
            "<image_01>": "<im_start><im_patch>*64<im_end>...",
        }

    出力:
        展開されたメッセージリスト

    スライス込みのプレースホルダ構造:
        <image_00> → <im_start><im_patch>*Q<im_end>
                     <slice_start>
                       <im_start><im_patch>*Q<im_end><im_start><im_patch>*Q<im_end>
                       <im_start><im_patch>*Q<im_end><im_start><im_patch>*Q<im_end>
                       <im_start><im_patch>*Q<im_end><im_start><im_patch>*Q<im_end>
                     <slice_end>
        (ソース画像 + 2x3グリッドのスライス)
    ========================================
    """
    import re

    pattern = r"<image_\d+>"

    for msg in question_text:
        content = msg["content"]
        parts = re.split(f"({pattern})", content)

        for i, part in enumerate(parts):
            if not part.strip():
                continue
            if re.match(pattern, part):
                if part in image_placeholder_dict:
                    parts[i] = image_placeholder_dict[part]
                else:
                    raise ValueError(f"Image placeholder {part} not found in dict")

        msg["content"] = "\n".join(parts)

    return question_text


# ========================================
# マルチモーダル融合 (Embedding-Space Fusion)
# ========================================
class MultimodalFusion(nn.Module):
    """
    視覚トークンとテキスト埋め込みの融合モジュール

    公式実装: omnilmm/model/omnilmm.py: OmniLMMModel.forward()

    テキスト入力中の <im_patch> プレースホルダ位置を実際の視覚埋め込みで
    置換する「Embedding-Space Fusion」方式。

    入力トークン列の構造:
        [text_tokens] <im_start> [<im_patch> × Q] <im_end> [text_tokens]

    処理後:
        [text_embeds] <im_start_embed> [vision_embeds × Q] <im_end_embed> [text_embeds]

    ========================================
    入力:
        input_ids: (B, L_text)
        inputs_embeds: (B, L_text, D_llm) - テキスト埋め込み
        image_features: List[Tensor] - 各画像の視覚特徴
            各要素: (Q, D_llm) = (64, 4096)
        im_patch_token_id: int - <im_patch> のトークンID
        im_start_token_id: int - <im_start> のトークンID
        im_end_token_id: int - <im_end> のトークンID

    出力:
        fused_embeds: (B, L_total, D_llm) - 融合後の埋め込み
            L_total = L_text (プレースホルダと視覚トークンは同数のため)
    ========================================
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor,          # (B, L_text)
        inputs_embeds: torch.FloatTensor,      # (B, L_text, D_llm)
        image_features: List[torch.Tensor],    # List of (Q, D_llm)
        im_patch_token_id: int,
        im_start_token_id: int,
        im_end_token_id: int,
        orig_embeds_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        視覚トークンをテキスト埋め込みに挿入する

        公式実装: omnilmm/model/omnilmm.py: OmniLMMModel.forward() L201-258

        ========================================
        処理の流れ:
        1. 各サンプルについて <im_patch> トークンの有無を確認
        2. <im_start> の位置を特定
        3. <im_start> の次の位置から Q 個のトークンを視覚埋め込みで置換
        4. <im_end> の位置を確認（整合性チェック）

        詳細な置換ロジック:
            元: [...text_embeds..., <im_start>, <im_patch>, ..., <im_patch>, <im_end>, ...text_embeds...]
            後: [...text_embeds..., <im_start>, vis_0, vis_1, ..., vis_Q-1, <im_end>, ...text_embeds...]

            <im_start> と <im_end> 自体のテキスト埋め込みは保持される。
            <im_patch> × Q 個分のみが視覚埋め込みに置換される。
        ========================================
        """
        new_input_embeds = []
        cur_image_idx = 0

        for batch_idx, (cur_input_ids, cur_input_embeds) in enumerate(
            zip(input_ids, inputs_embeds)
        ):
            # --- テキストのみのサンプルをスキップ ---
            if (cur_input_ids == im_patch_token_id).sum() == 0:
                # <im_patch> が存在しない → 画像なし
                new_input_embeds.append(cur_input_embeds)
                # cur_input_embeds: (L_text, D_llm)
                continue

            # --- <im_start> の位置を検索 ---
            image_start_tokens = torch.where(
                cur_input_ids == im_start_token_id
            )[0]
            # image_start_tokens: (N_images,) - 各画像の開始位置

            for image_start_token_pos in image_start_tokens:
                # 対応する視覚特徴を取得
                cur_image_features = image_features[cur_image_idx].to(
                    device=cur_input_embeds.device
                )
                num_patches = cur_image_features.shape[0]
                # num_patches: Q (64)
                # cur_image_features: (Q, D_llm) = (64, 4096)

                # --- 整合性チェック ---
                # <im_start> + Q個の<im_patch> + <im_end> の構造を確認
                if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token_id:
                    raise ValueError(
                        "The image end token should follow the image start token. "
                        f"Position: {image_start_token_pos}, "
                        f"num_patches: {num_patches}"
                    )

                # --- 埋め込みの再構築 ---
                if orig_embeds_params is not None:
                    # 事前学習時: プレースホルダ外のテキスト埋め込みを固定
                    cur_new_input_embeds = torch.cat([
                        # text before <im_start> (勾配なし)
                        cur_input_embeds[:image_start_token_pos].detach(),
                        # <im_start> (勾配あり)
                        cur_input_embeds[image_start_token_pos:image_start_token_pos + 1],
                        # 視覚埋め込み (Q, D_llm)
                        cur_image_features,
                        # <im_end> (勾配あり)
                        cur_input_embeds[
                            image_start_token_pos + num_patches + 1:
                            image_start_token_pos + num_patches + 2
                        ],
                        # text after <im_end> (勾配なし)
                        cur_input_embeds[image_start_token_pos + num_patches + 2:].detach(),
                    ], dim=0)
                else:
                    # 通常の推論/SFT:
                    cur_new_input_embeds = torch.cat([
                        # text + <im_start>
                        cur_input_embeds[:image_start_token_pos + 1],
                        # 視覚埋め込み (Q, D_llm)
                        cur_image_features,
                        # <im_end> + remaining text
                        cur_input_embeds[image_start_token_pos + num_patches + 1:],
                    ], dim=0)
                    # cur_new_input_embeds: (L_total, D_llm)
                    #   L_total = L_text (サイズ変化なし: <im_patch>×Qと視覚トークン×Qは同数)

                cur_image_idx += 1

            new_input_embeds.append(cur_new_input_embeds)

        fused_embeds = torch.stack(new_input_embeds, dim=0)
        # fused_embeds: (B, L_total, D_llm)

        return fused_embeds


# ========================================
# トークナイザ初期化
# ========================================
def initialize_vision_tokenizer(
    model: nn.Module,
    tokenizer,
    mm_use_im_start_end: bool = True,
    tune_mm_mlp_adapter: bool = False,
):
    """
    視覚関連の特殊トークンをトークナイザに追加する

    公式実装: omnilmm/model/omnilmm.py: OmniLMMForCausalLM.initialize_vision_tokenizer()

    ========================================
    追加されるトークン:
        1. <im_patch>  - パッチプレースホルダ
        2. <im_start>  - 画像開始マーカー
        3. <im_end>    - 画像終了マーカー
        4. <box>, </box>, <ref>, </ref>, <quad>, </quad> - グラウンディング用

    処理:
        1. トークンを追加
        2. モデルの埋め込み層をリサイズ
        3. 新トークンの埋め込みを既存トークンの平均で初期化
        4. vision_config に各トークンのIDを登録

    vision_config に設定される属性:
        - use_im_start_end: bool
        - im_patch_token: int (トークンID)
        - im_start_token: int (トークンID)
        - im_end_token: int (トークンID)
    ========================================
    """
    # --- 1. <im_patch> トークン追加 ---
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if mm_use_im_start_end:
        # --- 2. <im_start>, <im_end> トークン追加 ---
        num_new_tokens = tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
            special_tokens=True,
        )
        model.resize_token_embeddings(len(tokenizer))

        # im_start_token, im_end_token のIDを取得
        im_start_id, im_end_id = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )

        if num_new_tokens > 0:
            # --- 3. 新トークンの埋め込みを平均値で初期化 ---
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        # --- 4. グラウンディング用トークン追加 ---
        grounding_tokens = ["<box>", "</box>", "<ref>", "</ref>", "<quad>", "</quad>"]
        num_grounding = tokenizer.add_tokens(grounding_tokens, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        if num_grounding > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-num_grounding].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_grounding].mean(
                dim=0, keepdim=True
            )
            input_embeddings[-num_grounding:] = input_embeddings_avg
            output_embeddings[-num_grounding:] = output_embeddings_avg

        # --- 5. 事前学習時の埋め込み固定設定 ---
        if tune_mm_mlp_adapter:
            model.orig_embeds_params = [
                model.get_input_embeddings().weight.data.clone()
            ]
            for p in model.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in model.get_output_embeddings().parameters():
                p.requires_grad = False

    # --- 6. vision_config にトークンID設定 ---
    im_patch_id = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    # vision_config はモデルのlambdaオブジェクト
    # 公式実装: self.vision_config = lambda x: None


# ========================================
# 使用例
# ========================================
def example_fusion_flow():
    """
    マルチモーダル融合の全体フローデモ

    入力テキスト:
        "この画像の内容を説明してください。"

    → 展開:
        "<im_start><im_patch>*64<im_end>\nこの画像の内容を説明してください。"

    → トークン化:
        [BOS, im_start, im_patch, ...(64個)..., im_end, \n, こ, の, ...]

    → 埋め込み:
        [BOS_emb, im_start_emb, patch_emb, ..., im_end_emb, \n_emb, ...]

    → 融合 (im_patchを視覚埋め込みで置換):
        [BOS_emb, im_start_emb, vis_0, vis_1, ..., vis_63, im_end_emb, \n_emb, ...]
    """
    B = 1
    D_LLM = 4096
    Q = 64
    VOCAB_SIZE = 65536

    # --- ステップ1: テキスト展開 ---
    question = [{"role": "user", "content": "<image>\nこの画像の内容を説明してください。"}]
    question = expand_question_into_multimodal(question, image_token_len=Q)
    print(f"展開後: {question[0]['content'][:80]}...")
    # → "<im_start><im_patch><im_patch>...(64個)...<im_end>\nこの画像の..."

    # --- ステップ2: トークン化 (疑似) ---
    # 実際にはtokenizerで変換
    # ここではダミーのトークンIDを使用
    L_text = 10 + Q + 2  # text(10) + im_patch(64) + im_start(1) + im_end(1) = 76
    input_ids = torch.zeros(B, L_text, dtype=torch.long)
    # [BOS, text*5, im_start, im_patch*64, im_end, text*4]
    IM_START_ID = 32000
    IM_PATCH_ID = 32001
    IM_END_ID = 32002
    input_ids[0, 0:6] = torch.tensor([1, 100, 200, 300, 400, 500])  # BOS + text
    input_ids[0, 6] = IM_START_ID
    input_ids[0, 7:7 + Q] = IM_PATCH_ID
    input_ids[0, 7 + Q] = IM_END_ID
    input_ids[0, 7 + Q + 1:] = torch.tensor([600, 700, 800])  # text
    print(f"input_ids shape: {input_ids.shape}")
    # input_ids: (1, 76)

    # --- ステップ3: テキスト埋め込み ---
    embed_tokens = nn.Embedding(VOCAB_SIZE, D_LLM)
    inputs_embeds = embed_tokens(input_ids)
    print(f"inputs_embeds shape: {inputs_embeds.shape}")
    # inputs_embeds: (1, 76, 4096)

    # --- ステップ4: 視覚特徴 (Resampler出力) ---
    vision_features = [torch.randn(Q, D_LLM)]
    # vision_features[0]: (64, 4096)

    # --- ステップ5: 融合 ---
    fusion = MultimodalFusion(config={})
    fused = fusion(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_features=vision_features,
        im_patch_token_id=IM_PATCH_ID,
        im_start_token_id=IM_START_ID,
        im_end_token_id=IM_END_ID,
    )
    print(f"fused shape: {fused.shape}")
    # fused: (1, 76, 4096)
    # → im_patch位置が視覚埋め込みに置換済み

    # サイズが変わらないことを確認
    assert fused.shape == inputs_embeds.shape
    print("融合完了: テキスト+視覚の統合埋め込み")


if __name__ == "__main__":
    example_fusion_flow()
