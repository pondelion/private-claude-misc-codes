"""
Qwen3VL - Loss Computation
===========================

Qwen3-VL の学習損失計算を疑似コードで示します。

主要コンポーネント:
1. ラベルマスキング: 視覚トークン・プレフィックスを -100 でマスク
2. Cross-Entropy Loss: テキストトークンのみで計算
3. label 生成の詳細 (data_processor.py の preprocess_qwen_visual 参照)

論文: Qwen3-VL Technical Report (2025)
実装参照: qwen-vl-finetune/qwenvl/data/data_processor.py

============================================================
Shape Convention
============================================================
B: バッチサイズ
T_seq: LLMシーケンス長 (視覚トークン + テキストトークン)
vocab_size: 語彙サイズ = 151936
IGNORE_INDEX: -100 (損失計算から除外するマーカー)

============================================================
損失計算の基本方針
============================================================
- 損失はアシスタントの応答部分のみで計算
- 視覚トークン (IMAGE/VIDEO プレースホルダー) は損失から除外
- システムプロンプト・ユーザー入力は損失から除外
- アシスタント応答の EOS トークンを含めて損失計算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ============================================================
# 定数
# ============================================================

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655   # <image>
VIDEO_TOKEN_INDEX = 151656   # <video>

# Chat Template トークンID (Qwen3のChatML形式)
IM_START_TOKEN_ID = 151644   # <|im_start|>
IM_END_TOKEN_ID = 151645     # <|im_end|>
ASSISTANT_TOKEN_ID = 77091   # "assistant"


# ============================================================
# ラベル生成
# ============================================================

def create_labels_from_conversation(
    input_ids: torch.LongTensor,
    conversations: List[Dict],
) -> torch.LongTensor:
    """
    Conversation形式からラベルを生成

    アシスタントの応答部分のみをラベルとして使用し、
    それ以外 (system, user, 視覚トークン) は -100 でマスク

    ========================================
    Shape
    ========================================
    入力:
        input_ids: (1, T_seq) int64
            - トークン列全体
            - 视覚トークン位置には IMAGE_TOKEN_INDEX が入っている
        conversations: List[Dict]
            - 例: [{"role": "user", ...}, {"role": "assistant", ...}]

    出力:
        labels: (1, T_seq) int64
            - アシスタント応答部分: トークンID (損失計算対象)
            - それ以外: -100 (損失除外)

    ========================================
    処理詳細
    ========================================
    入力トークン列の構造 (ChatML形式):
    <|im_start|> system\n
    [システムプロンプト]
    <|im_end|>\n
    <|im_start|> user\n
    [<image> または <video> プレースホルダー]
    [ユーザーテキスト]
    <|im_end|>\n
    <|im_start|> assistant\n
    ← ここから ↓ アシスタント応答 (損失計算対象)
    [アシスタントのテキスト]
    <|im_end|>
    ← ここまで

    ラベル:
    -100, -100, ..., -100,    ← system + user 部分
    [tok_a1, tok_a2, ..., tok_aN, IM_END_TOKEN_ID]  ← assistant 応答 (+EOS)
    -100, -100, ...           ← 次のターン (マルチターンの場合)

    ========================================
    実装 (data_processor.py: preprocess_qwen_visual より)
    ========================================
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:  # "assistant" トークン
            ans_start = pos + 2           # "assistant\n" の次
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:  # <|im_end|>
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                # +2 は <|im_end|> と \n を含める
                pos = ans_end
        pos += 1
    """
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    # labels: (1, T_seq) 初期値 -100

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0

    while pos < L:
        # "assistant" トークン (77091) を探す
        if input_ids_flat[pos] == ASSISTANT_TOKEN_ID:
            ans_start = pos + 2  # "assistant\n" の2トークン後
            ans_end = ans_start

            # <|im_end|> (151645) まで探す
            while ans_end < L and input_ids_flat[ans_end] != IM_END_TOKEN_ID:
                ans_end += 1

            if ans_end < L:
                # アシスタント応答 + <|im_end|> + \n を損失計算対象に設定
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                # labels の該当範囲: -100 → トークンID

                pos = ans_end  # 探索位置を進める

        pos += 1

    return labels
    # labels: (1, T_seq) int64
    # アシスタント応答部分: トークンID
    # それ以外: -100


# ============================================================
# 具体的なラベル生成の例
# ============================================================

def example_label_generation():
    """
    ラベル生成の具体例 (数値付き)

    ========================================
    入力トークン列のイメージ
    ========================================
    ChatML形式 (簡略化):

    <|im_start|> user\n <image> 質問テキスト <|im_end|>\n
    <|im_start|> assistant\n 回答テキスト <|im_end|>

    実際のトークン列 (例):
    [151644, "user", "\n", 151655, "質問", 151645, "\n",   ← user部分
     151644, 77091, "\n", "回答", 151645]                  ← assistant部分

    ラベル:
    [-100,   -100,  -100, -100,   -100, -100,  -100,       ← user部分 (全て-100)
     -100,   -100,  -100, "回答", 151645]                  ← "回答"と<|im_end|>のみ有効
    """
    print("=== ラベル生成の例 ===\n")

    print("【トークン列構造】")
    print("位置: 0    1       2    3          4     5       6    7      8")
    print("ID:   SYS  user    \\n   IMG_TOKEN  質問  IM_END  \\n   ASST   \\n")
    print("値:   1644 user_id  13  151655     tok   1645    13   77091   13")
    print()
    print("位置: 9    10     11    12")
    print("ID:   回答  回答2  ...  IM_END")
    print("値:   tok   tok   ...   1645")
    print()
    print("【ラベル】")
    print("位置: 0    1    2    3    4    5    6    7    8    9     10    11   12")
    print("ラベル: -100 -100 -100 -100 -100 -100 -100 -100 -100 tok   tok  ... 1645")
    print("                                                       ↑ assistant応答から損失計算")
    print()
    print("【ポイント】")
    print("  - IMAGE_TOKEN (151655) は -100 → 視覚トークンは損失から除外")
    print("  - 'user' ロールは -100 → ユーザー入力は損失から除外")
    print("  - 'assistant' ロールの応答 + <|im_end|> のみが損失計算対象")
    print("  - マルチターン: 各 'assistant' ターンの応答が全て損失計算対象")


# ============================================================
# 損失計算: モデルの forward 内で実行
# ============================================================

def compute_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    num_items_in_batch: Optional[int] = None,
) -> torch.Tensor:
    """
    言語モデルの Cross-Entropy 損失計算

    ========================================
    Shape
    ========================================
    入力:
        logits: (B, T_seq, vocab_size) = (B, T_seq, 151936)
        labels: (B, T_seq) int64
            - -100 の位置は損失計算から除外
            - それ以外はトークンIDとして損失計算

    出力:
        loss: scalar float

    ========================================
    処理詳細 (次トークン予測)
    ========================================
    logits の各位置 t は位置 t+1 のトークンを予測する

    例:
        input:  [SYS, USR, IMG, "回答"] → input_ids
        logits: [L0,  L1,  L2,  L3   ] → 各位置からの予測
        labels: [-100,-100,-100,"次の"] → labels (shift)

    shift_logits: logits[:-1] = [L0, L1, L2]     # 最後を除く
    shift_labels: labels[1:]  = [-100, -100, "次"] # 最初を除く (シフト)

    loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    """
    # ==================================================
    # 1. 次トークン予測のためのシフト
    # ==================================================
    shift_logits = logits[..., :-1, :].contiguous()
    # shift_logits: (B, T_seq-1, vocab_size)

    shift_labels = labels[..., 1:].contiguous()
    # shift_labels: (B, T_seq-1)

    # ==================================================
    # 2. Cross-Entropy Loss
    # ==================================================
    # -100 は ignore_index により自動的に無視される
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        # (B×(T_seq-1), vocab_size)
        shift_labels.view(-1),
        # (B×(T_seq-1),)
        ignore_index=IGNORE_INDEX,  # = -100
        reduction="mean",           # 有効トークンのみの平均
    )
    # loss: scalar

    return loss


# ============================================================
# バッチコレーター内のラベル生成
# ============================================================

class Qwen3VLLabelGenerator:
    """
    バッチ内のラベル生成ロジック

    ========================================
    前処理フロー
    ========================================
    1. processor.apply_chat_template() でトークン化
    2. アシスタント応答部分を特定
    3. labels を生成 (-100 でマスク)
    4. パディング部分も -100 でマスク

    ========================================
    Shape
    ========================================
    入力:
        input_ids: (1, T_seq) per sample (バッチ化前)

    出力:
        labels: (1, T_seq) - -100 またはトークンID
    """

    def generate_labels(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        入力:
            input_ids: (1, T_seq)

        出力:
            labels: (1, T_seq)
        """
        return create_labels_from_conversation(input_ids, conversations=[])


def pad_and_batch_labels(
    input_ids_list: List[torch.LongTensor],
    labels_list: List[torch.LongTensor],
    pad_token_id: int,
    max_length: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    バッチ内のラベルをパディングしてバッチ化

    ========================================
    Shape
    ========================================
    入力:
        input_ids_list: List[(1, T_i)] - 各サンプルのトークン列
        labels_list: List[(1, T_i)] - 各サンプルのラベル

    出力:
        input_ids_batch: (B, T_max) int64 - パディング済み
        labels_batch: (B, T_max) int64
            - パディング位置: -100 (損失除外)
            - 有効位置: -100 (損失除外) または トークンID (損失計算対象)

    ========================================
    パディングの方針
    ========================================
    - input_ids: pad_token_id でパディング (右パディング)
    - labels: -100 でパディング (パディング位置は常に損失除外)
    """
    import torch.nn.utils.rnn as rnn_utils

    # 各 (1, T_i) から (T_i,) に変換
    input_ids_squeezed = [ids.squeeze(0) for ids in input_ids_list]
    labels_squeezed = [l.squeeze(0) for l in labels_list]

    # パディング
    input_ids_batch = rnn_utils.pad_sequence(
        input_ids_squeezed,
        batch_first=True,
        padding_value=pad_token_id,
    )
    # input_ids_batch: (B, T_max)

    labels_batch = rnn_utils.pad_sequence(
        labels_squeezed,
        batch_first=True,
        padding_value=IGNORE_INDEX,  # = -100
    )
    # labels_batch: (B, T_max)

    if max_length is not None:
        input_ids_batch = input_ids_batch[:, :max_length]
        labels_batch = labels_batch[:, :max_length]

    return input_ids_batch, labels_batch


# ============================================================
# 損失計算の全体フロー
# ============================================================

def full_training_step(
    model,
    batch: Dict,
    optimizer: torch.optim.Optimizer,
    scaler=None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> float:
    """
    1つの学習ステップ (Forward + Backward + Update)

    ========================================
    バッチ内のテンソル Shape
    ========================================
    batch:
        input_ids:              (B, T_seq) int64
            - テキスト + IMAGE/VIDEO プレースホルダー
        attention_mask:         (B, T_seq) int64
        pixel_values:           (N_patches_total, 588) float
            - 全バッチ・全画像のパッチ
        image_grid_thw:         (num_images, 3) int64
        pixel_values_videos:    (N_patches_video, 588) float  or None
        video_grid_thw:         (num_videos, 3) int64         or None
        position_ids:           (3, B, T_seq) int64
            - Interleaved MRoPE 用
        labels:                 (B, T_seq) int64
            - -100: 視覚トークン・システム・ユーザー (損失除外)
            - トークンID: アシスタント応答 (損失計算対象)

    ========================================
    Forward出力
    ========================================
    loss: scalar - アシスタント応答トークンのCross-Entropy
    logits: (B, T_seq, 151936) - (通常は損失計算後に破棄)

    ========================================
    重要なポイント
    ========================================
    labels の構成例:
    -100 -100 ... -100  tok1  tok2  ...  tokN  1645  -100 ... -100
     ↑   system/user   ↑    assistant応答      ↑IM_END ↑  padding
     ↑   + 視覚トークン                                  ↑  (または次turnのuser)
    """
    device = next(model.parameters()).device

    # バッチを GPU に移動
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    # 注: labels も一緒に GPU に移動

    # ==================================================
    # Forward Pass
    # ==================================================
    with torch.autocast(device_type="cuda", dtype=compute_dtype):
        outputs = model(**batch)
        # outputs["loss"]: scalar
        # outputs["logits"]: (B, T_seq, 151936)

        loss = outputs["loss"]

    # ==================================================
    # Backward Pass
    # ==================================================
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss.item()


# ============================================================
# 損失の内訳を可視化
# ============================================================

def analyze_loss_contribution(
    logits: torch.Tensor,
    labels: torch.LongTensor,
) -> Dict:
    """
    トークン種別ごとの損失寄与を分析

    ========================================
    Shape
    ========================================
    入力:
        logits: (B, T_seq, vocab_size)
        labels: (B, T_seq) - -100 または トークンID

    出力:
        analysis: Dict
            "total_tokens": int - 全トークン数
            "valid_tokens": int - 損失計算対象トークン数 (labels != -100)
            "ignored_tokens": int - 損失除外トークン数 (labels == -100)
            "per_token_loss": (valid_tokens,) float - 各トークンの損失
            "avg_loss": float - 平均損失

    ========================================
    典型的な比率 (画像1枚 + 短い応答の例)
    ========================================
    T_seq = 350 の場合:
        - system:          20 tokens  → -100 (損失なし)
        - user (text):     30 tokens  → -100 (損失なし)
        - visual tokens:  256 tokens  → -100 (損失なし)
        - assistant:       44 tokens  → 損失計算対象

    valid_tokens / total_tokens ≈ 44 / 350 ≈ 12.6%
    """
    # 次トークン予測のシフト
    shift_logits = logits[..., :-1, :].contiguous()  # (B, T_seq-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous()       # (B, T_seq-1)

    B, T, V = shift_logits.shape
    total_tokens = B * T

    # 有効トークン (labels != -100) の数
    valid_mask = (shift_labels != IGNORE_INDEX)
    valid_tokens = valid_mask.sum().item()
    ignored_tokens = total_tokens - valid_tokens

    # 各トークンの損失
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    # per_token_loss: (B×T,) - -100位置は0

    valid_losses = per_token_loss[valid_mask.view(-1)]
    # valid_losses: (valid_tokens,)

    avg_loss = valid_losses.mean().item() if valid_tokens > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "valid_tokens": int(valid_tokens),
        "ignored_tokens": int(ignored_tokens),
        "valid_ratio": valid_tokens / total_tokens if total_tokens > 0 else 0,
        "per_token_loss": valid_losses.detach(),
        "avg_loss": avg_loss,
    }


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    損失計算の使用例

    ========================================
    Shape Summary
    ========================================
    バッチ (B=2, 448×448画像2枚):
        input_ids:      (2, T_seq) where T_seq = text + 256 visual tokens
        labels:         (2, T_seq) with -100 for visual/system/user tokens
        pixel_values:   (512, 588) = 2 images × 256 patches
        logits:         (2, T_seq, 151936)
        loss:           scalar

    典型的なラベル比率:
        視覚トークン: -100  (256 tokens × 2 = 512, 全て除外)
        system/user:  -100  (50 tokens × 2 = 100, 全て除外)
        assistant:    損失計算対象 (20-100 tokens)
    """
    print("=== Loss Computation Example ===\n")

    example_label_generation()

    print("\n【損失計算フロー】")
    print("1. labels[-100 以外の位置] → アシスタント応答トークンのみ")
    print("2. shift: logits[:-1] vs labels[1:]  (次トークン予測)")
    print("3. F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)")
    print("4. 有効トークン (labels != -100) のみで平均を取る")
    print()

    # ダミーデータで動作確認
    B, T_seq, V = 2, 20, 100  # 簡略化
    logits = torch.randn(B, T_seq, V)
    labels = torch.full((B, T_seq), IGNORE_INDEX, dtype=torch.long)
    labels[0, 10:15] = torch.randint(0, V, (5,))  # サンプル0の応答5トークン
    labels[1, 8:14]  = torch.randint(0, V, (6,))  # サンプル1の応答6トークン

    loss = compute_loss(logits, labels)
    print(f"計算例 (ダミーデータ):")
    print(f"  logits: {logits.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  有効トークン数: {(labels != IGNORE_INDEX).sum().item()} / {B*T_seq}")
    print(f"  loss: {loss.item():.4f}")

    analysis = analyze_loss_contribution(logits, labels)
    print()
    print(f"損失分析:")
    print(f"  total_tokens:   {analysis['total_tokens']}")
    print(f"  valid_tokens:   {analysis['valid_tokens']}")
    print(f"  ignored_tokens: {analysis['ignored_tokens']}")
    print(f"  valid_ratio:    {analysis['valid_ratio']:.1%}")
    print(f"  avg_loss:       {analysis['avg_loss']:.4f}")


if __name__ == "__main__":
    example_usage()
