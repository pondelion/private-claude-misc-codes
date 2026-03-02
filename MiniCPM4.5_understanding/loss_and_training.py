"""
MiniCPM-V 4.5 - 損失関数・学習パイプライン
================================================

3段階事前学習・2段階SFT・ハイブリッドRL（GRPO）・RLAIF-V・
報酬整形のパイプライン実装。

論文: MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipes
公式実装:
    - finetune/trainer.py: CPMTrainer
    - finetune/finetune.py: train()

処理の流れ:
1. SFT: アシスタント応答部分のみにCrossEntropyLoss
2. RL: GRPOによるハイブリッド（短/長推論）学習
3. RLAIF-V: DPOによる幻覚低減
"""

"""
============================================================
Shape Convention (形状表記規則)
============================================================
B       : バッチサイズ
L       : シーケンス長
V       : 語彙サイズ
N_prompt: RLバッチ内のプロンプト数 (128)
K       : プロンプトあたりの応答数 (8)
============================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================
# 1. SFT損失関数 (CrossEntropyLoss)
# ========================================
class SFTLoss(nn.Module):
    """
    教師あり微調整（SFT）の損失計算

    公式実装: finetune/trainer.py: CPMTrainer.compute_loss()

    アシスタント応答部分のみにクロスエントロピー損失を適用する。
    ユーザーメッセージ・システムメッセージ部分は labels=-100 でマスクされ、
    損失計算から除外される。

    ========================================
    入力:
        logits: (B, L, V) - モデル出力のlogits
        labels: (B, L) - ターゲットラベル
            -100: マスク（損失計算から除外）
            0〜V-1: ターゲットトークンID

    出力:
        loss: スカラー - クロスエントロピー損失
    ========================================
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fct = nn.CrossEntropyLoss()  # ignore_index=-100 がデフォルト

    def forward(
        self,
        logits: torch.Tensor,   # (B, L, V)
        labels: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """
        ========================================
        処理:
            1. 1トークンシフト（次トークン予測）
               logits[t] → labels[t+1] を予測
            2. Flatten
            3. CrossEntropyLoss（-100部分を自動除外）

        数式:
            loss = -1/N_valid * Σ_{t: labels[t]≠-100} log P(labels[t] | logits[t-1])
        ========================================
        """
        # 1トークンシフト: tokens < n が n を予測
        shift_logits = logits[..., :-1, :].contiguous()
        # shift_logits: (B, L-1, V)

        shift_labels = labels[..., 1:].contiguous()
        # shift_labels: (B, L-1)

        # Flatten
        shift_logits = shift_logits.view(-1, self.vocab_size)
        # shift_logits: (B*(L-1), V)

        shift_labels = shift_labels.view(-1).long()
        # shift_labels: (B*(L-1),)

        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.loss_fct(shift_logits, shift_labels)
        # loss: スカラー

        return loss


# ========================================
# 2. ハイブリッドRL (GRPO)
# ========================================
@dataclass
class GRPOConfig:
    """
    GRPO (Group Relative Policy Optimization) のハイパーパラメータ

    論文 Section 2.4.3 & Appendix A:
        「We apply GRPO to optimize the model with these rollouts and
         remove the KL and entropy loss to improve stability.」
    """
    num_prompts: int = 128          # バッチ内のプロンプト数
    num_responses: int = 8          # プロンプトあたりの応答数
    max_response_length: int = 8192 # 最大応答トークン数
    temperature: float = 1.0        # ロールアウト温度
    long_reasoning_ratio: float = 0.5  # 長い推論モードの割合
    learning_rate: float = 1e-6     # 固定学習率
    # KL損失: なし、エントロピー損失: なし


class HybridRLTrainer:
    """
    ハイブリッド強化学習トレーナー

    論文 Section 2.4.3:
        短い推論モードと長い推論モードをランダムに交互実行し、
        GRPOで同時最適化する。

    ========================================
    学習ループ:
        1. プロンプトをランダムに短い/長い推論モードに割り当て
        2. 各プロンプトからK個の応答をサンプリング（ロールアウト）
        3. 報酬を計算
        4. GRPOの損失を計算
        5. パラメータ更新
    ========================================
    """

    def __init__(self, model: nn.Module, config: GRPOConfig = None):
        self.model = model
        self.config = config or GRPOConfig()

    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,          # (N_prompt, K, L_resp)
        ref_log_probs: torch.Tensor,      # (N_prompt, K, L_resp)
        rewards: torch.Tensor,            # (N_prompt, K)
        response_mask: torch.Tensor,      # (N_prompt, K, L_resp) - 有効トークンマスク
    ) -> torch.Tensor:
        """
        GRPO損失の計算

        論文 Section 2.4.3 + 参考: DeepSeekMath [44]

        ========================================
        入力:
            log_probs: (N_prompt, K, L_resp)
                - ポリシーモデルの対数確率
                - N_prompt=128, K=8

            ref_log_probs: (N_prompt, K, L_resp)
                - 参照モデル（SFTモデル）の対数確率

            rewards: (N_prompt, K)
                - 各応答の報酬スコア

            response_mask: (N_prompt, K, L_resp)
                - 1: 有効トークン, 0: パディング

        出力:
            loss: スカラー

        数式 (標準GRPO、KLなし):
            advantage_i = (R_i - mean(R)) / std(R)  (グループ内正規化)
            loss = -1/N Σ advantage_i * log_prob_i

        注: MiniCPM-V 4.5ではKL損失とエントロピー損失を除去
        ========================================
        """
        N_prompt = rewards.shape[0]
        K = rewards.shape[1]

        # --- グループ内でrewardを正規化（advantage計算）---
        reward_mean = rewards.mean(dim=1, keepdim=True)
        # reward_mean: (N_prompt, 1)
        reward_std = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        # reward_std: (N_prompt, 1)

        advantages = (rewards - reward_mean) / reward_std
        # advantages: (N_prompt, K)

        # --- トークンレベルの損失 ---
        # 各トークンの対数確率にadvantageを掛ける
        advantages_expanded = advantages.unsqueeze(-1).expand_as(log_probs)
        # advantages_expanded: (N_prompt, K, L_resp)

        token_loss = -advantages_expanded * log_probs * response_mask
        # token_loss: (N_prompt, K, L_resp)

        # 有効トークン数で正規化
        total_tokens = response_mask.sum()
        loss = token_loss.sum() / total_tokens.clamp(min=1)
        # loss: スカラー

        return loss

    def train_step(
        self,
        prompts: List[str],
        prompt_images: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        ハイブリッドRL の1訓練ステップ

        ========================================
        処理:
            1. プロンプトを短い/長い推論モードにランダム割り当て
            2. ロールアウト: 各プロンプトからK個の応答をサンプリング
            3. 報酬計算
            4. GRPO損失計算
            5. パラメータ更新

        入力:
            prompts: プロンプトテキストのリスト (128個)
            prompt_images: 各プロンプトの画像テンソル

        出力:
            metrics: {"loss": float, "reward_mean": float}
        ========================================
        """
        cfg = self.config
        N = len(prompts)

        # --- 1. 推論モードの割り当て ---
        # 50%を長い推論モードに割り当て
        is_long_reasoning = torch.rand(N) < cfg.long_reasoning_ratio
        # is_long_reasoning: (N,) - True=長い推論, False=短い推論

        # 長い推論モードのプロンプトには <think> プレフィックスを追加
        # 短い推論モードはそのまま
        modified_prompts = []
        for i, prompt in enumerate(prompts):
            if is_long_reasoning[i]:
                modified_prompts.append(prompt + "\n<think>")
            else:
                modified_prompts.append(prompt)

        # --- 2. ロールアウト ---
        # 各プロンプトからK個の応答をサンプリング
        # (実際にはmodel.generate()を使用)
        # responses: List[List[str]] - N×K の応答テキスト
        # log_probs: (N, K, L_resp) - ポリシーの対数確率
        # ref_log_probs: (N, K, L_resp) - 参照モデルの対数確率

        # (簡略化: 実際のロールアウト処理は省略)

        # --- 3. 報酬計算 ---
        # rewards: (N, K) = compute_hybrid_reward(...)
        # → 下記の compute_hybrid_reward() を使用

        # --- 4. GRPO損失計算 ---
        # loss = self.compute_grpo_loss(log_probs, ref_log_probs, rewards, mask)

        # --- 5. パラメータ更新 ---
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        return {"loss": 0.0, "reward_mean": 0.0}


# ========================================
# 3. 報酬計算 (Reward Shaping)
# ========================================
def compute_hybrid_reward(
    responses: List[str],
    ground_truths: List[str],
    reward_model: Optional[nn.Module] = None,
    is_long_reasoning: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    ハイブリッド報酬の計算

    論文 Section 2.4.4:
        R = R_acc + R_format + R_rep + 0.5 * R̃_rm

    ========================================
    入力:
        responses: 応答テキストのリスト
        ground_truths: 正解テキストのリスト
        reward_model: 嗜好報酬モデル (Optional)
        is_long_reasoning: 長い推論モードフラグ

    出力:
        rewards: (N,) - 各応答の報酬スコア

    報酬の構成要素:
        R_acc:    正確性報酬（ルールベース or 確率ベース）
        R_format: フォーマット報酬
        R_rep:    繰り返しペナルティ
        R̃_rm:    標準化された嗜好報酬
    ========================================
    """
    N = len(responses)
    rewards = torch.zeros(N)

    for i in range(N):
        response = responses[i]
        gt = ground_truths[i]

        # --- R_acc: 正確性報酬 ---
        r_acc = _compute_accuracy_reward(response, gt)
        # r_acc: スカラー (0.0 or 1.0 for rule-based)

        # --- R_format: フォーマット報酬 ---
        r_format = _compute_format_reward(response, is_long_reasoning[i] if is_long_reasoning is not None else False)
        # r_format: スカラー

        # --- R_rep: 繰り返しペナルティ ---
        r_rep = _compute_repetition_penalty(response)
        # r_rep: スカラー (≤ 0)

        # --- R_rm: 嗜好報酬 (Reward Model) ---
        r_rm = 0.0
        if reward_model is not None:
            r_rm = _compute_preference_reward(response, reward_model, is_long_reasoning)
            # 注: 長い推論モードでは最終回答部分のみをRMで評価
            # 論文: 「The RM scores only the final answer part of the response,
            #        completely bypassing the explicit thinking steps.」

        rewards[i] = r_acc + r_format + r_rep + 0.5 * r_rm

    return rewards


def _compute_accuracy_reward(response: str, ground_truth: str) -> float:
    """
    正確性報酬の計算

    論文 Section 2.4.2:
        「For straightforward answers containing only a few tokens, we employ
         a rule-based verification system, achieving 98% reward accuracy.
         For complex natural language answers, we use the probability-based
         rewards of RLPR [29].」

    ========================================
    ルールベース (短い回答):
        - 数値の完全一致 or 許容誤差内
        - 選択肢の一致
        - 98%の精度

    確率ベース RLPR (複雑な回答):
        - 参照モデルの条件付き確率で評価
        - P(ground_truth | response) を計算
    ========================================
    """
    # ルールベース: 短い回答の場合
    response_cleaned = response.strip().lower()
    gt_cleaned = ground_truth.strip().lower()

    # 数値比較
    try:
        resp_val = float(response_cleaned)
        gt_val = float(gt_cleaned)
        if abs(resp_val - gt_val) < 1e-6:
            return 1.0
        return 0.0
    except ValueError:
        pass

    # 完全一致
    if response_cleaned == gt_cleaned:
        return 1.0

    # 確率ベース RLPR (簡略化)
    # 実際にはP(gt | response)を参照モデルで計算
    return 0.0


def _compute_format_reward(response: str, is_long_reasoning: bool) -> float:
    """
    フォーマット報酬の計算

    長い推論モードの場合、<think>...</think> の存在を確認
    """
    if is_long_reasoning:
        if "<think>" in response and "</think>" in response:
            return 0.1
        return -0.1
    return 0.0


def _compute_repetition_penalty(response: str) -> float:
    """
    繰り返しペナルティ

    応答中の繰り返しパターンにペナルティを適用
    """
    # 簡略化: n-gram繰り返し率を計算
    words = response.split()
    if len(words) < 10:
        return 0.0

    # 3-gram繰り返し率
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    unique_ratio = len(set(trigrams)) / max(len(trigrams), 1)

    if unique_ratio < 0.5:
        return -0.5  # 繰り返しが多い
    return 0.0


def _compute_preference_reward(
    response: str,
    reward_model: nn.Module,
    is_long_reasoning: bool,
) -> float:
    """
    嗜好報酬 (Reward Model) の計算

    論文 Section 2.4.4:
        「The RM scores only the final answer part of the response,
         completely bypassing the explicit thinking steps.」

    ========================================
    処理:
        1. 長い推論モードの場合: <think>...</think> を除去し、
           最終回答部分のみをRMに入力
        2. RMスコアを正規化:
           R̃_rm = (R_rm - μ(R_rm)) / σ(R_rm)
           μ, σ は同一プロンプトの応答群から計算
    ========================================
    """
    # 長い推論の場合、思考過程を除去
    if is_long_reasoning and "</think>" in response:
        # </think> 以降のみをRMに入力
        answer_part = response.split("</think>")[-1].strip()
    else:
        answer_part = response

    # RMスコア計算 (簡略化)
    # score = reward_model(answer_part)
    score = 0.0

    return score


# ========================================
# 4. RLAIF-V (幻覚低減)
# ========================================
class RLAIFVTrainer:
    """
    RLAIF-V によるDPO学習

    論文 Section 2.4.5:
        「We integrate RLAIF-V [28] to make the responses more factually
         grounded to the visual input through alignment from scalable AI feedback.」

    ========================================
    パイプライン:
        1. Response Sampling: ポリシーモデルから複数回答をサンプリング
        2. Feedback Collection: 各回答を検証可能な原子的主張に分解
           → 主張レベルの事実検証
        3. Preference Pair Construction: 事実誤りが少ない回答を選好
        4. DPO Training: 選好ペアでDPOを実行

    DPO設定:
        - バッチサイズ: 256
        - 学習率: 1e-6
        - β: 0.1
        - ステップ数: 400
    ========================================
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
    ):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta

    def compute_dpo_loss(
        self,
        policy_logps_chosen: torch.Tensor,     # (B,)
        policy_logps_rejected: torch.Tensor,    # (B,)
        ref_logps_chosen: torch.Tensor,         # (B,)
        ref_logps_rejected: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        """
        DPO損失の計算

        ========================================
        入力:
            policy_logps_chosen: (B,)
                - ポリシーモデルの選好回答の対数確率
            policy_logps_rejected: (B,)
                - ポリシーモデルの非選好回答の対数確率
            ref_logps_chosen: (B,)
                - 参照モデルの選好回答の対数確率
            ref_logps_rejected: (B,)
                - 参照モデルの非選好回答の対数確率

        出力:
            loss: スカラー

        数式:
            loss = -log σ(β * (log π(y_w|x)/π_ref(y_w|x)
                            - log π(y_l|x)/π_ref(y_l|x)))
        ========================================
        """
        # log-ratio計算
        pi_logratios = policy_logps_chosen - policy_logps_rejected
        # pi_logratios: (B,)

        ref_logratios = ref_logps_chosen - ref_logps_rejected
        # ref_logratios: (B,)

        logits = self.beta * (pi_logratios - ref_logratios)
        # logits: (B,)

        loss = -F.logsigmoid(logits).mean()
        # loss: スカラー

        return loss


# ========================================
# 5. CPMTrainer (HuggingFace Trainer拡張)
# ========================================
class CPMTrainer:
    """
    MiniCPM-V 4.5 の学習トレーナー

    公式実装: finetune/trainer.py: CPMTrainer

    HuggingFace Trainerを拡張し、以下をカスタマイズ:
        - compute_loss: data dict形式でモデルにデータを渡す
        - training_step: 各ステップ後にCUDAキャッシュをクリア
        - _save: モデル保存ロジック

    ========================================
    学習設定:
        事前学習:
            - Stage 1: LR 5e-5 (WSD: Warmup-Stable-Decay)
            - Stage 2: LR 5e-5 (WSD)
            - Stage 3: LR 5e-5 (WSD, decay phase でデータ品質向上)

        SFT:
            - Stage 1: LR 1e-5 → 1e-6 (cosine decay)
            - Stage 2: LR 5e-6 → 1e-6 (Long-CoT & 3D-Resampler)

        RL:
            - GRPO: LR 1e-6 (固定)
            - RLAIF-V: LR 1e-6, batch=256, β=0.1, 400ステップ
    ========================================
    """

    def __init__(self, model: nn.Module, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        損失計算

        公式実装: finetune/trainer.py: CPMTrainer.compute_loss()

        ========================================
        入力:
            model: MiniCPM-V 4.5 モデル
            inputs: {
                "input_ids": (B, L),
                "labels": (B, L),
                "pixel_values": List[Tensor],
                "image_bound": List[Tensor],
                "tgt_sizes": List[Tensor],
                "attention_mask": (B, L),
                "position_ids": (B, L),
            }

        出力:
            loss: スカラー
        ========================================
        """
        labels = inputs.pop("labels", None)

        # モデルフォワード (data dict形式)
        # 公式: self.model(data=inputs, use_cache=False)
        outputs = model(data=inputs, use_cache=False)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1, model.config.vocab_size).contiguous()
            # logits: (B*L, V)

            labels = labels.view(-1).long().contiguous()
            # labels: (B*L,)

            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        1訓練ステップ

        公式実装: finetune/trainer.py: CPMTrainer.training_step()

        特徴: 各ステップ後に torch.cuda.empty_cache() を呼び出す
              → マルチモーダルデータの可変長による断片化を防止
        """
        model.train()

        loss = self.compute_loss(model, inputs)

        # メモリ断片化防止
        del inputs
        torch.cuda.empty_cache()

        loss.backward()

        return loss.detach()


# ========================================
# 6. 学習設定 (Training Configurations)
# ========================================
@dataclass
class PretrainingConfig:
    """
    3段階事前学習の設定

    論文 Section 2.2.1:
        Stage 1: Resamplerのみ学習（他はフリーズ）
        Stage 2: 視覚エンコーダ + Resampler学習（LLMフリーズ）
        Stage 3: 全パラメータEnd-to-End学習
    """
    # Stage 1: Resampler ウォームアップ
    stage1_lr: float = 5e-5
    stage1_frozen: List[str] = field(default_factory=lambda: ["vision_tower", "llm"])
    stage1_data: str = "image-caption pairs"

    # Stage 2: 視覚エンコーダ解放
    stage2_lr: float = 5e-5
    stage2_frozen: List[str] = field(default_factory=lambda: ["llm"])
    stage2_data: str = "OCR-rich data + image-caption pairs"

    # Stage 3: 全パラメータ学習
    stage3_lr: float = 5e-5
    stage3_frozen: List[str] = field(default_factory=list)
    stage3_data: str = "text-only + interleaved + video + document"

    # 共通
    lr_scheduler: str = "WSD (Warmup-Stable-Decay)"
    lr_decay_to: float = 1e-5


@dataclass
class SFTConfig:
    """
    2段階SFTの設定

    論文 Section 2.3.1:
        Stage 1: 汎用SFT
        Stage 2: Long-CoT & 3D-Resampler
    """
    # Stage 1: 汎用SFT
    stage1_lr_start: float = 1e-5
    stage1_lr_end: float = 1e-6
    stage1_data: str = "high-quality instruction-response + 10% text-only"

    # Stage 2: Long-CoT & 3D-Resampler
    stage2_lr_start: float = 5e-6
    stage2_lr_end: float = 1e-6
    stage2_data: str = "Long-CoT + high frame rate video"
    stage2_note: str = "2D→3D Resampler拡張、時間位置埋め込み追加"

    # 共通
    lr_scheduler: str = "cosine decay"


@dataclass
class RLConfig:
    """
    強化学習の設定

    論文 Section 2.4 & Appendix A
    """
    # GRPO
    grpo_lr: float = 1e-6
    grpo_batch_prompts: int = 128
    grpo_responses_per_prompt: int = 8
    grpo_max_response_length: int = 8192
    grpo_temperature: float = 1.0
    grpo_long_ratio: float = 0.5
    grpo_kl_loss: bool = False  # KL損失なし
    grpo_entropy_loss: bool = False  # エントロピー損失なし

    # RLAIF-V
    rlaifv_lr: float = 1e-6
    rlaifv_batch_size: int = 256
    rlaifv_beta: float = 0.1
    rlaifv_steps: int = 400

    # 報酬構成
    reward_formula: str = "R = R_acc + R_format + R_rep + 0.5 * R̃_rm"


# ========================================
# LoRA微調整設定
# ========================================
@dataclass
class LoRAConfig:
    """
    LoRA微調整の設定

    公式実装: finetune/finetune.py: LoraArguments

    ========================================
    ターゲットモジュール:
        llm.*.layers.*.self_attn.(q_proj|k_proj|v_proj)

    modules_to_save:
        embed_tokens, resampler, (vpm if tune_vision)
    ========================================
    """
    r: int = 64
    alpha: int = 64
    dropout: float = 0.05
    target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    modules_to_save: List[str] = field(
        default_factory=lambda: ["embed_tokens", "resampler"]
    )
    bias: str = "none"


# ========================================
# 使用例
# ========================================
def example_training_configs():
    """
    全学習設定の表示
    """
    print("=== 事前学習設定 ===")
    pt_cfg = PretrainingConfig()
    print(f"  Stage 1: LR={pt_cfg.stage1_lr}, frozen={pt_cfg.stage1_frozen}")
    print(f"  Stage 2: LR={pt_cfg.stage2_lr}, frozen={pt_cfg.stage2_frozen}")
    print(f"  Stage 3: LR={pt_cfg.stage3_lr}, frozen={pt_cfg.stage3_frozen}")

    print("\n=== SFT設定 ===")
    sft_cfg = SFTConfig()
    print(f"  Stage 1: LR={sft_cfg.stage1_lr_start}→{sft_cfg.stage1_lr_end}")
    print(f"  Stage 2: LR={sft_cfg.stage2_lr_start}→{sft_cfg.stage2_lr_end}")

    print("\n=== RL設定 ===")
    rl_cfg = RLConfig()
    print(f"  GRPO: LR={rl_cfg.grpo_lr}, batch={rl_cfg.grpo_batch_prompts}×{rl_cfg.grpo_responses_per_prompt}")
    print(f"  報酬: {rl_cfg.reward_formula}")
    print(f"  RLAIF-V: LR={rl_cfg.rlaifv_lr}, β={rl_cfg.rlaifv_beta}, steps={rl_cfg.rlaifv_steps}")

    print("\n=== LoRA設定 ===")
    lora_cfg = LoRAConfig()
    print(f"  r={lora_cfg.r}, alpha={lora_cfg.alpha}, dropout={lora_cfg.dropout}")
    print(f"  target: {lora_cfg.target_modules}")


if __name__ == "__main__":
    example_training_configs()
