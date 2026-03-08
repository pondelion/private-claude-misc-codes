"""
Qwen3-Omni ロス計算 - 簡略化疑似コード
==========================================

Pre-training (3段階) と Post-training (Thinker 3段階 + Talker 4段階) のロス計算

Qwen2.5-Omni との主な差分:
    - GSPO (Graduate-level Self-Preference Optimization) の追加
        - ルールベース報酬: 数学・コーディング・指示追従の客観的正確性
        - モデルベース報酬: LLM-as-judge (Qwen3汎用, Qwen2.5-VLビジョン用)
    - Strong-to-Weak Distillation の追加
        - Off-policy: 教師 (Qwen3-32B / Qwen3-235B-A22B) が応答生成 → 応答蒸留
        - On-policy: 生徒が応答生成 → 教師と生徒のlogits間でKLダイバージェンス最小化
    - Talker: 4段階 (2.5-Omniは3段階), CPTステージ追加
    - マルチコードブックロス (単一コードブック → 複数コードブック)
    - MTPモジュールロス (残差コードブック用)

公式論文: Section 4 (Pre-training), Section 5 (Post-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# ============================================
# Pre-training ロス
# ============================================

class PretrainingLoss(nn.Module):
    """
    Qwen3-Omni Pre-training のロス計算

    3段階のPre-training:

    Stage 1 (S1): エンコーダアライメント (LLM凍結)
        - LLMパラメータを凍結
        - Vision Encoder (Qwen3-VLから) + Audio Encoder (AuT) のアダプタのみ学習
        - AuT: 20M時間の教師あり音声データでスクラッチ学習された ~650M エンコーダ
        - ロス: Cross-Entropy (エンコーダ出力タスク)

    Stage 2 (S2): 全パラメータ学習 (~2T トークン)
        - 全パラメータを解凍
        - データ構成:
            - テキスト: 0.57T トークン (28.5%)
            - 音声:     0.77T トークン (38.5%)
            - 画像:     0.82T トークン (41%)
            - 動画:     0.05T トークン (2.5%)
            - 動画+音声: 0.05T トークン
        - max_length: 8,192 トークン
        - ロス: 標準 causal LM Cross-Entropy

    Stage 3 (S3): 長コンテキスト学習
        - コンテキスト拡張: 8192 → 32768
        - 長音声・長動画データを追加
        - ロス: S2と同一

    ★ Qwen2.5-Omni との差分:
        - AuT (スクラッチ学習) が Whisper-large-v3 を置き換え
        - S2データ量: ~2T (2.5-Omniは~1.2T)
        - S2のモダリティ比率がより均衡 (音声38.5%増加)
    """

    def __init__(self, vocab_size: int = 151936, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-training のロス計算

        入力:
            logits: (B, L, vocab_size) - モデルの出力ロジット
                B: バッチサイズ
                L: シーケンス長
                vocab_size: 151936

            labels: (B, L) - 教師ラベル
                -100: 無視 (入力部分、パディング)
                0-151935: 正解トークンID

        出力:
            Dict {
                'loss': scalar - Cross-Entropy Loss
                'num_tokens': scalar - 有効トークン数
            }

        ★ Next-Token Prediction:
            logits[t] → labels[t+1] を予測
            入力部分 (labels=-100) はロスに含めない
        """

        # シフト: logits[:-1] → labels[1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_logits: (B, L-1, vocab_size)
        # shift_labels: (B, L-1)

        # Cross-Entropy Loss
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),  # (B*(L-1), vocab_size)
            shift_labels.view(-1),                    # (B*(L-1),)
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        # 有効トークン数 (ignore_index でない数)
        num_tokens = (shift_labels != self.ignore_index).sum()

        return {
            'loss': loss,
            'num_tokens': num_tokens,
        }


# ============================================
# Thinker Post-training ロス: SFT
# ============================================

class ThinkerSFTLoss(nn.Module):
    """
    Thinker Post-training Stage 1: Supervised Fine-Tuning (SFT)

    ChatML形式の指示追従データで学習

    データ形式:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        <|vision_bos|><|IMAGE|><|vision_eos|>What is in this image?<|im_end|>
        <|im_start|>assistant
        A cat sitting on a windowsill.<|im_end|>

    ロス計算:
        assistant の応答部分のみでロスを計算
        system, user, 特殊トークンは labels=-100 で無視

    データ構成:
        - 純テキスト対話データ
        - 画像モダリティ会話データ (ビジョン)
        - 音声モダリティ会話データ
        - 混合モダリティ会話データ

    ★ Qwen2.5-Omni の ThinkerPosttrainingLoss と同一構造だが、
      Qwen3-Omni では SFT は Post-training の第1段階に位置付け
      (後に Distillation → GSPO と続く)
    """

    def __init__(self, vocab_size: int = 151936, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        入力:
            logits: (B, L, vocab_size) - Thinker出力
            labels: (B, L) - 教師ラベル
                assistant応答部分のみ有効なトークンID
                それ以外は -100

        出力:
            Dict { 'loss': scalar }
        """
        # Pre-trainingと同一のCross-Entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        return {'loss': loss}

    @staticmethod
    def create_labels_from_chatml(
        input_ids: torch.Tensor,
        assistant_start_token_id: int,
        assistant_end_token_id: int,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        ChatML形式のinput_idsからlabelsを作成

        入力:
            input_ids: (B, L) - 全トークンID
            assistant_start_token_id: <|im_start|>assistant の次のトークンID
            assistant_end_token_id: <|im_end|> のトークンID

        出力:
            labels: (B, L) - assistant応答部分のみ有効

        例:
            input_ids:  [sys, ..., sys_end, user, ..., user_end, asst, A, cat, ..., asst_end]
            labels:     [-100, ..., -100,   -100, ..., -100,    -100, A, cat, ..., asst_end]
        """
        B, L = input_ids.shape
        labels = torch.full_like(input_ids, ignore_index)

        for b in range(B):
            in_assistant = False
            for i in range(L):
                if input_ids[b, i] == assistant_start_token_id:
                    in_assistant = True
                    continue
                if in_assistant:
                    labels[b, i] = input_ids[b, i]
                if input_ids[b, i] == assistant_end_token_id:
                    in_assistant = False

        return labels


# ============================================
# Thinker Post-training ロス: Strong-to-Weak Distillation
# ============================================

class DistillationLoss(nn.Module):
    """
    Thinker Post-training Stage 2: Strong-to-Weak Distillation

    大規模教師モデルから小規模生徒モデルへの知識蒸留

    ★ Qwen3-Omni 新規追加 (Qwen2.5-Omni にはない)

    2つのモード:

    (A) Off-policy 蒸留 (応答蒸留):
        - 教師モデル (Qwen3-32B / Qwen3-235B-A22B) が応答を生成
        - 生成された応答を「正解」として生徒にSFT
        - ロス: 教師生成テキストに対する Cross-Entropy
        - 利点: 教師の高品質応答を直接学習
        - データ: 教師が生成したマルチモーダル対話応答

    (B) On-policy 蒸留 (KLダイバージェンス最小化):
        - 生徒モデルが応答を生成
        - 生徒の各トークン位置で教師と生徒のlogits分布を比較
        - ロス: KL(teacher_dist || student_dist) の最小化
        - 利点: 生徒の分布空間での最適化
        - 温度パラメータで分布のsoftnessを制御

    合計ロス:
        L_distill = α * L_off_policy + (1-α) * L_on_policy

    教師モデル:
        - Qwen3-32B: 高速推論用
        - Qwen3-235B-A22B: 高品質応答用 (MoE, 22B活性化)
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        ignore_index: int = -100,
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
        """
        パラメータ:
            vocab_size: 語彙サイズ
            ignore_index: 無視するラベルインデックス
            temperature: KLダイバージェンスの温度パラメータ
                大きい値 → 分布がソフトに (暗黙的知識をより多く伝達)
                小さい値 → 分布がシャープに (hard label に近づく)
            alpha: Off-policy と On-policy の混合比率
                alpha=1.0 → Off-policy のみ
                alpha=0.0 → On-policy のみ
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.alpha = alpha

    def off_policy_loss(
        self,
        student_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Off-policy 蒸留ロス: 教師生成応答に対するCross-Entropy

        教師 (Qwen3-32B / Qwen3-235B-A22B) が生成したテキストを
        正解ラベルとして生徒にSFT学習させる

        入力:
            student_logits: (B, L, vocab_size) - 生徒モデルの出力ロジット
            teacher_labels: (B, L) - 教師モデルが生成した応答のトークンID
                -100: 無視 (プロンプト部分)
                0-151935: 教師生成トークンID

        出力:
            loss: scalar - Cross-Entropy Loss

        計算:
            L_off = CE(student_logits, teacher_generated_tokens)
            ※ SFTと同一だが、正解が人手アノテーションではなく教師の生成応答
        """
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )
        return loss

    def on_policy_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        On-policy 蒸留ロス: 生徒と教師のlogits間のKLダイバージェンス

        生徒が応答を生成し、各トークン位置で教師のlogit分布との
        KLダイバージェンスを最小化する

        入力:
            student_logits: (B, L, vocab_size) - 生徒モデルの出力ロジット
            teacher_logits: (B, L, vocab_size) - 教師モデルの出力ロジット
                ※ 生徒が生成した同一トークン列に対する教師の出力
            labels: (B, L) - 有効位置の判定用
                -100: 無視 (プロンプト部分)
                それ以外: ロスを計算する位置

        出力:
            loss: scalar - KL Divergence Loss (温度スケーリング済み)

        計算:
            1. 温度 T で logits をソフト化:
               p_teacher = softmax(teacher_logits / T)
               log_p_student = log_softmax(student_logits / T)
            2. KLダイバージェンス:
               KL = Σ p_teacher * (log(p_teacher) - log_p_student)
            3. 温度の二乗でスケーリング:
               L_on = T^2 * KL
               ※ T^2 は勾配スケールを温度非依存にするため
        """
        # シフト: student/teacher ともに同じ位置でアライン
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 有効位置のマスク
        mask = (shift_labels != self.ignore_index).float()
        # mask: (B, L-1) - 1.0=有効, 0.0=無視

        # 温度スケーリングされた確率分布
        T = self.temperature
        teacher_probs = F.softmax(shift_teacher / T, dim=-1)
        # teacher_probs: (B, L-1, vocab_size) - 教師のソフトラベル
        student_log_probs = F.log_softmax(shift_student / T, dim=-1)
        # student_log_probs: (B, L-1, vocab_size) - 生徒の対数確率

        # KLダイバージェンス: KL(teacher || student) = Σ p_t * (log p_t - log p_s)
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='none',
            log_target=False,
        )
        # kl_div: (B, L-1, vocab_size)

        # 語彙次元で合計
        kl_per_token = kl_div.sum(dim=-1)
        # kl_per_token: (B, L-1)

        # マスク適用して平均
        kl_masked = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

        # T^2 スケーリング
        loss = (T ** 2) * kl_masked

        return loss

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        蒸留ロスの合計計算

        入力:
            student_logits: (B, L, vocab_size) - 生徒モデルの出力
            teacher_labels: (B, L) - 教師生成の応答ラベル (Off-policy用)
            teacher_logits: (B, L, vocab_size) - 教師の出力ロジット (On-policy用)
            labels: (B, L) - 有効位置マスク用ラベル (On-policy用)

        出力:
            Dict {
                'loss': scalar - α * L_off + (1-α) * L_on
                'off_policy_loss': scalar - Off-policy CE ロス
                'on_policy_loss': scalar - On-policy KL ロス
            }

        ★ 実運用では Off-policy と On-policy を交互またはバッチ内混合で実行
        """
        result = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)

        # Off-policy ロス
        if teacher_labels is not None:
            off_loss = self.off_policy_loss(student_logits, teacher_labels)
            result['off_policy_loss'] = off_loss
            total_loss = total_loss + self.alpha * off_loss

        # On-policy ロス
        if teacher_logits is not None and labels is not None:
            on_loss = self.on_policy_loss(student_logits, teacher_logits, labels)
            result['on_policy_loss'] = on_loss
            total_loss = total_loss + (1.0 - self.alpha) * on_loss

        result['loss'] = total_loss
        return result


# ============================================
# Thinker Post-training ロス: GSPO
# ============================================

class GSPOLoss(nn.Module):
    """
    Thinker Post-training Stage 3: GSPO (Graduate-level Self-Preference Optimization)

    ★ Qwen3-Omni 新規追加 (Qwen2.5-Omni にはない)

    自己選好最適化による報酬ベース強化学習

    2種類の報酬:

    (A) ルールベース報酬 (Rule-based Reward):
        - 数学: 最終回答の数値/式が正解と一致するか
        - コーディング: テストケース通過率
        - 指示追従: フォーマット制約の遵守率
        - 利点: 客観的・再現性のある報酬
        - 例: 数学問題 → 正解=1.0, 不正解=0.0

    (B) モデルベース報酬 (Model-based Reward):
        - LLM-as-judge パラダイム
        - 汎用タスク: Qwen3 が応答品質を評価 (0-10スコア)
        - ビジョンタスク: Qwen2.5-VL が画像理解の正確性を評価
        - 利点: ルール化困難なタスクにも適用可能

    GSPO ロス (DPO変種):
        L_GSPO = -E_{(x, y_w, y_l)} [
            log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
        ]

    変数:
        x: 入力プロンプト (マルチモーダル)
        y_w: 報酬が高い応答 (preferred)
        y_l: 報酬が低い応答 (rejected)
        π_θ: 学習中のポリシー
        π_ref: 参照ポリシー (SFT/蒸留後のスナップショット)
        β: 温度パラメータ

    選好ペア構築:
        1. ポリシー π_θ で N 個の応答をサンプリング
        2. 報酬モデルで各応答をスコアリング
        3. 最高スコア → y_w, 最低スコア → y_l
    """

    def __init__(self, beta: float = 0.1):
        """
        パラメータ:
            beta: GSPOの温度パラメータ
                小さい値 → 参照モデルに近く保守的
                大きい値 → 報酬差を強く反映し積極的
        """
        super().__init__()
        self.beta = beta

    def compute_rewards(
        self,
        responses: List[str],
        reward_type: str = "rule",
        ground_truth: Optional[str] = None,
    ) -> torch.Tensor:
        """
        報酬の計算 (疑似コード)

        入力:
            responses: N個の応答テキスト
            reward_type: "rule" (ルールベース) or "model" (モデルベース)
            ground_truth: 正解 (ルールベースの場合)

        出力:
            rewards: (N,) - 各応答のスコア

        ★ 実際の実装では外部の報酬モデルや評価関数を呼び出す
           ここではダミー実装
        """
        N = len(responses)
        if reward_type == "rule":
            # ルールベース: 正解一致 → 1.0, 不一致 → 0.0
            # 実装では数学の式比較、コードのテストケース実行等
            rewards = torch.zeros(N)
            if ground_truth is not None:
                for i, resp in enumerate(responses):
                    if ground_truth in resp:
                        rewards[i] = 1.0
            return rewards
        else:
            # モデルベース: LLM-as-judge (Qwen3 / Qwen2.5-VL)
            # 実装ではモデルにスコアリングプロンプトを送信
            rewards = torch.rand(N)  # ダミー: ランダムスコア
            return rewards

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        GSPO ロスの計算

        入力:
            policy_chosen_logps: (B,) - π_θ(y_w|x) の対数確率
            policy_rejected_logps: (B,) - π_θ(y_l|x) の対数確率
            reference_chosen_logps: (B,) - π_ref(y_w|x) の対数確率
            reference_rejected_logps: (B,) - π_ref(y_l|x) の対数確率

        出力:
            Dict {
                'loss': scalar - GSPO Loss
                'chosen_rewards': (B,) - 選好応答の暗黙的報酬
                'rejected_rewards': (B,) - 拒否応答の暗黙的報酬
                'reward_margin': (B,) - 暗黙的報酬の差分
                'accuracy': scalar - y_w の暗黙的報酬 > y_l の比率
            }

        計算:
            1. 対数確率比:
               log_ratio_w = log π_θ(y_w|x) - log π_ref(y_w|x)
               log_ratio_l = log π_θ(y_l|x) - log π_ref(y_l|x)
            2. GSPO Loss:
               loss = -log σ(β * (log_ratio_w - log_ratio_l))
        """
        # 対数確率比
        log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
        # log_ratio_chosen: (B,)
        log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
        # log_ratio_rejected: (B,)

        # GSPO ロス (DPO と同一形式)
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)
        # logits: (B,)

        loss = -F.logsigmoid(logits).mean()
        # loss: scalar

        # 暗黙的報酬 (モニタリング用)
        chosen_rewards = self.beta * log_ratio_chosen.detach()
        rejected_rewards = self.beta * log_ratio_rejected.detach()
        reward_margin = chosen_rewards - rejected_rewards

        # 正答率: chosen の暗黙的報酬が rejected より高い割合
        accuracy = (reward_margin > 0).float().mean()

        return {
            'loss': loss,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards,
            'reward_margin': reward_margin,
            'accuracy': accuracy,
        }

    @staticmethod
    def compute_sequence_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        シーケンスの対数確率を計算

        入力:
            logits: (B, L, vocab_size) - モデル出力
            labels: (B, L) - 正解ラベル

        出力:
            log_probs: (B,) - 各サンプルの対数確率

        計算:
            log P(y|x) = Σ_{t: labels[t]!=ignore} log P(y_t | y_{<t}, x)
        """
        # シフト
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # トークンごとの対数確率
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # log_probs: (B, L-1, vocab_size)

        # 正解トークンの対数確率を取得
        per_token_logps = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        # per_token_logps: (B, L-1)

        # 無視トークンをマスク
        mask = (shift_labels != ignore_index).float()
        per_token_logps = per_token_logps * mask

        # シーケンス全体の対数確率
        sequence_logps = per_token_logps.sum(dim=-1)
        # sequence_logps: (B,)

        return sequence_logps


# ============================================
# Talker Post-training ロス: マルチコードブック
# ============================================

class TalkerMultiCodebookLoss(nn.Module):
    """
    Talker Post-training Stage 1-2: マルチコードブックロス

    Qwen3-Omni の Talker は Multi-Token Prediction (MTP) モジュールを使い、
    複数のコードブックを同時に予測する

    ★ Qwen2.5-Omni との差分:
        - 単一コードブック → マルチコードブック (K個)
        - 第1コードブック: メインLM head で予測 (autoregressiveに次トークン予測)
        - 第2~K コードブック: MTP モジュール (残差予測ヘッド) で予測
        - 各コードブックは異なる抽象度の音声情報をエンコード

    Stage 1: マルチモーダルコンテキスト音声データ → 単調マッピング学習
        - 対話コンテキスト + 音声応答
        - 韻律・感情・アクセントの多様な表現獲得
        - 音色分離 (timbre disentanglement)
        - ロス: K個のコードブック全てのCE合計

    Stage 2 (CPT: Continued Pre-Training): 高品質データ + 長コンテキスト
        - 高品質音声データで継続事前学習
        - コンテキスト拡張で長い対話に対応
        - ロス: Stage 1と同一

    合計ロス:
        L_talker = L_main + Σ_{k=2}^{K} λ_k * L_residual_k
        - L_main: 第1コードブックの NTP Cross-Entropy
        - L_residual_k: 第kコードブックの MTP Cross-Entropy
        - λ_k: 各残差コードブックのロス重み (通常1.0)
    """

    def __init__(
        self,
        codebook_size: int = 8295,
        num_codebooks: int = 4,
        ignore_index: int = -100,
        residual_weights: Optional[List[float]] = None,
    ):
        """
        パラメータ:
            codebook_size: 各コードブックの語彙サイズ
            num_codebooks: コードブック数 K
            ignore_index: 無視インデックス
            residual_weights: 各残差コードブックのロス重み [λ_2, ..., λ_K]
                None の場合は全て 1.0
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.ignore_index = ignore_index

        if residual_weights is not None:
            assert len(residual_weights) == num_codebooks - 1, \
                f"残差ロス重みの長さは {num_codebooks - 1} である必要がある"
            self.residual_weights = residual_weights
        else:
            self.residual_weights = [1.0] * (num_codebooks - 1)

    def forward(
        self,
        main_logits: torch.Tensor,
        main_labels: torch.Tensor,
        residual_logits_list: Optional[List[torch.Tensor]] = None,
        residual_labels_list: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        マルチコードブックロスの計算

        入力:
            main_logits: (B, L, codebook_size) - 第1コードブックのロジット
            main_labels: (B, L) - 第1コードブックの正解トークン
            residual_logits_list: [(B, L, codebook_size)] * (K-1)
                - 第2~Kコードブックのロジット (MTPモジュール出力)
            residual_labels_list: [(B, L)] * (K-1)
                - 第2~Kコードブックの正解トークン

        出力:
            Dict {
                'loss': scalar - 全コードブックの合計ロス
                'main_loss': scalar - 第1コードブックのロス
                'residual_losses': [scalar] * (K-1) - 各残差コードブックのロス
            }
        """
        # 第1コードブック: メインNTPロス
        shift_main_logits = main_logits[..., :-1, :].contiguous()
        shift_main_labels = main_labels[..., 1:].contiguous()

        main_loss = F.cross_entropy(
            shift_main_logits.view(-1, self.codebook_size),
            shift_main_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        total_loss = main_loss
        residual_losses = []

        # 第2~Kコードブック: MTPモジュールによる残差ロス
        if residual_logits_list is not None and residual_labels_list is not None:
            assert len(residual_logits_list) == self.num_codebooks - 1, \
                f"残差ロジットは {self.num_codebooks - 1} 個必要"
            assert len(residual_labels_list) == self.num_codebooks - 1, \
                f"残差ラベルは {self.num_codebooks - 1} 個必要"

            for k, (res_logits, res_labels) in enumerate(
                zip(residual_logits_list, residual_labels_list)
            ):
                # 各残差コードブックのCross-Entropy
                shift_res_logits = res_logits[..., :-1, :].contiguous()
                shift_res_labels = res_labels[..., 1:].contiguous()

                res_loss = F.cross_entropy(
                    shift_res_logits.view(-1, self.codebook_size),
                    shift_res_labels.view(-1),
                    ignore_index=self.ignore_index,
                    reduction='mean',
                )
                residual_losses.append(res_loss)
                total_loss = total_loss + self.residual_weights[k] * res_loss

        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'residual_losses': residual_losses,
        }


# ============================================
# Talker Post-training ロス: DPO
# ============================================

class TalkerDPOLoss(nn.Module):
    """
    Talker Post-training Stage 3: DPO (Direct Preference Optimization)

    多言語話者選好ペアによるDPO学習

    DPO Loss:
        L_DPO(π_θ; π_ref) = -E_{(x, y_w, y_l) ~ D} [
            log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
        ]

    変数:
        x: 入力シーケンス (マルチモーダルコンテキスト + テキスト)
        y_w: 好ましい音声 (低WER, 自然な韻律, 高MOS)
        y_l: 好ましくない音声 (高WER, 不自然な韻律, 低MOS)
        π_θ: 現在のモデル (学習対象)
        π_ref: 参照モデル (Stage 2 CPT 終了時のスナップショット)
        β: 温度パラメータ

    ★ Qwen2.5-Omni との差分:
        - Stage 2 → Stage 3 (CPTステージが新たに挿入されたため)
        - 多言語対応の選好ペア生成
        - マルチコードブック対応のシーケンス対数確率計算

    ランキング基準:
        - WER (Word Error Rate): 書き起こし精度
        - 句読点停止エラー率
        - MOS (Mean Opinion Score): 主観的音声品質
    """

    def __init__(self, beta: float = 0.1):
        """
        パラメータ:
            beta: DPOの温度パラメータ
                小さい値 → 参照モデルに近い
                大きい値 → 報酬の差を強く反映
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        DPO ロスの計算

        入力:
            policy_chosen_logps: (B,) - π_θ(y_w|x) の対数確率
            policy_rejected_logps: (B,) - π_θ(y_l|x) の対数確率
            reference_chosen_logps: (B,) - π_ref(y_w|x) の対数確率
            reference_rejected_logps: (B,) - π_ref(y_l|x) の対数確率

        出力:
            Dict {
                'loss': scalar - DPO Loss
                'chosen_rewards': (B,) - 選好音声の暗黙的報酬
                'rejected_rewards': (B,) - 拒否音声の暗黙的報酬
            }
        """
        # 対数確率比
        log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
        # log_ratio_chosen: (B,)
        log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
        # log_ratio_rejected: (B,)

        # DPO ロス
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)
        # logits: (B,)
        loss = -F.logsigmoid(logits).mean()
        # loss: scalar

        # 暗黙的報酬 (モニタリング用)
        chosen_rewards = self.beta * log_ratio_chosen.detach()
        rejected_rewards = self.beta * log_ratio_rejected.detach()

        return {
            'loss': loss,
            'chosen_rewards': chosen_rewards,
            'rejected_rewards': rejected_rewards,
        }

    @staticmethod
    def compute_sequence_logps(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        シーケンスの対数確率を計算

        入力:
            logits: (B, L, vocab_size) - モデル出力
            labels: (B, L) - 正解ラベル

        出力:
            log_probs: (B,) - 各サンプルの対数確率

        計算:
            log P(y|x) = Σ_{t: labels[t]!=ignore} log P(y_t | y_{<t}, x)
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = (shift_labels != ignore_index).float()
        per_token_logps = per_token_logps * mask

        sequence_logps = per_token_logps.sum(dim=-1)
        return sequence_logps


# ============================================
# Talker Post-training ロス: Speaker Fine-tuning
# ============================================

class TalkerSpeakerFTLoss(nn.Module):
    """
    Talker Post-training Stage 4: Speaker Fine-tuning

    特定の話者の音声に適応するためのファインチューニング

    ★ Qwen2.5-Omni では Stage 3 だったが、Qwen3-Omni では
      CPTステージ挿入により Stage 4 に

    目的:
        - Talker が特定の音声を採用できるようにする
        - 自然さと制御可能性の向上
        - マルチスピーカー対応

    データ:
        - 複数話者の指示追従音声データ
        - 話者ごとの音声特徴を学習

    ロス:
        - マルチコードブック全体のCross-Entropy
        - speaker_ids による話者コンディショニング

    ★ マルチコードブック対応:
        第1コードブック + 残差コードブック全てでロスを計算
    """

    def __init__(
        self,
        codebook_size: int = 8295,
        num_codebooks: int = 4,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.ignore_index = ignore_index

    def forward(
        self,
        codec_logits: torch.Tensor,
        codec_labels: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        入力:
            codec_logits: (B, L, codebook_size) - Talker出力
                ※ 簡易実装: 第1コードブックのみ
            codec_labels: (B, L) - 正解コードトークン
            speaker_ids: (B,) - 話者ID (optional, コンディショニング用)

        出力:
            Dict { 'loss': scalar }
        """
        shift_logits = codec_logits[..., :-1, :].contiguous()
        shift_labels = codec_labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.codebook_size),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            reduction='mean',
        )

        return {'loss': loss}


# ============================================
# 使用例
# ============================================

def example_loss_computation():
    """
    ロス計算の使用例

    各ロスクラスを実際にインスタンス化し、ダミーデータで
    フォワードパスを実行してロス値と形状を確認する
    """

    # ========================================
    # 1. Pre-training ロス (Cross-Entropy)
    # ========================================
    # Pre-training 3段階:
    #   S1: エンコーダアライメント (LLM凍結), Vision=Qwen3-VL, Audio=AuT
    #   S2: 全パラメータ学習, ~2T トークン (テキスト28.5%, 音声38.5%, 画像41%, 動画2.5%)
    #   S3: 長コンテキスト学習, 8192→32768

    B, L, V = 2, 100, 151936
    logits = torch.randn(B, L, V)
    labels = torch.randint(0, V, (B, L))
    labels[:, :30] = -100  # 入力部分は無視

    pretrain_loss = PretrainingLoss(vocab_size=V)
    result = pretrain_loss(logits, labels)

    assert result['loss'].dim() == 0  # スカラー
    assert result['num_tokens'].item() == (labels[..., 1:] != -100).sum().item()

    # ========================================
    # 2. Thinker SFT ロス (ChatML形式)
    # ========================================
    # ChatML: assistant応答部分のみでロスを計算

    thinker_sft_loss = ThinkerSFTLoss(vocab_size=V)

    # create_labels_from_chatml でラベル生成
    input_ids_chatml = torch.randint(0, V, (B, L))
    ast_start = 50  # ダミーのassistant開始トークンID
    ast_end = 51    # ダミーのassistant終了トークンID
    input_ids_chatml[0, 40] = ast_start
    input_ids_chatml[0, 60] = ast_end
    input_ids_chatml[1, 50] = ast_start
    input_ids_chatml[1, 80] = ast_end

    labels_chatml = ThinkerSFTLoss.create_labels_from_chatml(
        input_ids_chatml, ast_start, ast_end
    )
    # assistant区間のみ有効、他は-100
    assert (labels_chatml[0, :41] == -100).all()  # start以前は無視
    assert (labels_chatml[0, 41:61] != -100).any()  # assistant応答部分は有効

    thinker_result = thinker_sft_loss(logits, labels_chatml)
    assert thinker_result['loss'].dim() == 0

    # ========================================
    # 3. Strong-to-Weak Distillation ロス
    # ========================================
    # ★ Qwen3-Omni 新規追加
    # Off-policy: 教師(Qwen3-32B/235B-A22B)生成応答でSFT
    # On-policy: 生徒生成 → 教師とのKLダイバージェンス最小化

    distill_loss = DistillationLoss(
        vocab_size=V, temperature=2.0, alpha=0.5
    )

    student_logits = torch.randn(B, L, V)
    teacher_logits = torch.randn(B, L, V)
    teacher_labels = torch.randint(0, V, (B, L))
    teacher_labels[:, :30] = -100  # プロンプト部分は無視
    distill_labels = torch.randint(0, V, (B, L))
    distill_labels[:, :30] = -100

    # (A) Off-policy のみ
    off_result = distill_loss(
        student_logits, teacher_labels=teacher_labels
    )
    assert off_result['loss'].dim() == 0
    assert 'off_policy_loss' in off_result

    # (B) On-policy のみ
    distill_loss_on = DistillationLoss(vocab_size=V, temperature=2.0, alpha=0.0)
    on_result = distill_loss_on(
        student_logits, teacher_logits=teacher_logits, labels=distill_labels
    )
    assert on_result['loss'].dim() == 0
    assert 'on_policy_loss' in on_result

    # (C) Off-policy + On-policy 合計
    combined_result = distill_loss(
        student_logits,
        teacher_labels=teacher_labels,
        teacher_logits=teacher_logits,
        labels=distill_labels,
    )
    assert combined_result['loss'].dim() == 0
    assert 'off_policy_loss' in combined_result
    assert 'on_policy_loss' in combined_result

    # ========================================
    # 4. GSPO ロス
    # ========================================
    # ★ Qwen3-Omni 新規追加
    # ルールベース報酬 (数学/コーディング) + モデルベース報酬 (LLM-as-judge)

    gspo_loss = GSPOLoss(beta=0.1)

    # 報酬計算のデモ (ルールベース)
    responses = ["答えは42です", "答えは100です", "答えは42"]
    rule_rewards = gspo_loss.compute_rewards(
        responses, reward_type="rule", ground_truth="42"
    )
    assert rule_rewards.shape == (3,)
    assert rule_rewards[0] == 1.0  # "42" を含む
    assert rule_rewards[1] == 0.0  # "42" を含まない
    assert rule_rewards[2] == 1.0  # "42" を含む

    # シーケンス対数確率の計算
    B_gspo = 4
    gspo_logits = torch.randn(B_gspo, 80, V)
    gspo_labels = torch.randint(0, V, (B_gspo, 80))
    seq_logps = GSPOLoss.compute_sequence_logps(gspo_logits, gspo_labels)
    assert seq_logps.shape == (B_gspo,)

    # GSPO ロス計算
    policy_chosen = torch.randn(B_gspo)
    policy_rejected = policy_chosen - 0.5  # 選好応答の方が高確率
    ref_chosen = torch.randn(B_gspo)
    ref_rejected = torch.randn(B_gspo)

    gspo_result = gspo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    assert gspo_result['loss'].dim() == 0
    assert gspo_result['chosen_rewards'].shape == (B_gspo,)
    assert gspo_result['rejected_rewards'].shape == (B_gspo,)
    assert gspo_result['reward_margin'].shape == (B_gspo,)
    assert gspo_result['accuracy'].dim() == 0

    # ========================================
    # 5. Talker マルチコードブックロス (Stage 1-2)
    # ========================================
    # ★ Qwen2.5-Omni の単一コードブック → マルチコードブック
    # 第1コードブック: メインLM head, 第2~K: MTPモジュール

    codebook_size = 8295
    num_codebooks = 4
    L_codec = 200

    multi_cb_loss = TalkerMultiCodebookLoss(
        codebook_size=codebook_size,
        num_codebooks=num_codebooks,
        residual_weights=[1.0, 0.5, 0.25],  # 残差コードブックは減衰重み
    )

    # メインコードブック
    main_logits = torch.randn(B, L_codec, codebook_size)
    main_labels = torch.randint(0, codebook_size, (B, L_codec))
    main_labels[:, :50] = -100  # コンテキスト部分は無視

    # 残差コードブック (K-1=3 個)
    residual_logits_list = [
        torch.randn(B, L_codec, codebook_size) for _ in range(num_codebooks - 1)
    ]
    residual_labels_list = [
        torch.randint(0, codebook_size, (B, L_codec)) for _ in range(num_codebooks - 1)
    ]
    for res_labels in residual_labels_list:
        res_labels[:, :50] = -100

    # メインのみ
    main_only_result = multi_cb_loss(main_logits, main_labels)
    assert main_only_result['loss'].dim() == 0
    assert main_only_result['main_loss'].dim() == 0
    assert len(main_only_result['residual_losses']) == 0

    # マルチコードブック全体
    multi_result = multi_cb_loss(
        main_logits, main_labels,
        residual_logits_list, residual_labels_list,
    )
    assert multi_result['loss'].dim() == 0
    assert multi_result['main_loss'].dim() == 0
    assert len(multi_result['residual_losses']) == num_codebooks - 1
    # 合計ロスはメイン + 重み付き残差の和
    expected_total = multi_result['main_loss']
    for k, res_loss in enumerate(multi_result['residual_losses']):
        expected_total = expected_total + [1.0, 0.5, 0.25][k] * res_loss
    assert torch.allclose(multi_result['loss'], expected_total, atol=1e-5)

    # ========================================
    # 6. Talker DPO ロス (Stage 3)
    # ========================================
    # 多言語話者選好ペアによる DPO

    B_dpo = 4
    talker_dpo_loss = TalkerDPOLoss(beta=0.1)

    # シーケンス対数確率の計算
    dpo_logits = torch.randn(B_dpo, 80, codebook_size)
    dpo_labels = torch.randint(0, codebook_size, (B_dpo, 80))
    dpo_seq_logps = TalkerDPOLoss.compute_sequence_logps(dpo_logits, dpo_labels)
    assert dpo_seq_logps.shape == (B_dpo,)

    # DPOロス計算
    dpo_policy_chosen = torch.randn(B_dpo)
    dpo_policy_rejected = dpo_policy_chosen - 0.5
    dpo_ref_chosen = torch.randn(B_dpo)
    dpo_ref_rejected = torch.randn(B_dpo)

    dpo_result = talker_dpo_loss(
        dpo_policy_chosen, dpo_policy_rejected,
        dpo_ref_chosen, dpo_ref_rejected,
    )
    assert dpo_result['loss'].dim() == 0
    assert dpo_result['chosen_rewards'].shape == (B_dpo,)
    assert dpo_result['rejected_rewards'].shape == (B_dpo,)

    # ========================================
    # 7. Talker Speaker Fine-tune ロス (Stage 4)
    # ========================================
    # 話者適応のCross-Entropy (マルチコードブック対応)

    speaker_ft_loss = TalkerSpeakerFTLoss(
        codebook_size=codebook_size, num_codebooks=num_codebooks,
    )
    speaker_result = speaker_ft_loss(main_logits, main_labels)
    assert speaker_result['loss'].dim() == 0

    # --- 結果表示 ---
    print(f"[Qwen3-Omni ロス計算 使用例]")
    print()
    print(f"  === Pre-training ===")
    print(f"  1. Pre-training Loss (S1/S2/S3 共通):")
    print(f"     入力: logits {logits.shape}, labels {labels.shape}")
    print(f"     loss={result['loss'].item():.4f}, 有効トークン={result['num_tokens'].item()}")
    print(f"     S1: エンコーダアライメント (LLM凍結, AuT+Vision Encoder)")
    print(f"     S2: 全パラメータ ~2T (テキスト0.57T, 音声0.77T, 画像0.82T, 動画0.05T)")
    print(f"     S3: 長コンテキスト 8192→32768")
    print()
    print(f"  === Thinker Post-training ===")
    print(f"  2. Thinker SFT Loss (ChatML):")
    print(f"     labels_chatml: assistant区間のみ有効 (他は-100)")
    print(f"     loss={thinker_result['loss'].item():.4f}")
    print()
    print(f"  3. Strong-to-Weak Distillation Loss: [新規]")
    print(f"     Off-policy loss={off_result['off_policy_loss'].item():.4f}")
    print(f"     On-policy loss={on_result['on_policy_loss'].item():.4f}")
    print(f"     合計 (alpha=0.5): loss={combined_result['loss'].item():.4f}")
    print(f"       教師: Qwen3-32B / Qwen3-235B-A22B")
    print()
    print(f"  4. GSPO Loss: [新規]")
    print(f"     ルールベース報酬: {rule_rewards.tolist()}")
    print(f"     loss={gspo_result['loss'].item():.4f}")
    print(f"     accuracy={gspo_result['accuracy'].item():.4f}")
    print(f"     reward_margin: {gspo_result['reward_margin'].tolist()}")
    print()
    print(f"  === Talker Post-training ===")
    print(f"  5. Talker マルチコードブックロス (Stage 1-2): [新規]")
    print(f"     メインロス={multi_result['main_loss'].item():.4f}")
    for k, res_loss in enumerate(multi_result['residual_losses']):
        print(f"     残差CB{k+2} ロス={res_loss.item():.4f} (重み={[1.0, 0.5, 0.25][k]})")
    print(f"     合計ロス={multi_result['loss'].item():.4f}")
    print()
    print(f"  6. Talker DPO Loss (Stage 3):")
    print(f"     loss={dpo_result['loss'].item():.4f}")
    print(f"     chosen_rewards:  {dpo_result['chosen_rewards'].tolist()}")
    print(f"     rejected_rewards: {dpo_result['rejected_rewards'].tolist()}")
    print()
    print(f"  7. Talker Speaker FT Loss (Stage 4):")
    print(f"     loss={speaker_result['loss'].item():.4f}")
    print()
    print(f"  === Qwen2.5-Omni からの主な変更点 ===")
    print(f"    [新規] Strong-to-Weak Distillation (Off-policy + On-policy)")
    print(f"    [新規] GSPO (ルールベース + モデルベース報酬)")
    print(f"    [新規] マルチコードブックロス (単一→K個)")
    print(f"    [変更] Talker 3段階→4段階 (CPTステージ追加)")
    print(f"    [変更] MTPモジュールによる残差コードブック予測")


if __name__ == "__main__":
    example_loss_computation()
