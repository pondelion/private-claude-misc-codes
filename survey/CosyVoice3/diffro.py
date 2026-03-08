"""
CosyVoice3 DiffRO (Differentiable Reward Optimization) - 簡略化疑似コード
==========================================================================

LLMの後処理 (Post-training) のための強化学習手法。
音声トークン上で直接最適化を行い、CFM/Vocoderを通さずに報酬を計算。

論文: CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
公式実装: cosyvoice/llm/llm.py (forward_dpo メソッド等)

【核心的アイデア】
従来のRL-for-TTS:
  テキスト → LLM → トークン → CFM → Vocoder → 音声 → 報酬モデル → 勾配
  問題: CFM/Vocoderは非微分可能 or 計算コストが高い

DiffRO:
  テキスト → LLM → Gumbel-Softmax → Token2Text報酬モデル → 勾配
  利点: CFM/Vocoderを完全にスキップ、離散トークン上で直接最適化

Shape Convention
============================================================
B: バッチサイズ
T_speech: 音声トークン長
L_text: テキストトークン長
Q: 音声トークン語彙サイズ (6561)
V_text: テキストトークン語彙サイズ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class DiffROTrainer:
    """
    DiffRO (Differentiable Reward Optimization) トレーナー

    学習フロー:
    ┌──────────────────────────────────────────────────────────┐
    │ 1. 事前学習済みLLM (pi_ref) を凍結                         │
    │ 2. 学習対象LLM (pi_theta) をpi_refからコピー               │
    │ 3. テキスト入力 → pi_thetaで音声トークン logits を生成      │
    │ 4. Gumbel-Softmax でソフトな離散トークンを取得              │
    │ 5. Token2Text報酬モデルで報酬 R(Y) を計算                  │
    │ 6. KLダイバージェンス D_KL(pi_theta || pi_ref) を計算      │
    │ 7. 目的関数: max E[R(Y)] - β × D_KL を最適化              │
    └──────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        policy_model,           # pi_theta: 学習対象LLM
        reference_model,        # pi_ref: 凍結された参照LLM
        reward_model,           # Token2Text報酬モデル (ASR)
        beta: float = 0.1,     # KL正則化の強度
        gumbel_temperature: float = 0.5,  # Gumbel-Softmax温度
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.reward = reward_model
        self.beta = beta
        self.gumbel_temperature = gumbel_temperature

        # 参照モデルは凍結
        for param in self.reference.parameters():
            param.requires_grad = False
        for param in self.reward.parameters():
            param.requires_grad = False

    def compute_diffro_loss(
        self,
        text_token_ids: torch.Tensor,         # (B, L_text)
        speech_tokens: torch.Tensor,           # (B, T_speech) - 正解トークン
    ) -> Dict[str, torch.Tensor]:
        """
        DiffROの学習ロス計算

        入力:
            text_token_ids: (B, L_text) - 入力テキストトークン
            speech_tokens: (B, T_speech) - 正解音声トークン (教師データ)

        出力:
            loss: スカラー - DiffROロス
            reward: スカラー - ASR報酬値
            kl_div: スカラー - KLダイバージェンス

        ========================================
        数式
        ========================================

        目的関数 (Equation 5):
            pi*_theta = argmax_{pi_theta} E[R(Y)] - β × D_KL[pi_theta || pi_ref]

        ASR報酬 (Equation 4):
            R_ASR(Y) = log P_ASR(Y_n = Y_n | Y_{1:n-1}; μ_bar_{1:T})
            μ_bar_t = GumbelSoftmax P_{pi_theta}(μ_t | μ_{1:t-1}; Y)

        KLダイバージェンス (Equation 6):
            D_KL = Σ_{t=1}^{T} Σ_{k=0}^{Q}
                   P_{pi_theta}(μ_t=k) × log(P_{pi_theta}(μ_t=k) / P_{pi_ref}(μ_t=k))
        """
        B, T_speech_len = speech_tokens.shape

        # ========================================
        # Step 1: ポリシーモデルのlogits取得
        # ========================================
        policy_logits = self.policy.forward_logits(
            text_token_ids=text_token_ids,
            speech_tokens=speech_tokens,
        )
        # policy_logits: (B, T_speech, Q)
        #   Q = 6561 (音声トークン語彙サイズ)

        # ========================================
        # Step 2: 参照モデルのlogits取得 (勾配不要)
        # ========================================
        with torch.no_grad():
            reference_logits = self.reference.forward_logits(
                text_token_ids=text_token_ids,
                speech_tokens=speech_tokens,
            )
        # reference_logits: (B, T_speech, Q)

        # ========================================
        # Step 3: Gumbel-Softmaxサンプリング (Equation 3)
        # ========================================
        # μ_bar_t = GumbelSoftmax P_{pi_theta}(μ_t | μ_{1:t-1}; Y)
        # 微分可能な離散サンプリング
        soft_tokens = F.gumbel_softmax(
            policy_logits,
            tau=self.gumbel_temperature,
            hard=False,           # ソフトサンプリング (勾配伝播可能)
            dim=-1,
        )
        # soft_tokens: (B, T_speech, Q)
        #   各行はほぼone-hotだが微分可能な連続値
        #   Σ_k soft_tokens[t, k] = 1 (正規化)

        # ========================================
        # Step 4: ASR報酬計算 (Equation 4)
        # ========================================
        # Token2Textモデルで音声トークン → テキストの対数尤度を計算
        reward_value = self.reward.compute_reward(
            soft_tokens=soft_tokens,        # (B, T_speech, Q) - ソフトトークン
            target_text=text_token_ids,     # (B, L_text) - 正解テキスト
        )
        # reward_value: (B,) - 各サンプルの報酬
        #   R_ASR(Y) = log P_ASR(Y_n | Y_{1:n-1}; μ_bar)
        #   高い値 = 音声トークンが正しくテキストにデコードされる

        # ========================================
        # Step 5: KLダイバージェンス計算 (Equation 6)
        # ========================================
        # トークンレベルのKL (シーケンスレベルではない)
        policy_probs = F.softmax(policy_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        # policy_probs, reference_probs: (B, T_speech, Q)

        # D_KL = Σ_t Σ_k P_theta(k) × log(P_theta(k) / P_ref(k))
        kl_per_token = torch.sum(
            policy_probs * (
                torch.log(policy_probs + 1e-10)
                - torch.log(reference_probs + 1e-10)
            ),
            dim=-1,
        )
        # kl_per_token: (B, T_speech)

        kl_div = kl_per_token.sum(dim=-1)
        # kl_div: (B,) - 各サンプルのKL

        # ========================================
        # Step 6: DiffROロス (Equation 5)
        # ========================================
        # max E[R(Y)] - β × D_KL  →  min -R(Y) + β × D_KL
        loss = -reward_value.mean() + self.beta * kl_div.mean()

        return {
            'loss': loss,
            'reward': reward_value.mean(),
            'kl_div': kl_div.mean(),
        }


class Token2TextRewardModel(nn.Module):
    """
    Token2Text 報酬モデル (ASRライクな構造)

    音声トークン (Gumbel-Softmaxソフトトークン) を入力として
    対応するテキストの対数尤度を計算。

    学習: ASR学習データで事前学習
    推論 (DiffRO): ソフトトークンからテキスト確率を出力

    アーキテクチャ:
    ┌──────────────────────────────────────────┐
    │ ソフトトークン (B, T_speech, Q)            │
    │     ↓                                    │
    │ 埋め込み変換 (Q → D_model)                │
    │     ↓                                    │
    │ Transformer Encoder (因果的)              │
    │     ↓                                    │
    │ テキスト予測ヘッド (D_model → V_text)      │
    │     ↓                                    │
    │ log P(text | speech_tokens): スカラー     │
    └──────────────────────────────────────────┘
    """

    def __init__(
        self,
        speech_vocab_size: int = 6561,    # 音声トークン語彙
        text_vocab_size: int = 151936,    # テキスト語彙
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()

        # 音声トークン → 埋め込み
        self.speech_proj = nn.Linear(speech_vocab_size, d_model)
        # (B, T_speech, Q) → (B, T_speech, D_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        # (B, T_speech, D_model) → (B, T_speech, D_model)

        # テキスト予測ヘッド
        self.text_head = nn.Linear(d_model, text_vocab_size)
        # (B, T_speech, D_model) → (B, T_speech, V_text)

    def compute_reward(
        self,
        soft_tokens: torch.Tensor,     # (B, T_speech, Q) - Gumbel-Softmaxソフトトークン
        target_text: torch.Tensor,     # (B, L_text) - 正解テキスト
    ) -> torch.Tensor:
        """
        ASR報酬計算

        入力:
            soft_tokens: (B, T_speech, Q) - ソフト音声トークン
                Gumbel-Softmax出力 (ほぼone-hot, 微分可能)
            target_text: (B, L_text) - 正解テキストトークンID

        出力:
            reward: (B,) - 対数尤度報酬
                R_ASR = log P(text | soft_tokens)

        処理:
            1. ソフトトークン → 連続埋め込み (行列積で微分可能)
            2. Transformer Encoder で特徴抽出
            3. テキスト予測 → 正解テキストとの対数尤度
        """
        # ソフトトークンを埋め込みに変換 (微分可能)
        h = self.speech_proj(soft_tokens)
        # h: (B, T_speech, D_model)

        # Encoderで文脈付き特徴
        h = self.encoder(h)
        # h: (B, T_speech, D_model)

        # テキスト予測
        text_logits = self.text_head(h)
        # text_logits: (B, T_speech, V_text)

        # CTC的な対数尤度 or Attention-based対数尤度
        # (簡略化: フレーム平均の対数尤度)
        log_probs = F.log_softmax(text_logits, dim=-1)
        # log_probs: (B, T_speech, V_text)

        # 正解テキストに対する報酬 (対数尤度の和)
        # 実装では CTC loss の負値 or Attention decoder の尤度
        reward = self._compute_ctc_reward(log_probs, target_text)
        # reward: (B,)

        return reward

    def _compute_ctc_reward(
        self,
        log_probs: torch.Tensor,   # (B, T_speech, V_text)
        target_text: torch.Tensor, # (B, L_text)
    ) -> torch.Tensor:
        """CTC対数尤度に基づく報酬"""
        # CTC loss の負値 = 報酬 (高い方が良い)
        input_lengths = torch.full(
            (log_probs.shape[0],), log_probs.shape[1],
            device=log_probs.device
        )
        target_lengths = torch.sum(target_text != 0, dim=1)

        ctc_loss = F.ctc_loss(
            log_probs.transpose(0, 1),  # (T, B, V)
            target_text,
            input_lengths,
            target_lengths,
            reduction='none',
        )
        # ctc_loss: (B,)

        return -ctc_loss  # 負値にして報酬に変換


class MultiTaskReward(nn.Module):
    """
    マルチタスク報酬 (MTR) (Equation 7)

    ASR報酬に加えて、複数の下流タスクからの報酬を統合。
    音声の属性制御 (感情、速度等) の指示追従能力を向上。

    R_MTR(Y, {A_i}) = Σ_i log P_{task_i}(A_hat_i = A_i | μ_bar)

    サポートタスク:
    1. ASR (音声認識) - コンテンツの正確性
    2. SER (感情認識) - 感情の正確性
    3. MOS予測 - 音質スコア
    4. AED (音響イベント) - 笑い声、呼吸等の正確性
    """

    def __init__(
        self,
        asr_model,        # ASR報酬モデル
        ser_model=None,   # 感情認識モデル (オプション)
        mos_model=None,   # MOS予測モデル (オプション)
        aed_model=None,   # 音響イベント検出モデル (オプション)
    ):
        super().__init__()

        self.asr_model = asr_model
        self.ser_model = ser_model
        self.mos_model = mos_model
        self.aed_model = aed_model

    def compute_reward(
        self,
        soft_tokens: torch.Tensor,         # (B, T_speech, Q)
        target_text: torch.Tensor,         # (B, L_text)
        target_emotion: Optional[torch.Tensor] = None,  # (B,) 感情ラベル
        target_events: Optional[torch.Tensor] = None,   # (B, num_events)
    ) -> torch.Tensor:
        """
        マルチタスク報酬の計算

        入力:
            soft_tokens: (B, T_speech, Q) - Gumbel-Softmaxソフトトークン
            target_text: (B, L_text) - 正解テキスト
            target_emotion: (B,) - 正解感情ラベル (オプション)
            target_events: (B, num_events) - 正解音響イベント (オプション)

        出力:
            total_reward: (B,) - マルチタスク報酬の合計

        数式 (Equation 7):
            R_MTR = Σ_i log P_{task_i}(A_hat_i = A_i | μ_bar)
        """
        total_reward = torch.zeros(soft_tokens.shape[0],
                                   device=soft_tokens.device)

        # ASR報酬 (常に使用)
        asr_reward = self.asr_model.compute_reward(soft_tokens, target_text)
        total_reward += asr_reward
        # asr_reward: (B,)

        # 感情認識報酬 (DiffRO-EMO)
        if self.ser_model is not None and target_emotion is not None:
            ser_reward = self.ser_model.compute_emotion_reward(
                soft_tokens, target_emotion
            )
            total_reward += ser_reward
            # ser_reward: (B,)

        # MOS予測報酬
        if self.mos_model is not None:
            mos_reward = self.mos_model.compute_mos_reward(soft_tokens)
            total_reward += mos_reward
            # mos_reward: (B,)

        # 音響イベント報酬
        if self.aed_model is not None and target_events is not None:
            aed_reward = self.aed_model.compute_event_reward(
                soft_tokens, target_events
            )
            total_reward += aed_reward
            # aed_reward: (B,)

        return total_reward


"""
========================================
DiffROの学習パイプライン (全体像)
========================================

Step 1: 事前学習 (Stage 1)
    - 約100万時間のデータで LLM + CFM をプレトレーニング
    - 標準的な Next-Token Prediction (交差エントロピー)

Step 2: DiffRO後処理 (Stage 2) ← このファイルの内容
    - 選択されたデータでRL最適化
    - Token2Text ASR報酬モデルを別途学習
    - Gumbel-Softmax で離散トークンを微分可能に
    - 目的: max E[R_ASR] - β × D_KL(pi_theta || pi_ref)

    効果:
    - SEED test-zh CER: 1.27% → 0.75% (41% 改善)
    - SEED test-en WER: 2.46% → 1.76% (28% 改善)
    - 低リソース言語 (韓国語): 68.7% 相対改善
    - クロスリンガル: >50% 改善 (半数の条件で)

Step 3: 継続事前学習 (Stage 3)
    - 感情・指示追従・多言語データでファインチューン
    - Token2Text LLM を活用

Step 4: 話者ファインチューン (Stage 4)
    - 特定話者のデータでファインチューン
    - 教師なしクラスタリングで話者埋め込みセンターを取得
"""
