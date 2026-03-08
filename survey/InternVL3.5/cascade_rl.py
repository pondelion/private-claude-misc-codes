"""
InternVL3.5 Cascade Reinforcement Learning
============================================

このファイルは InternVL3.5 の核心的学習革新である
「Cascade RL (Cascade Reinforcement Learning)」を実装します。

Cascade RL の概要:
  2段階の RL アルゴリズムを順番に適用することで、
  安定性・効率性・性能上限の3つを同時に改善する。

  Stage 1: Offline RL (MPO - Mixed Preference Optimization)
    - 既存ロールアウトを使用 → ロールアウト収集コスト不要
    - DPO + BCO + LM損失の組み合わせ
    - 報酬ハッキングを防止 (収集と更新を分離)
    - MPO 後のモデルが高品質なロールアウトを生成 → GSPO の初期化に最適

  Stage 2: Online RL (GSPO - Group Sampling Policy Optimization)
    - MPO 後の強化モデルからリアルタイムサンプリング
    - GRPOと類似: グループ内の正規化アドバンテージを使用
    - 参照モデル制約なし → Dense・MoE モデルで安定した学習
    - 幾何平均の重要サンプリング比でトークンレベルの偏りを防止

公式実装参考:
  internvl_chat/internvl/train/internvl_chat_mpo.py (MPO部分)
  GSPO は独自実装 (公式公開なし、論文記述から実装)

============================================================
テンソル形状記法
============================================================
  B  : バッチサイズ
  N  : 系列長 (テキスト + IMG_CONTEXT)
  G  : 1クエリあたりのサンプリング回答数 (GSPO)
  V  : 語彙サイズ
  D  : LLM hidden size
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. データ構造
# ============================================================

@dataclass
class PreferenceData:
    """
    MPO (Offline RL) の学習データ。
    選好ペア (chosen / rejected) と品質ラベルから構成。

    フィールド:
      chosen_input_ids   : (B, N_c)  選ばれた回答のトークン列
      rejected_input_ids : (B, N_r)  棄却された回答のトークン列
      pixel_values       : (B*P, 3, 448, 448) 対応する画像
      quality_labels     : (B,) float  品質スコア (1.0=高品質, 0.0=低品質)
      chosen_labels      : (B, N_c)  損失計算対象トークン (-100=マスク)
      rejected_labels    : (B, N_r)  損失計算対象トークン (-100=マスク)
    """
    chosen_input_ids: torch.Tensor
    rejected_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    quality_labels: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_labels: torch.Tensor


@dataclass
class OnlineRollout:
    """
    GSPO (Online RL) のロールアウトデータ。
    1クエリに対して G 個の回答をサンプリングした結果。

    フィールド:
      query_input_ids   : (B, N_q)    クエリ部のトークン列
      response_input_ids: (B, G, N_r) 各クエリへの G 個の回答
      pixel_values      : (B*P, 3, 448, 448) 画像
      rewards           : (B, G)      各回答への報酬スコア
      old_log_probs     : (B, G, N_r) 古いポリシーの log prob
    """
    query_input_ids: torch.Tensor
    response_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    rewards: torch.Tensor
    old_log_probs: torch.Tensor


# ============================================================
# 2. MPO 損失 (Offline RL Stage)
# ============================================================

class MPOLoss(nn.Module):
    """
    Mixed Preference Optimization (MPO) 損失。

    3つの損失の加重和:
      L_MPO = wp * L_DPO + wq * L_BCO + wg * L_LM

    L_DPO (Direct Preference Optimization):
      選好ペアに基づく選好損失。
      chosen の log prob を高め、rejected の log prob を下げる。

    L_BCO (Binary Classification Optimization):
      品質ラベルに基づく回答品質分類損失。
      高品質回答の log prob を高める。

    L_LM (Language Model Loss):
      通常の次トークン予測損失。
      生成能力の維持に使用。

    参考: MMPR-v1.2 (~200K ペア) を使用

    入力形状:
      policy_chosen_logps   : (B,)  ポリシーの chosen log prob
      policy_rejected_logps : (B,)  ポリシーの rejected log prob
      ref_chosen_logps      : (B,)  参照モデルの chosen log prob
      ref_rejected_logps    : (B,)  参照モデルの rejected log prob
      quality_labels        : (B,)  品質スコア [0, 1]
      lm_loss               : スカラー  標準 NTP 損失
    """
    def __init__(
        self,
        beta: float = 0.1,          # DPO の温度パラメータ
        wp: float = 1.0,            # 選好損失の重み
        wq: float = 0.5,            # 品質損失の重み
        wg: float = 0.1,            # 生成損失の重み
    ):
        super().__init__()
        self.beta = beta
        self.wp = wp
        self.wq = wq
        self.wg = wg

    def compute_sequence_log_prob(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        系列全体の平均 log probability を計算。

        入力:
          logits : (B, N, V)  未正規化ロジット
          labels : (B, N)    ターゲットトークン (-100=無視)
        出力:
          log_probs : (B,)   各系列の平均 log prob
        """
        # (B, N, V) → (B, N-1, V) / (B, N-1) でシフト
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # log softmax → (B, N-1, V)
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # ターゲットの log prob を選択: (B, N-1)
        selected = torch.gather(
            log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        # マスク (-100 の位置を0に)
        mask = (shift_labels != -100).float()
        # 平均 log prob: (B,)
        seq_log_probs = (selected * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)

        return seq_log_probs

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPO (Direct Preference Optimization) 損失。

        参照モデルからの相対的な log prob 比を最大化:
          L_DPO = -E[log σ(β * (logπ(y_c|x) - logπ_ref(y_c|x))
                               - β * (logπ(y_r|x) - logπ_ref(y_r|x)))]

        入力:
          policy_chosen_logps   : (B,)  ポリシー π の chosen 系列 log prob
          policy_rejected_logps : (B,)  ポリシー π の rejected 系列 log prob
          ref_chosen_logps      : (B,)  参照モデル π_ref の chosen 系列 log prob
          ref_rejected_logps    : (B,)  参照モデル π_ref の rejected 系列 log prob
        出力:
          dpo_loss : スカラー
        """
        # 参照モデルからの log 比の差
        # (B,)
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)

        # DPO 損失 = -log σ(chosen_rewards - rejected_rewards)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
        return loss.mean()

    def bco_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        quality_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        BCO (Binary Classification Optimization) 損失。
        品質ラベルに基づき高品質回答の生成を促進。

        入力:
          policy_chosen_logps : (B,)  chosen 系列の log prob
          quality_labels      : (B,)  [0, 1] の品質スコア
        出力:
          bco_loss : スカラー
        """
        # 高品質サンプルのみで損失計算
        high_quality = (quality_labels > 0.5).float()
        # 高品質サンプルの負の log prob を最小化
        loss = -policy_chosen_logps * high_quality
        # 有効サンプルが存在する場合のみ計算
        n_valid = high_quality.sum().clamp(min=1)
        return loss.sum() / n_valid

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        quality_labels: torch.Tensor,
        lm_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        MPO の総合損失を計算。

        入力:
          policy_chosen_logps   : (B,)  現在のポリシーの chosen log prob
          policy_rejected_logps : (B,)  現在のポリシーの rejected log prob
          ref_chosen_logps      : (B,)  参照モデルの chosen log prob
          ref_rejected_logps    : (B,)  参照モデルの rejected log prob
          quality_labels        : (B,)  品質スコア [0, 1]
          lm_loss               : スカラー  NTP 損失
        出力:
          {'total': スカラー, 'dpo': スカラー, 'bco': スカラー, 'lm': スカラー}
        """
        # 各損失コンポーネントを計算
        l_dpo = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )
        l_bco = self.bco_loss(policy_chosen_logps, quality_labels)
        l_lm = lm_loss

        # 加重和
        total = self.wp * l_dpo + self.wq * l_bco + self.wg * l_lm

        return {
            'total': total,
            'dpo': l_dpo,
            'bco': l_bco,
            'lm': l_lm,
        }


# ============================================================
# 3. GSPO 損失 (Online RL Stage)
# ============================================================

class GSPOLoss(nn.Module):
    """
    Group Sampling Policy Optimization (GSPO) 損失。

    GRPO との主な違い:
      1. 参照モデル制約 (KL penalty) なし → Dense/MoE で安定
      2. 重要サンプリング比に幾何平均 (per-token exp 平均) を使用
         → GRPOの s_i = π_θ/π_old ではなく
            s_i = exp(1/|y| * Σ_t log(π_θ(t)/π_ref(t)))

    処理フロー:
      1. グループ内の報酬を正規化してアドバンテージを計算
      2. 新旧ポリシーの幾何平均比を計算
      3. PPO スタイルのクリッピング付き目的関数で損失計算

    入力形状:
      logits         : (B*G, N_r, V)  現在のポリシーのロジット
      old_log_probs  : (B*G, N_r)     古いポリシーの log prob
      response_labels: (B*G, N_r)     ターゲットトークン (-100=無視)
      rewards        : (B, G)         各回答への報酬
      epsilon        : クリッピング範囲
    """
    def __init__(self, epsilon: float = 0.2):
        super().__init__()
        self.epsilon = epsilon

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        グループ内の報酬を標準化してアドバンテージを計算。

        入力: rewards  (B, G)   各クエリの G 個の回答への報酬
        出力: advantages (B, G)  標準化されたアドバンテージ

        計算式:
          Ā_i = (r_i - mean(r)) / (std(r) + ε)
          ※ mean/std はクエリ内 (G 個) で計算
        """
        # (B, G) → 各クエリ内で平均・標準偏差を計算
        r_mean = rewards.mean(dim=-1, keepdim=True)   # (B, 1)
        r_std = rewards.std(dim=-1, keepdim=True)     # (B, 1)
        advantages = (rewards - r_mean) / (r_std + 1e-8)  # (B, G)
        return advantages

    def compute_geometric_mean_ratio(
        self,
        logits: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        幾何平均重要サンプリング比を計算。

        通常の比 (GRPO):
          s_i = π_θ(y_i|x) / π_old(y_i|x)
            = exp(Σ_t log π_θ(t) - Σ_t log π_old(t))
          これはシーケンスが長いと非常に大きくなる可能性がある。

        幾何平均比 (GSPO):
          s_i = exp(1/|y_i| * Σ_t log(π_θ(t)/π_old(t)))
          トークン数で正規化することで長さへの依存を排除。

        入力:
          logits         : (B*G, N_r, V)  現在のポリシーのロジット
          old_log_probs  : (B*G, N_r)     古いポリシーの log prob
          response_labels: (B*G, N_r)     ターゲットトークン (-100=無視)
        出力:
          ratio          : (B*G,)         幾何平均重要サンプリング比
        """
        # 現在のポリシーの log prob を計算
        # (B*G, N_r, V) → (B*G, N_r)
        log_probs = F.log_softmax(logits, dim=-1)
        current_log_probs = torch.gather(
            log_probs, dim=-1,
            index=response_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)  # (B*G, N_r)

        # マスク作成 (-100 の位置を無視)
        mask = (response_labels != -100).float()  # (B*G, N_r)

        # per-token log 比: (B*G, N_r)
        log_ratio = (current_log_probs - old_log_probs) * mask

        # 幾何平均: Σ_t log_ratio / |y| → (B*G,)
        seq_len = mask.sum(dim=-1).clamp(min=1)
        geo_mean_log_ratio = log_ratio.sum(dim=-1) / seq_len

        # exp で比を計算: (B*G,)
        ratio = torch.exp(geo_mean_log_ratio)
        return ratio

    def forward(
        self,
        logits: torch.Tensor,
        old_log_probs: torch.Tensor,
        response_labels: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        GSPO 損失を計算。

        入力:
          logits         : (B*G, N_r, V)  現在のポリシーのロジット
          old_log_probs  : (B*G, N_r)     古いポリシーの log prob
          response_labels: (B*G, N_r)     ターゲットトークン (-100=無視)
          rewards        : (B, G)         報酬スコア
        出力:
          {'total': スカラー, 'policy_loss': スカラー, 'clip_fraction': スカラー}
        """
        B, G = rewards.shape

        # ステップ1: グループアドバンテージを計算
        # (B, G)
        advantages = self.compute_group_advantages(rewards)
        # (B*G,) に展開
        advantages_flat = advantages.reshape(B * G)

        # ステップ2: 幾何平均重要サンプリング比を計算
        # (B*G,)
        ratio = self.compute_geometric_mean_ratio(logits, old_log_probs, response_labels)

        # ステップ3: PPO クリッピング付き目的関数
        # clip(ratio, 1-ε, 1+ε) * advantage
        # (B*G,)
        surr1 = ratio * advantages_flat
        surr2 = ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_flat

        # 最小値 (pessimistic bound): (B*G,)
        policy_loss = -torch.min(surr1, surr2)

        # クリッピング割合 (デバッグ用)
        clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float().mean()

        total_loss = policy_loss.mean()

        return {
            'total': total_loss,
            'policy_loss': total_loss,
            'clip_fraction': clip_fraction,
        }


# ============================================================
# 4. Cascade RL トレーナー
# ============================================================

class CascadeRLTrainer:
    """
    Cascade RL の2段階学習を管理するトレーナー。

    Stage 1: MPO (Offline RL)
      - MMPR-v1.2 (~200K ペア) を使用
      - 参照モデル (ref_model) を固定
      - ロールアウトの再利用が可能

    Stage 2: GSPO (Online RL)
      - MPO 後のモデルを初期化として使用
      - MMPR-Tiny (~70K クエリ) を使用
      - 各ステップでモデル自身からサンプリング
    """
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        mpo_config: dict,
        gspo_config: dict,
        optimizer_cls=torch.optim.AdamW,
        lr: float = 1e-5,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model

        # MPO 損失
        self.mpo_loss = MPOLoss(
            beta=mpo_config.get('beta', 0.1),
            wp=mpo_config.get('wp', 1.0),
            wq=mpo_config.get('wq', 0.5),
            wg=mpo_config.get('wg', 0.1),
        )

        # GSPO 損失
        self.gspo_loss = GSPOLoss(
            epsilon=gspo_config.get('epsilon', 0.2),
        )

        self.optimizer = optimizer_cls(policy_model.parameters(), lr=lr)

        # 参照モデルは学習しない
        for param in ref_model.parameters():
            param.requires_grad = False

    # --------------------
    # Stage 1: MPO ステップ
    # --------------------
    def mpo_step(
        self,
        batch: PreferenceData,
    ) -> Dict[str, float]:
        """
        MPO (Offline RL) の1ステップ。

        引数:
          batch: PreferenceData
            - chosen_input_ids   : (B, N_c)
            - rejected_input_ids : (B, N_r)
            - pixel_values       : (B*P, 3, 448, 448)
            - quality_labels     : (B,)
            - chosen_labels      : (B, N_c)
            - rejected_labels    : (B, N_r)
        返値:
          損失値の辞書 {'total': float, 'dpo': float, 'bco': float, 'lm': float}
        """
        self.optimizer.zero_grad()
        self.policy_model.train()
        self.ref_model.eval()

        # ポリシーモデルのフォワード (chosen)
        # logits: (B, N_c, V)
        policy_chosen_output = self.policy_model(
            pixel_values=batch.pixel_values,
            input_ids=batch.chosen_input_ids,
            attention_mask=(batch.chosen_input_ids != 0).long(),
            image_flags=torch.ones(batch.pixel_values.shape[0], 1, dtype=torch.long),
            labels=batch.chosen_labels,
        )
        # ポリシーモデルのフォワード (rejected)
        policy_rejected_output = self.policy_model(
            pixel_values=batch.pixel_values,
            input_ids=batch.rejected_input_ids,
            attention_mask=(batch.rejected_input_ids != 0).long(),
            image_flags=torch.ones(batch.pixel_values.shape[0], 1, dtype=torch.long),
        )

        # 参照モデルのフォワード (勾配不要)
        with torch.no_grad():
            ref_chosen_output = self.ref_model(
                pixel_values=batch.pixel_values,
                input_ids=batch.chosen_input_ids,
                attention_mask=(batch.chosen_input_ids != 0).long(),
                image_flags=torch.ones(batch.pixel_values.shape[0], 1, dtype=torch.long),
            )
            ref_rejected_output = self.ref_model(
                pixel_values=batch.pixel_values,
                input_ids=batch.rejected_input_ids,
                attention_mask=(batch.rejected_input_ids != 0).long(),
                image_flags=torch.ones(batch.pixel_values.shape[0], 1, dtype=torch.long),
            )

        # 各系列の log prob を計算
        mpo_fn = self.mpo_loss.compute_sequence_log_prob
        policy_chosen_logps = mpo_fn(policy_chosen_output.logits, batch.chosen_labels)
        policy_rejected_logps = mpo_fn(policy_rejected_output.logits, batch.rejected_labels)
        ref_chosen_logps = mpo_fn(ref_chosen_output.logits, batch.chosen_labels)
        ref_rejected_logps = mpo_fn(ref_rejected_output.logits, batch.rejected_labels)

        # MPO 損失を計算
        losses = self.mpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            quality_labels=batch.quality_labels,
            lm_loss=policy_chosen_output.loss,
        )

        # バックプロパゲーション
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    # --------------------
    # Stage 2: GSPO ステップ
    # --------------------
    def gspo_step(
        self,
        rollout: OnlineRollout,
    ) -> Dict[str, float]:
        """
        GSPO (Online RL) の1ステップ。

        引数:
          rollout: OnlineRollout
            - query_input_ids   : (B, N_q)    クエリトークン列
            - response_input_ids: (B, G, N_r)  G 個の回答トークン列
            - pixel_values      : (B*P, 3, 448, 448)
            - rewards           : (B, G)       各回答への報酬
            - old_log_probs     : (B, G, N_r)  古いポリシーの log prob
        返値:
          損失値の辞書 {'total': float, 'policy_loss': float, 'clip_fraction': float}
        """
        self.optimizer.zero_grad()
        self.policy_model.train()

        B, G, N_r = rollout.response_input_ids.shape

        # (B, G, N_r) → (B*G, N_r) に展開
        responses_flat = rollout.response_input_ids.reshape(B * G, N_r)
        old_log_probs_flat = rollout.old_log_probs.reshape(B * G, N_r)

        # 現在のポリシーで回答のロジットを計算
        # pixel_values を各 G 個複製 (シンプル実装)
        # 実際は query + response を結合してフォワード
        n_patches = rollout.pixel_values.shape[0] // B
        pv_expanded = rollout.pixel_values.unsqueeze(1).expand(
            B, G, n_patches, 3, 448, 448
        ).reshape(B * G * n_patches, 3, 448, 448)

        image_flags = torch.ones(B * G * n_patches, 1, dtype=torch.long,
                                 device=rollout.pixel_values.device)

        # フォワードパス: (B*G, N_r, V)
        # 注意: 実際の実装ではクエリ+回答を結合した full sequence でフォワード
        outputs = self.policy_model(
            pixel_values=pv_expanded,
            input_ids=responses_flat,
            attention_mask=(responses_flat != 0).long(),
            image_flags=image_flags,
        )
        # (B*G, N_r, V)
        logits = outputs.logits

        # 回答トークンのラベル (マスクなし: 全トークンに対して計算)
        response_labels = responses_flat.clone()
        # パディング位置をマスク
        response_labels[response_labels == 0] = -100

        # GSPO 損失を計算
        losses = self.gspo_loss(
            logits=logits,
            old_log_probs=old_log_probs_flat,
            response_labels=response_labels,
            rewards=rollout.rewards,
        )

        # バックプロパゲーション
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}


# ============================================================
# 5. Square Averaging (学習損失の再重み付け)
# ============================================================

def compute_square_averaging_weights(sequence_lengths: List[int], power: float = 0.6) -> torch.Tensor:
    """
    Square Averaging: 長い系列への偏りを防ぐ損失重み付け。

    論文 Eq. (2):
      w_i = 1 / N^0.6
      L'_t = (Σ_j w_j / Σ_j w_j) * L_t  ← 正規化済み

    直感:
      長い系列ほどトークン数が多く NTP 損失が小さくなりがち。
      1/N^0.6 で重み付けることで短い/長い系列間のバランスを取る。
      power=0.6 は 0 (均等) と 1 (1/N 正規化) の中間。

    入力:
      sequence_lengths : List[int]  各サンプルのシーケンス長 N
      power            : float      正規化指数 (デフォルト 0.6)
    出力:
      weights          : (len(sequence_lengths),)  正規化前の重み
    """
    weights = torch.tensor(
        [1.0 / (n ** power) for n in sequence_lengths],
        dtype=torch.float32,
    )
    return weights


# ============================================================
# 使用例
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Cascade RL (MPO + GSPO) 動作確認")
    print("=" * 60)

    torch.manual_seed(42)
    B = 2   # バッチサイズ
    G = 4   # GSPO グループサイズ (1クエリあたりのサンプル数)
    N = 64  # 系列長
    V = 1000  # 語彙サイズ (テスト用に小さく)

    # --- 1. MPOLoss テスト ---
    print("\n[1] MPOLoss テスト")
    mpo_loss_fn = MPOLoss(beta=0.1, wp=1.0, wq=0.5, wg=0.1)

    # ダミーデータ
    policy_chosen_logps = torch.randn(B) * 0.5 - 2.0    # (B,) 負値
    policy_rejected_logps = torch.randn(B) * 0.5 - 2.5  # (B,) 負値
    ref_chosen_logps = torch.randn(B) * 0.5 - 2.0       # (B,) 負値
    ref_rejected_logps = torch.randn(B) * 0.5 - 2.5     # (B,) 負値
    quality_labels = torch.tensor([1.0, 0.0])            # (B,)
    lm_loss = torch.tensor(2.5)                          # スカラー

    losses = mpo_loss_fn(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        ref_chosen_logps=ref_chosen_logps,
        ref_rejected_logps=ref_rejected_logps,
        quality_labels=quality_labels,
        lm_loss=lm_loss,
    )
    print(f"  L_DPO:  {losses['dpo'].item():.4f}")
    print(f"  L_BCO:  {losses['bco'].item():.4f}")
    print(f"  L_LM:   {losses['lm'].item():.4f}")
    print(f"  L_total:{losses['total'].item():.4f}")

    # --- 2. DPO ロジック検証 ---
    print("\n[2] DPO 損失の方向性確認")
    # chosen > rejected の場合、損失は小さくなるべき
    logps_chosen_better = torch.tensor([-1.0, -1.2])    # 選ばれた方が高 log prob
    logps_rejected_worse = torch.tensor([-2.0, -2.2])   # 棄却の方が低 log prob
    loss_good = mpo_loss_fn.dpo_loss(
        logps_chosen_better, logps_rejected_worse,
        torch.zeros(B), torch.zeros(B)  # ref は 0 にして比較
    )

    logps_chosen_worse = torch.tensor([-2.0, -2.2])    # 逆の場合
    logps_rejected_better = torch.tensor([-1.0, -1.2])
    loss_bad = mpo_loss_fn.dpo_loss(
        logps_chosen_worse, logps_rejected_better,
        torch.zeros(B), torch.zeros(B)
    )
    print(f"  chosen > rejected の場合の損失: {loss_good.item():.4f}")
    print(f"  chosen < rejected の場合の損失: {loss_bad.item():.4f}")
    assert loss_good < loss_bad, "DPO 損失の方向性が誤り"
    print("  OK: chosen が高品質の場合に損失が小さい")

    # --- 3. GSPOLoss テスト ---
    print("\n[3] GSPOLoss テスト")
    gspo_loss_fn = GSPOLoss(epsilon=0.2)

    # ダミーデータ
    logits = torch.randn(B * G, N, V)            # (B*G, N, V)
    old_log_probs = torch.randn(B * G, N) - 3.0  # (B*G, N)
    response_labels = torch.randint(1, V, (B * G, N))  # (B*G, N)
    response_labels[:, N // 2:] = -100            # 後半をマスク
    rewards = torch.randn(B, G)                  # (B, G)

    gspo_losses = gspo_loss_fn(
        logits=logits,
        old_log_probs=old_log_probs,
        response_labels=response_labels,
        rewards=rewards,
    )
    print(f"  GSPO total_loss:   {gspo_losses['total'].item():.4f}")
    print(f"  Clip fraction:     {gspo_losses['clip_fraction'].item():.4f}")

    # --- 4. グループアドバンテージの確認 ---
    print("\n[4] グループアドバンテージ計算確認")
    rewards_test = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # (1, G=4)
    advantages = gspo_loss_fn.compute_group_advantages(rewards_test)
    print(f"  報酬: {rewards_test.squeeze().tolist()}")
    print(f"  アドバンテージ: {[f'{a:.3f}' for a in advantages.squeeze().tolist()]}")
    assert abs(advantages.mean().item()) < 1e-5, "アドバンテージの平均が 0 でない"
    print("  OK: グループ内の平均が 0 に正規化")

    # --- 5. Square Averaging テスト ---
    print("\n[5] Square Averaging テスト")
    seq_lens = [10, 50, 100, 500, 1000]
    weights = compute_square_averaging_weights(seq_lens, power=0.6)
    print(f"  系列長: {seq_lens}")
    print(f"  重み (1/N^0.6): {[f'{w:.4f}' for w in weights.tolist()]}")
    # 長い系列ほど重みが小さいことを確認
    for i in range(len(weights) - 1):
        assert weights[i] > weights[i + 1], "長い系列ほど重みが小さいべき"
    print("  OK: 長い系列の損失が相対的に小さく重み付け")

    print("\n全テスト完了!")
