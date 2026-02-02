"""
BitDistill 蒸留損失関数

Logits蒸留 (KL divergence) と Multi-Head Attention蒸留 (MiniLM方式) の実装。

参考: BitNet Distillation (arXiv:2510.13998)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==============================================================================
# Logits蒸留損失
# ==============================================================================

class LogitsDistillationLoss(nn.Module):
    """
    Logits蒸留損失 (Eq. 8-9)

    Teacher (FP16) のソフトな出力分布をStudent (1.58-bit) に転写。

    数学的表現:
      L_LD = (1/N) Σᵢ D_KL(P^FP16_θ(yᵢ|xᵢ) || P^1.58-bit_θ(yᵢ|xᵢ))
      P_θ(y|x) = exp(z_y/τ) / Σ_{y'} exp(z_{y'}/τ)

    温度パラメータ τ の役割:
      τ=1.0: 通常のsoftmax (ハードな分布)
      τ>1.0: ソフトな分布 (クラス間の関係を保持)
      → τ=5.0 が論文での設定

    なぜLogits蒸留が必要か:
      - Hard label (one-hot) だけでは類似クラスの関係が失われる
      - Teacherの"暗黙知"（類似カテゴリの相対的スコア）を伝達
      - 例: "猫"の予測で"犬"のスコアが高い → 視覚的類似性の知識

    効果:
      蒸留なし: 86.73% → +Logits蒸留: 87.32% (MNLI, +0.59pt)
    """

    def __init__(self, temperature: float = 5.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            teacher_logits: (B, T, V) Teacher出力logits
            student_logits: (B, T, V) Student出力logits
            mask: (B, T) 有効トークンマスク (optional)

        Returns:
            loss: KL divergence損失 (スカラー)

        計算フロー:
            t_logits/τ → softmax → P_teacher  (ソフトラベル)
            s_logits/τ → log_softmax → log_P_student
            loss = KL(P_teacher || P_student) × τ²
        """
        tau = self.temperature

        # 温度スケーリング
        t_probs = F.softmax(teacher_logits / tau, dim=-1)      # (B, T, V)
        s_log_probs = F.log_softmax(student_logits / tau, dim=-1)  # (B, T, V)

        # KL divergence
        # D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)) = Σ P(x) (log P(x) - log Q(x))
        kl_div = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=-1)  # (B, T)

        if mask is not None:
            kl_div = kl_div * mask
            loss = kl_div.sum() / mask.sum().clamp(min=1)
        else:
            loss = kl_div.mean()

        # τ² でスケール補正
        # (勾配の大きさをτに依存しないようにするため)
        loss = loss * (tau ** 2)

        return loss


# ==============================================================================
# Multi-Head Attention蒸留損失 (MiniLM方式)
# ==============================================================================

class AttentionDistillationLoss(nn.Module):
    """
    Multi-Head Attention蒸留損失 (Eq. 10-12)

    MiniLM (NeurIPS 2020) をベースにした蒸留手法。
    Q, K, V それぞれの自己関係行列を Teacher→Student で蒸留。

    数学的表現:
      A ∈ {Q, K, V}  (各attention成分)
      R = Softmax(A · Aᵀ / √d_r)  (関係行列)
      L_AD = (1/|Υ|) Σᵢ Σ_{j∈{Q,K,V}} αᵢ × (1/(A_r·|x|)) Σₐ Σₜ D_KL(R^T || R^S)

    設計上の重要なポイント:

    1. 単一レイヤーのみ蒸留 (|Υ|=1):
       - 全レイヤー蒸留 < 単一レイヤー蒸留
       - 理由: Studentに最適化の自由度を与える
       - 後半レイヤーの方が効果的

    2. Q, K, V 全てを蒸留:
       - Attention重み (QKᵀ) だけでなく
       - V (値表現) の関係も蒸留
       → より豊富な構造的知識を伝達

    3. 関係行列 R = Softmax(A·Aᵀ/√d):
       - 各トークン間の相対的関係を捉える
       - Softmaxで正規化 → スケール不変

    効果:
      Logits蒸留のみ: 87.32% → +Attention蒸留: 88.17% (MNLI, +0.85pt)
    """

    def __init__(self, distill_layer_idx: int = -1, alpha: float = 1.0):
        """
        Args:
            distill_layer_idx: 蒸留するレイヤーのインデックス
                              -1 = 最終レイヤー (推奨)
            alpha: レイヤー重み αᵢ
        """
        super().__init__()
        self.distill_layer_idx = distill_layer_idx
        self.alpha = alpha

    def compute_relation_matrix(
        self, A: torch.Tensor
    ) -> torch.Tensor:
        """
        関係行列の計算 (Eq. 12)

        R = Softmax(A · Aᵀ / √d_r)

        Args:
            A: Q, K, or V テンソル (B, H, T, d)

        Returns:
            R: 関係行列 (B, H, T, T)

        各 (i,j) 要素は「トークンiとトークンjの
        Q/K/V表現がどれだけ類似しているか」を表す
        """
        d = A.shape[-1]

        # A · Aᵀ (内積 = 類似度)
        similarity = torch.matmul(A, A.transpose(-1, -2))  # (B, H, T, T)

        # スケーリング + Softmax
        R = F.softmax(similarity / (d ** 0.5), dim=-1)  # (B, H, T, T)

        return R

    def forward(
        self,
        teacher_qkv: dict,
        student_qkv: dict,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            teacher_qkv: {
                layer_idx: {
                    'Q': (B, H, T, d),
                    'K': (B, H, T, d),
                    'V': (B, H, T, d),
                }
            }
            student_qkv: 同上
            mask: (B, T) 有効トークンマスク

        Returns:
            loss: Attention蒸留損失 (スカラー)

        計算フロー:
            各 A ∈ {Q, K, V}:
              R_teacher = Softmax(A_t · A_tᵀ / √d)
              R_student = Softmax(A_s · A_sᵀ / √d)
              loss += KL(R_teacher || R_student)
            loss /= 3  (Q, K, V の平均)
        """
        layer_idx = self.distill_layer_idx
        loss = torch.tensor(0.0, device=next(iter(
            teacher_qkv[layer_idx].values()
        )).device)

        for key in ['Q', 'K', 'V']:
            # Teacher/Student の Q/K/V
            t_A = teacher_qkv[layer_idx][key]  # (B, H, T, d)
            s_A = student_qkv[layer_idx][key]  # (B, H, T, d)

            # 関係行列
            R_teacher = self.compute_relation_matrix(t_A)  # (B, H, T, T)
            R_student = self.compute_relation_matrix(s_A)  # (B, H, T, T)

            # KL divergence
            kl = F.kl_div(
                R_student.log().clamp(min=-100),
                R_teacher,
                reduction='none',
            )  # (B, H, T, T)

            if mask is not None:
                # マスク適用 (パディングトークンを除外)
                mask_2d = mask.unsqueeze(1).unsqueeze(-1) * mask.unsqueeze(1).unsqueeze(-2)
                kl = kl * mask_2d.unsqueeze(1)

            loss += self.alpha * kl.mean()

        return loss / 3.0  # Q, K, V の平均


# ==============================================================================
# 統合損失
# ==============================================================================

class BitDistillLoss(nn.Module):
    """
    BitDistill 統合損失 (Eq. 13-14)

    L = L_CE + λ × L_LD + γ × L_AD

    L_CE: Cross-Entropy損失
      -(1/N) Σᵢ Σₜ log P_θ(yᵢᵗ | xᵢ)
      → タスクの正解ラベルに基づく損失

    L_LD: Logits蒸留損失
      (1/N) Σᵢ D_KL(P^FP16(τ) || P^1.58-bit(τ))
      → Teacherの出力分布を模倣

    L_AD: Attention蒸留損失
      MiniLM方式の関係行列蒸留
      → Teacherの内部表現構造を模倣

    ハイパーパラメータ設定:
      | タスク | τ | λ | γ |
      |--------|---|---|---|
      | 分類   | 5.0 | 10 | 1e5 |
      | 要約   | 5.0 | 1  | 1e3 |
    """

    def __init__(
        self,
        temperature: float = 5.0,
        lambda_ld: float = 10.0,
        gamma_ad: float = 1e5,
        distill_layer_idx: int = -1,
    ):
        super().__init__()
        self.logits_loss = LogitsDistillationLoss(temperature)
        self.attention_loss = AttentionDistillationLoss(distill_layer_idx)
        self.lambda_ld = lambda_ld
        self.gamma_ad = gamma_ad

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_qkv: Optional[dict] = None,
        teacher_qkv: Optional[dict] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            student_logits: (B, T, V) Student出力
            teacher_logits: (B, T, V) Teacher出力
            labels: (B, T) 正解ラベル
            student_qkv: Student Q/K/V (optional)
            teacher_qkv: Teacher Q/K/V (optional)
            mask: (B, T) 有効トークンマスク

        Returns:
            dict: {
                'total': 総合損失,
                'ce': Cross-Entropy損失,
                'ld': Logits蒸留損失,
                'ad': Attention蒸留損失,
            }
        """
        # 1. Cross-Entropy損失 (Eq. 14)
        loss_ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 2. Logits蒸留損失 (Eq. 8)
        loss_ld = self.logits_loss(teacher_logits, student_logits, mask)

        # 3. Attention蒸留損失 (Eq. 11)
        loss_ad = torch.tensor(0.0, device=student_logits.device)
        if student_qkv is not None and teacher_qkv is not None:
            loss_ad = self.attention_loss(teacher_qkv, student_qkv, mask)

        # 4. 総合損失 (Eq. 13)
        total = loss_ce + self.lambda_ld * loss_ld + self.gamma_ad * loss_ad

        return {
            'total': total,
            'ce': loss_ce,
            'ld': loss_ld,
            'ad': loss_ad,
        }


# ==============================================================================
# 使用例
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("BitDistill 蒸留損失デモ")
    print("=" * 60)

    B, T, V = 2, 8, 100  # バッチ2, シーケンス8, 語彙100
    H, d = 4, 32          # ヘッド4, ヘッド次元32

    # ダミーデータ
    teacher_logits = torch.randn(B, T, V)
    student_logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))

    # --- Logits蒸留 ---
    print("\n--- Logits蒸留損失 ---")
    ld_loss = LogitsDistillationLoss(temperature=5.0)
    loss = ld_loss(teacher_logits, student_logits)
    print(f"L_LD (τ=5.0): {loss.item():.4f}")

    ld_loss_t1 = LogitsDistillationLoss(temperature=1.0)
    loss_t1 = ld_loss_t1(teacher_logits, student_logits)
    print(f"L_LD (τ=1.0): {loss_t1.item():.4f}")
    print("→ 温度が高いほどソフトな分布で蒸留")

    # --- Attention蒸留 ---
    print("\n--- Attention蒸留損失 ---")
    teacher_qkv = {
        -1: {
            'Q': torch.randn(B, H, T, d),
            'K': torch.randn(B, H, T, d),
            'V': torch.randn(B, H, T, d),
        }
    }
    student_qkv = {
        -1: {
            'Q': torch.randn(B, H, T, d),
            'K': torch.randn(B, H, T, d),
            'V': torch.randn(B, H, T, d),
        }
    }

    ad_loss = AttentionDistillationLoss(distill_layer_idx=-1)
    loss = ad_loss(teacher_qkv, student_qkv)
    print(f"L_AD: {loss.item():.4f}")

    # --- 統合損失 ---
    print("\n--- BitDistill統合損失 ---")
    criterion = BitDistillLoss(
        temperature=5.0,
        lambda_ld=10.0,
        gamma_ad=1e5,
    )

    losses = criterion(
        student_logits, teacher_logits, labels,
        student_qkv, teacher_qkv,
    )

    print(f"L_total: {losses['total'].item():.4f}")
    print(f"  L_CE:  {losses['ce'].item():.4f}")
    print(f"  λ×L_LD: {10.0 * losses['ld'].item():.4f} (λ=10)")
    print(f"  γ×L_AD: {1e5 * losses['ad'].item():.4f} (γ=1e5)")
