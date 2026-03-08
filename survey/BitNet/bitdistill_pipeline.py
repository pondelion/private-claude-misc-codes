"""
BitDistill 3段階蒸留パイプライン

BitNet Distillation (arXiv:2510.13998) の全体フローを簡略化して実装。
既存のFP16 LLMを1.58-bit BitNetに変換する3段階パイプライン。

Stage 1: SubLN挿入 (活性化分散の安定化)
Stage 2: 継続事前学習 (重み分布の変換, 10B tokens)
Stage 3: 蒸留ベースファインチューニング (Teacher→Student)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import copy

from bitnet_quantization import BitLinear, SubLayerNorm


# ==============================================================================
# Stage 1: SubLN挿入によるモデリング改良
# ==============================================================================

def insert_subln(model: nn.Module, hidden_dim: int) -> nn.Module:
    """
    Stage 1: 既存Transformerモデルに SubLN を挿入

    変更箇所:
      1. MHSA出力投影前: SubLN(Concat(heads)) × W_out
      2. FFN出力投影前:  SubLN(gate ⊙ up) × W_down

    効果:
      - 1.58-bit量子化での活性化分散爆発を防止
      - 学習の安定化

    SubLNなし: 74.09% → SubLNあり: 76.30% (MNLI, +2.2pt)

    Args:
        model: 元のTransformerモデル
        hidden_dim: 隠れ層次元

    Returns:
        model: SubLN挿入済みモデル
    """
    for layer in model.layers:
        # MHSA出力投影前にSubLN追加
        if not hasattr(layer, 'sub_ln_attn'):
            layer.sub_ln_attn = SubLayerNorm(hidden_dim)

        # FFN出力投影前にSubLN追加
        ffn_dim = layer.up_proj.out_features if hasattr(layer, 'up_proj') else hidden_dim * 4
        if not hasattr(layer, 'sub_ln_ffn'):
            layer.sub_ln_ffn = SubLayerNorm(ffn_dim)

    print(f"[Stage 1] SubLN挿入完了: {len(model.layers)} layers")
    return model


# ==============================================================================
# Stage 1b: Linear → BitLinear 変換
# ==============================================================================

def convert_linear_to_bitlinear(model: nn.Module) -> nn.Module:
    """
    モデル内の全nn.LinearをBitLinearに置換

    対象: Q, K, V, O 投影 + Gate, Up, Down 投影
    除外: Embedding層, LM Head

    変換手順:
      1. nn.Linear の重みを取得
      2. 同サイズの BitLinear を作成
      3. 重みをコピー (FP16マスター重みとして)
      4. 元のLinearをBitLinearに置換

    注意: Embedding層とLM Headは高精度が必要なため除外
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Embedding層とLM Headは除外
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                continue

            # BitLinearに変換
            bit_linear = BitLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None
            )
            # FP16重みをコピー
            bit_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                bit_linear.bias.data = module.bias.data.clone()

            setattr(model, name, bit_linear)
        else:
            # 再帰的に変換
            convert_linear_to_bitlinear(module)

    return model


# ==============================================================================
# Stage 2: 継続事前学習
# ==============================================================================

def continual_pretraining(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_steps: int = 10000,
    device: str = "cuda",
    log_interval: int = 100,
):
    """
    Stage 2: 継続事前学習で重み分布を変換

    目的:
      FP16の重み分布 (ガウス分布) → BitNet向き分布に変換
      → 遷移境界 (0↔±1) 付近に重みを集中

    スケーラビリティ問題の解決:
      直接QATでは、モデルサイズ↑ → 精度ギャップ↑
        0.6B: 13.9pt差
        1.7B: 14.3pt差
        4B:   15.3pt差

      継続事前学習で重み分布を変換することで解消:
        → 小さな勾配でも量子化値がシフト可能に

    コスト:
      10B tokens (FALCON corpus)
      ≈ フルスクラッチ学習 (4T tokens) の 0.25%
      → "virtually negligible"

    効果:
      SubLNのみ: 76.30% → +CT: 86.73% (MNLI, +10.4pt)

    損失関数 (Eq. 7):
      L_CT = -(1/N) Σᵢ Σₜ log P_θ(c_{i,t} | c_{i,<t})

    Args:
        model: SubLN + BitLinear 変換済みモデル
        dataloader: 事前学習コーパス (例: FALCON)
        optimizer: オプティマイザ (AdamW推奨)
        num_steps: 学習ステップ数
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    step = 0

    for batch in dataloader:
        if step >= num_steps:
            break

        input_ids = batch['input_ids'].to(device)        # (B, T)
        labels = batch['labels'].to(device)               # (B, T)

        # 順伝播 (BitLinear → STE → 三値量子化)
        logits = model(input_ids).logits                  # (B, T, V)

        # 標準言語モデリング損失 (Eq. 7)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 逆伝播 + 更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            print(f"[Stage 2] Step {step}/{num_steps}, Loss: {avg_loss:.4f}")
            total_loss = 0.0

    print(f"[Stage 2] 継続事前学習完了: {step} steps")
    return model


# ==============================================================================
# Stage 3: 蒸留ベースファインチューニング
# ==============================================================================

class BitDistillTrainer:
    """
    Stage 3: Teacher (FP16) → Student (1.58-bit) の蒸留ファインチューニング

    3つの損失を組み合わせ:
      L = L_CE + λ × L_LD + γ × L_AD   (Eq. 13)

    L_CE: Cross-Entropy損失 (タスク損失)
    L_LD: Logits蒸留 (KL divergence)
    L_AD: Multi-Head Attention蒸留 (MiniLM方式)

    ハイパーパラメータ:
      τ = 5.0 (温度)
      λ = 10 (分類), 1 (要約)
      γ = 1e5 (分類), 1e3 (要約)
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        temperature: float = 5.0,
        lambda_ld: float = 10.0,
        gamma_ad: float = 1e5,
        distill_layer_idx: int = -1,
        device: str = "cuda",
    ):
        """
        Args:
            teacher: FP16 fine-tuned モデル (frozen)
            student: 1.58-bit BitNet モデル (学習対象)
            temperature: Logits蒸留の温度 τ
            lambda_ld: Logits蒸留の重み λ
            gamma_ad: Attention蒸留の重み γ
            distill_layer_idx: Attention蒸留を行うレイヤー
                              (デフォルト: -1 = 最終レイヤー)
        """
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.optimizer = optimizer
        self.temperature = temperature
        self.lambda_ld = lambda_ld
        self.gamma_ad = gamma_ad
        self.distill_layer_idx = distill_layer_idx
        self.device = device

        # Teacherのパラメータを凍結
        for param in self.teacher.parameters():
            param.requires_grad = False

    def compute_logits_distillation_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Logits蒸留損失 (Eq. 8-9)

        L_LD = (1/N) Σᵢ D_KL(P^FP16(τ) || P^1.58-bit(τ))

        P(y|x) = exp(z_y/τ) / Σ_{y'} exp(z_{y'}/τ)

        温度τを上げることで:
          - ソフトな確率分布を生成
          - Teacherの"暗黙知"を伝達
          - 類似クラス間の関係性を保持

        Args:
            teacher_logits: (B, T, V) Teacher出力
            student_logits: (B, T, V) Student出力

        Returns:
            loss: KL divergence損失
        """
        tau = self.temperature

        # 温度スケーリング後のソフトmax
        t_probs = F.softmax(teacher_logits / tau, dim=-1)
        s_log_probs = F.log_softmax(student_logits / tau, dim=-1)

        # KL divergence (tau^2 でスケール補正)
        loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (tau ** 2)

        return loss

    def compute_attention_distillation_loss(
        self,
        teacher_attentions: dict,
        student_attentions: dict,
    ) -> torch.Tensor:
        """
        Multi-Head Attention蒸留損失 (Eq. 10-12)

        MiniLM方式: Q, K, V それぞれの関係行列を蒸留

        R = Softmax(A · Aᵀ / √d)   where A ∈ {Q, K, V}
        L_AD = Σ_{A∈{Q,K,V}} KL(R_teacher || R_student)

        設計上の工夫:
          - 単一レイヤーのみ蒸留 (全レイヤーより効果的)
          - 後半レイヤーの方が性能が良い
          → Studentの最適化自由度を確保

        効果:
          Logits蒸留のみ: 87.32% → +Attention蒸留: 88.17% (MNLI, +0.85pt)

        Args:
            teacher_attentions: {layer_idx: {'Q': tensor, 'K': tensor, 'V': tensor}}
            student_attentions: 同上

        Returns:
            loss: Attention蒸留損失
        """
        layer_idx = self.distill_layer_idx
        loss = torch.tensor(0.0, device=self.device)

        for key in ['Q', 'K', 'V']:
            # Teacher/Student の Q/K/V を取得
            t_A = teacher_attentions[layer_idx][key]   # (B, H, T, d)
            s_A = student_attentions[layer_idx][key]   # (B, H, T, d)

            d = t_A.shape[-1]

            # 関係行列の計算 (Eq. 12)
            # R = Softmax(A · Aᵀ / √d)
            R_teacher = F.softmax(
                torch.matmul(t_A, t_A.transpose(-1, -2)) / (d ** 0.5),
                dim=-1
            )  # (B, H, T, T)

            R_student = F.softmax(
                torch.matmul(s_A, s_A.transpose(-1, -2)) / (d ** 0.5),
                dim=-1
            )  # (B, H, T, T)

            # KL divergence
            loss += F.kl_div(
                R_student.log(),
                R_teacher,
                reduction='batchmean'
            )

        return loss / 3.0  # Q, K, V の平均

    def train_step(self, batch: dict) -> dict:
        """
        1回の学習ステップ

        総合損失 (Eq. 13):
          L = L_CE + λ × L_LD + γ × L_AD

        Args:
            batch: {'input_ids': (B,T), 'attention_mask': (B,T), 'labels': (B,T)}

        Returns:
            losses: 各損失値の辞書
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        labels = batch['labels'].to(self.device)

        # --- Teacher forward (frozen) ---
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            t_logits = teacher_outputs.logits     # (B, T, V)
            t_attentions = teacher_outputs.attentions  # tuple of (B, H, T, T)

        # --- Student forward ---
        student_outputs = self.student(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        s_logits = student_outputs.logits         # (B, T, V)
        s_attentions = student_outputs.attentions

        # --- 損失計算 ---

        # 1. Cross-Entropy損失 (Eq. 14)
        loss_ce = F.cross_entropy(
            s_logits.view(-1, s_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # 2. Logits蒸留損失 (Eq. 8)
        loss_ld = self.compute_logits_distillation_loss(t_logits, s_logits)

        # 3. Attention蒸留損失 (Eq. 11)
        # 注: 簡略化のためattention weightsを使用
        # 実際にはQ, K, Vの関係行列を使用
        loss_ad = torch.tensor(0.0, device=self.device)
        if t_attentions is not None and s_attentions is not None:
            t_attn = t_attentions[self.distill_layer_idx]  # (B, H, T, T)
            s_attn = s_attentions[self.distill_layer_idx]
            loss_ad = F.kl_div(
                s_attn.log().clamp(min=-100),
                t_attn,
                reduction='batchmean'
            )

        # 4. 総合損失 (Eq. 13)
        total_loss = loss_ce + self.lambda_ld * loss_ld + self.gamma_ad * loss_ad

        # --- 逆伝播 + 更新 ---
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'ce': loss_ce.item(),
            'ld': loss_ld.item(),
            'ad': loss_ad.item(),
        }

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 3,
        log_interval: int = 50,
    ):
        """
        蒸留ファインチューニングのメインループ

        Args:
            dataloader: タスクデータ
            num_epochs: エポック数
            log_interval: ログ出力間隔
        """
        self.student.train()

        for epoch in range(num_epochs):
            total_losses = {'total': 0, 'ce': 0, 'ld': 0, 'ad': 0}
            step = 0

            for batch in dataloader:
                losses = self.train_step(batch)

                for k in total_losses:
                    total_losses[k] += losses[k]
                step += 1

                if step % log_interval == 0:
                    avg = {k: v / log_interval for k, v in total_losses.items()}
                    print(
                        f"[Stage 3] Epoch {epoch+1}, Step {step}, "
                        f"Loss: {avg['total']:.4f} "
                        f"(CE: {avg['ce']:.4f}, LD: {avg['ld']:.4f}, AD: {avg['ad']:.6f})"
                    )
                    total_losses = {k: 0 for k in total_losses}

            print(f"[Stage 3] Epoch {epoch+1} 完了")

        print("[Stage 3] 蒸留ファインチューニング完了")


# ==============================================================================
# 全体パイプライン
# ==============================================================================

class BitDistillPipeline:
    """
    BitDistill 完全パイプライン

    3段階の処理を統合:
      Stage 1: SubLN挿入 + Linear → BitLinear変換
      Stage 2: 継続事前学習 (重み分布変換)
      Stage 3: 蒸留ファインチューニング

    使い方:
      pipeline = BitDistillPipeline(teacher_model, hidden_dim=1024)
      student = pipeline.run(pretrain_loader, finetune_loader)
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        hidden_dim: int,
        temperature: float = 5.0,
        lambda_ld: float = 10.0,
        gamma_ad: float = 1e5,
        device: str = "cuda",
    ):
        self.teacher = teacher_model
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.lambda_ld = lambda_ld
        self.gamma_ad = gamma_ad
        self.device = device

    def run(
        self,
        pretrain_dataloader: DataLoader,
        finetune_dataloader: DataLoader,
        pretrain_steps: int = 10000,
        finetune_epochs: int = 3,
        pretrain_lr: float = 1e-4,
        finetune_lr: float = 2e-5,
    ) -> nn.Module:
        """
        全3段階を実行

        処理フロー:
          FP16 Teacher
               ↓ (コピー)
          Student (FP16)
               ↓ Stage 1
          Student + SubLN + BitLinear
               ↓ Stage 2 (10B tokens)
          Student (重み分布変換済み)
               ↓ Stage 3 (Teacher→Student蒸留)
          Student (1.58-bit, タスク特化)

        Returns:
            student: 蒸留済み1.58-bitモデル
        """
        # --- Stage 1: モデリング改良 ---
        print("=" * 60)
        print("Stage 1: SubLN挿入 + BitLinear変換")
        print("=" * 60)

        # TeacherをコピーしてStudentの初期化
        student = copy.deepcopy(self.teacher)

        # SubLN挿入
        student = insert_subln(student, self.hidden_dim)

        # Linear → BitLinear変換
        student = convert_linear_to_bitlinear(student)

        # --- Stage 2: 継続事前学習 ---
        print("\n" + "=" * 60)
        print("Stage 2: 継続事前学習 (重み分布変換)")
        print("=" * 60)

        optimizer_ct = torch.optim.AdamW(
            student.parameters(), lr=pretrain_lr, weight_decay=0.01
        )
        student = continual_pretraining(
            student, pretrain_dataloader, optimizer_ct,
            num_steps=pretrain_steps, device=self.device,
        )

        # --- Stage 3: 蒸留ファインチューニング ---
        print("\n" + "=" * 60)
        print("Stage 3: 蒸留ベースファインチューニング")
        print("=" * 60)

        # Teacherをタスクデータでファインチューニング（事前に済んでいる前提）
        optimizer_ft = torch.optim.AdamW(
            student.parameters(), lr=finetune_lr, weight_decay=0.01
        )

        trainer = BitDistillTrainer(
            teacher=self.teacher,
            student=student,
            optimizer=optimizer_ft,
            temperature=self.temperature,
            lambda_ld=self.lambda_ld,
            gamma_ad=self.gamma_ad,
            device=self.device,
        )
        trainer.train(finetune_dataloader, num_epochs=finetune_epochs)

        print("\n" + "=" * 60)
        print("BitDistill パイプライン完了!")
        print("=" * 60)

        return student


# ==============================================================================
# 使用例
# ==============================================================================

if __name__ == "__main__":
    print("BitDistill パイプライン概要")
    print("=" * 60)
    print("""
    Stage 1: SubLN挿入 + BitLinear変換
      - MHSA/FFN出力投影前にLayerNorm追加
      - 全Linear層をBitLinear (1.58-bit) に変換
      - 効果: 活性化分散の安定化

    Stage 2: 継続事前学習
      - 10B tokensで言語モデリング
      - 重み分布: ガウス分布 → BitNet向き分布
      - 遷移境界付近に重みを集中
      - コスト: フルスクラッチの0.25%

    Stage 3: 蒸留ファインチューニング
      - L = L_CE + λ×L_LD + γ×L_AD
      - Logits蒸留 (KL div, τ=5.0)
      - Attention蒸留 (MiniLM方式, 単一レイヤー)
      - FP16 Teacherの知識を1.58-bit Studentに転写

    結果:
      FP16:     88.01% (MNLI)
      直接QAT:  74.09% (-13.9pt)
      BitDistill: 88.17% (+0.16pt vs FP16!)
    """)
