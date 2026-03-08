"""
DINOv3 Gram Anchoring - 簡略化疑似コード
==========================================

Gram Anchoring: DINOv3の核心的新手法
密特徴の劣化を防ぐための二次統計量正則化

論文: https://arxiv.org/abs/2508.10104 (Section 4)

問題:
  SSL を長時間学習すると、CLS token (グローバル) の精度は向上し続けるが、
  パッチ特徴 (密特徴) の品質は劣化する:
    - CLS と Patch の cosine similarity が増加
    - パッチ特徴の空間局所性が失われる
    - 類似度マップがノイズ化 (~600k iter)

解決:
  学習初期のモデルスナップショット (Gram Teacher) の Gram 行列構造を維持
  → 特徴自体は自由に移動可能、パッチ間の類似度「構造」のみ保存

数式:
  L_Gram = || X_S @ X_S^T - X_G @ X_G^T ||_F^2
  - X_S: (P, D) Student L2正規化パッチ特徴
  - X_G: (P, D) Gram Teacher L2正規化パッチ特徴
  - ||.||_F: Frobenius ノルム

公式実装参照: dinov3/loss/gram_loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# Gram Loss
# ============================================================
class GramLoss(nn.Module):
    """
    Gram Anchoring 損失

    二次統計量 (Gram 行列) のマッチングによる密特徴正則化

    Gram 行列 G = X @ X^T は全パッチペアの類似度を表す:
      G[i,j] = <x_i, x_j>  (L2正規化後は cosine similarity)

    Student の G_S と Gram Teacher の G_G の MSE を最小化することで、
    パッチ間の「類似度構造」を保存する。

    重要な洞察:
      - 特徴ベクトル自体をマッチングすると、特徴空間の発展を阻害
      - Gram 行列は回転・反転に不変 → 特徴は自由に移動可能
      - 「どのパッチが似ているか」の関係性のみを保存
    """

    def __init__(
        self,
        apply_norm: bool = True,
        img_level: bool = True,
        remove_neg: bool = True,
        remove_only_teacher_neg: bool = False,
    ):
        """
        Args:
            apply_norm: L2正規化を適用するか
            img_level: 画像単位で Gram 行列を計算するか
                True: (B, P, D) → 各画像で (P, P) の Gram 行列
                False: (B*P, D) → バッチ全体で (B*P, B*P) の Gram 行列
            remove_neg: 負の類似度をゼロに置換するか
            remove_only_teacher_neg: Teacher の負値のみ除去
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.apply_norm = apply_norm
        self.img_level = img_level
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

    def forward(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gram 行列間の MSE 損失

        Args:
            student_feats: (B, P, D) - Student パッチ特徴
                B=バッチサイズ, P=パッチ数 (例: 256), D=次元 (例: 4096)
            teacher_feats: (B, P, D) - Gram Teacher パッチ特徴

        Returns:
            loss: scalar - Gram 行列の MSE

        Shape flow:
            student_feats: (B, P, D=4096)
            → L2 normalize: (B, P, D)
            → Gram matrix: (B, P, P)  ※img_level=True
            → clamp(min=0): (B, P, P)
            → MSE with teacher Gram: scalar
        """
        # float32 に変換 (精度のため)
        student_feats = student_feats.float()
        teacher_feats = teacher_feats.float()

        # 1. L2 正規化 (オプション)
        if self.apply_norm:
            student_feats = F.normalize(student_feats, dim=-1)  # (B, P, D)
            teacher_feats = F.normalize(teacher_feats, dim=-1)  # (B, P, D)

        if self.img_level:
            # === 画像単位の Gram 行列計算 ===
            # (B, P, D) @ (B, D, P) → (B, P, P)
            student_gram = torch.bmm(
                student_feats, student_feats.transpose(1, 2)
            )  # (B, P, P) - 各画像のパッチ間類似度

            teacher_gram = torch.bmm(
                teacher_feats, teacher_feats.transpose(1, 2)
            )  # (B, P, P)

        else:
            # === バッチ全体の Gram 行列計算 ===
            B, P, D = student_feats.shape
            s_flat = student_feats.reshape(B * P, D)  # (B*P, D)
            t_flat = teacher_feats.reshape(B * P, D)  # (B*P, D)

            student_gram = torch.mm(s_flat, s_flat.T)  # (B*P, B*P)
            teacher_gram = torch.mm(t_flat, t_flat.T)  # (B*P, B*P)

        # 2. 負値の処理
        if self.remove_neg:
            # 負の類似度をゼロに (正の相関のみ保存)
            student_gram = torch.clamp(student_gram, min=0)
            teacher_gram = torch.clamp(teacher_gram, min=0)
        elif self.remove_only_teacher_neg:
            # Teacher の負値箇所のみゼロに
            neg_mask = teacher_gram < 0
            teacher_gram = torch.clamp(teacher_gram, min=0)
            student_gram[neg_mask] = 0

        # 3. MSE 損失 (Frobenius ノルム)
        loss = self.mse_loss(student_gram, teacher_gram)

        return loss


# ============================================================
# Gram Teacher Manager
# ============================================================
class GramTeacherManager:
    """
    Gram Teacher の管理

    Gram Teacher は学習初期 (~200k iter) のモデルスナップショット
    学習中に段階的に更新される:
      - 最初: ~200k iter のスナップショット
      - 10k step ごとに EMA Teacher に更新 (最大3回)

    高解像度 Gram (L_HRef):
      - Teacher に 2x 解像度で入力
      - 出力を bicubic で 1x にダウンサンプル
      - より滑らかな Gram 行列ターゲット (+2 mIoU)
    """

    def __init__(
        self,
        gram_teacher_backbone: nn.Module,
        update_interval: int = 10_000,    # 10k step ごとに更新
        max_updates: int = 3,              # 最大更新回数
        high_res_factor: float = 2.0,      # 高解像度倍率
    ):
        self.backbone = gram_teacher_backbone
        self.update_interval = update_interval
        self.max_updates = max_updates
        self.high_res_factor = high_res_factor
        self.update_count = 0

    @torch.no_grad()
    def get_gram_features(
        self,
        images: torch.Tensor,
        target_patch_grid: int = 16,
    ) -> torch.Tensor:
        """
        Gram Teacher から特徴を取得

        高解像度入力 → ダウンサンプルにより滑らかな特徴を生成

        Args:
            images: (B, 3, H, W) - 入力画像
            target_patch_grid: 目標パッチグリッドサイズ

        Returns:
            features: (B, target_P, D) - ダウンサンプルされたパッチ特徴
                target_P = target_patch_grid^2

        Shape flow:
            images: (B, 3, 256, 256)
            → resize 2x: (B, 3, 512, 512)
            → backbone: (B, 1024, D)  ※32x32 パッチ
            → reshape: (B, 32, 32, D)
            → bicubic downsample: (B, 16, 16, D)
            → flatten: (B, 256, D)
        """
        B = images.shape[0]

        if self.high_res_factor > 1.0:
            # 高解像度にリサイズ
            H, W = images.shape[2], images.shape[3]
            H_hr = int(H * self.high_res_factor)
            W_hr = int(W * self.high_res_factor)
            images_hr = F.interpolate(
                images, size=(H_hr, W_hr), mode="bicubic", align_corners=False
            )  # (B, 3, 512, 512)

            # Backbone forward
            out = self.backbone(images_hr)
            patches = out["x_norm_patchtokens"]  # (B, P_hr, D)
            # P_hr = (512/16)^2 = 1024

            # パッチグリッドにリシェイプ
            H_patch_hr = H_hr // 16  # 32
            W_patch_hr = W_hr // 16  # 32
            patches_2d = patches.view(B, H_patch_hr, W_patch_hr, -1)
            # (B, 32, 32, D)

            # チャネル次元を先にして bicubic ダウンサンプル
            patches_2d = patches_2d.permute(0, 3, 1, 2)  # (B, D, 32, 32)
            patches_down = F.interpolate(
                patches_2d.float(),
                size=(target_patch_grid, target_patch_grid),
                mode="bicubic",
                align_corners=False,
            )  # (B, D, 16, 16)

            # Flatten
            patches_down = patches_down.permute(0, 2, 3, 1)  # (B, 16, 16, D)
            features = patches_down.flatten(1, 2)              # (B, 256, D)

        else:
            # 通常解像度
            out = self.backbone(images)
            features = out["x_norm_patchtokens"]  # (B, P, D)

        return features

    def maybe_update(
        self,
        iteration: int,
        ema_teacher_backbone: nn.Module,
    ):
        """
        必要に応じて Gram Teacher を更新

        update_interval ステップごとに EMA Teacher の重みをコピー
        最大 max_updates 回まで

        Args:
            iteration: 現在のイテレーション
            ema_teacher_backbone: EMA Teacher のバックボーン
        """
        if self.update_count >= self.max_updates:
            return

        if iteration > 0 and iteration % self.update_interval == 0:
            # EMA Teacher の重みを Gram Teacher にコピー
            self.backbone.load_state_dict(
                ema_teacher_backbone.state_dict()
            )
            self.update_count += 1
            print(
                f"Gram Teacher updated at iter {iteration} "
                f"({self.update_count}/{self.max_updates})"
            )


# ============================================================
# Gram Anchoring 統合例
# ============================================================
def compute_gram_anchoring_loss(
    student_backbone_output: dict,
    gram_teacher_manager: GramTeacherManager,
    images: torch.Tensor,
    gram_loss_fn: GramLoss,
    n_global_crops: int = 2,
) -> torch.Tensor:
    """
    Gram Anchoring 損失の計算フロー

    Args:
        student_backbone_output: Student バックボーン出力
            patch_pre_head: (n_global, B, P, D)
        gram_teacher_manager: Gram Teacher マネージャー
        images: (n_global*B, 3, H, W) - 元画像
        gram_loss_fn: GramLoss インスタンス
        n_global_crops: グローバルクロップ数

    Returns:
        loss: scalar

    学習フロー全体:
        Phase 1 (0 ~ 1M iter): DINO + iBOT + KoLeo のみ
        Phase 2 (1M ~ iter): + Gram Anchoring (weight=2.0)

        L_Ref = w_D * L_DINO + L_iBOT + w_DK * L_DKoleo + w_Gram * L_Gram

        where w_Gram = 2.0

    Gram Teacher の更新スケジュール:
        初期: ~200k iter のスナップショット (密特徴が最良の時期)
        以降: 10k step ごとに EMA Teacher に更新 (最大3回)
    """
    B = images.shape[0] // n_global_crops
    P = student_backbone_output["x_norm_patchtokens"].shape[1]

    # 1. Student のパッチ特徴
    student_patches = student_backbone_output["x_norm_patchtokens"]
    # (n_global*B, P, D)

    # 2. Gram Teacher のパッチ特徴 (高解像度入力 + ダウンサンプル)
    with torch.no_grad():
        gram_teacher_patches = gram_teacher_manager.get_gram_features(
            images,
            target_patch_grid=int(P ** 0.5),  # sqrt(256) = 16
        )
        # (n_global*B, P, D)

    # 3. Gram 損失計算
    loss = gram_loss_fn(student_patches, gram_teacher_patches)

    return loss


# ============================================================
# 密特徴劣化の可視化ヘルパー
# ============================================================
def analyze_feature_degradation(
    cls_token: torch.Tensor,
    patch_tokens: torch.Tensor,
) -> dict:
    """
    密特徴の劣化指標を計算

    DINOv3論文で指摘された劣化現象:
      1. CLS-Patch cosine similarity の増加
      2. パッチ特徴の空間局所性の喪失
      3. 類似度マップのノイズ化

    Args:
        cls_token: (B, D) - CLS token
        patch_tokens: (B, P, D) - Patch tokens

    Returns:
        metrics: dict
            cls_patch_sim: float - CLS-Patch 平均 cosine similarity
            patch_patch_locality: float - 近傍パッチとの類似度
            diversity: float - パッチ特徴の多様性
    """
    B, P, D = patch_tokens.shape

    # 1. CLS-Patch cosine similarity
    cls_norm = F.normalize(cls_token, dim=-1)        # (B, D)
    patch_norm = F.normalize(patch_tokens, dim=-1)   # (B, P, D)
    cls_patch_sim = torch.einsum(
        "bd,bpd->bp", cls_norm, patch_norm
    ).mean().item()
    # 劣化時: この値が 1.0 に近づく (全パッチが CLS と似る)

    # 2. パッチ間類似度の空間構造
    # 隣接パッチ間の類似度 vs 遠方パッチ間の類似度
    gram = torch.bmm(patch_norm, patch_norm.transpose(1, 2))  # (B, P, P)
    H = W = int(P ** 0.5)

    # 簡易的な局所性指標
    mean_sim = gram.mean().item()
    # 劣化時: 全パッチの類似度が均一化 (局所性喪失)

    # 3. パッチ特徴の多様性 (分散)
    diversity = patch_norm.var(dim=1).mean().item()
    # 劣化時: 多様性低下

    return {
        "cls_patch_sim": cls_patch_sim,
        "mean_patch_sim": mean_sim,
        "diversity": diversity,
    }
