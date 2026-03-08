"""
DINOv3 Loss Computation - 簡略化疑似コード
=============================================

DINOv3の全損失関数の詳細実装

損失構成:
  Phase 1 (事前学習, 0~1M iter):
    L = L_DINO + L_iBOT + 0.1 * L_DKoleo

  Phase 2 (Gram Anchoring 精緻化):
    L = w_D * L_DINO + L_iBOT + w_DK * L_DKoleo + 2.0 * L_Gram

各損失:
  1. L_DINO: CLS token の自己蒸留 (画像レベル)
  2. L_iBOT: マスクパッチの再構成 (パッチレベル)
  3. L_DKoleo: 分散 Kozachenko-Leonenko 均一性正則化
  4. L_Gram: Gram 行列の二次統計量マッチング

公式実装参照:
  - dinov3/loss/dino_clstoken_loss.py
  - dinov3/loss/ibot_patch_loss.py
  - dinov3/loss/koleo_loss.py
  - dinov3/loss/gram_loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ============================================================
# 1. DINO Loss (画像レベル自己蒸留)
# ============================================================
class DINOLoss(nn.Module):
    """
    DINO CLS Token 自己蒸留損失

    Teacher と Student の CLS token 出力間の交差エントロピー

    概要:
      - Student: softmax(cls_logits / student_temp) で確率分布を計算
      - Teacher: Sinkhorn-Knopp 正規化で均一な確率分布を生成
      - 損失: -sum(teacher_probs * log(student_probs))

    クロップペアの組み合わせ:
      - Global-Global: (A→B, B→A) = 2ペア (対角無視)
      - Global-Local: (A→L1..L8, B→L1..L8) = 16ペア
      - 合計: 18ペア

    DINOv1 からの改善:
      - センタリング → Sinkhorn-Knopp に変更 (より安定)
      - 専用 LayerNorm の分離
    """

    def __init__(
        self,
        out_dim: int,               # プロトタイプ数 K (256K)
        student_temp: float = 0.1,   # Student softmax 温度
        center_momentum: float = 0.9,  # センター EMA momentum
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # センター (Sinkhorn-Knopp のバイアス補正用)
        self.register_buffer("center", torch.zeros(1, out_dim))
        # center: (1, K=256K) - プロトタイプの平均活性

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        ignore_diagonal: bool = False,
    ) -> torch.Tensor:
        """
        DINO 交差エントロピー損失

        Args:
            student_logits: (n_student_crops, B, K) - Student のプロトタイプロジット
                K = 256,000 (プロトタイプ数)
            teacher_probs: (n_teacher_crops, B, K) - Teacher の確率分布
                Sinkhorn-Knopp 正規化済み (各行が sum=1)
            ignore_diagonal: True → 同一クロップペア (A→A, B→B) を無視

        Returns:
            loss: scalar

        Shape flow:
            student_logits: (n_s, B, K=256K)
            → log_softmax(/ student_temp): (n_s, B, K)
            → einsum with teacher_probs: (n_s, n_t)
            → average: scalar
        """
        n_s, B, K = student_logits.shape
        n_t = teacher_probs.shape[0]

        # Student の log softmax
        student_log_probs = F.log_softmax(
            student_logits.float() / self.student_temp, dim=-1
        )  # (n_s, B, K)

        # 全クロップペアの交差エントロピー
        # einsum: (n_s, B, K) × (n_t, B, K) → (n_s, n_t)
        loss_matrix = -torch.einsum(
            "sbk,tbk->st", student_log_probs, teacher_probs.float()
        )  # (n_s, n_t) - 各ペアの損失合計 (B で合計済み)

        if ignore_diagonal:
            # Global-Global の場合: A→A, B→B を除外
            min_st = min(n_s, n_t)
            loss_matrix[:min_st, :min_st].fill_diagonal_(0)
            total_pairs = B * (n_s * n_t - min_st)
        else:
            total_pairs = B * n_s * n_t

        return loss_matrix.sum() / total_pairs

    def sinkhorn_knopp_teacher(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """
        Sinkhorn-Knopp 正規化

        Teacher のロジットを均一な確率分布に変換
        → 各プロトタイプが均等に使用される (mode collapse 防止)

        Args:
            teacher_output: (B, K) - Teacher のプロトタイプロジット
                K = 256,000
            teacher_temp: float - 温度 (ウォームアップ: 0.04 → 0.07)
            n_iterations: 反復回数 (デフォルト: 3)

        Returns:
            Q: (B, K) - 正規化された確率分布 (各行 sum=1)

        Shape flow:
            teacher_output: (B, K)
            → exp(/ teacher_temp): (B, K)
            → transpose → Q: (K, B)
            → row/col normalize × n_iter
            → transpose → (B, K)
        """
        B, K = teacher_output.shape

        # 1. 温度付き softmax (exp のみ)
        Q = torch.exp(teacher_output.float() / teacher_temp).T  # (K, B)

        # 2. 全体正規化
        Q /= Q.sum()

        # 3. Sinkhorn-Knopp 反復
        for _ in range(n_iterations):
            # 行正規化: 各プロトタイプの使用量を 1/K に
            Q /= Q.sum(dim=1, keepdim=True) * K    # (K, B) - 各行 sum = 1/K

            # 列正規化: 各サンプルの割り当て量を 1/B に
            Q /= Q.sum(dim=0, keepdim=True) * B    # (K, B) - 各列 sum = 1/B

        return Q.T  # (B, K)

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        センターの EMA 更新

        Args:
            teacher_output: (n_crops*B, K) - Teacher 出力
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)  # (1, K)
        # 分散学習の場合は all_reduce

        self.center = (
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )


# ============================================================
# 2. iBOT Loss (パッチレベル再構成)
# ============================================================
class iBOTPatchLoss(nn.Module):
    """
    iBOT マスクパッチ再構成損失

    マスクされたパッチの Student 予測と Teacher ターゲットの交差エントロピー

    マスキング戦略:
      - ランダムブロックマスク: 矩形領域をマスク
      - マスク率: ランダム [0.1, 0.5] (10~50%)
      - 適用確率: 50% (半分のイテレーションではマスクなし)

    BEiT/MAE との違い:
      - BEiT: 離散トークン (VQ-VAE) を予測
      - MAE: ピクセルを再構成
      - iBOT: Teacher のパッチ特徴分布を予測 (自己蒸留)
    """

    def __init__(
        self,
        patch_out_dim: int,           # iBOT プロトタイプ数 (96K)
        student_temp: float = 0.1,    # Student 温度
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum

        # iBOT センター
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        # center: (1, 1, K_ibot=96K)

    def forward_masked(
        self,
        student_patch_tokens_masked: torch.Tensor,
        teacher_patch_tokens_masked: torch.Tensor,
        masks_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        マスクパッチの損失計算

        Args:
            student_patch_tokens_masked: (n_masked, K) - Student のマスクパッチ予測
                n_masked = マスクされたパッチ数 (可変)
                K = 96,000 (iBOT プロトタイプ数)
            teacher_patch_tokens_masked: (n_masked, K) - Teacher のマスクパッチ確率
                Sinkhorn-Knopp 正規化済み
            masks_weight: (n_masked,) optional - パッチ重み

        Returns:
            loss: scalar

        Shape flow:
            student: (n_masked, K=96K)
            → log_softmax(/ 0.1): (n_masked, K)
            → element-wise × teacher: (n_masked, K)
            → sum(dim=-1): (n_masked,)
            → weighted mean: scalar
        """
        # Student の log softmax
        student_log_probs = F.log_softmax(
            student_patch_tokens_masked.float() / self.student_temp, dim=-1
        )  # (n_masked, K)

        # パッチごとの交差エントロピー
        loss_per_patch = -torch.sum(
            teacher_patch_tokens_masked.float() * student_log_probs, dim=-1
        )  # (n_masked,)

        if masks_weight is not None:
            # 重み付き平均 (各画像のマスク数で正規化)
            loss = (loss_per_patch * masks_weight).sum()
        else:
            loss = loss_per_patch.mean()

        return loss

    def sinkhorn_knopp_teacher(
        self,
        teacher_output: torch.Tensor,
        teacher_temp: float,
        n_iterations: int = 3,
    ) -> torch.Tensor:
        """
        iBOT 用 Sinkhorn-Knopp

        DINO と同様だが、マスクパッチ数が可変

        Args:
            teacher_output: (n_masked, K) - Teacher ロジット
            teacher_temp: float - 温度

        Returns:
            Q: (n_masked, K) - 正規化確率
        """
        N, K = teacher_output.shape

        Q = torch.exp(teacher_output.float() / teacher_temp).T  # (K, N)
        Q /= Q.sum()

        for _ in range(n_iterations):
            Q /= Q.sum(dim=1, keepdim=True) * K
            Q /= Q.sum(dim=0, keepdim=True) * N

        return Q.T  # (N, K)


# ============================================================
# 3. KoLeo Loss (均一性正則化)
# ============================================================
class KoLeoLoss(nn.Module):
    """
    KoLeo (Kozachenko-Leonenko) 均一性正則化損失

    特徴空間での CLS token の均一分布を促進

    原理:
      - 各サンプルの最近傍との距離を計算
      - 距離が小さい → 特徴が密集 → 損失大
      - 距離が大きい → 特徴が分散 → 損失小
      - 結果として mode collapse を防止

    数式:
      L_KoLeo = -mean(log(distance_to_nearest_neighbor))
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-8)

    def forward(
        self,
        student_output: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        KoLeo 損失

        Args:
            student_output: (B, D) - CLS token 特徴
                B = バッチサイズ (小バッチ: 16)
                D = 埋め込み次元 (4096)

        Returns:
            loss: scalar

        Shape flow:
            student_output: (B=16, D=4096)
            → L2 normalize: (B, D)
            → pairwise dot product: (B, B)
            → argmax (exclude self): (B,) - 最近傍インデックス
            → L2 distance to NN: (B,)
            → -log(distance).mean(): scalar
        """
        B, D = student_output.shape

        # 1. L2 正規化
        x = F.normalize(student_output, dim=-1, p=2, eps=eps)  # (B, D)

        # 2. 全ペアのドット積 (cosine similarity)
        dots = torch.mm(x, x.T)  # (B, B)
        dots.fill_diagonal_(-1)   # 自分自身を除外 (最小値に)

        # 3. 最近傍のインデックス
        nn_indices = dots.argmax(dim=1)  # (B,) - 各サンプルの最近傍

        # 4. 最近傍との L2 距離
        nn_features = x[nn_indices]  # (B, D) - 最近傍の特徴
        distances = self.pdist(x, nn_features)  # (B,) - L2 距離

        # 5. 負対数距離の平均
        loss = -torch.log(distances + eps).mean()  # scalar
        # 距離が近い → log 値小 → 負で大きい → 損失大 → 分散を促進

        return loss


class KoLeoLossDistributed(nn.Module):
    """
    分散 KoLeo 損失 (DKoLeo)

    複数 GPU にまたがる全サンプルで最近傍を探索
    → より正確な均一性推定

    DINOv3 での使用:
      - 各 GPU の first global crop CLS (16サンプル) を使用
      - all_gather で全 GPU の特徴を収集
      - Top-K 最近傍 (K=1) で距離を計算
    """

    def __init__(
        self,
        topk: int = 1,
        loss_group_size: Optional[int] = None,
    ):
        super().__init__()
        self.pdist = nn.PairwiseDistance(p=2, eps=1e-8)
        self.topk = topk
        self.loss_group_size = loss_group_size

    def forward(
        self,
        student_output: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        分散 KoLeo 損失

        Args:
            student_output: (B_local, D) - ローカルバッチの CLS token

        Returns:
            loss: scalar

        Shape flow (概念的):
            student_output: (B_local=16, D=4096)
            → L2 normalize: (B_local, D)
            → all_gather: (B_global, D)  ※全 GPU から収集
            → dot product (local × global): (B_local, B_global)
            → topk=1: (B_local,) - 最近傍インデックス
            → L2 distance: (B_local,)
            → -log(distance).mean(): scalar
        """
        # 1. L2 正規化
        x = F.normalize(student_output.float(), dim=-1, p=2, eps=eps)
        # (B_local, D)

        # 2. 全 GPU から特徴を収集 (概念的)
        # 実際は torch.distributed.all_gather を使用
        all_features = x  # 単一 GPU の場合はそのまま
        # 分散時: (B_global, D) where B_global = B_local * world_size

        # 3. ドット積で類似度計算
        dots = torch.mm(x, all_features.T)  # (B_local, B_global)

        # 自分自身を除外
        dots.fill_diagonal_(-1)

        # 4. Top-K 最近傍
        _, nn_indices = dots.topk(self.topk, dim=1)  # (B_local, topk)

        # 5. 距離計算
        x_expanded = x.unsqueeze(1).expand(-1, self.topk, -1)  # (B_local, topk, D)
        nn_feats = all_features[nn_indices.flatten()]  # (B_local*topk, D)
        nn_feats = nn_feats.view(-1, self.topk, x.shape[-1])  # (B_local, topk, D)

        x_flat = x_expanded.reshape(-1, x.shape[-1])         # (B_local*topk, D)
        nn_flat = nn_feats.reshape(-1, x.shape[-1])           # (B_local*topk, D)
        distances = self.pdist(x_flat, nn_flat)                # (B_local*topk,)

        # 6. 負対数距離
        loss = -torch.log(distances.float() + eps).mean()

        return loss


# ============================================================
# 4. Gram Loss (二次統計量マッチング)
# ============================================================
class GramLoss(nn.Module):
    """
    Gram Anchoring 損失

    詳細は gram_anchoring.py を参照

    L_Gram = MSE(G_student, G_teacher)
    where G = X_norm @ X_norm^T  (Gram matrix of L2-normalized features)

    入力: Student/Teacher のパッチ特徴 (B, P, D)
    出力: scalar 損失
    """

    def __init__(
        self,
        apply_norm: bool = True,
        remove_neg: bool = True,
    ):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg

    def forward(
        self,
        student_feats: torch.Tensor,
        teacher_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_feats: (B, P, D) - Student パッチ特徴
            teacher_feats: (B, P, D) - Gram Teacher パッチ特徴

        Returns:
            loss: scalar
        """
        s = student_feats.float()
        t = teacher_feats.float()

        if self.apply_norm:
            s = F.normalize(s, dim=-1)  # (B, P, D)
            t = F.normalize(t, dim=-1)  # (B, P, D)

        # Gram 行列
        G_s = torch.bmm(s, s.transpose(1, 2))  # (B, P, P)
        G_t = torch.bmm(t, t.transpose(1, 2))  # (B, P, P)

        if self.remove_neg:
            G_s = torch.clamp(G_s, min=0)
            G_t = torch.clamp(G_t, min=0)

        return self.mse_loss(G_s, G_t)


# ============================================================
# 5. マスク生成
# ============================================================
class MaskingGenerator:
    """
    iBOT 用ランダムブロックマスク生成

    ランダムな矩形ブロックでパッチをマスク
    BEiT スタイルのブロックマスキング

    設定 (DINOv3):
      - マスク率: uniform [0.1, 0.5] (10~50%)
      - 適用確率: 50%
      - ブロックアスペクト比: [0.3, 1/0.3]
      - 最小ブロックサイズ: 4 パッチ
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
        mask_probability: float = 0.5,
        min_num_patches: int = 4,
        min_aspect: float = 0.3,
    ):
        """
        Args:
            input_size: (H_patch, W_patch) - パッチグリッドサイズ (例: (16, 16))
            mask_ratio_min: 最小マスク率
            mask_ratio_max: 最大マスク率
            mask_probability: マスクを適用する確率
            min_num_patches: 1ブロックの最小パッチ数
            min_aspect: 最小アスペクト比
        """
        self.H, self.W = input_size
        self.num_patches = self.H * self.W
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.mask_probability = mask_probability
        self.min_num_patches = min_num_patches
        self.min_aspect = min_aspect
        self.max_aspect = 1.0 / min_aspect

    def __call__(self) -> torch.Tensor:
        """
        マスクを生成

        Returns:
            mask: (H, W) - バイナリマスク (1=マスク, 0=可視)

        例 (16×16 グリッド, マスク率30%):
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0
            0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0
            0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0
            0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0
            ...
            マスクされた部分 (1) の Student 出力を Teacher と比較
        """
        # マスクを適用するか (50%の確率)
        if torch.rand(1).item() > self.mask_probability:
            return torch.zeros(self.H, self.W)

        # マスク率をランダム決定
        mask_ratio = torch.empty(1).uniform_(
            self.mask_ratio_min, self.mask_ratio_max
        ).item()
        num_masking = int(self.num_patches * mask_ratio)

        mask = torch.zeros(self.H, self.W)
        masked_count = 0

        # ランダムブロックを配置
        while masked_count < num_masking:
            # ブロックサイズとアスペクト比をランダム決定
            remaining = num_masking - masked_count
            block_size = min(remaining, torch.randint(
                self.min_num_patches, remaining + 1, (1,)
            ).item())

            aspect = torch.empty(1).uniform_(
                self.min_aspect, self.max_aspect
            ).item()

            block_h = max(1, int(round((block_size * aspect) ** 0.5)))
            block_w = max(1, int(round((block_size / aspect) ** 0.5)))
            block_h = min(block_h, self.H)
            block_w = min(block_w, self.W)

            # ランダム位置に配置
            top = torch.randint(0, self.H - block_h + 1, (1,)).item()
            left = torch.randint(0, self.W - block_w + 1, (1,)).item()

            mask[top:top+block_h, left:left+block_w] = 1
            masked_count = mask.sum().int().item()

        return mask  # (H, W)


# ============================================================
# 6. 損失統合
# ============================================================
class DINOv3LossWrapper(nn.Module):
    """
    DINOv3 の全損失を統合するラッパー

    学習フェーズに応じた損失構成:
      Phase 1: L_DINO + L_iBOT + 0.1 * L_KoLeo
      Phase 2: L_DINO + L_iBOT + 0.1 * L_KoLeo + 2.0 * L_Gram
    """

    def __init__(
        self,
        dino_out_dim: int = 256_000,
        ibot_out_dim: int = 96_000,
        student_temp: float = 0.1,
        dino_loss_weight: float = 1.0,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        gram_loss_weight: float = 2.0,
        use_gram: bool = False,
    ):
        super().__init__()

        self.dino_loss = DINOLoss(dino_out_dim, student_temp)
        self.ibot_loss = iBOTPatchLoss(ibot_out_dim, student_temp)
        self.koleo_loss = KoLeoLossDistributed(topk=1)
        self.gram_loss = GramLoss() if use_gram else None

        self.dino_loss_weight = dino_loss_weight
        self.ibot_loss_weight = ibot_loss_weight
        self.koleo_loss_weight = koleo_loss_weight
        self.gram_loss_weight = gram_loss_weight

    def forward(
        self,
        # Teacher 出力
        teacher_cls_centered: torch.Tensor,     # (n_global, B, K_dino)
        teacher_patch_centered: torch.Tensor,   # (n_masked, K_ibot)
        # Student 出力
        student_global_cls: torch.Tensor,       # (n_global, B, K_dino)
        student_local_cls: torch.Tensor,        # (n_local, B, K_dino)
        student_masked_patches: torch.Tensor,   # (n_masked, K_ibot)
        student_cls_features: torch.Tensor,     # (n_global, B, D)
        # Gram (オプション)
        student_patch_feats: Optional[torch.Tensor] = None,   # (n_global*B, P, D)
        gram_teacher_feats: Optional[torch.Tensor] = None,    # (n_global*B, P, D)
        # マスク情報
        masks_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        全損失の計算

        Returns:
            total_loss: scalar
            loss_dict: 各損失値
        """
        n_global = student_global_cls.shape[0]
        n_local = student_local_cls.shape[0]

        # --- スケーリング ---
        dino_global_terms = n_global * (n_global - 1)  # 対角無視: 2
        dino_local_terms = n_global * n_local            # 16
        total = dino_global_terms + dino_local_terms     # 18

        # --- 1. DINO Global Loss ---
        l_dino_global = self.dino_loss(
            student_global_cls, teacher_cls_centered, ignore_diagonal=True
        ) * self.dino_loss_weight * dino_global_terms / total

        # --- 2. DINO Local Loss ---
        l_dino_local = self.dino_loss(
            student_local_cls, teacher_cls_centered, ignore_diagonal=False
        ) * self.dino_loss_weight * dino_local_terms / total

        # --- 3. iBOT Loss ---
        l_ibot = self.ibot_loss.forward_masked(
            student_masked_patches, teacher_patch_centered, masks_weight
        ) * self.ibot_loss_weight

        # --- 4. KoLeo Loss ---
        l_koleo = 0.0
        for i in range(n_global):
            l_koleo += self.koleo_loss(student_cls_features[i])
        l_koleo = l_koleo * self.koleo_loss_weight * n_global

        # --- 5. Gram Loss ---
        l_gram = torch.tensor(0.0)
        if self.gram_loss is not None and student_patch_feats is not None:
            l_gram = self.gram_loss(
                student_patch_feats, gram_teacher_feats
            ) * self.gram_loss_weight

        # --- 合計 ---
        total_loss = l_dino_global + l_dino_local + l_ibot + l_koleo + l_gram

        loss_dict = {
            "dino_global": float(l_dino_global),
            "dino_local": float(l_dino_local),
            "ibot": float(l_ibot),
            "koleo": float(l_koleo),
            "gram": float(l_gram),
            "total": float(total_loss),
        }

        return total_loss, loss_dict
