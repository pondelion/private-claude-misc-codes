"""
LightGlue - Loss Computation
============================

LightGlueの学習損失を疑似コードで示します。

主要コンポーネント:
1. Correspondence Loss (対応関係損失)
2. Deep Supervision (全レイヤーで損失計算)
3. Confidence Loss (確信度分類器の学習)
4. Ground Truth Computation (教師データ生成)

論文: LightGlue: Local Feature Matching at Light Speed (ICCV 2023)

Shape Convention:
    B: バッチサイズ
    M: Image A のキーポイント数
    N: Image B のキーポイント数
    H_img, W_img: 画像の高さ・幅
    L: 総レイヤー数 (default 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


# ============================================================
# Ground Truth Computation
# ============================================================

def compute_ground_truth_matches_homography(
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    H: torch.Tensor,
    match_threshold: float = 3.0,
    unmatch_threshold: float = 5.0
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Homography から Ground Truth マッチを計算

    ========================================
    処理フロー
    ========================================

    1. kpts0 を H で変換 → kpts0_warped
    2. kpts0_warped と kpts1 の距離を計算
    3. 距離 < match_threshold → マッチ
    4. 全ての点との距離 > unmatch_threshold → unmatchable

    ========================================
    入力
    ========================================
        kpts0: (M, 2) Image A のキーポイント [x, y]
        kpts1: (N, 2) Image B のキーポイント [x, y]
        H: (3, 3) Homography行列 (A→B)
        match_threshold: マッチ判定閾値 (default: 3px), スカラー
        unmatch_threshold: unmatchable判定閾値 (default: 5px), スカラー

    ========================================
    出力
    ========================================
        matches: List[(i, j)] マッチペア (長さ K_match)
        unmatchable_A: List[i] Aのunmatchable点 (長さ K_unA)
        unmatchable_B: List[j] Bのunmatchable点 (長さ K_unB)
    """
    # kpts0: (M, 2), kpts1: (N, 2), H: (3, 3)
    M, N = kpts0.shape[0], kpts1.shape[0]

    # ========================================
    # Step 1: キーポイントを変換
    # ========================================
    # Homogeneous coordinates
    ones = torch.ones(M, 1, device=kpts0.device)
    # ones: (M, 1)

    # kpts0: (M, 2) + ones: (M, 1) → cat → (M, 3)
    kpts0_homo = torch.cat([kpts0, ones], dim=1)
    # kpts0_homo: (M, 3) = [x, y, 1]

    # H で変換: (M, 3) @ (3, 3)^T → (M, 3)
    kpts0_warped_homo = kpts0_homo @ H.T
    # kpts0_warped_homo: (M, 3) = [x', y', w']

    # 正規化 (ホモジニアス → ユークリッド)
    # [:, :2]: (M, 2), [:, 2:3]: (M, 1) → (M, 2) / (M, 1) → broadcast → (M, 2)
    kpts0_warped = kpts0_warped_homo[:, :2] / kpts0_warped_homo[:, 2:3]
    # kpts0_warped: (M, 2) = [x'/w', y'/w']

    # ========================================
    # Step 2: 距離行列を計算
    # ========================================
    # L2距離
    # kpts0_warped: (M, 2) → unsqueeze(1) → (M, 1, 2)
    # kpts1: (N, 2) → unsqueeze(0) → (1, N, 2)
    # diff: (M, 1, 2) - (1, N, 2) → broadcast → (M, N, 2)
    diff = kpts0_warped.unsqueeze(1) - kpts1.unsqueeze(0)
    # diff: (M, N, 2)

    # norm: (M, N, 2) → dim=-1 → (M, N)
    dist = torch.norm(diff, dim=-1)
    # dist: (M, N) - 各ペア (i, j) 間のL2距離

    # ========================================
    # Step 3: マッチを抽出
    # ========================================
    matches = []

    # 各A点について最近傍B点を見つける
    # dist: (M, N) → dim=1で最小 → min_dist_a: (M,), argmin_a: (M,)
    min_dist_a, argmin_a = dist.min(dim=1)
    # min_dist_a: (M,) - 各A点の最近傍距離
    # argmin_a: (M,) - 各A点の最近傍B点インデックス

    # 各B点について最近傍A点を見つける
    # dist: (M, N) → dim=0で最小 → min_dist_b: (N,), argmin_b: (N,)
    min_dist_b, argmin_b = dist.min(dim=0)
    # min_dist_b: (N,) - 各B点の最近傍距離
    # argmin_b: (N,) - 各B点の最近傍A点インデックス

    for i in range(M):
        j = argmin_a[i].item()
        # Mutual nearest neighbor check
        if argmin_b[j].item() == i:
            # 距離が閾値以下
            if min_dist_a[i] < match_threshold:
                matches.append((i, j))

    # ========================================
    # Step 4: Unmatchable を判定
    # ========================================
    matched_A = set(m[0] for m in matches)
    matched_B = set(m[1] for m in matches)

    # A点がunmatchable: マッチされておらず、全B点との距離 > unmatch_threshold
    unmatchable_A = []
    for i in range(M):
        if i not in matched_A:
            # dist[i]: (N,) → min → スカラー
            if dist[i].min() > unmatch_threshold:
                unmatchable_A.append(i)

    # B点がunmatchable (逆変換が必要だが、簡略化)
    unmatchable_B = []
    for j in range(N):
        if j not in matched_B:
            # dist[:, j]: (M,) → min → スカラー
            if dist[:, j].min() > unmatch_threshold:
                unmatchable_B.append(j)

    return matches, unmatchable_A, unmatchable_B


def compute_ground_truth_matches_depth(
    kpts0: torch.Tensor,
    kpts1: torch.Tensor,
    depth0: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
    match_threshold: float = 3.0,
    unmatch_threshold: float = 5.0
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Depth + Pose から Ground Truth マッチを計算

    ========================================
    処理フロー
    ========================================

    1. kpts0 を depth で 3D に変換
    2. T_0to1 で カメラ1 の座標系に変換
    3. K1 で Image B に投影
    4. 距離でマッチ判定

    ========================================
    入力
    ========================================
        kpts0: (M, 2) Image A のキーポイント [x, y]
        kpts1: (N, 2) Image B のキーポイント [x, y]
        depth0: (H_img, W_img) Image A の深度マップ
        K0: (3, 3) Image A のカメラ内部パラメータ
        K1: (3, 3) Image B のカメラ内部パラメータ
        T_0to1: (4, 4) 相対ポーズ (カメラ0→カメラ1)

    ========================================
    数式
    ========================================

    3D point in camera 0:
        p_cam0 = K0^{-1} @ [x, y, 1]^T * depth[y, x]

    3D point in camera 1:
        p_cam1 = R @ p_cam0 + t
        where T_0to1 = [R | t]

    2D point in image 1:
        p_img1 = K1 @ p_cam1
        [x', y'] = p_img1[:2] / p_img1[2]

    ========================================
    出力
    ========================================
        matches: List[(i, j)] マッチペア
        unmatchable_A: List[i] Aのunmatchable点
        unmatchable_B: List[j] Bのunmatchable点
    """
    # kpts0: (M, 2), kpts1: (N, 2), depth0: (H_img, W_img)
    # K0: (3, 3), K1: (3, 3), T_0to1: (4, 4)
    M, N = kpts0.shape[0], kpts1.shape[0]

    # 深度を取得
    kpts0_int = kpts0.long()
    # kpts0_int: (M, 2) - 整数座標
    # depth0[y, x]: (H_img, W_img) → インデックス → (M,)
    depths = depth0[kpts0_int[:, 1], kpts0_int[:, 0]]
    # depths: (M,) - 各キーポイントの深度

    # 有効な深度を持つ点のみ
    # depths > 0: (M,) bool
    valid_depth = depths > 0
    # valid_depth: (M,) bool

    # ========================================
    # Step 1: 3D点に変換 (カメラ0座標系)
    # ========================================
    # K0: (3, 3) → inverse → (3, 3)
    K0_inv = torch.inverse(K0)
    # K0_inv: (3, 3)

    ones = torch.ones(M, 1, device=kpts0.device)
    # ones: (M, 1)

    # kpts0: (M, 2) + ones: (M, 1) → cat → (M, 3)
    kpts0_homo = torch.cat([kpts0, ones], dim=1)
    # kpts0_homo: (M, 3) = [x, y, 1]

    # ピクセル → カメラ座標 (正規化平面)
    # K0_inv: (3, 3) @ kpts0_homo^T: (3, M) → (3, M) → ^T → (M, 3)
    kpts0_cam_norm = (K0_inv @ kpts0_homo.T).T
    # kpts0_cam_norm: (M, 3) - 正規化カメラ座標

    # 深度でスケーリング → 3D点
    # kpts0_cam_norm: (M, 3) * depths: (M,) → unsqueeze(1) → (M, 1) → broadcast → (M, 3)
    p_cam0 = kpts0_cam_norm * depths.unsqueeze(1)
    # p_cam0: (M, 3) - カメラ0座標系の3D点

    # ========================================
    # Step 2: カメラ1座標系に変換
    # ========================================
    # T_0to1: (4, 4) → R: (3, 3), t: (3,)
    R = T_0to1[:3, :3]
    # R: (3, 3)
    t = T_0to1[:3, 3]
    # t: (3,)

    # R @ p_cam0^T: (3, 3) @ (3, M) → (3, M) → ^T → (M, 3) + t: (3,) → broadcast → (M, 3)
    p_cam1 = (R @ p_cam0.T).T + t
    # p_cam1: (M, 3) - カメラ1座標系の3D点

    # ========================================
    # Step 3: Image B に投影
    # ========================================
    # K1: (3, 3) @ p_cam1^T: (3, M) → (3, M) → ^T → (M, 3)
    p_img1_homo = (K1 @ p_cam1.T).T
    # p_img1_homo: (M, 3) = [x*z, y*z, z]

    # 正の深度のみ (カメラ前方)
    # p_img1_homo[:, 2]: (M,) > 0 → (M,) bool
    valid_z = p_img1_homo[:, 2] > 0
    # valid_z: (M,) bool

    # valid_depth & valid_z: (M,) bool
    valid = valid_depth & valid_z
    # valid: (M,) bool

    # 正規化 (ホモジニアス → ユークリッド)
    # [:, :2]: (M, 2) / [:, 2:3]: (M, 1) → (M, 2)
    kpts0_warped = p_img1_homo[:, :2] / p_img1_homo[:, 2:3]
    # kpts0_warped: (M, 2) - Image Bに投影された座標

    # ========================================
    # Step 4: マッチ判定
    # ========================================
    # kpts0_warped: (M, 2) → unsqueeze(1) → (M, 1, 2)
    # kpts1: (N, 2) → unsqueeze(0) → (1, N, 2)
    # diff: (M, 1, 2) - (1, N, 2) → (M, N, 2)
    diff = kpts0_warped.unsqueeze(1) - kpts1.unsqueeze(0)
    # diff: (M, N, 2)

    # norm: (M, N, 2) → (M, N)
    dist = torch.norm(diff, dim=-1)
    # dist: (M, N)

    # 無効な点は大きな距離に設定
    # ~valid: (M,) bool → dist[~valid]: 各行全体をinfに
    dist[~valid] = float('inf')
    # dist: (M, N)

    # dist: (M, N) → dim=1で最小 → min_dist_a: (M,), argmin_a: (M,)
    matches = []
    min_dist_a, argmin_a = dist.min(dim=1)
    # min_dist_a: (M,), argmin_a: (M,)
    # dist: (M, N) → dim=0で最小 → min_dist_b: (N,), argmin_b: (N,)
    min_dist_b, argmin_b = dist.min(dim=0)
    # min_dist_b: (N,), argmin_b: (N,)

    for i in range(M):
        if not valid[i]:
            continue
        j = argmin_a[i].item()
        if argmin_b[j].item() == i and min_dist_a[i] < match_threshold:
            matches.append((i, j))

    # Unmatchable
    matched_A = set(m[0] for m in matches)
    matched_B = set(m[1] for m in matches)

    unmatchable_A = [i for i in range(M) if i not in matched_A and
                    (not valid[i] or dist[i].min() > unmatch_threshold)]
    unmatchable_B = [j for j in range(N) if j not in matched_B and
                    dist[:, j].min() > unmatch_threshold]

    return matches, unmatchable_A, unmatchable_B


# ============================================================
# Correspondence Loss
# ============================================================

def compute_correspondence_loss(
    scores: torch.Tensor,
    matches: List[Tuple[int, int]],
    unmatchable_A: List[int],
    unmatchable_B: List[int]
) -> torch.Tensor:
    """
    対応関係損失を計算

    ========================================
    損失関数
    ========================================

    L = L_positive + L_negative

    L_positive = -1/|M| × Σ_{(i,j)∈M} log P_{ij}
        → 正しいマッチの対数尤度

    L_negative = -1/|Ā| × Σ_{i∈Ā} log(1 - σ_A^i)
               + -1/|B̄| × Σ_{j∈B̄} log(1 - σ_B^j)
        → unmatchable点の対数尤度

    ========================================
    入力
    ========================================
        scores: (M+1, N+1) log assignment matrix (single sample, バッチなし)
        matches: List[(i, j)] ground truth マッチ (長さ K_match)
        unmatchable_A: List[i] A の unmatchable 点 (長さ K_unA)
        unmatchable_B: List[j] B の unmatchable 点 (長さ K_unB)

    ========================================
    出力
    ========================================
        loss: スカラー損失値 (shape: ())
    """
    # scores: (M+1, N+1)

    # ========================================
    # Positive Loss (正しいマッチ)
    # ========================================
    if len(matches) > 0:
        # matches: List[(i, j)] → Tensor: (K_match, 2)
        match_indices = torch.tensor(matches, dtype=torch.long, device=scores.device)
        # match_indices: (K_match, 2)

        # scores[i, j] for each match: (K_match,)
        match_scores = scores[match_indices[:, 0], match_indices[:, 1]]
        # match_scores: (K_match,) - 各正解マッチのlog確率

        # -mean: (K_match,) → スカラー
        loss_positive = -match_scores.mean()
        # loss_positive: スカラー
    else:
        loss_positive = scores.new_tensor(0.0)
        # loss_positive: スカラー (0.0)

    # ========================================
    # Negative Loss (unmatchable)
    # ========================================
    M, N = scores.shape[0] - 1, scores.shape[1] - 1

    # A点のunmatchable: 最後の列 (dustbin)
    if len(unmatchable_A) > 0:
        # unmatchable_A: List[i] → Tensor: (K_unA,)
        unmatch_A_indices = torch.tensor(unmatchable_A, dtype=torch.long, device=scores.device)
        # unmatch_A_indices: (K_unA,)

        # scores[i, -1]: dustbin列 → (K_unA,)
        unmatch_A_scores = scores[unmatch_A_indices, -1]
        # unmatch_A_scores: (K_unA,) - A点がunmatchableであるlog確率

        # -mean: (K_unA,) → スカラー
        loss_neg_A = -unmatch_A_scores.mean()
        # loss_neg_A: スカラー
    else:
        loss_neg_A = scores.new_tensor(0.0)
        # loss_neg_A: スカラー (0.0)

    # B点のunmatchable: 最後の行 (dustbin)
    if len(unmatchable_B) > 0:
        # unmatchable_B: List[j] → Tensor: (K_unB,)
        unmatch_B_indices = torch.tensor(unmatchable_B, dtype=torch.long, device=scores.device)
        # unmatch_B_indices: (K_unB,)

        # scores[-1, j]: dustbin行 → (K_unB,)
        unmatch_B_scores = scores[-1, unmatch_B_indices]
        # unmatch_B_scores: (K_unB,) - B点がunmatchableであるlog確率

        # -mean: (K_unB,) → スカラー
        loss_neg_B = -unmatch_B_scores.mean()
        # loss_neg_B: スカラー
    else:
        loss_neg_B = scores.new_tensor(0.0)
        # loss_neg_B: スカラー (0.0)

    # スカラー + スカラー → スカラー
    loss_negative = (loss_neg_A + loss_neg_B) / 2
    # loss_negative: スカラー

    return loss_positive + loss_negative
    # return: スカラー


# ============================================================
# Deep Supervision Loss
# ============================================================

class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision による損失計算

    ========================================
    SuperGlueとの違い
    ========================================

    SuperGlue:
        - 最終レイヤーのみで損失計算
        - Sinkhorn が重いため中間出力困難

    LightGlue:
        - 全レイヤーで損失計算
        - 軽量なヘッドで中間予測が可能
        - 収束が速い

    ========================================
    損失関数
    ========================================

    L = (1/L) × Σ_ℓ L_correspondence(ℓ)

    各レイヤーで:
        1. Assignment matrix を計算
        2. Ground truth と比較
        3. 損失を累積

    これにより:
        - 早期レイヤーでも意味のある予測を学習
        - Early stopping の前提条件
    """

    def __init__(self, n_layers: int = 9):
        """
        Args:
            n_layers: Transformerレイヤー数 (L)
        """
        super().__init__()
        self.n_layers = n_layers

    def forward(
        self,
        scores_per_layer: List[torch.Tensor],
        matches: List[Tuple[int, int]],
        unmatchable_A: List[int],
        unmatchable_B: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        全レイヤーの損失を計算

        入力:
            scores_per_layer: List[(M+1, N+1)] 各レイヤーのassignment (長さ L)
                scores_per_layer[ℓ]: (M+1, N+1) レイヤーℓのlog assignment matrix
            matches: List[(i, j)] ground truth マッチ (長さ K_match)
            unmatchable_A: List[i] unmatchable点 (長さ K_unA)
            unmatchable_B: List[j] unmatchable点 (長さ K_unB)

        処理:
            各レイヤーで compute_correspondence_loss を呼び出し平均

        出力:
            dict with:
                'total_loss': スカラー (全レイヤー平均)
                'loss_layer_0': スカラー (レイヤー0の損失)
                ...
                'loss_layer_{L-1}': スカラー (レイヤーL-1の損失)
        """
        total_loss = 0
        losses = {}

        for layer_idx, scores in enumerate(scores_per_layer):
            # scores: (M+1, N+1)
            layer_loss = compute_correspondence_loss(
                scores, matches, unmatchable_A, unmatchable_B
            )
            # layer_loss: スカラー
            losses[f'loss_layer_{layer_idx}'] = layer_loss
            total_loss = total_loss + layer_loss

        # 全レイヤー平均: スカラー / L → スカラー
        total_loss = total_loss / len(scores_per_layer)
        # total_loss: スカラー
        losses['total_loss'] = total_loss

        return losses


# ============================================================
# Confidence Loss (2段階目)
# ============================================================

class ConfidenceLoss(nn.Module):
    """
    確信度分類器の損失

    ========================================
    学習戦略
    ========================================

    2段階学習:
        Stage 1: 対応関係予測を学習 (上記のDeepSupervisionLoss)
        Stage 2: 確信度分類器を学習 (このクラス)

    Stage 2では:
        - マッチング部分の重みを固定
        - 確信度分類器のみ学習

    ========================================
    Ground Truth
    ========================================

    label_i = (match_at_layer_ℓ == match_at_layer_L)

    つまり:
        - Layer ℓ での予測が最終レイヤーと同じ → 1 (confident)
        - Layer ℓ での予測が最終レイヤーと異なる → 0 (not confident)

    ========================================
    損失
    ========================================

    L_conf = BCE(confidence_i, label_i)

    重要:
        - 勾配は記述子埋め込みに伝播させない (detachされている)
        - マッチング精度に影響しない
    """

    def __init__(self, n_layers: int = 9):
        """
        Args:
            n_layers: Transformerレイヤー数 (L)
        """
        super().__init__()
        self.n_layers = n_layers

    def forward(
        self,
        confidences_per_layer: List[Tuple[torch.Tensor, torch.Tensor]],
        matches_per_layer: List[torch.Tensor],
        final_matches: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        確信度損失を計算

        入力:
            confidences_per_layer: List[(conf0, conf1)] 各レイヤーの確信度 (長さ L-1)
                conf0: (B, M) Image A の確信度
                conf1: (B, N) Image B の確信度
            matches_per_layer: List[(M,)] 各レイヤーのマッチ結果 (長さ L-1)
                matches_per_layer[ℓ]: (M,) レイヤーℓでの各A点のマッチ先B点index
            final_matches: (M,) 最終レイヤーのマッチ結果

        処理:
            1. 各レイヤーのマッチと最終レイヤーのマッチを比較 → labels: (M,) ∈ {0, 1}
            2. BCE(conf0, labels) で損失計算

        出力:
            dict with:
                'total_conf_loss': スカラー (全レイヤー平均)
                'conf_loss_layer_0': スカラー
                ...
        """
        total_loss = 0
        losses = {}

        # 最終レイヤー以外で計算
        for layer_idx in range(len(confidences_per_layer)):
            conf0, conf1 = confidences_per_layer[layer_idx]
            # conf0: (B, M), conf1: (B, N) ※ここではconf0のみ使用 (簡略化)

            match_at_layer = matches_per_layer[layer_idx]
            # match_at_layer: (M,) - レイヤーℓでの各A点のマッチ先

            # Ground truth: 予測が最終レイヤーと同じか
            # match_at_layer: (M,) == final_matches: (M,) → (M,) bool → float → (M,) ∈ {0.0, 1.0}
            labels = (match_at_layer == final_matches).float()
            # labels: (M,) ∈ {0.0, 1.0}

            # BCE loss
            # conf0: (B, M) → conf0[最初のバッチ]を使用 (簡略化)
            # conf0: (B, M), labels: (M,) → スカラー
            loss = F.binary_cross_entropy(conf0, labels)
            # loss: スカラー

            losses[f'conf_loss_layer_{layer_idx}'] = loss
            total_loss = total_loss + loss

        # 全レイヤー平均: スカラー / (L-1) → スカラー
        total_loss = total_loss / len(confidences_per_layer)
        # total_loss: スカラー
        losses['total_conf_loss'] = total_loss

        return losses


# ============================================================
# 統合: Training Loss Wrapper
# ============================================================

class LightGlueLoss(nn.Module):
    """
    LightGlue 学習損失の統合クラス

    2段階学習をサポート:
        Stage 1: 対応関係 + Deep Supervision
        Stage 2: 確信度分類器
    """

    def __init__(self, n_layers: int = 9, stage: int = 1):
        """
        Args:
            n_layers: Transformerレイヤー数 (L)
            stage: 学習ステージ (1: correspondence, 2: confidence)
        """
        super().__init__()
        self.n_layers = n_layers
        self.stage = stage

        self.correspondence_loss = DeepSupervisionLoss(n_layers)
        self.confidence_loss = ConfidenceLoss(n_layers)

    def forward(
        self,
        predictions: Dict,
        ground_truth: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        損失を計算

        入力:
            predictions: モデルの出力
                Stage 1:
                    'scores_per_layer': List[(M+1, N+1)] 長さ L
                Stage 2:
                    'confidences_per_layer': List[(conf0: (B, M), conf1: (B, N))] 長さ L-1
                    'matches_per_layer': List[(M,)] 長さ L-1
                    'final_matches': (M,)

            ground_truth: 教師データ
                'matches': List[(i, j)] 長さ K_match
                'unmatchable_A': List[i] 長さ K_unA
                'unmatchable_B': List[j] 長さ K_unB

        出力:
            dict with all loss components (各値はスカラー)
        """
        losses = {}

        if self.stage == 1:
            # Stage 1: Correspondence Loss
            corr_losses = self.correspondence_loss(
                predictions['scores_per_layer'],
                ground_truth['matches'],
                ground_truth['unmatchable_A'],
                ground_truth['unmatchable_B']
            )
            losses.update(corr_losses)

        elif self.stage == 2:
            # Stage 2: Confidence Loss
            conf_losses = self.confidence_loss(
                predictions['confidences_per_layer'],
                predictions['matches_per_layer'],
                predictions['final_matches']
            )
            losses.update(conf_losses)

        return losses


# ============================================================
# 可視化・デバッグ用
# ============================================================

def explain_training():
    """
    LightGlueの学習プロセスを解説
    """
    print("=" * 60)
    print("LightGlue Training Process")
    print("=" * 60)

    print("""
    === 学習データ ===

    Pre-training: Oxford-Paris 1M (Synthetic Homographies)
        - 170K 画像
        - ランダムなHomography変換
        - 強い光学的拡張
        - 完全なground truth（ノイズなし）

    Fine-tuning: MegaDepth
        - 196 観光地ランドマーク
        - 1M crowd-sourced 画像
        - SfM + MVS による depth/pose
        - 現実的な変化を含む

    === 2段階学習 ===

    Stage 1: 対応関係予測 (2 GPU-days)
        目的: 正しいマッチを予測

        損失:
            L = (1/L) × Σ_ℓ [L_pos(ℓ) + L_neg(ℓ)]

            L_pos = -mean(log P[gt_matches])
            L_neg = -mean(log(1-σ[unmatchable]))

        設定:
            - Batch size: 32
            - Keypoints: 2048 per image
            - Learning rate: 1e-4 → 1e-5 (decay)
            - Optimizer: Adam

    Stage 2: 確信度分類器 (0.5 GPU-days)
        目的: 早期終了の判断を学習

        前提:
            - Stage 1の重みを固定
            - 確信度分類器のみ学習

        損失:
            L_conf = BCE(confidence, label)
            label = (match_at_layer == match_at_final)

    === Deep Supervision の効果 ===

    SuperGlue: 最終レイヤーのみ
        - 収束に7+ GPU-days
        - 中間レイヤーの予測は不安定

    LightGlue: 全レイヤー
        - 収束に2 GPU-days (3.5倍高速)
        - 中間レイヤーでも意味のある予測
        - Early stopping が可能に

    === Ground Truth の計算 ===

    Homography:
        kpts0_warped = H @ kpts0
        match if ||kpts0_warped - kpts1|| < 3px

    Depth + Pose:
        p_3d = unproject(kpts0, depth0)
        p_3d_cam1 = T_0to1 @ p_3d
        kpts0_warped = project(p_3d_cam1, K1)
        match if ||kpts0_warped - kpts1|| < 3px

    Unmatchable:
        - reprojection error > 5px for all points
        - no depth available
        - large epipolar error
    """)


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    Loss computation の使用例
    """
    print("\n=== Loss Computation Example ===\n")

    B, M, N = 1, 64, 48
    n_layers = 9  # L

    # ========================================
    # Ground Truth の生成 (ダミー)
    # ========================================
    print("1. Ground Truth Computation")

    # ダミーのキーポイント
    kpts0 = torch.rand(M, 2) * 100  # (M, 2) = (64, 2)
    kpts1 = torch.rand(N, 2) * 100  # (N, 2) = (48, 2)

    # ダミーのHomography (ほぼ恒等変換)
    H = torch.eye(3) + torch.randn(3, 3) * 0.01  # (3, 3)

    # kpts0: (64, 2), kpts1: (48, 2), H: (3, 3)
    # → matches: List[(i,j)], unmatchable_A: List[i], unmatchable_B: List[j]
    matches, unmatchable_A, unmatchable_B = compute_ground_truth_matches_homography(
        kpts0, kpts1, H,
        match_threshold=5.0,
        unmatch_threshold=10.0
    )

    print(f"   Found {len(matches)} matches")
    print(f"   Unmatchable A: {len(unmatchable_A)} points")
    print(f"   Unmatchable B: {len(unmatchable_B)} points")

    # ========================================
    # Correspondence Loss
    # ========================================
    print("\n2. Correspondence Loss (Single Layer)")

    # ダミーの assignment matrix
    scores = torch.randn(M + 1, N + 1)  # (M+1, N+1) = (65, 49)
    scores = F.log_softmax(scores, dim=-1)  # 正規化 → (65, 49)

    # scores: (65, 49), matches, unmatchable_A, unmatchable_B → スカラー
    loss = compute_correspondence_loss(
        scores, matches, unmatchable_A, unmatchable_B
    )
    print(f"   Loss: {loss.item():.4f}")

    # ========================================
    # Deep Supervision Loss
    # ========================================
    print("\n3. Deep Supervision Loss")

    # 各レイヤーの scores: List[(M+1, N+1)] 長さ L
    scores_per_layer = [
        F.log_softmax(torch.randn(M + 1, N + 1), dim=-1)  # (65, 49)
        for _ in range(n_layers)
    ]

    deep_loss = DeepSupervisionLoss(n_layers)
    # scores_per_layer: List[(65, 49)] × 9
    # → losses dict (各値はスカラー)
    losses = deep_loss(
        scores_per_layer, matches, unmatchable_A, unmatchable_B
    )

    print(f"   Total loss: {losses['total_loss'].item():.4f}")
    for i in range(min(3, n_layers)):
        print(f"   Layer {i}: {losses[f'loss_layer_{i}'].item():.4f}")

    # ========================================
    # 統合クラス
    # ========================================
    print("\n4. LightGlueLoss (Stage 1)")

    loss_fn = LightGlueLoss(n_layers=9, stage=1)

    predictions = {'scores_per_layer': scores_per_layer}
    ground_truth = {
        'matches': matches,
        'unmatchable_A': unmatchable_A,
        'unmatchable_B': unmatchable_B
    }

    # predictions, ground_truth → losses dict (各値はスカラー)
    all_losses = loss_fn(predictions, ground_truth)
    print(f"   Total loss: {all_losses['total_loss'].item():.4f}")


if __name__ == "__main__":
    explain_training()
    print("\n" + "=" * 60 + "\n")
    example_usage()
