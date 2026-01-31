"""
ALIKED Differentiable Keypoint Detection - 簡略化疑似コード
========================================================

DKD (Differentiable Keypoint Detection)

主要特徴:
- Sub-pixel精度のキーポイント検出
- 微分可能 → end-to-end学習可能
- Soft-argmaxによる位置推定
- Score Dispersity (スコアの集中度) 計算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DKD(nn.Module):
    """
    Differentiable Keypoint Detection (DKD)

    処理フロー:
    1. NMS (Non-Maximum Suppression) - 局所最大値検出
    2. Thresholding - スコア閾値適用
    3. Soft-argmax - Sub-pixel位置推定
    4. Score Dispersity - 信頼度計算

    特徴:
    - 完全に微分可能
    - Sub-pixel精度
    - スコア分散度を考慮
    """

    def __init__(
        self,
        radius: int = 2,          # NMSカーネル半径
        top_k: int = 5000,        # 最大キーポイント数
        scores_th: float = 0.2,   # スコア閾値
        n_limit: int = 20000,     # 制限
        temperature: float = 0.1   # Soft-argmax temperature
    ):
        super().__init__()

        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        self.temperature = temperature

    def forward(
        self,
        score_map: torch.Tensor,
        top_k: int = None,
        scores_th: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DKD フォワードパス

        入力:
            score_map: (B, 1, H, W) - スコアマップ

        出力:
            keypoints: (B, N, 2) - Sub-pixel座標 [x, y]
            scores: (B, N) - キーポイントスコア
        """

        if top_k is None:
            top_k = self.top_k
        if scores_th is None:
            scores_th = self.scores_th

        B, _, H, W = score_map.shape

        # ========================================
        # Step 1: NMS (Non-Maximum Suppression)
        # ========================================

        # 2回NMS適用 (精度向上のため)
        nms_map = score_map
        for _ in range(2):
            nms_map = self._simple_nms(nms_map, kernel_size=2 * self.radius + 1)
        # nms_map: (B, 1, H, W) - NMS後のスコア

        # ========================================
        # Step 2: Thresholding & Top-K Selection
        # ========================================

        # スコアマップをフラット化
        scores_flat = nms_map.view(B, -1)  # (B, H*W)

        # 閾値適用
        mask = scores_flat > scores_th

        # Top-K選択
        topk_scores, topk_indices = torch.topk(
            scores_flat,
            k=min(top_k, scores_flat.shape[1]),
            dim=1
        )

        # 閾値を満たすもののみ
        valid_mask = topk_scores > scores_th

        keypoints_list = []
        scores_list = []

        for b in range(B):
            # バッチごとに処理
            valid_idx = topk_indices[b][valid_mask[b]]
            valid_scores = topk_scores[b][valid_mask[b]]

            # インデックス → (y, x) 座標
            y_pix = valid_idx // W
            x_pix = valid_idx % W

            # Pixel-level keypoints
            kpts_pix = torch.stack([x_pix, y_pix], dim=-1).float()
            # kpts_pix: (N, 2)

            # ========================================
            # Step 3: Soft-argmax Refinement
            # ========================================

            # Sub-pixel refinement
            kpts_refined = self._soft_argmax_refine(
                score_map[b, 0],
                kpts_pix,
                window_size=2 * self.radius + 1
            )
            # kpts_refined: (N, 2) - Sub-pixel座標 [x, y]

            keypoints_list.append(kpts_refined)
            scores_list.append(valid_scores)

        # Padding to same length
        max_kpts = max(k.shape[0] for k in keypoints_list)

        keypoints = torch.zeros(B, max_kpts, 2, device=score_map.device)
        scores = torch.zeros(B, max_kpts, device=score_map.device)

        for b in range(B):
            n = keypoints_list[b].shape[0]
            keypoints[b, :n] = keypoints_list[b]
            scores[b, :n] = scores_list[b]

        return keypoints, scores

    def _simple_nms(
        self,
        score_map: torch.Tensor,
        kernel_size: int = 5
    ) -> torch.Tensor:
        """
        Simple Non-Maximum Suppression

        入力:
            score_map: (B, 1, H, W)
            kernel_size: int - NMSウィンドウサイズ

        出力:
            nms_map: (B, 1, H, W) - NMS後のスコア
        """

        # Max pooling
        max_pooled = F.max_pool2d(
            score_map,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        # 局所最大値のみ保持
        nms_map = torch.where(
            score_map == max_pooled,
            score_map,
            torch.zeros_like(score_map)
        )

        return nms_map

    def _soft_argmax_refine(
        self,
        score_map: torch.Tensor,
        keypoints_pix: torch.Tensor,
        window_size: int = 5
    ) -> torch.Tensor:
        """
        Soft-argmax による Sub-pixel refinement

        原理:
        ピクセルレベルのキーポイント周辺のウィンドウで、
        スコアに基づく重み付き平均を計算してsub-pixel位置を推定

        数式:
        p_refined = Σ (p_i × exp(s_i / T)) / Σ exp(s_i / T)

        where:
          p_i: ウィンドウ内の各ピクセル座標
          s_i: スコア
          T: temperature

        入力:
            score_map: (H, W) - スコアマップ
            keypoints_pix: (N, 2) - ピクセルレベルキーポイント [x, y]
            window_size: int - ウィンドウサイズ

        出力:
            keypoints_refined: (N, 2) - Sub-pixel座標 [x, y]
        """

        H, W = score_map.shape
        N = keypoints_pix.shape[0]

        if N == 0:
            return keypoints_pix

        half = window_size // 2

        # ウィンドウグリッド生成
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=score_map.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=score_map.device),
            indexing='ij'
        )
        # grid_x, grid_y: (window_size, window_size)

        grid_coords = torch.stack([grid_x, grid_y], dim=-1)
        # grid_coords: (window_size, window_size, 2)

        keypoints_refined = []

        for i in range(N):
            x_pix, y_pix = keypoints_pix[i]
            x_pix = int(x_pix.item())
            y_pix = int(y_pix.item())

            # ウィンドウ範囲
            y_min = max(0, y_pix - half)
            y_max = min(H, y_pix + half + 1)
            x_min = max(0, x_pix - half)
            x_max = min(W, x_pix + half + 1)

            # スコアパッチ抽出
            score_patch = score_map[y_min:y_max, x_min:x_max]
            # score_patch: (patch_h, patch_w)

            # 対応するグリッド座標
            grid_h = y_max - y_min
            grid_w = x_max - x_min

            offset_y = y_min - (y_pix - half)
            offset_x = x_min - (x_pix - half)

            grid_patch = grid_coords[
                offset_y:offset_y + grid_h,
                offset_x:offset_x + grid_w
            ]
            # grid_patch: (patch_h, patch_w, 2)

            # Softmaxウェイト計算
            weights = F.softmax(
                score_patch.flatten() / self.temperature,
                dim=0
            )
            # weights: (patch_h * patch_w,)

            # 重み付き平均
            grid_flat = grid_patch.reshape(-1, 2)
            refined_offset = (weights.unsqueeze(1) * grid_flat).sum(dim=0)
            # refined_offset: (2,) - [dx, dy]

            # 絶対座標
            x_refined = x_pix + refined_offset[0]
            y_refined = y_pix + refined_offset[1]

            keypoints_refined.append(torch.stack([x_refined, y_refined]))

        keypoints_refined = torch.stack(keypoints_refined, dim=0)
        # keypoints_refined: (N, 2)

        return keypoints_refined

    def compute_score_dispersity(
        self,
        score_map: torch.Tensor,
        keypoints: torch.Tensor,
        window_size: int = 5
    ) -> torch.Tensor:
        """
        Score Dispersity 計算

        スコアの分散度を測定:
        - 高分散 → キーポイント位置が不確実
        - 低分散 → キーポイント位置が確実

        数式:
        dispersity = Σ (||p_i - p_center||^2 × softmax(s_i / T))

        入力:
            score_map: (B, 1, H, W)
            keypoints: (B, N, 2) - [x, y]
            window_size: int

        出力:
            dispersity: (B, N) - 分散度スコア
        """

        B, _, H, W = score_map.shape
        B, N, _ = keypoints.shape

        half = window_size // 2

        # グリッド生成
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=score_map.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=score_map.device),
            indexing='ij'
        )

        # 距離計算
        distances = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        # distances: (window_size, window_size)

        dispersity_list = []

        for b in range(B):
            batch_dispersity = []

            for n in range(N):
                x_pix = int(keypoints[b, n, 0].item())
                y_pix = int(keypoints[b, n, 1].item())

                # ウィンドウ抽出
                y_min = max(0, y_pix - half)
                y_max = min(H, y_pix + half + 1)
                x_min = max(0, x_pix - half)
                x_max = min(W, x_pix + half + 1)

                score_patch = score_map[b, 0, y_min:y_max, x_min:x_max]

                # Softmax weights
                weights = F.softmax(score_patch.flatten() / self.temperature, dim=0)

                # Dispersity
                dist_flat = distances[:score_patch.shape[0], :score_patch.shape[1]].flatten()
                dispersity_val = (weights * dist_flat).sum()

                batch_dispersity.append(dispersity_val)

            dispersity_list.append(torch.stack(batch_dispersity))

        dispersity = torch.stack(dispersity_list, dim=0)
        # dispersity: (B, N)

        return dispersity

# ============================================
# 使用例
# ============================================

def example_dkd():
    """DKD使用例"""

    # DKD作成
    dkd = DKD(
        radius=2,
        top_k=1000,
        scores_th=0.2,
        temperature=0.1
    )

    # ダミースコアマップ
    score_map = torch.rand(2, 1, 160, 120)

    # いくつかの高スコア領域を作成
    score_map[0, 0, 50:55, 60:65] = 0.9
    score_map[0, 0, 100:105, 80:85] = 0.85
    score_map[1, 0, 70:75, 50:55] = 0.92

    # キーポイント検出
    keypoints, scores = dkd(score_map, top_k=500, scores_th=0.2)

    print(f"Score map: {score_map.shape}")
    print(f"Keypoints: {keypoints.shape}")  # (2, 500, 2)
    print(f"Scores: {scores.shape}")         # (2, 500)

    # 有効なキーポイント数
    valid_counts = (scores > 0).sum(dim=1)
    print(f"Valid keypoints per batch: {valid_counts}")

    # Score dispersity計算
    dispersity = dkd.compute_score_dispersity(score_map, keypoints)
    print(f"Dispersity: {dispersity.shape}")  # (2, 500)
    print(f"Mean dispersity: {dispersity[dispersity > 0].mean():.4f}")

if __name__ == "__main__":
    example_dkd()
