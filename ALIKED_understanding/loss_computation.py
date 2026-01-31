"""
ALIKED Loss Computation - 簡略化疑似コード
=========================================

5つの損失関数:
1. Reprojection Loss - キーポイント位置の幾何的整合性
2. Peaky Loss - スコアマップの鋭さ
3. Sparse NRE Loss - スパース記述子のマッチング (KEY INNOVATION)
4. Reliable Loss - 記述子の信頼性
5. (Optional) Triplet Loss - Hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class ALIKEDLossWrapper(nn.Module):
    """
    ALIKED 損失計算ラッパー

    損失重み (デフォルト):
    - w_rp: 1.0  (Reprojection Loss)
    - w_pk: 0.5  (Peaky Loss)
    - w_ds: 5.0  (Sparse NRE Loss)
    - w_re: 1.0  (Reliable Loss)
    - w_triplet: 0.0  (通常は未使用)
    """

    def __init__(
        self,
        w_rp: float = 1.0,
        w_pk: float = 0.5,
        w_ds: float = 5.0,
        w_re: float = 1.0,
        w_triplet: float = 0.0,
        tdes: float = 0.1,      # Descriptor temperature
        trel: float = 1.0       # Reliability temperature
    ):
        super().__init__()

        self.w_rp = w_rp
        self.w_pk = w_pk
        self.w_ds = w_ds
        self.w_re = w_re
        self.w_triplet = w_triplet

        self.tdes = tdes
        self.trel = trel

    def forward(
        self,
        outputs_a: Dict[str, torch.Tensor],
        outputs_b: Dict[str, torch.Tensor],
        homography_ab: torch.Tensor,
        depth_a: torch.Tensor = None,
        R_ab: torch.Tensor = None,
        t_ab: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        損失計算

        入力:
            outputs_a: Image Aの出力
                {
                    'keypoints': (B, N_a, 2)
                    'descriptors': (B, N_a, dim)
                    'scores': (B, N_a)
                    'score_map': (B, 1, H, W)
                }
            outputs_b: Image Bの出力 (同様)
            homography_ab: (B, 3, 3) - Homography変換 (Homographyデータセット用)
            depth_a: (B, H, W) - Depth map (Perspectiveデータセット用)
            R_ab, t_ab: Rotation & translation (Perspectiveデータセット用)

        出力:
            losses: {
                'loss_rp': scalar
                'loss_pk': scalar
                'loss_ds': scalar
                'loss_re': scalar
                'total_loss': scalar
            }
        """

        # ========================================
        # 1. Reprojection Loss
        # ========================================
        loss_rp = self._reprojection_loss(
            outputs_a, outputs_b,
            homography_ab, depth_a, R_ab, t_ab
        )

        # ========================================
        # 2. Peaky Loss (Score Dispersity)
        # ========================================
        loss_pk = self._peaky_loss(outputs_a, outputs_b)

        # ========================================
        # 3. Sparse NRE Loss (Descriptor Matching)
        # ========================================
        loss_ds = self._sparse_nre_loss(
            outputs_a, outputs_b,
            homography_ab, depth_a, R_ab, t_ab
        )

        # ========================================
        # 4. Reliable Loss
        # ========================================
        loss_re = self._reliable_loss(
            outputs_a, outputs_b,
            homography_ab, depth_a, R_ab, t_ab
        )

        # ========================================
        # Total Loss
        # ========================================
        total_loss = (
            self.w_rp * loss_rp +
            self.w_pk * loss_pk +
            self.w_ds * loss_ds +
            self.w_re * loss_re
        )

        return {
            'loss_rp': loss_rp,
            'loss_pk': loss_pk,
            'loss_ds': loss_ds,
            'loss_re': loss_re,
            'total_loss': total_loss
        }

    def _reprojection_loss(
        self,
        outputs_a: Dict,
        outputs_b: Dict,
        H_ab: torch.Tensor,
        depth_a: torch.Tensor = None,
        R_ab: torch.Tensor = None,
        t_ab: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Reprojection Location Loss

        キーポイントの幾何的整合性を保証:
        - pAをImage Bに投影 → pAB
        - pBに最も近いキーポイントとマッチング
        - 逆方向も同様
        - 双方向の距離を最小化

        数式:
        L_rp = 1/2 * (||pA - pBA|| + ||pB - pAB||)

        入力:
            outputs_a, outputs_b: キーポイント情報
            H_ab: (B, 3, 3) - Homography
            (または depth_a, R_ab, t_ab for perspective)

        出力:
            loss: scalar
        """

        kpts_a = outputs_a['keypoints']  # (B, N_a, 2)
        kpts_b = outputs_b['keypoints']  # (B, N_b, 2)

        B = kpts_a.shape[0]

        total_loss = 0.0
        num_matches = 0

        for b in range(B):
            # ========================================
            # A → B 投影
            # ========================================
            kpts_a_warped = self._warp_keypoints(
                kpts_a[b],
                H_ab[b] if H_ab is not None else None,
                depth_a[b] if depth_a is not None else None,
                R_ab[b] if R_ab is not None else None,
                t_ab[b] if t_ab is not None else None
            )
            # kpts_a_warped: (N_a, 2)

            # 最近傍マッチング
            matches_ab = self._find_nearest_neighbors(
                kpts_a_warped,
                kpts_b[b],
                distance_threshold=5.0  # pixels
            )
            # matches_ab: (M, 2) - [idx_a, idx_b]

            # ========================================
            # B → A 投影
            # ========================================
            H_ba = torch.inverse(H_ab[b]) if H_ab is not None else None
            kpts_b_warped = self._warp_keypoints(
                kpts_b[b],
                H_ba,
                None,  # depth not needed for inverse
                R_ab[b].T if R_ab is not None else None,
                -t_ab[b] if t_ab is not None else None
            )

            # Reprojection error計算
            for idx_a, idx_b in matches_ab:
                # Forward: A → B
                err_ab = torch.norm(kpts_a_warped[idx_a] - kpts_b[b, idx_b])

                # Backward: B → A
                err_ba = torch.norm(kpts_a[b, idx_a] - kpts_b_warped[idx_b])

                total_loss += (err_ab + err_ba) / 2.0
                num_matches += 1

        if num_matches > 0:
            return total_loss / num_matches
        else:
            return torch.tensor(0.0, device=kpts_a.device)

    def _peaky_loss(
        self,
        outputs_a: Dict,
        outputs_b: Dict
    ) -> torch.Tensor:
        """
        Peaky Loss (Dispersity Peak Loss)

        スコアマップの鋭さを強化:
        - キーポイント位置でスコアが鋭くピークを持つように訓練
        - Score dispersity (分散度) を最小化

        数式:
        L_pk = mean(softmax(s_patch) · ||p - c||)

        where:
          s_patch: キーポイント周辺のスコアパッチ
          p: パッチ内の各ピクセル座標
          c: キーポイント座標

        入力:
            outputs_a, outputs_b: キーポイント情報

        出力:
            loss: scalar
        """

        score_map_a = outputs_a['score_map']  # (B, 1, H, W)
        kpts_a = outputs_a['keypoints']        # (B, N_a, 2)

        score_map_b = outputs_b['score_map']
        kpts_b = outputs_b['keypoints']

        window_size = 5
        half = window_size // 2

        # グリッド生成
        grid_y, grid_x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=score_map_a.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=score_map_a.device),
            indexing='ij'
        )

        distances = torch.sqrt(grid_x ** 2 + grid_y ** 2)
        # distances: (window_size, window_size)

        def compute_dispersity(score_map, kpts):
            B, _, H, W = score_map.shape
            total_dispersity = 0.0
            count = 0

            for b in range(B):
                for n in range(kpts.shape[1]):
                    x_pix = int(kpts[b, n, 0].item())
                    y_pix = int(kpts[b, n, 1].item())

                    if x_pix < half or x_pix >= W - half or \
                       y_pix < half or y_pix >= H - half:
                        continue

                    # スコアパッチ抽出
                    score_patch = score_map[b, 0,
                                           y_pix - half:y_pix + half + 1,
                                           x_pix - half:x_pix + half + 1]

                    # Softmax weights
                    weights = F.softmax(score_patch.flatten(), dim=0)

                    # Dispersity
                    dispersity = (weights * distances.flatten()).sum()

                    total_dispersity += dispersity
                    count += 1

            return total_dispersity / count if count > 0 else 0.0

        loss_a = compute_dispersity(score_map_a, kpts_a)
        loss_b = compute_dispersity(score_map_b, kpts_b)

        return (loss_a + loss_b) / 2.0

    def _sparse_nre_loss(
        self,
        outputs_a: Dict,
        outputs_b: Dict,
        H_ab: torch.Tensor,
        depth_a: torch.Tensor = None,
        R_ab: torch.Tensor = None,
        t_ab: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sparse Neural Reprojection Error Loss

        🔑 ALIKEDの主要イノベーション:
        ========================================

        従来のNRE Loss (Dense):
        - 密な記述子マップが必要
        - 2D確率マップを構築
        - Cross-entropy loss

        Sparse NRE Loss:
        - スパース記述子のみ使用
        - 1D確率ベクトルを構築
        - メモリ使用量を大幅削減

        処理:
        1. 幾何的対応関係から Reprojection Probability Vector 構築
        2. 記述子類似度から Matching Probability Vector 構築
        3. 2つの確率ベクトル間のCross-Entropy最小化

        数式:
        q_r(pA, P_B) = binary vector (matching=1, others=0)
        q_m(dA, D_B) = softmax((sim(dA, D_B) - 1) / t_des)
        L_ds = -log(q_m(dA, dB))

        入力:
            outputs_a, outputs_b: キーポイント・記述子情報
            H_ab: Homography

        出力:
            loss: scalar
        """

        kpts_a = outputs_a['keypoints']      # (B, N_a, 2)
        desc_a = outputs_a['descriptors']    # (B, N_a, dim)

        kpts_b = outputs_b['keypoints']      # (B, N_b, 2)
        desc_b = outputs_b['descriptors']    # (B, N_b, dim)

        B = kpts_a.shape[0]

        total_loss = 0.0
        num_matches = 0

        for b in range(B):
            # ========================================
            # Step 1: 幾何的対応からReprojection Probability構築
            # ========================================

            # A → B 投影
            kpts_a_warped = self._warp_keypoints(
                kpts_a[b],
                H_ab[b] if H_ab is not None else None,
                depth_a[b] if depth_a is not None else None,
                R_ab[b] if R_ab is not None else None,
                t_ab[b] if t_ab is not None else None
            )

            # 最近傍マッチング
            matches_ab = self._find_nearest_neighbors(
                kpts_a_warped,
                kpts_b[b],
                distance_threshold=5.0
            )

            if len(matches_ab) == 0:
                continue

            # ========================================
            # Step 2: 各マッチペアに対してSparse NRE Loss計算
            # ========================================

            for idx_a, idx_b in matches_ab:
                dA = desc_a[b, idx_a]  # (dim,)
                DB = desc_b[b]          # (N_b, dim)

                # ========================================
                # Matching Similarity Vector
                # ========================================
                # Cosine similarity
                sim = torch.matmul(DB, dA)  # (N_b,)

                # Matching probability vector
                q_m = F.softmax((sim - 1.0) / self.tdes, dim=0)
                # q_m: (N_b,) - 全キーポイントに対する確率

                # ========================================
                # Loss: -log(q_m[matching_idx])
                # ========================================
                loss = -torch.log(q_m[idx_b] + 1e-8)

                total_loss += loss
                num_matches += 1

        if num_matches > 0:
            return total_loss / num_matches
        else:
            return torch.tensor(0.0, device=kpts_a.device)

    def _reliable_loss(
        self,
        outputs_a: Dict,
        outputs_b: Dict,
        H_ab: torch.Tensor,
        depth_a: torch.Tensor = None,
        R_ab: torch.Tensor = None,
        t_ab: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Reliable Loss (論文 Section V-D, 式12-13)

        マッチング対応点での記述子の信頼性に基づいてスコアを調整:
        - 対応点の記述子が明確にマッチする → 高信頼性 → 高スコア維持
        - 対応点の記述子が曖昧 → 低信頼性 → スコアを下げる

        数式:
        r(pA, I_B) = softmax(sim(dA, D_B) / t_rel)[idx_b]  (式12, 対応点のindex)
        L_re = (1 / ŜA) * Σ (1 - r(pA, I_B)) * sA           (式13)

        入力:
            outputs_a, outputs_b: キーポイント・記述子・スコア情報
            H_ab: Homography行列
            depth_a, R_ab, t_ab: Perspective projection用（オプション）

        出力:
            loss: scalar
        """

        kpts_a = outputs_a['keypoints']      # (B, N_a, 2)
        desc_a = outputs_a['descriptors']    # (B, N_a, dim)
        scores_a = outputs_a['scores']       # (B, N_a)

        kpts_b = outputs_b['keypoints']      # (B, N_b, 2)
        desc_b = outputs_b['descriptors']    # (B, N_b, dim)

        B = kpts_a.shape[0]

        total_loss = 0.0

        for b in range(B):
            SA = scores_a[b]  # (N_a,)
            DB = desc_b[b]    # (N_b, dim)

            # ========================================
            # Step 1: 幾何的対応からマッチングペアを見つける
            # ========================================
            kpts_a_warped = self._warp_keypoints(
                kpts_a[b],
                H_ab[b] if H_ab is not None else None,
                depth_a[b] if depth_a is not None else None,
                R_ab[b] if R_ab is not None else None,
                t_ab[b] if t_ab is not None else None
            )

            matches_ab = self._find_nearest_neighbors(
                kpts_a_warped,
                kpts_b[b],
                distance_threshold=5.0
            )

            if len(matches_ab) == 0:
                continue

            # ========================================
            # Step 2: マッチングペアごとにReliability計算
            # ========================================
            weighted_loss = 0.0
            score_sum = 0.0

            for idx_a, idx_b in matches_ab:
                dA = desc_a[b, idx_a]   # (dim,)
                sA = SA[idx_a]          # scalar

                # Matching similarity vector (式9)
                sim = torch.matmul(DB, dA)  # (N_b,)

                # Reliability: 対応点でのsoftmax値 (式12)
                r_vec = F.softmax(sim / self.trel, dim=0)  # (N_b,)
                r = r_vec[idx_b]  # 対応点のreliability (scalar)

                # 重み付きloss (式13)
                weighted_loss += (1.0 - r) * sA
                score_sum += sA

            # 正規化 (式13: 1/ŜA)
            if score_sum > 0:
                loss_b = weighted_loss / (score_sum + 1e-8)
                total_loss += loss_b

        return total_loss / B

    def _warp_keypoints(
        self,
        keypoints: torch.Tensor,
        H: torch.Tensor = None,
        depth: torch.Tensor = None,
        R: torch.Tensor = None,
        t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        キーポイントをワープ

        Homography mode: H適用
        Perspective mode: 3D projection

        入力:
            keypoints: (N, 2) - [x, y]
            H: (3, 3) - Homography matrix
            (or depth, R, t for perspective)

        出力:
            warped: (N, 2) - [x', y']
        """

        if H is not None:
            # Homography変換
            kpts_homo = torch.cat([
                keypoints,
                torch.ones(keypoints.shape[0], 1, device=keypoints.device)
            ], dim=1)  # (N, 3)

            kpts_warped_homo = torch.matmul(kpts_homo, H.T)  # (N, 3)

            # 正規化
            kpts_warped = kpts_warped_homo[:, :2] / kpts_warped_homo[:, 2:3]

            return kpts_warped

        else:
            # Perspective変換 (簡略版)
            # 実装では depth map と R, t を使用
            return keypoints  # 簡略化のため

    def _find_nearest_neighbors(
        self,
        kpts_src: torch.Tensor,
        kpts_tgt: torch.Tensor,
        distance_threshold: float = 5.0
    ) -> List[Tuple[int, int]]:
        """
        最近傍マッチング

        入力:
            kpts_src: (N_src, 2)
            kpts_tgt: (N_tgt, 2)
            distance_threshold: float - ピクセル

        出力:
            matches: List[(idx_src, idx_tgt)]
        """

        # 距離行列
        dist_matrix = torch.cdist(kpts_src, kpts_tgt)  # (N_src, N_tgt)

        # 最近傍
        min_dists, min_indices = dist_matrix.min(dim=1)  # (N_src,)

        # 閾値適用
        valid_mask = min_dists < distance_threshold

        matches = []
        for i in range(kpts_src.shape[0]):
            if valid_mask[i]:
                matches.append((i, min_indices[i].item()))

        return matches

# ============================================
# 使用例
# ============================================

def example_loss():
    """損失計算の使用例"""

    # 損失ラッパー
    loss_wrapper = ALIKEDLossWrapper(
        w_rp=1.0,
        w_pk=0.5,
        w_ds=5.0,
        w_re=1.0
    )

    # ダミー出力
    outputs_a = {
        'keypoints': torch.rand(2, 500, 2) * 100,
        'descriptors': F.normalize(torch.randn(2, 500, 128), p=2, dim=-1),
        'scores': torch.rand(2, 500),
        'score_map': torch.rand(2, 1, 160, 120)
    }

    outputs_b = {
        'keypoints': torch.rand(2, 500, 2) * 100,
        'descriptors': F.normalize(torch.randn(2, 500, 128), p=2, dim=-1),
        'scores': torch.rand(2, 500),
        'score_map': torch.rand(2, 1, 160, 120)
    }

    # Homography
    H_ab = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    H_ab[:, 0, 2] = 10  # Translation

    # 損失計算
    losses = loss_wrapper(outputs_a, outputs_b, H_ab)

    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

if __name__ == "__main__":
    example_loss()
