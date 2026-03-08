"""
LightGlue - Adaptive Inference
==============================

LightGlueの適応的推論メカニズムを疑似コードで示します。

主要コンポーネント:
1. Token Confidence (確信度分類器)
2. Adaptive Depth (Early Stopping)
3. Adaptive Width (Point Pruning)

論文: LightGlue: Local Feature Matching at Light Speed (ICCV 2023)

Shape Convention:
    B: バッチサイズ
    M: Image A のキーポイント数
    N: Image B のキーポイント数
    C: 特徴記述子の埋め込み次元 (dim, default 256)
    H: ヘッド数 (default 4)
    head_dim: ヘッド次元 (C / H = 64)
    N': プルーニング後のキーポイント数 (N' ≤ N)
    L: 総レイヤー数 (default 9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ============================================================
# Token Confidence (確信度分類器)
# ============================================================

class TokenConfidence(nn.Module):
    """
    各レイヤーでの予測確信度を推定

    ========================================
    目的
    ========================================

    LightGlueは各レイヤーで予測を行える (Deep Supervision)。
    問題: いつ推論を止めるか？

    解決: 確信度分類器
        - 各点の「予測が最終レイヤーと同じか」を予測
        - 確信度が高い点が十分多ければ → 早期終了

    ========================================
    学習
    ========================================

    Ground truth:
        label_i = (match_at_layer_ℓ == match_at_layer_L)

        例:
        - Layer 3 で点iが点jにマッチ
        - Layer 9 (最終) でも点iが点jにマッチ
        → label = 1 (confident)

        - Layer 3 で点iが点jにマッチ
        - Layer 9 で点iが点kにマッチ (変わった)
        → label = 0 (not confident)

    損失:
        L_conf = BCE(confidence_i, label_i)

    重要:
        - 勾配は記述子埋め込みに伝播させない (detach)
        - マッチング精度に影響させない

    ========================================
    アーキテクチャ
    ========================================

    Linear(C, 1) → Sigmoid

    シンプルで計算オーバーヘッドが小さい (~2%)
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: 記述子埋め込みの次元 (C)

        パラメータ:
            token: Sequential(Linear(C, 1), Sigmoid)
        """
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(dim, 1),   # Linear(C, 1): (B, N, C) → (B, N, 1)
            nn.Sigmoid()         # (B, N, 1) → (B, N, 1) ∈ [0, 1]
        )

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        確信度を予測

        入力:
            desc0: (B, M, C) Image A の特徴記述子埋め込み
            desc1: (B, N, C) Image B の特徴記述子埋め込み

        処理:
            1. detach() で勾配切断
            2. Linear(C, 1) + Sigmoid で確信度予測
            3. squeeze で次元削除

        出力:
            conf0: (B, M) Image A の各点の確信度 [0, 1]
            conf1: (B, N) Image B の各点の確信度 [0, 1]

        注意:
            detach() により勾配を切断
            → 確信度学習がマッチング精度に影響しない
        """
        # desc0: (B, M, C) → detach → (B, M, C) (勾配なし)
        # → Linear(C, 1) → (B, M, 1) → Sigmoid → (B, M, 1)
        # → squeeze(-1) → (B, M)
        conf0 = self.token(desc0.detach()).squeeze(-1)
        # conf0: (B, M) ∈ [0, 1]

        # desc1: (B, N, C) → detach → (B, N, C) (勾配なし)
        # → Linear(C, 1) → (B, N, 1) → Sigmoid → (B, N, 1)
        # → squeeze(-1) → (B, N)
        conf1 = self.token(desc1.detach()).squeeze(-1)
        # conf1: (B, N) ∈ [0, 1]

        return conf0, conf1
        # conf0: (B, M), conf1: (B, N)


# ============================================================
# Adaptive Depth (Early Stopping)
# ============================================================

def compute_confidence_threshold(
    layer_index: int,
    n_layers: int
) -> float:
    """
    層ごとの確信度閾値

    ========================================
    背景
    ========================================

    観察: 確信度分類器自体が早期レイヤーでは不正確
        - Layer 1 の予測は不安定
        - Layer 8 の予測は安定

    解決: 層に応じて閾値を減衰
        - 早期レイヤー: 高い閾値 (厳しい判定)
        - 後半レイヤー: 低い閾値 (緩い判定)

    ========================================
    数式
    ========================================

    λ_l = 0.8 + 0.1 × exp(-4l/L)

    例 (L=9):
        Layer 0: λ = 0.9
        Layer 4: λ = 0.82
        Layer 8: λ = 0.8

    ========================================
    入力・出力
    ========================================
        入力:
            layer_index: 現在のレイヤー (0-indexed), スカラー
            n_layers: 総レイヤー数, スカラー

        出力:
            threshold: 確信度閾値 [0.8, 0.9], スカラー
    """
    threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / n_layers)
    return np.clip(threshold, 0, 1)
    # return: スカラー float


def check_if_stop(
    confidences0: torch.Tensor,
    confidences1: torch.Tensor,
    layer_index: int,
    n_layers: int,
    depth_confidence: float = 0.95
) -> bool:
    """
    早期終了の判定

    ========================================
    アルゴリズム
    ========================================

    1. 両画像の確信度を結合
    2. 閾値 λ_l を超える点の割合を計算
    3. 割合が α (depth_confidence) を超えたら停止

    ========================================
    数式
    ========================================

    exit = (1/(M+N) × Σ [c_i > λ_l]) > α

    where:
        c_i: 点iの確信度
        λ_l: 層lの確信度閾値
        α: depth_confidence (default: 0.95)

    ========================================
    効果 (MegaDepth)
    ========================================

    | ペアタイプ | 平均停止層 | 速度向上 |
    |-----------|-----------|---------|
    | Easy      | 4.7       | 1.86x   |
    | Medium    | 5.5       | 1.33x   |
    | Hard      | 6.9       | 1.16x   |

    ========================================
    入力・出力
    ========================================
        入力:
            confidences0: (B, M) Image A の確信度
            confidences1: (B, N) Image B の確信度
            layer_index: 現在のレイヤー, スカラー
            n_layers: 総レイヤー数, スカラー
            depth_confidence: 終了判定閾値 (default: 0.95), スカラー

        出力:
            should_stop: bool
    """
    # confidences0: (B, M), confidences1: (B, N)

    # 確信度を結合
    # cat along dim=-1: (B, M) + (B, N) → (B, M+N)
    confidences = torch.cat([confidences0, confidences1], dim=-1)
    # confidences: (B, M+N)

    num_points = confidences.shape[-1]
    # num_points: M+N (スカラー)

    # 層ごとの閾値
    threshold = compute_confidence_threshold(layer_index, n_layers)
    # threshold: スカラー float

    # 確信度が閾値を超える点の割合
    # (confidences >= threshold): (B, M+N) bool → float → sum → スカラー
    num_confident = (confidences >= threshold).float().sum()
    # num_confident: スカラー (バッチ全体で合計)

    ratio_confident = num_confident / num_points
    # ratio_confident: スカラー

    # 割合が α を超えたら終了
    return ratio_confident > depth_confidence
    # return: bool


# ============================================================
# Adaptive Width (Point Pruning)
# ============================================================

def get_pruning_mask(
    confidences: Optional[torch.Tensor],
    matchability: torch.Tensor,
    layer_index: int,
    n_layers: int,
    width_confidence: float = 0.99
) -> torch.Tensor:
    """
    Point Pruning のマスク生成

    ========================================
    目的
    ========================================

    問題: マッチ不可能な点も後続レイヤーで処理される
        - Attentionの計算量: O(N²)
        - 無駄な計算

    解決: マッチ不可能と判断された点を早期に除外
        - 探索空間を縮小
        - 計算量削減

    ========================================
    除外条件
    ========================================

    点を除外する条件:
        1. 確信度が高い (confident)
        2. かつ Matchability が低い (unmatchable)

    つまり:
        「この点にはマッチ相手がいない」と確信している

    保持する条件 (keep):
        1. Matchability が高い (マッチ可能性あり)
        2. または 確信度が低い (まだ不確実)

    ========================================
    数式
    ========================================

    keep = (matchability > 1 - β) OR (confidence ≤ λ_l)

    where:
        β = 1 - width_confidence = 0.01
        λ_l = confidence_threshold(l)

    除外される点:
        - matchability < 0.01
        - かつ confidence > λ_l

    ========================================
    効果 (MegaDepth)
    ========================================

    | Difficulty | Unmatchable % | Speedup |
    |------------|---------------|---------|
    | Easy       | 19.8%         | (depth主導) |
    | Medium     | 23.4%         | (mixed) |
    | Hard       | 27.9%         | (width主導) |

    Average: 23.7% の点が除外される

    ========================================
    入力・出力
    ========================================
        入力:
            confidences: (B, N) 確信度 (None if early stopping disabled)
            matchability: (B, N) マッチ可能性 [0, 1]
            layer_index: 現在のレイヤー, スカラー
            n_layers: 総レイヤー数, スカラー
            width_confidence: 除外閾値 (default: 0.99), スカラー

        出力:
            keep_mask: (B, N) 保持する点はTrue (bool)
    """
    # matchability: (B, N)

    # Matchability 閾値
    # width_confidence = 0.99 → β = 0.01
    matchability_threshold = 1 - width_confidence  # 0.01

    # Matchability が高い点は保持
    # matchability: (B, N) > 0.01 → (B, N) bool
    keep = matchability > matchability_threshold
    # keep: (B, N) bool

    # 確信度が低い点も保持 (まだ判断を下すには早い)
    if confidences is not None:
        # confidences: (B, N)
        confidence_threshold = compute_confidence_threshold(layer_index, n_layers)
        # confidence_threshold: スカラー float

        # (confidences <= threshold): (B, N) bool
        # keep | ...: (B, N) bool
        keep = keep | (confidences <= confidence_threshold)
        # keep: (B, N) bool
        # True = matchability高い OR 確信度低い → 保持
        # False = matchability低い AND 確信度高い → 除外

    return keep
    # return: (B, N) bool


def apply_pruning(
    desc: torch.Tensor,
    encoding: torch.Tensor,
    keep_mask: torch.Tensor,
    indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Point Pruning を適用

    入力:
        desc: (B, N, C) 特徴記述子埋め込み
        encoding: (2, B, 1, N, head_dim) 位置エンコーディング
        keep_mask: (B, N) 保持する点のマスク (bool)
        indices: (B, N) 元のインデックス

    処理:
        keep_maskがTrueの点のみを選択してテンソルを縮小

    出力:
        desc_pruned: (B, N', C)  where N' = keep_maskのTrue数
        encoding_pruned: (2, B, 1, N', head_dim)
        indices_updated: (B, N')
    """
    # desc: (B, N, C)
    # encoding: (2, B, 1, N, head_dim)
    # keep_mask: (B, N) bool
    # indices: (B, N)

    # 保持するインデックスを取得
    # keep_mask: (B, N) → where → (バッチ内のインデックス, 点インデックス)
    keep_indices = torch.where(keep_mask)[1]  # バッチ内でflatten
    # keep_indices: (N',) ※簡略化 (実際はバッチごとに異なるが、ここではB=1を想定)

    # 要素を選択
    desc_pruned = desc.index_select(1, keep_indices)
    # desc_pruned: (B, N', C)

    encoding_pruned = encoding.index_select(-2, keep_indices)
    # encoding_pruned: (2, B, 1, N', head_dim)

    indices_updated = indices.index_select(1, keep_indices)
    # indices_updated: (B, N')

    return desc_pruned, encoding_pruned, indices_updated
    # desc_pruned: (B, N', C), encoding_pruned: (2, B, 1, N', head_dim), indices_updated: (B, N')


# ============================================================
# 統合: Adaptive Inference Manager
# ============================================================

class AdaptiveInferenceManager:
    """
    適応的推論の管理クラス

    Early Stopping と Point Pruning を統合管理

    処理の流れ (各レイヤーで):
        1. Transformer処理
        2. 確信度・matchability を計算
        3. check_stop() → True なら終了 (Adaptive Depth)
        4. should_prune() → True なら pruning (Adaptive Width)
           get_pruning_mask() → apply_pruning() で点を除外
        5. 次のレイヤーへ
    """

    def __init__(
        self,
        n_layers: int = 9,
        depth_confidence: float = 0.95,
        width_confidence: float = 0.99,
        min_keypoints_for_pruning: int = 1024
    ):
        """
        Args:
            n_layers: 総レイヤー数 (L)
            depth_confidence: Early stopping 閾値 α (-1 で無効)
            width_confidence: Point pruning 閾値 β (-1 で無効)
            min_keypoints_for_pruning: Pruning を有効にする最小キーポイント数
        """
        self.n_layers = n_layers
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence
        self.min_keypoints_for_pruning = min_keypoints_for_pruning

        # 閾値をキャッシュ: List[float] of length L
        self.confidence_thresholds = [
            compute_confidence_threshold(i, n_layers)
            for i in range(n_layers)
        ]

    @property
    def do_early_stop(self) -> bool:
        return self.depth_confidence > 0

    @property
    def do_point_pruning(self) -> bool:
        return self.width_confidence > 0

    def check_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int
    ) -> bool:
        """
        Early stopping check

        入力:
            confidences0: (B, M) Image A の確信度
            confidences1: (B, N) Image B の確信度
            layer_index: 現在のレイヤー, スカラー

        出力:
            bool: 終了すべきか
        """
        if not self.do_early_stop:
            return False

        return check_if_stop(
            confidences0, confidences1,
            layer_index, self.n_layers,
            self.depth_confidence
        )

    def get_pruning_mask(
        self,
        confidences: Optional[torch.Tensor],
        matchability: torch.Tensor,
        layer_index: int
    ) -> torch.Tensor:
        """
        Point pruning mask

        入力:
            confidences: (B, N) 確信度 (None if early stopping disabled)
            matchability: (B, N) マッチ可能性 [0, 1]
            layer_index: 現在のレイヤー, スカラー

        出力:
            keep_mask: (B, N) bool
        """
        return get_pruning_mask(
            confidences, matchability,
            layer_index, self.n_layers,
            self.width_confidence
        )

    def should_prune(self, num_keypoints: int) -> bool:
        """
        Pruning を実行すべきか

        入力:
            num_keypoints: 現在のキーポイント数 N, スカラー

        出力:
            bool: N > min_keypoints_for_pruning かつ pruning有効
        """
        return (
            self.do_point_pruning and
            num_keypoints > self.min_keypoints_for_pruning
        )


# ============================================================
# 可視化・デバッグ用
# ============================================================

def visualize_pruning(
    original_points: torch.Tensor,
    keep_mask: torch.Tensor,
    image_size: Tuple[int, int] = (640, 480),
    title: str = "Point Pruning"
):
    """
    Point Pruning の可視化

    入力:
        original_points: (N, 2) 元のキーポイント座標
        keep_mask: (N,) 保持する点のマスク (bool)
        image_size: (W, H) 画像サイズ
        title: タイトル
    """
    import matplotlib.pyplot as plt

    # original_points: (N, 2) → numpy
    points = original_points.cpu().numpy()
    # mask: (N,) → numpy
    mask = keep_mask.cpu().numpy()

    kept_points = points[mask]     # (N_kept, 2)
    pruned_points = points[~mask]  # (N_pruned, 2)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 画像境界
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)

    # Pruned points (赤)
    if len(pruned_points) > 0:
        ax.scatter(
            pruned_points[:, 0], pruned_points[:, 1],
            c='red', s=20, alpha=0.5, label=f'Pruned ({len(pruned_points)})'
        )

    # Kept points (緑)
    if len(kept_points) > 0:
        ax.scatter(
            kept_points[:, 0], kept_points[:, 1],
            c='green', s=20, alpha=0.7, label=f'Kept ({len(kept_points)})'
        )

    ax.set_title(f'{title}\nKept: {len(kept_points)} / Total: {len(points)}')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def explain_adaptive_inference():
    """
    Adaptive Inference の解説
    """
    print("=" * 60)
    print("LightGlue Adaptive Inference")
    print("=" * 60)

    print("""
    === なぜ適応的推論が必要か？ ===

    観察:
        - 画像ペアの「難易度」は様々
        - Easy pairs: 高オーバーラップ、変化小 → 早期に収束
        - Hard pairs: 低オーバーラップ、変化大 → 全レイヤー必要

    固定深度の問題:
        - Easy pairs でも9層全て実行 → 無駄
        - Hard pairs では適切だが、全体の大部分は easy

    解決: 適応的推論
        1. Adaptive Depth: 確信度が高ければ早期終了
        2. Adaptive Width: マッチ不可能な点を除外

    === Adaptive Depth (Early Stopping) ===

    仕組み:
        1. 各レイヤーで確信度を予測
           - 「この予測は最終レイヤーと同じか？」
        2. 確信度 > λ_l の点が α% を超えたら終了

    閾値の減衰:
        λ_l = 0.8 + 0.1 × exp(-4l/L)

        理由: 早期レイヤーの確信度予測自体が不安定
        → 厳しい閾値で誤った早期終了を防ぐ

    効果:
        - Easy pairs: ~4.7層で終了 → 1.86x高速化
        - Hard pairs: ~6.9層で終了 → 1.16x高速化
        - 平均: 1.45x高速化

    === Adaptive Width (Point Pruning) ===

    仕組み:
        1. 各点のmatchabilityを予測
           - 「この点にはマッチ相手がいるか？」
        2. 確信度が高く、matchabilityが低い点を除外

    除外条件:
        (confidence > λ_l) AND (matchability < 0.01)

    どんな点が除外される？:
        - オクルージョン領域
        - 視野外の点
        - テクスチャレス領域
        - 動的物体

    効果:
        - 平均23.7%の点が除外
        - Attention計算量: O(N²) → O((0.76N)²) = 0.58 × O(N²)

    === 2つの相補性 ===

    Easy pairs:
        → 高オーバーラップ → ほとんどの点がマッチ可能
        → Point pruning の効果小
        → Early stopping の効果大

    Hard pairs:
        → 低オーバーラップ → 多くの点がマッチ不可能
        → Point pruning の効果大
        → Early stopping の効果小

    → どちらの場合でも高速化が得られる!
    """)


# ============================================================
# 使用例
# ============================================================

def example_usage():
    """
    Adaptive Inference の使用例
    """
    print("\n=== Adaptive Inference Example ===\n")

    B, M, N = 1, 256, 240
    dim = 256  # C
    n_layers = 9  # L

    # ダミーデータ
    desc0 = torch.randn(B, M, dim)  # (B, M, C) = (1, 256, 256)
    desc1 = torch.randn(B, N, dim)  # (B, N, C) = (1, 240, 256)

    # ========================================
    # Token Confidence
    # ========================================
    print("1. Token Confidence")

    token_conf = TokenConfidence(dim)
    # desc0: (B, M, C) = (1, 256, 256) → conf0: (B, M) = (1, 256)
    # desc1: (B, N, C) = (1, 240, 256) → conf1: (B, N) = (1, 240)
    conf0, conf1 = token_conf(desc0, desc1)

    print(f"   Confidence A: shape={conf0.shape}, mean={conf0.mean():.3f}")
    # (1, 256)
    print(f"   Confidence B: shape={conf1.shape}, mean={conf1.mean():.3f}")
    # (1, 240)

    # ========================================
    # Confidence Thresholds
    # ========================================
    print("\n2. Confidence Thresholds by Layer")

    for i in range(n_layers):
        thresh = compute_confidence_threshold(i, n_layers)
        print(f"   Layer {i}: λ = {thresh:.4f}")

    # ========================================
    # Early Stopping Check
    # ========================================
    print("\n3. Early Stopping Check")

    # 低い確信度 → 続行
    # low_conf0: (B, M) = (1, 256) ∈ [0, 0.5]
    low_conf0 = torch.rand(B, M) * 0.5
    # low_conf1: (B, N) = (1, 240) ∈ [0, 0.5]
    low_conf1 = torch.rand(B, N) * 0.5
    # cat: (1, 256) + (1, 240) → (1, 496) → ratio < 0.95 → False
    should_stop = check_if_stop(low_conf0, low_conf1, layer_index=3, n_layers=9)
    print(f"   Low confidence (mean=0.25): stop={should_stop}")

    # 高い確信度 → 終了
    # high_conf0: (B, M) = (1, 256) ∈ [0.9, 1.0]
    high_conf0 = torch.rand(B, M) * 0.1 + 0.9
    # high_conf1: (B, N) = (1, 240) ∈ [0.9, 1.0]
    high_conf1 = torch.rand(B, N) * 0.1 + 0.9
    # cat: (1, 256) + (1, 240) → (1, 496) → ratio > 0.95 → True
    should_stop = check_if_stop(high_conf0, high_conf1, layer_index=5, n_layers=9)
    print(f"   High confidence (mean=0.95): stop={should_stop}")

    # ========================================
    # Point Pruning
    # ========================================
    print("\n4. Point Pruning")

    # Matchability: 一部の点は低い
    matchability = torch.rand(B, M)  # (B, M) = (1, 256)
    matchability[0, :50] = 0.001     # 50点は低matchability → 除外候補

    # 確信度: 全体的に高い
    confidences = torch.ones(B, M) * 0.95  # (B, M) = (1, 256) ≈ 0.95

    # matchability: (1, 256), confidences: (1, 256) → keep_mask: (1, 256) bool
    keep_mask = get_pruning_mask(
        confidences, matchability,
        layer_index=5, n_layers=9,
        width_confidence=0.99
    )
    # keep_mask: (1, 256) bool

    n_kept = keep_mask.sum().item()
    n_pruned = M - n_kept
    print(f"   Total points: {M}")         # 256
    print(f"   Kept: {n_kept}")             # ~206 (256 - 50)
    print(f"   Pruned: {n_pruned}")         # ~50

    # ========================================
    # Adaptive Inference Manager
    # ========================================
    print("\n5. Adaptive Inference Manager")

    manager = AdaptiveInferenceManager(
        n_layers=9,
        depth_confidence=0.95,
        width_confidence=0.99,
        min_keypoints_for_pruning=1024
    )

    print(f"   Early stopping enabled: {manager.do_early_stop}")
    print(f"   Point pruning enabled: {manager.do_point_pruning}")
    print(f"   Should prune (256 points): {manager.should_prune(256)}")    # False (< 1024)
    print(f"   Should prune (2048 points): {manager.should_prune(2048)}")  # True (> 1024)


if __name__ == "__main__":
    explain_adaptive_inference()
    print("\n" + "=" * 60 + "\n")
    example_usage()
