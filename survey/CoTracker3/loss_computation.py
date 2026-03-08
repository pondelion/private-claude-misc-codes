"""
CoTracker3 - 損失関数
論文: https://arxiv.org/abs/2410.11831

【このファイルの概要】
CoTracker3の訓練で使用される3種類の損失関数を詳細に解説。

1. sequence_loss: 座標予測の損失 (Huber Loss + 指数減衰重み)
2. sequence_BCE_loss: 可視性の損失 (Binary Cross-Entropy)
3. sequence_prob_loss: 信頼度の損失 (12pxしきい値ベースのBCE)

【公式コードの対応箇所】
- cotracker/models/core/cotracker/losses.py

========================================
訓練の全体損失
========================================

total_loss = λ_coord × L_coord + λ_vis × L_vis + λ_conf × L_conf

論文より:
  λ_coord = 1.0 (メイン損失)
  λ_vis   = 適切な重み
  λ_conf  = 適切な重み

CoTracker2からの変更点:
  - L_conf (信頼度損失) が新たに追加
  - 可視点と不可視点の重み付け (loss_only_for_visible=False, 不可視は1/5)

========================================
反復予測に対する指数減衰重み
========================================

M回の反復予測に対して、最終反復の予測に最も高い重みを付ける:

  L = (1/M) × Σ_{m=1}^{M} γ^{M-m} × L_m

  γ = 0.8 (デフォルト)

  m=1 (最初): γ^{M-1} = 0.8^3 = 0.512
  m=2:        γ^{M-2} = 0.8^2 = 0.64
  m=3:        γ^{M-3} = 0.8^1 = 0.8
  m=4 (最後): γ^{M-4} = 0.8^0 = 1.0

  → 最終予測の重みが最大

========================================
ウィンドウ損失の平均化
========================================

Onlineモードでは複数のウィンドウに対して損失を計算し、
ウィンドウ数で平均化する。(Offlineではウィンドウ=1)

  total_loss = (1/num_windows) × Σ_{j} L_window_j
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ========================================
# 補助関数: マスク付き平均
# ========================================
def reduce_masked_mean(
    data: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    マスク付き平均値の計算

    ========================================
    Shape
    ========================================
    data: (B, S, N) 損失値
    mask: (B, S, N) 有効マスク (1=有効, 0=無効)
    dim:  None (全体平均) or int

    ========================================
    処理
    ========================================
    mean = sum(data * mask) / max(sum(mask), 1)

    ※ max(., 1) でゼロ除算を防止
    ※ mask は valid マスク (クエリフレーム以降のみ1, それ以前は0)
    """
    if dim is None:
        return (data * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    else:
        return (data * mask).sum(dim=dim) / torch.clamp(mask.sum(dim=dim), min=1.0)


# ========================================
# 1. 座標予測損失 (sequence_loss)
# ========================================
def sequence_loss(
    flow_preds: List[List[torch.Tensor]],
    flow_gt: List[torch.Tensor],
    valids: List[torch.Tensor],
    vis: Optional[List[torch.Tensor]] = None,
    gamma: float = 0.8,
    add_huber_loss: bool = False,
    loss_only_for_visible: bool = False,
) -> torch.Tensor:
    """
    座標予測に対する損失関数

    ========================================
    概要
    ========================================
    各ウィンドウ × 各反復の予測に対して、
    指数減衰重みを付けた座標損失の平均を計算。

    ========================================
    Shape
    ========================================
    flow_preds: List (num_windows) of List (M iters) of (B, S, N, 2)
    flow_gt:    List (num_windows) of (B, S, N, 2)
    valids:     List (num_windows) of (B, S, N)  ← 有効マスク
    vis:        List (num_windows) of (B, S, N)  ← 可視性GT (optional)

    ========================================
    計算フロー
    ========================================
    for j in windows:  # ウィンドウごと
      for i in iterations:  # 反復ごと
        weight = γ^{M - i - 1}

        # ピクセル単位の損失
        if add_huber_loss:
          loss = huber_loss(pred, gt, δ=6.0)  # 大きなずれにロバスト
        else:
          loss = |pred - gt|  # L1損失

        loss = mean(loss, dim=channel)  # (B, S, N, 2) → (B, S, N)

        # マスク付き平均
        if loss_only_for_visible:
          valid = valid * vis  # 可視点のみで損失計算
        loss = reduce_masked_mean(loss, valid)

        flow_loss += weight * loss

      flow_loss /= M
    total /= num_windows

    ========================================
    Huber Loss (δ=6.0)
    ========================================
    huber(x, y, δ) =
      0.5 * (x-y)² / δ        if |x-y| <= δ
      |x-y| - 0.5 * δ         otherwise

    → 大きな外れ値の影響を緩和
    → δ=6.0 → 6ピクセル以上のずれはL1的に扱う

    ========================================
    可視点 vs 不可視点の重み付け
    ========================================
    loss_only_for_visible=False (デフォルト):
    - 不可視点も損失計算に含める
    - ただし、不可視点は1/5の重みで計算 (論文の記述)
    - これにより、オクルージョン時の予測精度も向上

    loss_only_for_visible=True:
    - 可視点のみで損失計算
    - valid = valid * vis
    """
    total_flow_loss = 0.0

    for j in range(len(flow_gt)):
        # --- ウィンドウ j ---
        B, S, N, D = flow_gt[j].shape  # D = 2 (x, y)
        n_predictions = len(flow_preds[j])  # M回の反復予測

        flow_loss = 0.0
        for i in range(n_predictions):
            # 指数減衰重み: 最後の予測が最も重い
            i_weight = gamma ** (n_predictions - i - 1)

            flow_pred = flow_preds[j][i]  # (B, S, N, 2)

            # 損失計算
            if add_huber_loss:
                i_loss = huber_loss(flow_pred, flow_gt[j], delta=6.0)  # (B, S, N, 2)
            else:
                i_loss = (flow_pred - flow_gt[j]).abs()  # L1損失: (B, S, N, 2)

            # チャネル方向の平均: (B, S, N, 2) → (B, S, N)
            i_loss = torch.mean(i_loss, dim=3)

            # マスク処理
            valid_ = valids[j].clone()  # (B, S, N)
            if loss_only_for_visible and vis is not None:
                valid_ = valid_ * vis[j]  # 可視点のみ

            # マスク付き平均
            flow_loss += i_weight * reduce_masked_mean(i_loss, valid_)

        # 反復数で平均化
        flow_loss = flow_loss / n_predictions
        total_flow_loss += flow_loss

    # ウィンドウ数で平均化
    return total_flow_loss / len(flow_gt)


# ========================================
# Huber Loss
# ========================================
def huber_loss(
    x: torch.Tensor, y: torch.Tensor, delta: float = 6.0
) -> torch.Tensor:
    """
    要素ごとのHuber Loss

    ========================================
    数式
    ========================================
    L(x, y, δ) =
      0.5 * (x - y)²                    if |x - y| <= δ
      δ * (|x - y| - 0.5 * δ)          otherwise

    → |x - y| <= δ: 二次関数 (MSE的)
    → |x - y| > δ:  線形関数 (L1的)

    ========================================
    Shape
    ========================================
    入力: x, y: (B, S, N, 2)
    出力: (B, S, N, 2)

    ========================================
    δ=6.0 の意味
    ========================================
    6ピクセル以内の誤差 → 二次的に罰する (精密な位置合わせ)
    6ピクセル以上の誤差 → 線形に罰する (外れ値のロバスト性)
    """
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).float()
    return flag * 0.5 * diff ** 2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


# ========================================
# 2. 可視性損失 (sequence_BCE_loss)
# ========================================
def sequence_BCE_loss(
    vis_preds: List[List[torch.Tensor]],
    vis_gts: List[torch.Tensor],
) -> torch.Tensor:
    """
    可視性の Binary Cross-Entropy 損失

    ========================================
    概要
    ========================================
    各ウィンドウ × 各反復の可視性予測に対して BCE を計算。
    ※ CoTracker3では sigmoid(vis_pred) が既に適用された値を受け取る。

    ========================================
    Shape
    ========================================
    vis_preds: List (num_windows) of List (M iters) of (B, S, N)
               ※ sigmoid済み [0, 1]
    vis_gts:   List (num_windows) of (B, S, N)
               ※ 1=可視, 0=不可視

    ========================================
    計算
    ========================================
    for j in windows:
      for i in iterations:
        loss += BCE(vis_pred[j][i], vis_gt[j])
      loss /= M
    total /= num_windows

    ========================================
    注意点
    ========================================
    - 指数減衰重みなし (全反復で同じ重み)
    - CoTracker2との違い: CoTracker2は最終反復のvis_predのみで計算
    - CoTracker3は全反復のsigmoid(vis)で計算
    """
    total_bce_loss = 0.0

    for j in range(len(vis_preds)):
        n_predictions = len(vis_preds[j])
        bce_loss = 0.0

        for i in range(n_predictions):
            vis_loss = F.binary_cross_entropy(
                vis_preds[j][i],  # sigmoid済み: (B, S, N)
                vis_gts[j],       # GT: (B, S, N)
            )
            bce_loss += vis_loss

        bce_loss = bce_loss / n_predictions
        total_bce_loss += bce_loss

    return total_bce_loss / len(vis_preds)


# ========================================
# 3. 信頼度損失 (sequence_prob_loss)
# ========================================
def sequence_prob_loss(
    tracks: List[List[torch.Tensor]],
    confidence: List[List[torch.Tensor]],
    target_points: List[torch.Tensor],
    visibility: List[torch.Tensor],
    expected_dist_thresh: float = 12.0,
) -> torch.Tensor:
    """
    信頼度の Binary Cross-Entropy 損失

    ========================================
    概要 (論文 Section 3.3)
    ========================================
    予測座標がGTから12ピクセル以内かどうかを分類する損失。
    "12ピクセル以上離れた点は使い物にならない"という仮定に基づく。

    ========================================
    Shape
    ========================================
    tracks:        List (windows) of List (M iters) of (B, S, N, 2)
                   ※ 座標予測 (画像スケール)
    confidence:    List (windows) of List (M iters) of (B, S, N)
                   ※ sigmoid済み信頼度 [0, 1]
    target_points: List (windows) of (B, S, N, 2)
                   ※ GT座標 (画像スケール)
    visibility:    List (windows) of (B, S, N)
                   ※ GT可視性
    expected_dist_thresh: 12.0 ピクセル

    ========================================
    計算フロー
    ========================================
    for j in windows:
      for i in iterations:
        # 1. 座標誤差の計算 (detach: 座標の勾配を信頼度に流さない)
        err = sum((pred.detach() - gt)^2, dim=-1)   # (B, S, N)

        # 2. 12ピクセルしきい値で二値ラベル生成
        valid = (err <= 12^2).float()                 # (B, S, N)
        # err <= 144 → valid=1 (信頼できる)
        # err > 144  → valid=0 (信頼できない)

        # 3. 信頼度のBCE損失
        loss = BCE(confidence, valid)                 # (B, S, N)

        # 4. 可視点のみで計算 (不可視点の信頼度は学習しない)
        loss = loss * visibility                      # マスク

        # 5. バッチ・時間・点の平均
        loss = mean(loss, dim=[1, 2])                 # (B,)

    ========================================
    detach() の重要性
    ========================================
    tracks[j][i].detach() により:
    - 座標予測の勾配が信頼度ヘッドに逆流しない
    - 信頼度は「現在の予測の良し悪し」を独立に学習
    - 座標更新とは独立した信頼度キャリブレーション

    ========================================
    12ピクセルしきい値の意味
    ========================================
    TAP-Vid ベンチマークの標準的なしきい値。
    - 12px以内: 追跡成功とみなす
    - 12px以上: オクルージョンマーキングした方が
      Jaccard指標が改善する
    """
    total_logprob_loss = 0.0

    for j in range(len(tracks)):
        n_predictions = len(tracks[j])
        logprob_loss = 0.0

        for i in range(n_predictions):
            # 1. 座標誤差 (二乗距離)
            err = torch.sum(
                (tracks[j][i].detach() - target_points[j]) ** 2,
                dim=-1,
            )
            # err: (B, S, N)  ※ ピクセル二乗距離

            # 2. しきい値判定
            valid = (err <= expected_dist_thresh ** 2).float()
            # valid: (B, S, N)
            # err <= 144 → 1.0 (12px以内)
            # err > 144  → 0.0 (12px以上)

            # 3. BCE損失
            logprob = F.binary_cross_entropy(
                confidence[j][i],  # sigmoid済み: (B, S, N)
                valid,             # 自動生成ラベル: (B, S, N)
                reduction="none",
            )
            # logprob: (B, S, N)

            # 4. 可視性マスク
            logprob = logprob * visibility[j]  # 不可視点は損失0
            # logprob: (B, S, N)

            # 5. 平均 (時間とトラック方向)
            logprob = torch.mean(logprob, dim=[1, 2])  # (B,)
            logprob_loss += logprob

        logprob_loss = logprob_loss / n_predictions
        total_logprob_loss += logprob_loss

    return total_logprob_loss / len(tracks)


# ========================================
# 全体損失の計算デモ
# ========================================
def demo_loss_computation():
    """
    CoTracker3の損失計算デモ

    ========================================
    訓練データの構造
    ========================================
    model.forward(video, queries, is_train=True) の出力:
      train_data = (
        all_coords_predictions,     # List[List[Tensor]]
        all_vis_predictions,        # List[List[Tensor]]
        all_confidence_predictions, # List[List[Tensor]]
        valid_mask,                 # Tensor
      )

    all_coords_predictions:
      [window_0_preds, window_1_preds, ...]
      window_i_preds = [iter_0_pred, iter_1_pred, iter_2_pred, iter_3_pred]
      iter_j_pred: (B, S_trimmed, N, 2) 画像座標

    ========================================
    GT データの構造
    ========================================
    trajectory:  (B, T, N, 2)  GT座標
    visibility:  (B, T, N)     GT可視性 (0 or 1)
    valid:       (B, T, N)     有効マスク (クエリフレーム以降=1)
    """
    print("=" * 60)
    print("CoTracker3 損失関数 デモ")
    print("=" * 60)

    # === ダミーデータ生成 ===
    B, S, N = 2, 16, 50
    M = 4  # 反復回数
    num_windows = 1

    # GT
    flow_gt = [torch.randn(B, S, N, 2) * 100]  # ピクセル座標
    vis_gt = [(torch.rand(B, S, N) > 0.2).float()]  # 80%可視
    valids = [torch.ones(B, S, N)]

    # 予測 (M回の反復)
    flow_preds = [[torch.randn(B, S, N, 2) * 100 for _ in range(M)]]
    vis_preds = [[torch.sigmoid(torch.randn(B, S, N)) for _ in range(M)]]
    conf_preds = [[torch.sigmoid(torch.randn(B, S, N)) for _ in range(M)]]

    # === 1. 座標損失 ===
    print("\n[1. 座標損失 (sequence_loss)]")
    coord_loss_l1 = sequence_loss(
        flow_preds, flow_gt, valids,
        gamma=0.8, add_huber_loss=False,
    )
    coord_loss_huber = sequence_loss(
        flow_preds, flow_gt, valids,
        gamma=0.8, add_huber_loss=True,
    )
    print(f"  L1 Loss:    {coord_loss_l1.item():.4f}")
    print(f"  Huber Loss: {coord_loss_huber.item():.4f}")

    # 指数減衰重みの表示
    print(f"\n  指数減衰重み (γ=0.8, M={M}):")
    for m in range(M):
        w = 0.8 ** (M - m - 1)
        print(f"    反復{m+1}: γ^{M-m-1} = {w:.3f}")

    # === 2. 可視性損失 ===
    print(f"\n[2. 可視性損失 (sequence_BCE_loss)]")
    vis_loss = sequence_BCE_loss(vis_preds, vis_gt)
    print(f"  BCE Loss: {vis_loss.item():.4f}")

    # === 3. 信頼度損失 ===
    print(f"\n[3. 信頼度損失 (sequence_prob_loss)]")
    prob_loss = sequence_prob_loss(
        flow_preds, conf_preds, flow_gt, vis_gt,
        expected_dist_thresh=12.0,
    )
    print(f"  Prob Loss: {prob_loss.mean().item():.4f}")
    print(f"  しきい値: 12.0 ピクセル (12² = 144 二乗距離)")

    # === Huber Loss の挙動 ===
    print(f"\n[Huber Loss の挙動 (δ=6.0)]")
    errors = [1.0, 3.0, 6.0, 10.0, 20.0, 50.0]
    for e in errors:
        x = torch.tensor([e])
        y = torch.tensor([0.0])
        h = huber_loss(x, y, delta=6.0)
        l1 = (x - y).abs()
        print(f"  |error|={e:5.1f}: Huber={h.item():8.2f}, L1={l1.item():5.1f}")

    # === 全体損失 ===
    print(f"\n[全体損失の構成]")
    print(f"  L_total = λ_coord × L_coord + λ_vis × L_vis + λ_conf × L_conf")
    print(f"")
    print(f"  Synthetic訓練:")
    print(f"    - L_coord: Huber Loss (γ=0.8)")
    print(f"    - L_vis:   BCE Loss")
    print(f"    - L_conf:  Prob Loss (12px)")
    print(f"")
    print(f"  Pseudo-Label訓練:")
    print(f"    - L_coord: Huber Loss (γ=0.8)")
    print(f"    - L_vis:   フリーズ (vis_conf_head固定)")
    print(f"    - L_conf:  フリーズ (vis_conf_head固定)")
    print(f"")
    print(f"  不可視点の重み付け:")
    print(f"    - 可視点:   重み 1.0")
    print(f"    - 不可視点: 重み 0.2 (1/5)")


if __name__ == "__main__":
    demo_loss_computation()
