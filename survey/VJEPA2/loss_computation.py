"""
V-JEPA 2.1 Dense Prediction Loss - 簡略化疑似コード
=====================================================

V-JEPA 2.1の損失関数は2つのコンポーネントから成る:

  L_dense = L_predict + λ * L_context

  L_predict: マスクトークン予測損失 (V-JEPA 2 と同じ)
    - Predictorがマスクされた位置を正しく予測できるか
    - L1損失: (1/|M|) Σ_{i∈M} |z_pred_i - h_i|

  L_context: コンテキストトークン予測損失 (V-JEPA 2.1 の新規追加)
    - Predictorが可視トークンの表現も予測できるか (Dense特徴の強制)
    - 距離重み付きL1損失: (1/|C|) Σ_{i∈C} λ_i |z_ctx_i - h_i|
    - λ_i = λ / sqrt(d_min(i, M))  ← マスクに近いほど高い重み

  両損失ともTeacherエンコーダ出力 h を正解ターゲットとして使用。
  hにはstop-gradient (torch.no_grad) が適用されている。

対応する公式実装:
  - app/vjepa_2_1/train.py の loss_fn 関数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def apply_masks(x: torch.Tensor, masks: list, concat: bool = True):
    """
    マスクインデックスに対応するトークンを選択する。

    入力:
        x:      (B, N, D)
        masks:  list of (B, K)  K: 選択するトークン数

    出力:
        concat=True:  (B*len(masks), K, D)
        concat=False: list of (B, K, D)
    """
    all_x = []
    for m in masks:
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))  # (B, K, D)
        all_x.append(torch.gather(x, dim=1, index=idx))
    if concat:
        return torch.cat(all_x, dim=0)
    return all_x


# ============================================================
# L_predict: マスクトークン予測損失 (元のV-JEPA損失)
# ============================================================

class MaskedPredictionLoss(nn.Module):
    """
    マスクトークン予測損失 (V-JEPA 2 / V-JEPA 2.1 共通)

    Predictorが予測したマスクトークン z_pred と
    Teacherエンコーダの対応出力 h の差を最小化する。

    数式:
        L_predict = (1/|M|) Σ_{i∈M} |z_pred_i - sg(h_i)|^exp / exp
        (exp=1.0 の場合: L1損失)

    入力:
        z_pred:     (B*n_masks, N_pred, D_out)  Predictorのマスク予測
        h:          (B, N_total, D)             Teacherエンコーダ出力 (全トークン)
        masks_pred: list of (B, N_pred)          ターゲットインデックス
        loss_exp:   損失の指数 (通常1.0 = L1)

    出力:
        loss: スカラー
    """

    def __init__(self, loss_exp: float = 1.0):
        super().__init__()
        self.loss_exp = loss_exp

    def forward(
        self,
        z_pred: torch.Tensor,       # (B*n_masks, N_pred, D_out)
        h: list,                    # list of (B, N_total, D) ← マルチレベル対応
        masks_pred: list,           # list of (B, N_pred)
    ) -> torch.Tensor:
        """
        入力:
            z_pred:     list of list of (B, N_pred, D)
                        外側: マスク設定数, 内側: レベル数
            h:          list of (B, N_total, D)   Teacherの各レベル出力
            masks_pred: list of (B, N_pred)

        注意:
            公式実装ではz_predはネストされたlist構造:
            z_pred[mask_idx][level_idx]: (B, N_pred, D)
        """
        # ターゲット h からマスク位置のトークンを選択
        # h_pred[i]: list of (B, N_pred, D) ← apply_masks(concat=False)で分割
        h_target = [
            apply_masks(hi, masks_pred, concat=False)
            for hi in h
        ]
        # h_target: list(levels) of list(n_masks) of (B, N_pred, D)

        loss = 0.0
        n = 0
        for zi, hi_list in zip(z_pred, h_target):
            # zi: list of (B, N_pred, D) [1つのマスク設定の各レベル]
            # hi_list: list of (B, N_pred, D) [各マスクの各レベル]
            for zij, hij in zip(zi, hi_list):
                # zij: (B, N_pred, D)
                # hij: (B, N_pred, D)
                loss += (
                    torch.mean(torch.abs(zij - hij) ** self.loss_exp)
                    / self.loss_exp
                )
                n += 1

        return loss / max(n, 1)


# ============================================================
# L_context: コンテキストトークン予測損失 (V-JEPA 2.1 新規)
# ============================================================

class ContextPredictionLoss(nn.Module):
    """
    コンテキストトークン予測損失 (V-JEPA 2.1 の核心的な追加)

    Predictorが出力したコンテキストトークン z_context と
    Teacherエンコーダの対応出力 h の差を最小化する。

    距離重み λ_i により、マスク境界付近のトークンに高い重みをかける:
        λ_i = λ / sqrt(d_min(i, M))

    数式:
        L_context = (1/|C|) Σ_{i∈C} λ_i * |z_ctx_i - sg(h_i)|^exp / exp

    入力:
        z_context:  list of list of (B, N_ctx, D)  Predictorのコンテキスト出力
        h:          list of (B, N_total, D)        Teacherエンコーダ出力
        masks_enc:  list of (B, N_ctx)             コンテキストインデックス
        d_weights:  list of list of (B, N_ctx)     距離重み (None=一様重み)
        loss_exp:   損失の指数

    出力:
        loss: スカラー
    """

    def __init__(self, loss_exp: float = 1.0):
        super().__init__()
        self.loss_exp = loss_exp

    def forward(
        self,
        z_context: list,           # list(mask_configs) of list(levels) of (B, N_ctx, D)
        h: list,                   # list(levels) of (B, N_total, D)
        masks_enc: list,           # list of (B, N_ctx)
        d_weights: list = None,    # list(mask_configs) of list(B) of (N_ctx,)  or None
    ) -> torch.Tensor:
        """
        距離重み付きL1損失を計算する。

        d_weights が None の場合は一様重み (通常の L1)。
        d_weights がある場合は各トークンに重みをかける。
        """
        # ターゲット h からコンテキスト位置のトークンを選択
        h_target = [
            apply_masks(hi, masks_enc, concat=False)
            for hi in h
        ]
        # h_target: list(levels) of list(n_masks) of (B, N_ctx, D)

        loss = 0.0
        n = 0

        for i_mask, (zi, hi_list) in enumerate(zip(z_context, h_target)):
            # zi: list of (B, N_ctx, D) [1つのマスク設定の各レベル]
            for j_level, (zij, hij) in enumerate(zip(zi, hi_list)):
                # zij: (B, N_ctx, D)
                # hij: (B, N_ctx, D)

                if d_weights is not None:
                    # 距離重み付き損失
                    # d_weights[i_mask]: list(B) of (N_ctx,)
                    # → バッチを結合して (B, N_ctx) に
                    dw = torch.stack(d_weights[i_mask], dim=0)  # (B, N_ctx)
                    dw_inv = (1.0 / dw).unsqueeze(2)            # (B, N_ctx, 1)

                    # |zij - hij|^exp * (1/d_i)
                    loss_n = torch.abs(zij - hij) ** self.loss_exp * dw_inv
                    loss += torch.mean(loss_n) / self.loss_exp
                else:
                    # 一様重み (通常のL1)
                    loss += (
                        torch.mean(torch.abs(zij - hij) ** self.loss_exp)
                        / self.loss_exp
                    )
                n += 1

        return loss / max(n, 1)


# ============================================================
# Dense Prediction Loss: L_predict + λ * L_context
# ============================================================

class DensePredictionLoss(nn.Module):
    """
    V-JEPA 2.1 の Dense Prediction Loss の統合クラス

    L_dense = L_predict + λ(t) * L_context

    λ(t) は Progressive Warmup スケジュールで段階的に増加:
      - epoch 0~50:   λ(t) = 0
      - epoch 50~100: λ(t) = 0 → λ (線形増加)
      - epoch 100~:   λ(t) = λ (固定)

    入力:
        z_pred:     list of list of (B, N_pred, D)  マスク予測
        z_context:  list of list of (B, N_ctx, D)   コンテキスト予測
        h:          list of (B, N_total, D)          Teacher出力 (各レベル)
        masks_pred: list of (B, N_pred)
        masks_enc:  list of (B, N_ctx)
        d_weights:  list of list of (B, N_ctx) or None
        epoch, total_itr: スケジュール計算用
    """

    def __init__(
        self,
        loss_exp: float = 1.0,
        lambda_value: float = 0.5,
        lambda_progressive: bool = True,
        warmup_start_epoch: int = 50,   # λのwarmup開始epoch
        warmup_end_epoch: int = 100,    # λのwarmup終了epoch
        weight_distance: bool = True,   # 距離重みを使用するか
    ):
        super().__init__()
        self.predict_loss = MaskedPredictionLoss(loss_exp)
        self.context_loss = ContextPredictionLoss(loss_exp)
        self.lambda_value = lambda_value
        self.lambda_progressive = lambda_progressive
        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_end_epoch = warmup_end_epoch
        self.weight_distance = weight_distance

    def get_lambda(self, epoch: int) -> float:
        """
        Progressive Warmup スケジュールに従って λ を計算する。

        入力:
            epoch: 現在のエポック数

        出力:
            lambda_step: 現在ステップでのλ値
        """
        if not self.lambda_progressive:
            return self.lambda_value

        if epoch < self.warmup_start_epoch:
            return 0.0
        elif epoch >= self.warmup_end_epoch:
            return self.lambda_value
        else:
            # 線形 warmup
            progress = (epoch - self.warmup_start_epoch) / (
                self.warmup_end_epoch - self.warmup_start_epoch
            )
            return self.lambda_value * progress

    def forward(
        self,
        z_pred: list,
        z_context: list,
        h: list,
        masks_pred: list,
        masks_enc: list,
        d_weights: list = None,
        epoch: int = 0,
    ) -> dict:
        """
        Dense Prediction Loss を計算する。

        入力:
            z_pred:     list(mask_cfgs) of list(levels) of (B, N_pred, D)
            z_context:  list(mask_cfgs) of list(levels) of (B, N_ctx, D)
            h:          list(levels) of (B, N_total, D)  Teacher出力
            masks_pred: list of (B, N_pred)
            masks_enc:  list of (B, N_ctx)
            d_weights:  list(mask_cfgs) of list(B) of (N_ctx,)  or None
            epoch:      現在エポック数 (λスケジュール用)

        出力:
            dict:
              "loss_total":   スカラー   総損失
              "loss_predict": スカラー   L_predict のみ
              "loss_context": スカラー   L_context のみ (重みなし)
              "lambda":       float      現在の λ 値
        """
        # L_predict: マスクトークン予測損失
        loss_pred = self.predict_loss(z_pred, h, masks_pred)

        # L_context: コンテキストトークン予測損失
        dw = d_weights if self.weight_distance else None
        loss_ctx = self.context_loss(z_context, h, masks_enc, dw)

        # λのスケジュール
        lambda_step = self.get_lambda(epoch)

        # 合計損失
        loss_total = loss_pred + lambda_step * loss_ctx

        return {
            "loss_total":   loss_total,
            "loss_predict": loss_pred,
            "loss_context": loss_ctx,
            "lambda":       lambda_step,
        }


# ============================================================
# LayerNorm正規化ユーティリティ
# ============================================================

def normalize_multilevel(h: list, embed_dim: int) -> list:
    """
    マルチレベルエンコーダ出力を各レベルでLayerNorm正規化する。

    V-JEPA 2.1では4つの中間層出力を連結するが、
    各レベルを個別にLayerNormしてから連結する。

    入力:
        h:         list of (B, N, D*K) または (B, N, D)
                   K層分が連結されたTensor
        embed_dim: 1レベルの次元 D

    出力:
        list of (B, N, D*K) 各レベルがLN正規化済み
    """
    normalized = []
    for hi in h:
        if hi.shape[-1] == embed_dim:
            # 単一レベル
            normalized.append(F.layer_norm(hi, (embed_dim,)))
        else:
            # 複数レベル連結: 各スライスを個別にLN
            K = hi.shape[-1] // embed_dim
            chunks = []
            for k in range(K):
                chunk = hi[..., k * embed_dim:(k + 1) * embed_dim]  # (B, N, D)
                chunks.append(F.layer_norm(chunk, (embed_dim,)))
            normalized.append(torch.cat(chunks, dim=-1))  # (B, N, D*K)
    return normalized


# ============================================================
# 動作確認 example
# ============================================================

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("V-JEPA 2.1 Dense Prediction Loss 動作確認")
    print("=" * 60)

    # ----------------------------------------
    # テスト設定
    # ----------------------------------------
    B = 2
    N_total = 2048   # 総パッチ数
    N_ctx = 700      # コンテキストパッチ数
    N_pred = 1200    # ターゲットパッチ数
    D = 1024         # 次元
    K = 4            # 中間層数 (Deep Self-Supervision)

    # ランダムなインデックス生成
    perm = torch.randperm(N_total)
    ctx_idx  = perm[:N_ctx].unsqueeze(0).expand(B, -1)    # (B, N_ctx)
    pred_idx = perm[N_ctx:N_ctx + N_pred].unsqueeze(0).expand(B, -1)  # (B, N_pred)
    masks_enc  = [ctx_idx]
    masks_pred = [pred_idx]

    # Teacher出力 (各レベル, 全トークン)
    h = [torch.randn(B, N_total, D * K) for _ in range(1)]  # 1つのマスク設定に対応

    # ----------------------------------------
    # L_predict のテスト
    # ----------------------------------------
    print("\n[1] L_predict (マスクトークン予測損失)")
    predict_loss_fn = MaskedPredictionLoss(loss_exp=1.0)

    # Predictor出力 (マスク用): 1マスク設定 × K レベル
    z_pred = [[torch.randn(B, N_pred, D * K)]]  # list(1 mask) of list(K levels)
    # ただし簡略化: D*K次元のまま1レベルとして扱う
    z_pred_simple = [[torch.randn(B, N_pred, D * K)]]

    loss_p = predict_loss_fn(z_pred_simple, h, masks_pred)
    print(f"  z_pred[0][0]: {z_pred_simple[0][0].shape}")
    print(f"  h[0]:         {h[0].shape}")
    print(f"  L_predict:    {loss_p.item():.4f}")
    assert loss_p.item() >= 0

    # ----------------------------------------
    # L_context のテスト (距離重みなし)
    # ----------------------------------------
    print("\n[2] L_context (コンテキストトークン予測損失、均一重み)")
    context_loss_fn = ContextPredictionLoss(loss_exp=1.0)

    z_context = [[torch.randn(B, N_ctx, D * K)]]  # list(1 mask) of list(1 level)

    loss_c = context_loss_fn(z_context, h, masks_enc, d_weights=None)
    print(f"  z_context[0][0]: {z_context[0][0].shape}")
    print(f"  L_context:       {loss_c.item():.4f}")
    assert loss_c.item() >= 0

    # ----------------------------------------
    # L_context のテスト (距離重みあり)
    # ----------------------------------------
    print("\n[3] L_context (距離重み付き、λ_i = λ/sqrt(d_min))")
    # 各バッチ要素に対して距離重み (N_ctx,) を生成
    d_weights = [[
        torch.rand(N_ctx) * 5.0 + 1.0  # d_min ∈ [1, 6] のランダム距離
        for _ in range(B)
    ]]

    loss_c_weighted = context_loss_fn(z_context, h, masks_enc, d_weights=d_weights)
    print(f"  d_weights[0][0]: {d_weights[0][0].shape}")
    print(f"  L_context (重み付き): {loss_c_weighted.item():.4f}")

    # ----------------------------------------
    # Dense Prediction Loss 統合テスト
    # ----------------------------------------
    print("\n[4] Dense Prediction Loss 統合 (L_dense = L_pred + λ * L_ctx)")
    dense_loss = DensePredictionLoss(
        loss_exp=1.0,
        lambda_value=0.5,
        lambda_progressive=True,
        warmup_start_epoch=50,
        warmup_end_epoch=100,
        weight_distance=True,
    )

    # epoch 0: λ=0 (warmupまだ始まっていない)
    result_e0 = dense_loss(z_pred_simple, z_context, h, masks_pred, masks_enc,
                            d_weights, epoch=0)
    print(f"  epoch=0:   λ={result_e0['lambda']:.3f}, L_total={result_e0['loss_total'].item():.4f}")
    assert result_e0["lambda"] == 0.0

    # epoch 75: λ=0.25 (warmup中)
    result_e75 = dense_loss(z_pred_simple, z_context, h, masks_pred, masks_enc,
                             d_weights, epoch=75)
    print(f"  epoch=75:  λ={result_e75['lambda']:.3f}, L_total={result_e75['loss_total'].item():.4f}")
    assert abs(result_e75["lambda"] - 0.25) < 1e-6

    # epoch 100+: λ=0.5 (完全)
    result_e200 = dense_loss(z_pred_simple, z_context, h, masks_pred, masks_enc,
                              d_weights, epoch=200)
    print(f"  epoch=200: λ={result_e200['lambda']:.3f}, L_total={result_e200['loss_total'].item():.4f}")
    assert result_e200["lambda"] == 0.5

    # ----------------------------------------
    # LayerNorm正規化テスト
    # ----------------------------------------
    print("\n[5] マルチレベル LayerNorm 正規化")
    h_multilevel = [torch.randn(B, N_total, D * K)]  # (B, N, D*K)
    h_normalized = normalize_multilevel(h_multilevel, embed_dim=D)
    print(f"  入力:  {h_multilevel[0].shape}")
    print(f"  出力:  {h_normalized[0].shape}")
    # 各 D-blockが正規化されているか確認
    chunk0 = h_normalized[0][..., :D]  # 最初のレベル
    print(f"  chunk0 mean≈0: {chunk0.mean().item():.4f}, std≈1: {chunk0.std().item():.4f}")

    print("\n全テスト通過!")
