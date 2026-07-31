"""
V-JEPA 2.1 メインフロー - 簡略化疑似コード
==========================================

JEPA の自己教師あり学習全体フローを示す。

コンポーネント:
  - Encoder (Student): 可視トークンを処理し特徴を出力
  - Target Encoder (Teacher/EMA): 全トークンを処理し予測ターゲットを提供
  - Predictor: エンコーダ出力からマスク位置を予測
  - MaskCollator: 3Dブロックマスクを生成

V-JEPA 2.1 の学習ステップ:
  1. 動画/画像クリップを読み込む
  2. MaskCollatorで (masks_enc, masks_pred) を生成
  3. Target Encoder で全トークンの表現 h を計算 (no_grad, EMAパラメータ)
  4. Encoder で可視トークンの表現 z_enc を計算 (勾配あり)
  5. Predictor で z_pred, z_context を計算
  6. L_dense = L_predict + λ * L_context を計算
  7. backwardと最適化ステップ
  8. EMAでTarget Encoderを更新: θ_t ← m*θ_t + (1-m)*θ_s

対応する公式実装:
  - app/vjepa_2_1/train.py (main training loop)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from encoder import VisionTransformer, vit_large, vit_gigantic
from predictor import VisionTransformerPredictor
from mask_generator import MaskGenerator, MaskCollator, compute_mask_distance
from loss_computation import DensePredictionLoss, normalize_multilevel


def apply_masks(x: torch.Tensor, masks: list, concat: bool = True):
    """torch.gatherでマスクインデックスのトークンを選択"""
    all_x = []
    for m in masks:
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=idx))
    if concat:
        return torch.cat(all_x, dim=0)
    return all_x


# ============================================================
# VJEPA2 モデルクラス (Encoder + Predictor + EMA Target)
# ============================================================

class VJEPA2(nn.Module):
    """
    V-JEPA 2 / V-JEPA 2.1 の Student側モデル。

    Target Encoder (EMA Teacher) は外部で管理する。
    このクラスはEncoder + Predictorの順伝播のみを担当する。

    入力 (forward):
        clips:      list of (B, 3, T, H, W) or (B, 3, H, W)
                    (マルチFPCに対応するためlist)
        masks_enc:  list of list of (B, N_ctx)
                    外側: FPCごと, 内側: マスク設定ごと
        masks_pred: list of list of (B, N_pred)
        mod:        "video" or "image"

    出力:
        z_pred:     list(FPC) of list(mask_cfg) of list(levels) of (B, N_pred, D)
        z_context:  list(FPC) of list(mask_cfg) of list(levels) of (B, N_ctx, D)
                    (return_all_tokens=Trueの場合のみ)
    """

    def __init__(
        self,
        encoder: VisionTransformer,
        predictor: VisionTransformerPredictor,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(
        self,
        clips: list,        # list of (B, 3, T, H, W) or (B, 3, H, W)
        masks_enc: list,    # list of list of (B, N_ctx)
        masks_pred: list,   # list of list of (B, N_pred)
        mod: str = "video",
    ):
        """
        1クリップ分の順伝播。

        入力:
            clips[i]:        (B, 3, T, H, W) または (B, 3, H, W)
            masks_enc[i]:    list of (B, N_ctx)   マスク設定ごとのエンコーダマスク
            masks_pred[i]:   list of (B, N_pred)  マスク設定ごとのPredictorマスク

        出力:
            z_pred[i]:    list(mask_cfg) of list(levels) of (B, N_pred, D)
            z_context[i]: list(mask_cfg) of list(levels) of (B, N_ctx, D)
        """
        all_z_pred = []
        all_z_context = []

        for clip, me, mp in zip(clips, masks_enc, masks_pred):
            # ============================================
            # Step 1: エンコーダで可視トークンを処理
            # ============================================
            # clip: (B, 3, T, H, W) または (B, 3, H, W)
            # me:   list of (B, N_ctx)  各マスク設定の可視パッチインデックス

            # エンコーダ出力:
            #   out_layers=None の場合: (B*n_masks, N_ctx, D)
            #   out_layers指定の場合: list of (B*n_masks, N_ctx, D) ← 各中間層
            z_enc = self.encoder(clip, masks=me)

            # ============================================
            # Step 2: Predictorでマスク位置を予測
            # ============================================
            # z_pred:    (B, N_pred, D) or (B, N_pred, D*K)
            # z_context: (B, N_ctx, D) or (B, N_ctx, D*K)
            result = self.predictor(z_enc, me, mp, mod=mod)

            if isinstance(result, tuple):
                z_pred, z_context = result
            else:
                z_pred = result
                z_context = None

            all_z_pred.append(z_pred)
            all_z_context.append(z_context)

        return all_z_pred, all_z_context


# ============================================================
# EMA アップデート
# ============================================================

def update_ema(student: nn.Module, teacher: nn.Module, momentum: float):
    """
    Exponential Moving Average でTeacherエンコーダを更新する。

    θ_teacher ← m * θ_teacher + (1 - m) * θ_student

    入力:
        student:  Studentエンコーダ (勾配あり)
        teacher:  Teacherエンコーダ (勾配なし, frozen)
        momentum: EMAモメンタム m (通常 0.99925)

    実装ノート:
        公式実装では torch._foreach_mul_ と torch._foreach_add_ を使用
        (インプレース演算で高速化)
    """
    with torch.no_grad():
        student_params = list(student.parameters())
        teacher_params = list(teacher.parameters())

        # インプレース演算で効率的にEMAアップデート
        torch._foreach_mul_(teacher_params, momentum)
        torch._foreach_add_(teacher_params, student_params, alpha=1.0 - momentum)


# ============================================================
# Target Encoder (EMA Teacher) の順伝播
# ============================================================

def forward_target(
    clips: list,
    target_encoder: nn.Module,
    embed_dim: int,
    levels_predictor: int = 4,
) -> list:
    """
    Target Encoder (EMA Teacher) で全トークンを処理する。

    勾配なし。出力を LayerNorm で正規化する。

    入力:
        clips:           list of (B, 3, T, H, W) または (B, 3, H, W)
        target_encoder:  EMAエンコーダ (no_grad)
        embed_dim:       1レベルの次元 D
        levels_predictor: 中間層数 K

    出力:
        h: list of (B, N_total, D*K) または (B, N_total, D)
           各クリップのTeacher出力

    処理:
        1. target_encoder を全トークンに対して実行
        2. 各中間層出力を D ごとにLayerNorm
        3. 連結して (B, N, D*K) に
    """
    with torch.no_grad():
        all_h = []
        for clip in clips:
            # clip: (B, 3, T, H, W)
            # out_layers 指定時: list of (B, N, D)
            # 指定なし時: (B, N, D)
            h_raw = target_encoder(clip)  # list of (B, N, D) [各中間層]

            if isinstance(h_raw, list):
                # 複数中間層出力を LayerNorm → 連結
                h_normalized = []
                for hi in h_raw:
                    h_normalized.append(F.layer_norm(hi, (embed_dim,)))
                h = torch.cat(h_normalized, dim=-1)  # (B, N, D*K)
            else:
                # 単一出力
                h = F.layer_norm(h_raw, (embed_dim,))  # (B, N, D)

            all_h.append(h)

    return all_h  # list of (B, N, D*K)


# ============================================================
# 1ステップの学習処理
# ============================================================

def train_step(
    clips: list,                  # list of (B, 3, T, H, W)
    masks_enc: list,              # list of list of (B, N_ctx)
    masks_pred: list,             # list of list of (B, N_pred)
    model: VJEPA2,
    target_encoder: nn.Module,
    dense_loss_fn: DensePredictionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    momentum: float,
    embed_dim: int,
    device: torch.device,
    epoch: int = 0,
    grid_size: int = 16,
    weight_distance_loss: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    1ステップの学習を実行する。

    入力:
        clips:       list of (B, 3, T, H, W)  各FPCのクリップ
        masks_enc:   list(FPC) of list(mask_cfg) of (B, N_ctx)
        masks_pred:  list(FPC) of list(mask_cfg) of (B, N_pred)
        model:       VJEPA2 (encoder + predictor)
        target_encoder: EMA Teacher
        dense_loss_fn:  DensePredictionLoss
        optimizer:   AdamW等
        scaler:      混合精度用GradScaler
        momentum:    EMA更新係数 m
        embed_dim:   エンコーダの次元 D
        device:      "cuda:0" 等
        epoch:       現在エポック (λスケジュール用)
        grid_size:   空間グリッドサイズ (H/patch)
        weight_distance_loss: 距離重みを使うか
        dtype:       混合精度の型 (bfloat16/float16)

    出力:
        dict:
          "loss": float    損失値
          "lr":   float    現在学習率
    """
    with torch.cuda.amp.autocast(dtype=dtype, enabled=(dtype != torch.float32)):

        # ============================================
        # Step 1: Target Encoder で全トークン処理
        # ============================================
        # no_grad: Teacherは勾配なし
        h = forward_target(clips, target_encoder, embed_dim)
        # h: list(FPC) of (B, N_total, D) または (B, N_total, D*K)

        # ============================================
        # Step 2: Student Encoder + Predictor
        # ============================================
        mod = "video" if clips[0].ndim == 5 else "image"
        z_pred, z_context = model(clips, masks_enc, masks_pred, mod=mod)
        # z_pred[fpc_i]:    (B, N_pred, D) または list[levels] of (B, N_pred, D)
        # z_context[fpc_i]: (B, N_ctx, D)  または list[levels] of (B, N_ctx, D)

        # ============================================
        # Step 3: 距離重みの計算 (L_context用)
        # ============================================
        if weight_distance_loss:
            # 各FPCのマスクに対して距離重みを計算
            all_d_weights = []
            for me, mp in zip(masks_enc, masks_pred):
                d_w = compute_mask_distance(mp, me, grid_size)
                all_d_weights.append(d_w)
        else:
            all_d_weights = [None] * len(masks_enc)

        # ============================================
        # Step 4: 損失計算
        # ============================================
        total_loss = torch.tensor(0.0, device=device)

        for fpc_idx, (zp, zc, h_fpc, me, mp, dw) in enumerate(
            zip(z_pred, z_context, h, masks_enc, masks_pred, all_d_weights)
        ):
            # z_pred, z_context を list of list 形式に変換 (損失関数の期待する形式)
            # 簡略化: 単一レベル・単一マスク設定として扱う
            zp_wrapped = [[zp]] if not isinstance(zp, list) else zp
            zc_wrapped = [[zc]] if not isinstance(zc, list) else zc
            h_wrapped  = [h_fpc]

            result = dense_loss_fn(
                z_pred=zp_wrapped,
                z_context=zc_wrapped,
                h=h_wrapped,
                masks_pred=me,  # list of (B, N_pred)
                masks_enc=mp,   # list of (B, N_ctx)
                d_weights=dw,
                epoch=epoch,
            )
            total_loss = total_loss + result["loss_total"]

        total_loss = total_loss / len(clips)

    # ============================================
    # Step 5: 逆伝播とパラメータ更新
    # ============================================
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    # (勾配クリッピング等があれば ここで実施)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # ============================================
    # Step 6: EMA でTarget Encoderを更新
    # ============================================
    update_ema(model.encoder, target_encoder, momentum)

    current_lr = optimizer.param_groups[0]["lr"]
    return {"loss": total_loss.item(), "lr": current_lr}


# ============================================================
# 学習ループ (簡略版)
# ============================================================

def train_loop(
    model: VJEPA2,
    target_encoder: nn.Module,
    dataloader,
    num_epochs: int = 100,
    device: torch.device = None,
    embed_dim: int = 1024,
    base_lr: float = 5.25e-4,
    warmup_epochs: int = 10,
    ema_momentum: float = 0.99925,
    weight_distance_loss: bool = True,
    grid_size: int = 16,
    lambda_value: float = 0.5,
    lambda_warmup_start: int = 50,
    lambda_warmup_end: int = 100,
):
    """
    V-JEPA 2.1 の学習ループ (簡略版)。

    データローダーはMaskCollatorを使用していること:
        for sample in dataloader:
            # sample: list of (collated_batch, masks_enc_list, masks_pred_list)

    入力:
        model:          VJEPA2 (encoder + predictor)
        target_encoder: EMA Teacher (初期状態はencoderのコピー)
        dataloader:     MaskCollator付きデータローダー
        num_epochs:     エポック数
        device:         学習デバイス
        embed_dim:      エンコーダ次元 D
        base_lr:        基本学習率
        warmup_epochs:  学習率warmupエポック数
        ema_momentum:   EMAモメンタム m
        weight_distance_loss: 距離重みを使うか
        grid_size:      空間グリッドサイズ
        lambda_value:   コンテキスト損失係数 λ
        lambda_warmup_start: λのwarmup開始epoch
        lambda_warmup_end:   λのwarmup終了epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    target_encoder = target_encoder.to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    optimizer = AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=base_lr, weight_decay=0.04, betas=(0.9, 0.999),
    )
    scaler = torch.cuda.amp.GradScaler()

    # コサインスケジューラ (warmup込み)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dense_loss_fn = DensePredictionLoss(
        loss_exp=1.0,
        lambda_value=lambda_value,
        lambda_progressive=True,
        warmup_start_epoch=lambda_warmup_start,
        warmup_end_epoch=lambda_warmup_end,
        weight_distance=weight_distance_loss,
    )

    # ----------------------------
    # 学習ループ
    # ----------------------------
    for epoch in range(num_epochs):
        model.train()
        target_encoder.eval()

        epoch_loss = 0.0
        n_iters = 0

        for batch in dataloader:
            # batch: list of (collated_data, masks_enc_list, masks_pred_list)
            # 各要素が1つのFPCグループ

            clips = []
            all_masks_enc = []
            all_masks_pred = []

            for fpc_sample in batch:
                udata, masks_enc_list, masks_pred_list = fpc_sample
                # udata[0][0]: (B, 3, T, H, W) 動画テンソル
                clip = udata[0][0].to(device, non_blocking=True)
                me = [m.to(device) for m in masks_enc_list]
                mp = [m.to(device) for m in masks_pred_list]
                clips.append(clip)
                all_masks_enc.append(me)
                all_masks_pred.append(mp)

            result = train_step(
                clips=clips,
                masks_enc=all_masks_enc,
                masks_pred=all_masks_pred,
                model=model,
                target_encoder=target_encoder,
                dense_loss_fn=dense_loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                momentum=ema_momentum,
                embed_dim=embed_dim,
                device=device,
                epoch=epoch,
                grid_size=grid_size,
                weight_distance_loss=weight_distance_loss,
            )

            epoch_loss += result["loss"]
            n_iters += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_iters, 1)
        print(f"  Epoch [{epoch+1}/{num_epochs}] loss={avg_loss:.4f}, lr={result['lr']:.2e}")


# ============================================================
# モデル初期化ユーティリティ
# ============================================================

def build_vjepa2(
    model_name: str = "vit_large",
    img_size: int = 256,
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    pred_depth: int = 12,
    pred_embed_dim: int = 384,
    use_mask_tokens: bool = True,
    num_mask_tokens: int = 10,   # マスク設定数 × FPC数
    out_layers: list = None,     # Deep Self-Supervision用中間層インデックス
    modality_embedding: bool = False,
    levels_encoder: int = 1,
    return_all_tokens: bool = True,  # V-JEPA 2.1: コンテキストも予測
):
    """
    V-JEPA 2 / V-JEPA 2.1 モデルを構築する。

    入力:
        model_name:     "vit_large" / "vit_gigantic" 等
        img_size:       入力解像度 (px)
        patch_size:     パッチサイズ (px)
        num_frames:     クリップフレーム数
        tubelet_size:   チューブレットサイズ
        pred_depth:     Predictorのレイヤー数
        pred_embed_dim: Predictor内部次元
        out_layers:     [None] for V-JEPA 2, [l1, l2, l3, l4] for V-JEPA 2.1

    出力:
        (encoder, predictor, target_encoder)
    """
    EMBED_DIMS = {
        "vit_base": 768,
        "vit_large": 1024,
        "vit_giant": 1408,
        "vit_gigantic": 1664,
    }
    embed_dim = EMBED_DIMS.get(model_name, 1024)

    # Encoder
    encoder = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        depth={"vit_base": 12, "vit_large": 24, "vit_giant": 40, "vit_gigantic": 48}[model_name],
        num_heads={"vit_base": 12, "vit_large": 16, "vit_giant": 22, "vit_gigantic": 26}[model_name],
        out_layers=out_layers,
        modality_embedding=modality_embedding,
    )

    # Target Encoder (EMA) = Encoderのディープコピー
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Predictor
    predictor = VisionTransformerPredictor(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        out_embed_dim=embed_dim,
        depth=pred_depth,
        num_heads=pred_embed_dim // 64,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens,
        return_all_tokens=return_all_tokens,
        modality_embedding=modality_embedding,
        levels_encoder=levels_encoder,
    )

    return encoder, predictor, target_encoder


# ============================================================
# 動作確認 example
# ============================================================

if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("V-JEPA 2.1 メインフロー 動作確認")
    print("=" * 60)

    device = torch.device("cpu")  # テストはCPUで実行
    B = 2
    T, H, W = 16, 256, 256

    # ----------------------------------------
    # V-JEPA 2.1 モデル構築 (ViT-L ベース、簡略化)
    # ----------------------------------------
    print("\n[1] モデル構築 (ViT-L 簡略版, depth=4)")
    encoder = VisionTransformer(
        img_size=256, patch_size=16, num_frames=16, tubelet_size=2,
        embed_dim=1024, depth=4, num_heads=16,
        out_layers=[1, 2, 3],  # Deep Self-Supervision: 3中間層 + 最終層 = 4レベル
        modality_embedding=True,
    )
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    predictor = VisionTransformerPredictor(
        img_size=256, patch_size=16, num_frames=16, tubelet_size=2,
        embed_dim=1024, predictor_embed_dim=384, out_embed_dim=1024,
        depth=4, num_heads=8,
        use_mask_tokens=True, num_mask_tokens=2,
        return_all_tokens=True,
        modality_embedding=True,
        levels_encoder=4,   # 4レベル連結入力
    )

    model = VJEPA2(encoder, predictor)
    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")
    print(f"  Predictor params: {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M")

    # ----------------------------------------
    # マスク生成
    # ----------------------------------------
    print("\n[2] マスク生成")
    N_total = (16 // 2) * (256 // 16) * (256 // 16)  # = 2048
    N_ctx = 700
    N_pred = 1200
    perm = torch.randperm(N_total)
    ctx_idx  = perm[:N_ctx].unsqueeze(0).expand(B, -1)
    pred_idx = perm[N_ctx:N_ctx + N_pred].unsqueeze(0).expand(B, -1)
    masks_enc  = [[ctx_idx],  [ctx_idx]]   # 2マスク設定 (同じを2つ)
    masks_pred = [[pred_idx], [pred_idx]]

    print(f"  N_total={N_total}, N_ctx={N_ctx}, N_pred={N_pred}")

    # ----------------------------------------
    # Teacher Forward Pass
    # ----------------------------------------
    print("\n[3] Target Encoder (Teacher) 順伝播")
    x_vid = torch.randn(B, 3, T, H, W)
    h = forward_target([x_vid], target_encoder, embed_dim=1024, levels_predictor=4)
    print(f"  入力:  {x_vid.shape}")
    print(f"  Teacher出力 h[0]: {h[0].shape}")
    # out_layers=[1,2,3] → 4レベル出力 → (B, N, D*4)
    assert h[0].shape == (B, N_total, 1024 * 4), f"Got {h[0].shape}"

    # ----------------------------------------
    # Student (Encoder + Predictor) Forward Pass
    # ----------------------------------------
    print("\n[4] Student (Encoder + Predictor) 順伝播")
    model.eval()
    with torch.no_grad():
        z_pred, z_context = model([x_vid], masks_enc[:1], masks_pred[:1], mod="video")

    print(f"  z_pred[0]:    {z_pred[0].shape}")
    print(f"  z_context[0]: {z_context[0].shape}")
    # predictor output: (B, N_pred, D_out) = (B, N_pred, 1024)
    assert z_pred[0].shape == (B, N_pred, 1024)
    assert z_context[0].shape == (B, N_ctx, 1024)

    # ----------------------------------------
    # 損失計算
    # ----------------------------------------
    print("\n[5] Dense Prediction Loss 計算")
    dense_loss = DensePredictionLoss(
        loss_exp=1.0,
        lambda_value=0.5,
        lambda_progressive=True,
        weight_distance=False,  # テスト用に距離重みはオフ
    )

    # 損失関数はネスト構造: list(mask_cfg) of list(levels)
    zp_wrapped = [[z_pred[0]]]
    zc_wrapped = [[z_context[0]]]
    h_wrapped  = [h[0]]

    result = dense_loss(
        z_pred=zp_wrapped,
        z_context=zc_wrapped,
        h=h_wrapped,
        masks_pred=masks_pred[0],
        masks_enc=masks_enc[0],
        d_weights=None,
        epoch=150,  # warmup後
    )
    print(f"  L_predict: {result['loss_predict'].item():.4f}")
    print(f"  L_context: {result['loss_context'].item():.4f}")
    print(f"  λ:         {result['lambda']:.3f}")
    print(f"  L_dense:   {result['loss_total'].item():.4f}")

    # ----------------------------------------
    # EMAアップデート
    # ----------------------------------------
    print("\n[6] EMA アップデート")
    m = 0.99925
    # アップデート前のパラメータを記録
    param_before = list(target_encoder.parameters())[0].data.clone()
    update_ema(model.encoder, target_encoder, m)
    param_after = list(target_encoder.parameters())[0].data.clone()
    # パラメータが変わっていることを確認
    assert not torch.allclose(param_before, param_after, atol=1e-8)
    print(f"  momentum={m}")
    print(f"  パラメータが更新されたことを確認 ✓")

    print("\n全テスト通過!")
