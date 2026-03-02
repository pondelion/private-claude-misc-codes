"""
SiT - 学習ループ + 推論パイプライン

対応:
  学習: https://github.com/willisma/SiT/blob/main/train.py
  推論: https://github.com/willisma/SiT/blob/main/sample.py
  引数: https://github.com/willisma/SiT/blob/main/train_utils.py

このファイルでは以下を疑似コードで説明:
1. 学習ループ (ImageNet + VAE + Interpolant Framework)
2. 推論パイプライン (ODE / SDE + CFG + VAEデコード)

========================================
DiTとの差異 (学習)
========================================
DiT:
  diffusion = GaussianDiffusion(1000)
  t = randint(0, 1000)           ← 離散
  loss = diffusion.training_losses(model, x, t, {"y": y})

SiT:
  transport = create_transport("Linear", "velocity")
  # t のサンプリングは transport.training_losses 内で自動
  loss = transport.training_losses(model, x, {"y": y})

→ transport.training_losses が t ~ U[0,1] のサンプリング、
  x_t の補間、速度場ターゲットの計算、損失計算を全て内包。

========================================
DiTとの差異 (推論)
========================================
DiT:
  p_sample_loop(model_fn, shape, noise, ...)  ← DDPM逆拡散

SiT:
  sampler = Sampler(transport)
  sample_fn = sampler.sample_ode(method="dopri5", num_steps=50)
  # or
  sample_fn = sampler.sample_sde(method="Euler", num_steps=250)
  samples = sample_fn(noise, model_fn, y=labels, cfg_scale=4.0)
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy

from models import SiT, SiT_configs
from transport import create_transport, Sampler


# ============================================================
# EMA (DiTと同一)
# ============================================================

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    """
    EMAモデルの更新

    ema_param = decay × ema_param + (1 - decay) × model_param

    対応: 公式 train.py L39-L49
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# ============================================================
# 学習ループ
# ============================================================

def train(args):
    """
    SiT学習メインループ

    ========================================
    学習設定 (論文デフォルト: SiT-XL/2)
    ========================================
    - データ: ImageNet (1.28M images, 1000 classes)
    - 入力: 256×256 → VAE Encode → 32×32×4 潜在
    - バッチサイズ: 256 (global)
    - オプティマイザ: AdamW (lr=1e-4, weight_decay=0, β=(0.9, 0.999))
    - スケジューラ: なし (定数学習率)
    - エポック: 1400 (≈ 7M steps on 8 GPUs)
    - EMA decay: 0.9999
    - CFG dropout: 10% (ラベルを無条件トークンに置換)

    ========================================
    DiTと全く同じ学習設定
    ========================================
    アーキテクチャ、ハイパーパラメータ、データ前処理、
    VAE、EMA、CFG dropout は全て DiT から変更なし。
    唯一変わるのは diffusion → transport のフレームワーク。

    ========================================
    SiT固有の追加引数
    ========================================
    --path-type:    Linear | GVP | VP         (デフォルト: Linear)
    --prediction:   velocity | score | noise  (デフォルト: velocity)
    --loss-weight:  none | velocity | likelihood (デフォルト: none)
    --train-eps:    学習時の端点ε              (auto)
    --sample-eps:   推論時の端点ε              (auto)

    対応: 公式 train.py L110-L315
    """
    # --- DDP初期化 (DiTと同一) ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # --- モデル作成 (DiTと同一構造) ---
    latent_size = args.image_size // 8  # 256 // 8 = 32
    model = SiT(
        input_size=latent_size,
        num_classes=args.num_classes,
        **SiT_configs[args.model]
    )
    ema = deepcopy(model).to(device)
    model = DDP(model.to(device), device_ids=[device])

    # --- ★ Transport (DiTの GaussianDiffusion に代わる) ---
    transport = create_transport(
        path_type=args.path_type,       # "Linear" (default)
        prediction=args.prediction,     # "velocity" (default)
        loss_weight=args.loss_weight,   # None (default)
        train_eps=args.train_eps,       # auto
        sample_eps=args.sample_eps,     # auto
    )
    # velocity + Linear の場合:
    #   train_eps = 0, sample_eps = 0
    #   path_sampler = ICPlan()
    #   model_type = ModelType.VELOCITY

    # 学習中のサンプリング用
    transport_sampler = Sampler(transport)

    # --- VAE (DiTと同一, 事前学習済み, 固定) ---
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae = None  # 疑似コードのため省略

    # --- オプティマイザ (DiTと同一) ---
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # --- データローダー (DiTと同一) ---
    # transform = Compose([CenterCrop(256), RandomHFlip, ToTensor, Normalize(0.5, 0.5)])
    # dataset = ImageFolder(args.data_path, transform=transform)
    # sampler = DistributedSampler(dataset)
    # loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)
    loader = []  # 疑似コードのため省略

    # --- EMA初期化 ---
    update_ema(ema, model.module, decay=0)  # 同期初期化
    model.train()
    ema.eval()

    # --- 学習中のサンプリング設定 ---
    sample_labels = torch.randint(1000, size=(local_batch_size,), device=device)
    sample_noise = torch.randn(local_batch_size, 4, latent_size, latent_size, device=device)

    use_cfg = args.cfg_scale > 1.0
    if use_cfg:
        sample_noise = torch.cat([sample_noise, sample_noise], 0)
        y_null = torch.tensor([1000] * local_batch_size, device=device)
        sample_labels_cfg = torch.cat([sample_labels, y_null], 0)
        sample_model_kwargs = dict(y=sample_labels_cfg, cfg_scale=args.cfg_scale)
        sample_model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=sample_labels)
        sample_model_fn = ema.forward

    # --- 学習ループ ---
    train_steps = 0
    for epoch in range(args.epochs):
        for x, y in loader:
            x = x.to(device)  # (B, 3, 256, 256)
            y = y.to(device)  # (B,) クラスラベル {0,...,999}

            # --- VAE Encode (DiTと同一) ---
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # x: (B, 4, 32, 32) ← VAE潜在空間 (×0.18215 正規化)

            # --- ★ 損失計算 (SiT固有) ---
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            # transport.training_losses 内部:
            #   1. t ~ U[0, 1]                     (B,)
            #   2. x0 ~ N(0, I)                    (B, 4, 32, 32)
            #   3. x_t = t×x_1 + (1-t)×x_0         (B, 4, 32, 32) [Linear]
            #   4. u_t = x_1 - x_0                  (B, 4, 32, 32) [Linear]
            #   5. pred = model(x_t, t, y)           (B, 4, 32, 32)
            #   6. loss = MSE(pred, u_t)             (B,) [velocity]

            loss = loss_dict["loss"].mean()
            # loss: スカラー

            # --- 更新 (DiTと同一) ---
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            train_steps += 1

            # --- チェックポイント保存 ---
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    # torch.save(checkpoint, f"checkpoints/{train_steps:07d}.pt")

            # --- 学習中サンプリング (SiT固有: ODE) ---
            if train_steps % args.sample_every == 0 and train_steps > 0:
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode()
                    # デフォルト: dopri5, 50ステップ, atol=1e-6, rtol=1e-3
                    samples = sample_fn(sample_noise, sample_model_fn, **sample_model_kwargs)[-1]
                    # samples[-1]: 最終時刻の出力
                    # samples: (2B, 4, 32, 32) if cfg, else (B, 4, 32, 32)

                    if use_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    # samples: (B, 4, 32, 32)

                    # samples = vae.decode(samples / 0.18215).sample
                    # → (B, 3, 256, 256) 生成画像

    dist.destroy_process_group()


# ============================================================
# 推論パイプライン
# ============================================================

def sample_ode(args):
    """
    SiT ODE推論 (クラス条件付き画像生成)

    ========================================
    ODE推論設定 (デフォルト)
    ========================================
    - モデル: SiT-XL/2 (EMAパラメータ使用)
    - Transport: Linear path, velocity prediction
    - ソルバー: dopri5 (適応的ステップ)
    - ステップ: 250 (保存点数, 内部ステップは適応的)
    - CFG scale: 4.0
    - VAE: stabilityai/sd-vae-ft-mse

    ========================================
    処理フロー
    ========================================
    1. 初期ノイズ z ~ N(0, I): (B, 4, 32, 32)  ← t=0
    2. ODE求解 dx/dt = v_θ(x, t)  t: 0 → 1
    3. VAEデコード: (B, 4, 32, 32) → (B, 3, 256, 256)

    DiTとの比較:
      DiT:  DDPM逆拡散 (250ステップ, 固定)     → 250回のモデル呼び出し
      SiT:  dopri5 (適応ステップ)               → ~100-150回程度
             Euler (固定250ステップ)              → 250回

    対応: 公式 sample.py L21-L109 (mode="ODE")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # --- モデル読み込み ---
    latent_size = args.image_size // 8  # 32
    model = SiT(
        input_size=latent_size,
        num_classes=args.num_classes,
        **SiT_configs[args.model]
    ).to(device)
    # state_dict = torch.load("SiT-XL-2-256x256.pt")
    # model.load_state_dict(state_dict)
    model.eval()

    # --- Transport + Sampler ---
    transport = create_transport(
        path_type=args.path_type,       # "Linear"
        prediction=args.prediction,     # "velocity"
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
    )
    sampler = Sampler(transport)

    # --- ODE サンプリング関数を構成 ---
    sample_fn = sampler.sample_ode(
        sampling_method=args.sampling_method,  # "dopri5" (default)
        num_steps=args.num_sampling_steps,      # 250 (default)
        atol=args.atol,                         # 1e-6 (default)
        rtol=args.rtol,                         # 1e-3 (default)
        reverse=False,                          # noise→data
    )

    # --- VAE ---
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # --- サンプリング ---
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)

    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    # z: (8, 4, 32, 32) ← 初期ノイズ (t=0)

    y = torch.tensor(class_labels, device=device)
    # y: (8,)

    # --- CFG用にバッチを2倍 (DiTと同一) ---
    z = torch.cat([z, z], 0)
    # z: (16, 4, 32, 32)

    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    # y: (16,) ← [class_labels, null_labels]

    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # --- ODE求解 ---
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    # sample_fn 内部:
    #   odeint(drift_fn, z, t=[0.0, ..., 1.0], method="dopri5")
    #   drift_fn = model.forward_with_cfg(x, t, y, cfg_scale)
    #   → velocity をそのまま dx/dt として使用
    #
    # samples: (num_steps, 16, 4, 32, 32)
    # [-1] → 最終時刻 (t=1): (16, 4, 32, 32)

    # --- null サンプル除去 ---
    samples, _ = samples.chunk(2, dim=0)
    # samples: (8, 4, 32, 32)

    # --- VAEデコード ---
    # images = vae.decode(samples / 0.18215).sample
    # images: (8, 3, 256, 256)

    # save_image(images, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    return samples


def sample_sde(args):
    """
    SiT SDE推論 (クラス条件付き画像生成)

    ========================================
    SDE推論設定 (デフォルト)
    ========================================
    - ソルバー: Euler-Maruyama
    - ステップ: 250
    - Diffusion形式: SBDM (Score-Based Diffusion Matching)
    - Diffusion norm: 1.0
    - Last step: Mean (最終ステップはドリフトのみで更新)
    - Last step size: 0.04

    ========================================
    SDE の構成
    ========================================
    dx = [drift + g²(t) × score(x, t)] dt + √(2g²(t)) dW

    drift:     ODE velocity → そのまま v_θ(x, t)
    score:     velocity → score に変換
    g(t):      diffusion 係数 (SBDM: drift_var に一致)

    ========================================
    処理フロー
    ========================================
    1. 初期ノイズ z ~ N(0, I): (B, 4, 32, 32)     t=0
    2. SDE Euler-Maruyama (249ステップ)             t: 0 → 0.96
    3. Last step: Mean (決定的な最終ステップ)        t: 0.96 → 1.0
    4. VAEデコード: (B, 4, 32, 32) → (B, 3, 256, 256)

    ========================================
    ODEとの比較
    ========================================
    ODE: 決定的 → 再現性あり、少ないステップでOK
    SDE: 確率的 → 多様性が高い、ステップ数多めが必要

    短い学習 (400K steps): SDE > ODE (FID)
    長い学習 (7M steps):   SDE ≈ ODE (FID)

    対応: 公式 sample.py L21-L109 (mode="SDE")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # --- モデル読み込み (ODE推論と同一) ---
    latent_size = args.image_size // 8
    model = SiT(
        input_size=latent_size,
        num_classes=args.num_classes,
        **SiT_configs[args.model]
    ).to(device)
    model.eval()

    # --- Transport + Sampler ---
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
    )
    sampler = Sampler(transport)

    # --- SDE サンプリング関数を構成 ---
    sample_fn = sampler.sample_sde(
        sampling_method=args.sampling_method,  # "Euler" (default)
        diffusion_form=args.diffusion_form,    # "SBDM" (default)
        diffusion_norm=args.diffusion_norm,    # 1.0 (default)
        last_step=args.last_step,              # "Mean" (default)
        last_step_size=args.last_step_size,    # 0.04 (default)
        num_steps=args.num_sampling_steps,      # 250 (default)
    )

    # --- サンプリング (ODE推論と同一のラベル/ノイズ設定) ---
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    n = len(class_labels)

    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # --- SDE求解 ---
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    # sample_fn 内部:
    #   1. Euler-Maruyama ステップ × 249回:
    #      mean_x = x + [v_θ + g²×score] × dt
    #      x = mean_x + √(2g²) × dW
    #
    #   2. Last step (Mean):
    #      x_final = x + [v_θ + g²×score] × 0.04
    #
    # samples: [(16, 4, 32, 32)] × 250
    # [-1] → 最終 (last_step 適用後): (16, 4, 32, 32)

    samples, _ = samples.chunk(2, dim=0)
    # samples: (8, 4, 32, 32)

    # images = vae.decode(samples / 0.18215).sample
    return samples


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=== SiT Training & Sampling Pipeline ===")
    print()
    print("学習設定 (SiT-XL/2, DiTと同一のハイパーパラメータ):")
    print("  データ: ImageNet 256x256 (1.28M images, 1000 classes)")
    print("  VAE潜在空間: 32x32x4 (x0.18215)")
    print("  最適化: AdamW lr=1e-4, weight_decay=0")
    print("  バッチ: 256 global")
    print("  EMA: decay=0.9999")
    print("  CFG dropout: 10%")
    print()
    print("★ DiTとの違い (Transport Framework):")
    print("  DiT:  t = randint(0, 1000)  → diffusion.training_losses(model, x, t, kwargs)")
    print("  SiT:  transport.training_losses(model, x, kwargs)")
    print("        └─ 内部: t~U[0,1], x_t=α_t*x_1+σ_t*x_0, loss=MSE(model(x_t,t), u_t)")
    print()
    print("SiT固有の引数:")
    print("  --path-type:   Linear (=FM) | GVP (cosine) | VP")
    print("  --prediction:  velocity (推奨) | score | noise")
    print("  --loss-weight: none | velocity | likelihood")
    print()
    print("推論設定:")
    print("  ODE:")
    print("    ソルバー: dopri5 (適応的, デフォルト) / Euler / Heun")
    print("    ステップ: 250 (保存点数)")
    print("    atol=1e-6, rtol=1e-3")
    print("    CFG scale: 4.0 (論文), 1.80 (最良FID)")
    print()
    print("  SDE:")
    print("    ソルバー: Euler-Maruyama (デフォルト) / Heun")
    print("    ステップ: 250")
    print("    diffusion: SBDM (デフォルト)")
    print("    last_step: Mean (デフォルト)")
    print("    last_step_size: 0.04")
    print("    CFG scale: 1.80 (最良FID)")
    print()
    print("最良結果 (SiT-XL/2, 7M steps):")
    print("  Linear + velocity + ODE (dopri5, cfg=1.80): FID 2.06")
    print("  Linear + velocity + SDE (Euler, cfg=1.80):  FID 2.06")
    print("  → DiT-XL/2 (DDPM, cfg=1.50):               FID 2.27")
