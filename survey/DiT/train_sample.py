"""
DiT - 学習ループ + 推論パイプライン

対応:
  学習: https://github.com/facebookresearch/DiT/blob/main/train.py
  推論: https://github.com/facebookresearch/DiT/blob/main/sample.py
  拡散: https://github.com/facebookresearch/DiT/blob/main/diffusion/

このファイルでは以下を疑似コードで説明:
1. Gaussian Diffusion (Forward/Reverse, β schedule)
2. 学習ループ (ImageNet + VAE + DDPM)
3. 推論パイプライン (DDPM逆拡散 + CFG + VAEデコード)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy

from models import DiT, DiT_configs


# ============================================================
# Gaussian Diffusion
# ============================================================

class GaussianDiffusion:
    """
    DDPM / Improved DDPM のGaussian Diffusionプロセス

    ========================================
    Forward Process (ノイズ付加)
    ========================================
    q(x_t | x_0) = N(x_t; √ᾱ_t × x_0, (1-ᾱ_t) × I)

    x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε    (ε ~ N(0, I))

    ========================================
    β Schedule (Linear)
    ========================================
    β_1 = 0.0001, β_T = 0.02, T = 1000
    β_t = β_1 + (β_T - β_1) × (t-1) / (T-1)  線形補間

    α_t = 1 - β_t
    ᾱ_t = Π_{s=1}^{t} α_s  (累積積)

    ========================================
    Reverse Process (ノイズ除去)
    ========================================
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ²_θ(x_t, t) × I)

    μ_θ = 1/√α_t × (x_t - β_t/√(1-ᾱ_t) × ε_θ(x_t, t))
    σ²_θ = exp(v × log(β_t) + (1-v) × log(β_tilde_t))  (LEARNED_RANGE)

    対応: 公式 diffusion/gaussian_diffusion.py + diffusion/__init__.py
    """

    def __init__(self, num_timesteps: int = 1000):
        self.num_timesteps = num_timesteps

        # --- Linear β schedule ---
        beta_start = 0.0001
        beta_end = 0.02
        self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        # betas: (1000,) β_1, ..., β_T

        self.alphas = 1.0 - self.betas
        # alphas: (1000,) α_t = 1 - β_t

        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        # alphas_cumprod: (1000,) ᾱ_t = Π α_s

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # alphas_cumprod_prev: (1000,) ᾱ_{t-1} (先頭に1.0を追加)

        # --- Posterior variance (forward process posterior) ---
        # q(x_{t-1} | x_t, x_0) の分散
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # β_tilde_t = β_t × (1 - ᾱ_{t-1}) / (1 - ᾱ_t)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward process: x_0 にノイズを付加して x_t を得る

        入力:
          x_start: (B, 4, 32, 32) - 元データ (VAE潜在)
          t: (B,)                  - 時刻ステップ
          noise: (B, 4, 32, 32)   - ε ~ N(0, I)

        出力:
          x_t: (B, 4, 32, 32) - ノイズ付きデータ

        計算:
          x_t = √ᾱ_t × x_0 + √(1-ᾱ_t) × ε
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_t

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: dict,
    ) -> dict:
        """
        学習損失の計算

        入力:
          model: DiTモデル
          x_start: (B, 4, 32, 32) - 元データ (VAE潜在)
          t: (B,)                  - ランダム時刻ステップ
          model_kwargs: dict       - {"y": class_labels}

        出力:
          {"loss": (B,)} - バッチ内各サンプルの損失

        ========================================
        損失の内訳
        ========================================
        model_output = DiT(x_t, t, y)   → (B, 8, 32, 32)
        ε_θ = model_output[:, :4]       → ノイズ予測
        σ_θ = model_output[:, 4:]       → 分散予測

        1. L_simple = MSE(ε_θ, ε)       メイン損失
        2. L_vlb = KL(q || p_θ)         分散学習のVLB項 (LEARNED_RANGE)
        3. L = L_simple + L_vlb

        対応: 公式 diffusion/gaussian_diffusion.py の training_losses()
        """
        noise = torch.randn_like(x_start)
        # noise: (B, 4, 32, 32)

        x_t = self.q_sample(x_start, t, noise)
        # x_t: (B, 4, 32, 32) ← ノイズ付きデータ

        model_output = model(x_t, t, **model_kwargs)
        # model_output: (B, 8, 32, 32) ← ε予測(4ch) + 分散予測(4ch)

        # --- MSE損失 (ε予測) ---
        B, C = x_start.shape[0], self.betas.shape[0]
        eps_pred = model_output[:, :4]
        # eps_pred: (B, 4, 32, 32)
        mse_loss = F.mse_loss(eps_pred, noise, reduction='none')
        mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))
        # mse_loss: (B,)

        # --- VLB損失 (分散学習) ---
        # LEARNED_RANGE: モデル出力を [β_tilde_t, β_t] の対数空間で内挿
        # v = (σ_θ + 1) / 2  → [0, 1]
        # log_var = v × log(β_t) + (1-v) × log(β_tilde_t)
        # L_vlb = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))
        # (この部分は公式コードでは gaussian_diffusion.py の _vb_terms_bpd で計算)
        # 実装は約200行の数値計算であり省略。概念は上記の通り。

        # 全体損失
        loss = mse_loss  # + vlb_loss (実際にはVLB項も加算される)
        return {"loss": loss}

    def p_sample_loop(
        self,
        model_fn,
        shape: tuple,
        noise: torch.Tensor,
        model_kwargs: dict,
        device: str,
    ) -> torch.Tensor:
        """
        DDPM逆拡散ループ (推論)

        入力:
          model_fn: forward_with_cfg (CFG付きforward)
          shape: (B, 4, 32, 32)
          noise: (B, 4, 32, 32) - 初期ノイズ z_T
          model_kwargs: {"y": labels, "cfg_scale": 4.0}
          device: "cuda"

        出力:
          x_0: (B, 4, 32, 32) - デノイズされた潜在表現

        ========================================
        アルゴリズム (DDPM p_sample)
        ========================================
        for t = T-1, T-2, ..., 0:
            ε_θ = model(x_t, t, y, cfg_scale)  # CFG付き
            μ_θ = 1/√α_t × (x_t - β_t/√(1-ᾱ_t) × ε_θ)
            σ² = posterior_variance_t  (or learned)
            x_{t-1} = μ_θ + σ × z    (z ~ N(0,I), t>0のとき)

        対応: 公式 diffusion/gaussian_diffusion.py の p_sample_loop()
        """
        x = noise
        # x: (B, 4, 32, 32) ← z_T ~ N(0, I)

        # Timestep Respacing: 1000ステップ → 250ステップに間引き
        # 例: [0, 4, 8, 12, ..., 996] の250個を使用
        timesteps = list(range(self.num_timesteps))[::-1]  # [999, 998, ..., 0]
        # 実際のrespacingでは均等間引きされたステップを使用

        for t_idx in timesteps:
            t = torch.tensor([t_idx] * shape[0], device=device)
            # t: (B,)

            with torch.no_grad():
                model_output = model_fn(x, t, **model_kwargs)
                # model_output: (2B, 8, 32, 32) ← CFGで2倍バッチ
                # forward_with_cfg内でCFG処理済み

            eps_pred = model_output[:, :4]
            # eps_pred: (B, 4, 32, 32)

            # --- posterior mean ---
            alpha_t = self._extract(self.alphas, t, x.shape)
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
            beta_t = self._extract(self.betas, t, x.shape)

            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * eps_pred) / torch.sqrt(alpha_cumprod_t)
            # pred_x0: (B, 4, 32, 32)

            alpha_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t, x.shape)
            posterior_mean = (
                torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t) * pred_x0
                + torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * x
            )
            # posterior_mean: (B, 4, 32, 32)

            posterior_var = self._extract(self.posterior_variance, t, x.shape)

            # --- サンプリング ---
            noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x = posterior_mean + torch.sqrt(posterior_var) * noise
            # x: (B, 4, 32, 32) ← x_{t-1}

        return x

    @staticmethod
    def _extract(arr, timesteps, broadcast_shape):
        """配列から時刻に対応する値を取得し、broadcast_shapeに合わせる"""
        vals = torch.tensor(arr, dtype=torch.float32, device=timesteps.device)
        vals = vals[timesteps]
        while len(vals.shape) < len(broadcast_shape):
            vals = vals.unsqueeze(-1)
        return vals


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    """
    EMAモデルの更新

    ema_param = decay × ema_param + (1 - decay) × model_param

    入力:
      ema_model: EMAモデル (推論に使用)
      model: 学習中のモデル
      decay: 0.9999 (デフォルト)

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
    DiT学習メインループ

    ========================================
    学習設定 (論文デフォルト: DiT-XL/2)
    ========================================
    - データ: ImageNet (1.28M images, 1000 classes)
    - 入力: 256×256 → VAE Encode → 32×32×4 潜在
    - バッチサイズ: 256 (global)
    - オプティマイザ: AdamW (lr=1e-4, weight_decay=0, β=(0.9, 0.999))
    - スケジューラ: なし (定数学習率)
    - エポック: 1400 (≈ 7M steps on 8 GPUs)
    - EMA decay: 0.9999
    - CFG dropout: 10% (ラベルを無条件トークンに置換)
    - Diffusion: 1000 steps, linear β schedule, learn_sigma=True

    対応: 公式 train.py
    """
    # --- DDP初期化 ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(args.global_seed * dist.get_world_size() + rank)
    torch.cuda.set_device(device)

    # --- モデル作成 ---
    latent_size = args.image_size // 8  # 256 // 8 = 32
    model = DiT(
        input_size=latent_size,
        num_classes=args.num_classes,
        **DiT_configs[args.model]
    )
    ema = deepcopy(model).to(device)
    model = DDP(model.to(device), device_ids=[rank])

    # --- Diffusion ---
    diffusion = GaussianDiffusion(num_timesteps=1000)

    # --- VAE (事前学習済み, 固定) ---
    # Stable Diffusion の VAE: stabilityai/sd-vae-ft-ema
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae = None  # 疑似コードのため省略 (AutoencoderKLの読み込み)

    # --- オプティマイザ ---
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # --- データローダー ---
    # ImageFolder形式: data_path/{class_name}/{image_file}
    # 前処理: CenterCrop(256) → RandomHFlip → ToTensor → Normalize(0.5, 0.5)
    # sampler = DistributedSampler(dataset)
    # loader = DataLoader(dataset, batch_size=256//world_size, sampler=sampler)
    loader = []  # 疑似コードのため省略

    # --- EMA初期化 ---
    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # --- 学習ループ ---
    train_steps = 0
    for epoch in range(args.epochs):
        for x, y in loader:
            x = x.to(device)  # (B, 3, 256, 256)
            y = y.to(device)  # (B,) クラスラベル

            # --- VAE Encode (勾配不要) ---
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # x: (B, 4, 32, 32) ← VAE潜在空間
                # ×0.18215: 標準偏差を~1に正規化する公式のスケーリング

            # --- ランダム時刻サンプリング ---
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # t: (B,) ∈ {0, ..., 999}

            # --- 損失計算 ---
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            # loss: スカラー

            # --- 更新 ---
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            train_steps += 1

            # --- チェックポイント保存 ---
            if train_steps % 50000 == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                # torch.save(checkpoint, f"checkpoints/{train_steps:07d}.pt")

    dist.destroy_process_group()


# ============================================================
# 推論パイプライン
# ============================================================

def sample(args):
    """
    DiT推論 (クラス条件付き画像生成)

    ========================================
    推論設定 (デフォルト)
    ========================================
    - モデル: DiT-XL/2 (EMAパラメータ使用)
    - Diffusionステップ: 250 (1000からrespacing)
    - CFG scale: 4.0
    - VAE: stabilityai/sd-vae-ft-mse

    ========================================
    処理フロー
    ========================================
    1. 初期ノイズ z_T ~ N(0, I): (B, 4, 32, 32)
    2. DDPM逆拡散 (250ステップ, CFG付き)
    3. VAEデコード: (B, 4, 32, 32) → (B, 3, 256, 256)

    対応: 公式 sample.py
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # --- モデル読み込み ---
    latent_size = args.image_size // 8  # 32
    model = DiT(
        input_size=latent_size,
        num_classes=args.num_classes,
        **DiT_configs[args.model]
    ).to(device)
    # EMAチェックポイントを読み込み
    # state_dict = torch.load("DiT-XL-2-256x256.pt")
    # model.load_state_dict(state_dict)
    model.eval()

    # --- Diffusion ---
    diffusion = GaussianDiffusion(num_timesteps=1000)
    # 推論時は250ステップにrespacing

    # --- VAE ---
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # --- サンプリング ---
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # ImageNetクラス: golden retriever, otter, panda, volcano, macaw, valley, balloon, mushroom
    n = len(class_labels)

    # 初期ノイズ
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    # z: (8, 4, 32, 32)

    y = torch.tensor(class_labels, device=device)
    # y: (8,)

    # --- CFG用にバッチを2倍にする ---
    z = torch.cat([z, z], 0)
    # z: (16, 4, 32, 32) ← [cond, uncond]
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    # y: (16,) ← [class_labels, null_labels]
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # --- DDPM逆拡散 ---
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg,
        z.shape,
        z,
        model_kwargs=model_kwargs,
        device=device,
    )
    # samples: (16, 4, 32, 32)

    # --- nullサンプル除去 ---
    samples, _ = samples.chunk(2, dim=0)
    # samples: (8, 4, 32, 32)

    # --- VAEデコード ---
    # images = vae.decode(samples / 0.18215).sample
    # images: (8, 3, 256, 256)

    # --- 保存 ---
    # save_image(images, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    return samples


# ============================================================
# メイン
# ============================================================

if __name__ == "__main__":
    print("=== DiT Training & Sampling Pipeline ===")
    print()
    print("学習設定 (DiT-XL/2):")
    print("  データ: ImageNet 256×256 (1.28M images, 1000 classes)")
    print("  VAE潜在空間: 32×32×4 (×0.18215)")
    print("  Diffusion: 1000 steps, linear β (0.0001 → 0.02)")
    print("  損失: MSE(ε_θ, ε) + VLB (LEARNED_RANGE)")
    print("  最適化: AdamW lr=1e-4, weight_decay=0")
    print("  バッチ: 256 global")
    print("  EMA: decay=0.9999")
    print("  CFG dropout: 10%")
    print()
    print("推論設定:")
    print("  ステップ: 250 (respacing)")
    print("  CFG scale: 4.0")
    print("  VAE: sd-vae-ft-mse")
    print()
    print("β schedule (linear, 1000 steps):")
    betas = np.linspace(0.0001, 0.02, 1000)
    alphas_cumprod = np.cumprod(1 - betas)
    for t in [0, 100, 250, 500, 750, 999]:
        print(f"  t={t:4d}: β={betas[t]:.6f}, ᾱ_t={alphas_cumprod[t]:.6f}, "
              f"√ᾱ_t={np.sqrt(alphas_cumprod[t]):.4f}, √(1-ᾱ_t)={np.sqrt(1-alphas_cumprod[t]):.4f}")
