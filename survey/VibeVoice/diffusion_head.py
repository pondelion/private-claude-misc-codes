"""
VibeVoice Diffusion Head - トークンレベル拡散ヘッド

公式実装: vibevoice/modular/modular_vibevoice_diffusion_head.py

LLM の隠れ状態を条件として、音声 VAE の潜在変数を予測する軽量拡散モデル。
わずか4層の Transformer ブロックで構成。

アーキテクチャ:
  1. noisy_images_proj: ノイズ付き潜在変数 → ヘッド次元
  2. cond_proj: LLM 隠れ状態 → 条件次元
  3. t_embedder: タイムステップ → 条件次元の埋め込み
  4. HeadLayer × 4: AdaLN + SwiGLU FFN
  5. FinalLayer: 最終射影 → 潜在次元 (64)

重み初期化:
  - AdaLN 変調: ゼロ初期化（恒等写像からスタート）
  - FinalLayer 出力: ゼロ初期化（予測を0からスタート）
  - TimestepEmbedder MLP: normal(std=0.02)

参照: modular_vibevoice_diffusion_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 正規化とユーティリティ
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    数式: x_norm = x / sqrt(mean(x²) + eps) * weight

    参照: modular_vibevoice_diffusion_head.py の RMSNorm
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, dim]
        Returns:
            [*, dim]
        """
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x * norm
        if self.elementwise_affine:
            out = out * self.weight
        return out


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Layer Normalization (AdaLN) の変調関数。

    数式: output = x * (1 + scale) + shift

    FiLM (Feature-wise Linear Modulation) スタイルの変調。
    scale=0, shift=0 で恒等写像（ゼロ初期化の根拠）。

    Args:
        x:     [B, T, D] 正規化済み入力
        shift: [B, T, D] or [B, 1, D] シフト量
        scale: [B, T, D] or [B, 1, D] スケール量

    Returns:
        [B, T, D] 変調された出力
    """
    return x * (1 + scale) + shift


# ============================================================================
# TimestepEmbedder: 拡散タイムステップの埋め込み
# ============================================================================

class TimestepEmbedder(nn.Module):
    """
    スカラーのタイムステップを学習可能な埋め込みベクトルに変換。

    構造:
      1. 正弦波位置エンコーディング（固定、dim=256）
      2. MLP: Linear(256 → hidden_size) → SiLU → Linear(hidden_size → hidden_size)

    参照: modular_vibevoice_diffusion_head.py の TimestepEmbedder
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        Args:
            hidden_size: 出力次元（= cond_dim, LLM隠れ次元）
            frequency_embedding_size: 正弦波の次元 (256)
        """
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,         # [N] タイムステップ (0~999)
        dim: int,                 # 出力次元 (256)
        max_period: int = 10000,  # 最大周期
    ) -> torch.Tensor:
        """
        正弦波位置エンコーディング（Transformer と同じ原理）。

        数式:
          freq_k = exp(-log(max_period) * 2k / dim)
          PE(t, 2k)   = cos(t * freq_k)
          PE(t, 2k+1) = sin(t * freq_k)

        Args:
            t: [N] スカラータイムステップ（整数 or 小数）
            dim: 埋め込み次元 (256)
            max_period: 正弦波の最大周期 (10000)

        Returns:
            embedding: [N, dim] 位置埋め込み
        """
        half = dim // 2
        # freqs: [half] = exp(-log(10000) * [0, 1, ..., half-1] / half)
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device).float() / half
        )
        # args: [N, half] = t[:, None] * freqs[None, :]
        args = t[:, None].float() * freqs[None, :]
        # embedding: [N, dim] = [cos(args), sin(args)]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding  # [N, dim]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [N] タイムステップ (整数, 0~999)

        Returns:
            [N, hidden_size] 学習可能なタイムステップ埋め込み

        データフロー:
            [N] → sinusoidal PE → [N, 256] → MLP → [N, hidden_size]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # [N, 256]
        t_emb = self.mlp(t_freq)
        # [N, hidden_size]
        return t_emb


# ============================================================================
# FeedForwardNetwork (SwiGLU)
# ============================================================================

class FeedForwardNetwork(nn.Module):
    """
    SwiGLU 活性化を使用した Feed-Forward Network。

    数式:
      gate = SiLU(gate_proj(x))
      up = up_proj(x)
      output = down_proj(gate ⊙ up)

    SwiGLU は GLU の Swish 活性化版で、Llama/Qwen でも使用される
    効率的な FFN アーキテクチャ。

    参照: modular_vibevoice_diffusion_head.py の FeedForwardNetwork
    """

    def __init__(self, embed_dim: int, ffn_dim: int):
        """
        Args:
            embed_dim: 入力/出力次元 (= latent_size = 64)
            ffn_dim: 中間次元 (= hidden_size * head_ffn_ratio)
                     1.5B: 768 * 3.0 = 2304
        """
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim]

        Returns:
            [B, T, embed_dim]

        データフロー:
            [B, T, 64] → gate_proj → [B, T, ffn_dim] → SiLU
                       → up_proj   → [B, T, ffn_dim]
                       → element-wise multiply
                       → down_proj → [B, T, 64]
        """
        gate = F.silu(self.gate_proj(x))  # [B, T, ffn_dim]
        up = self.up_proj(x)               # [B, T, ffn_dim]
        return self.down_proj(gate * up)   # [B, T, embed_dim]


# ============================================================================
# HeadLayer: AdaLN + SwiGLU の中間層
# ============================================================================

class HeadLayer(nn.Module):
    """
    Diffusion Head の中間層。
    Adaptive Layer Normalization (AdaLN) + SwiGLU FFN で構成。

    条件ベクトル c から shift, scale, gate の3つのパラメータを生成し、
    入力を変調した上で FFN を適用する。

    構造:
      1. 条件 c → adaLN_modulation → (shift, scale, gate)
      2. norm(x) → modulate(norm_x, shift, scale) → FFN
      3. x = x + gate * FFN(modulated)

    gate はゼロ初期化されるため、学習初期は x = x + 0（恒等写像）。

    参照: modular_vibevoice_diffusion_head.py の HeadLayer
    """

    def __init__(
        self,
        embed_dim: int,   # 潜在次元 (64)
        ffn_dim: int,      # FFN 中間次元 (2304)
        cond_dim: int,     # 条件次元 (hidden_size: 768 or 1536 or 3584)
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)

        # AdaLN 変調: 条件 → (shift, scale, gate) の3つ
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )
        # 最終層はゼロ初期化（initialize_weights で設定）

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim]  現在の潜在表現
            c: [B, T, cond_dim]   条件ベクトル (LLM隠れ状態 + タイムステップ)

        Returns:
            [B, T, embed_dim] 更新された潜在表現

        データフロー:
            c: [B, T, cond_dim]
            → adaLN_modulation: SiLU → Linear
            → [B, T, 3 * embed_dim]
            → chunk(3): shift_ffn, scale_ffn, gate_ffn  各 [B, T, embed_dim]

            x: [B, T, embed_dim]
            → norm: RMSNorm
            → modulate(norm_x, shift, scale): norm_x * (1 + scale) + shift
            → FFN (SwiGLU)
            → gate * FFN_output
            → x + gate * FFN_output (残差接続)
        """
        # 条件から変調パラメータを生成
        modulation = self.adaLN_modulation(c)
        # [B, T, 3 * embed_dim]
        shift_ffn, scale_ffn, gate_ffn = modulation.chunk(3, dim=-1)
        # 各 [B, T, embed_dim]

        # AdaLN + FFN + 残差接続
        norm_x = self.norm(x)                                     # [B, T, embed_dim]
        modulated = modulate(norm_x, shift_ffn, scale_ffn)        # [B, T, embed_dim]
        ffn_out = self.ffn(modulated)                             # [B, T, embed_dim]
        x = x + gate_ffn * ffn_out                                # [B, T, embed_dim]

        return x


# ============================================================================
# FinalLayer: 出力射影
# ============================================================================

class FinalLayer(nn.Module):
    """
    Diffusion Head の最終層。
    AdaLN 変調 + 線形射影で潜在変数次元に戻す。

    HeadLayer と異なり、gate パラメータなし（shift + scale のみ）。
    出力の linear 層もゼロ初期化。

    参照: modular_vibevoice_diffusion_head.py の FinalLayer
    """

    def __init__(
        self,
        hidden_size: int,  # ヘッド内部次元 (= embed_dim = 64)
        output_size: int,  # 出力次元 (= latent_size = 64)
        cond_dim: int,     # 条件次元 (hidden_size)
    ):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

        # AdaLN 変調: (shift, scale) の2つ
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_size] HeadLayer 群の出力
            c: [B, T, cond_dim]    条件ベクトル

        Returns:
            [B, T, output_size] 予測ノイズ or 予測速度

        データフロー:
            c → adaLN_modulation → (shift, scale)
            x → norm_final → modulate → linear → output
            [B, T, 64] → norm → modulate → [B, T, 64] → linear → [B, T, 64]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        # 各 [B, T, hidden_size]
        x = modulate(self.norm_final(x), shift, scale)
        # [B, T, hidden_size]
        x = self.linear(x)
        # [B, T, output_size]
        return x


# ============================================================================
# VibeVoiceDiffusionHead: 完全な拡散予測ヘッド
# ============================================================================

class VibeVoiceDiffusionHead(nn.Module):
    """
    VibeVoice の拡散予測ヘッド。
    LLM の隠れ状態を条件として、ノイズ付き潜在変数からノイズ/速度を予測。

    構成:
      noisy_images_proj: Linear(latent_size → hidden_size)
      cond_proj: Linear(hidden_size → cond_dim)
      t_embedder: TimestepEmbedder(cond_dim)
      layers: HeadLayer × head_layers (4)
      final_layer: FinalLayer

    設定値 (1.5B モデル):
      - hidden_size: 1536 (LLM hidden dim = cond_dim)
      - latent_size: 64 (acoustic VAE dim)
      - head_layers: 4
      - head_ffn_ratio: 3.0 → ffn_dim = 1536 * 3 = 4608
        ※ ヘッド内部は embed_dim=latent_size=64 で動作
        ※ ffn_dim は hidden_size * head_ffn_ratio = 1536 * 3 = 4608
           ただし実際には latent_size 次元で動作するため
           ffn_dim = max(int(hidden_size * head_ffn_ratio), latent_size * 4) 等
      - rms_norm_eps: 1e-5
      - prediction_type: "v_prediction"

    参照: modular_vibevoice_diffusion_head.py の VibeVoiceDiffusionHead
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        cond_dim = config.hidden_size      # LLM隠れ次元 (1536 or 3584)
        latent_size = config.latent_size    # 64
        hidden_size = latent_size           # ヘッド内部次元 = latent_size
        ffn_dim = int(config.hidden_size * config.head_ffn_ratio)  # 1536*3=4608

        # --- ノイズ付き潜在変数の射影 ---
        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        # [B, T, 64] → [B, T, 64]

        # --- 条件の射影 ---
        self.cond_proj = nn.Linear(config.hidden_size, cond_dim, bias=False)
        # [B, T, 1536] → [B, T, cond_dim]

        # --- タイムステップ埋め込み ---
        self.t_embedder = TimestepEmbedder(cond_dim)
        # [B] → [B, cond_dim]

        # --- 中間層 ×4 ---
        self.layers = nn.ModuleList([
            HeadLayer(
                embed_dim=hidden_size,        # 64
                ffn_dim=ffn_dim,              # 4608
                cond_dim=cond_dim,            # 1536
                norm_eps=config.rms_norm_eps,  # 1e-5
            )
            for _ in range(config.head_layers)  # 4
        ])

        # --- 最終層 ---
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,   # 64
            output_size=latent_size,   # 64
            cond_dim=cond_dim,         # 1536
        )

        # 重み初期化
        self.initialize_weights()

    def initialize_weights(self):
        """
        安定した学習のための重み初期化戦略。

        1. TimestepEmbedder の MLP: normal(std=0.02)
        2. 各 HeadLayer の adaLN_modulation 最終層: ゼロ初期化
           → shift=0, scale=0, gate=0 → modulate(x, 0, 0) = x（恒等写像）
        3. FinalLayer の adaLN_modulation 最終層: ゼロ初期化
        4. FinalLayer の linear: ゼロ初期化
           → 初期出力は全て 0（予測が 0 からスタート）

        これにより、学習初期は入力をほぼそのまま通す安定した状態から始まる。
        """
        # TimestepEmbedder MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # AdaLN 変調のゼロ初期化
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)

        # FinalLayer のゼロ初期化
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.linear.weight)

    def forward(
        self,
        noisy_images: torch.Tensor,  # [B, T_latent, latent_size]
        timesteps: torch.Tensor,      # [B]
        condition: torch.Tensor,       # [B, T_latent, hidden_size]
    ) -> torch.Tensor:
        """
        拡散ヘッドのフォワードパス。

        ノイズ付き潜在変数 + タイムステップ + LLM条件 → 予測速度/ノイズ

        Args:
            noisy_images: [B, T_latent, 64]
                拡散プロセスでノイズが追加された音声潜在変数
                学習時: x_t = α_t * x_0 + σ_t * ε
                推論時: 純粋ノイズ z ~ N(0,1) から開始

            timesteps: [B]
                拡散タイムステップ (0~999)
                0: ほぼノイズなし（元データに近い）
                999: ほぼ純粋ノイズ

            condition: [B, T_latent, hidden_size]
                LLM の隠れ状態（各音声トークン位置）
                テキスト内容・話者性・韻律の情報を含む

        Returns:
            output: [B, T_latent, 64]
                prediction_type = "v_prediction" の場合:
                    速度 v = α_t * ε - σ_t * x_0
                prediction_type = "epsilon" の場合:
                    ノイズ ε

        データフロー:
            noisy_images [B, T, 64]
              → noisy_images_proj → [B, T, 64]  (= x)

            timesteps [B]
              → t_embedder → [B, cond_dim]  (= t)

            condition [B, T, hidden_size]
              → cond_proj → [B, T, cond_dim]
              → + t.unsqueeze(1) → [B, T, cond_dim]  (= c)

            x, c を4層の HeadLayer に通す:
              HeadLayer 1: x = x + gate₁ * FFN(AdaLN(x, c))
              HeadLayer 2: x = x + gate₂ * FFN(AdaLN(x, c))
              HeadLayer 3: x = x + gate₃ * FFN(AdaLN(x, c))
              HeadLayer 4: x = x + gate₄ * FFN(AdaLN(x, c))

            FinalLayer: x → AdaLN(x, c) → Linear → [B, T, 64]
        """
        # === ノイズ付き潜在変数の射影 ===
        x = self.noisy_images_proj(noisy_images)
        # [B, T, latent_size] → [B, T, hidden_size(=latent_size)]

        # === タイムステップ埋め込み ===
        t = self.t_embedder(timesteps)
        # [B] → [B, cond_dim]

        # === 条件射影 + タイムステップ結合 ===
        c = self.cond_proj(condition)
        # [B, T, hidden_size] → [B, T, cond_dim]
        c = c + t.unsqueeze(1)
        # [B, T, cond_dim] + [B, 1, cond_dim] → [B, T, cond_dim]
        # タイムステップ情報が全トークン位置に加算される

        # === 中間層 (HeadLayer × 4) ===
        for layer in self.layers:
            x = layer(x, c)
        # [B, T, hidden_size] (各層で AdaLN + SwiGLU + 残差)

        # === 最終射影 ===
        x = self.final_layer(x, c)
        # [B, T, latent_size] = [B, T, 64]

        return x


# ============================================================================
# 使用例（擬似コード）
# ============================================================================

def example_diffusion_training():
    """拡散ヘッドの学習ループ（擬似コード）"""

    # モデル構成
    diffusion_head = VibeVoiceDiffusionHead(config)
    scheduler = DPMSolverScheduler(num_train_timesteps=1000, beta_schedule="cosine")

    # --- 学習ステップ ---
    # target: [B, T, 64] 正解の音声潜在変数（Acoustic Tokenizer の出力）
    # condition: [B, T, hidden_size] LLM の隠れ状態

    # 1. ランダムタイムステップ
    timesteps = torch.randint(0, 1000, (B,))  # [B]

    # 2. ランダムノイズ
    noise = torch.randn_like(target)  # [B, T, 64]

    # 3. ノイズ追加
    noisy = scheduler.add_noise(target, noise, timesteps)
    # x_t = α_t * x_0 + σ_t * ε

    # 4. 予測
    pred = diffusion_head(noisy, timesteps, condition)
    # [B, T, 64]

    # 5. ターゲット（v-prediction）
    target_v = scheduler.get_velocity(target, noise, timesteps)
    # v = α_t * ε - σ_t * x_0

    # 6. MSE損失
    loss = F.mse_loss(pred, target_v)


def example_diffusion_inference():
    """拡散ヘッドの推論（CFG付き）"""

    diffusion_head = VibeVoiceDiffusionHead(config)
    scheduler = DPMSolverScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(20)  # 推論は20ステップ

    cfg_scale = 1.3
    condition = lm_hidden_state  # [B, 1, hidden_size]

    # 純粋ノイズから開始
    z = torch.randn(B, 1, 64)  # [B, 1, 64]

    for t in scheduler.timesteps:
        # CFG: 条件付き + 無条件 を同時に予測
        z_doubled = torch.cat([z, z], dim=0)          # [2B, 1, 64]
        cond_doubled = torch.cat([
            condition,                                  # 条件付き
            torch.zeros_like(condition),                # 無条件
        ], dim=0)                                       # [2B, 1, hidden_size]
        t_doubled = t.expand(2 * B)                    # [2B]

        # 予測
        v_pred = diffusion_head(z_doubled, t_doubled, cond_doubled)
        v_cond, v_uncond = v_pred.chunk(2, dim=0)

        # CFG ガイダンス
        v = v_uncond + cfg_scale * (v_cond - v_uncond)

        # スケジューラで1ステップ進める
        z = scheduler.step(v, t, z).prev_sample

    # z: [B, 1, 64] デノイズされた音声潜在変数
    audio = acoustic_tokenizer.decode(z)  # → [B, 1, T_audio]


class DPMSolverScheduler:
    """DPM-Solver++ のプレースホルダ。詳細は loss_and_training.py を参照。"""
    def __init__(self, **kwargs): pass
    def add_noise(self, original, noise, timesteps): pass
    def get_velocity(self, original, noise, timesteps): pass
    def set_timesteps(self, num_steps): pass
    def step(self, model_output, timestep, sample): pass
