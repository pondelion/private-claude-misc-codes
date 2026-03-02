"""
SiT (Scalable Interpolant Transformers) - モデルアーキテクチャ

対応: https://github.com/willisma/SiT/blob/main/models.py

SiTのモデルアーキテクチャはDiTと**完全同一**です。
唯一の違いは forward() の最後で learn_sigma=True の場合に前半4chのみ返す点。

DiTとの差異を明確にするため、DiTのmodels.pyからの変更箇所にコメントで注記しています。
DiTの models.py と同じクラス (TimestepEmbedder, LabelEmbedder, DiTBlock 等) は
DiT_understanding/models.py を参照してください。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# ユーティリティ (DiTと同一)
# ============================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DiTと同一: x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================
# 位置符号化 (DiTと同一)
# ============================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """DiTと同一: 2D sin-cos位置符号化"""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)

    half_dim = embed_dim // 2
    omega = np.arange(half_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (half_dim / 2.0)))

    pos_h = grid_h.reshape(-1)
    out_h = np.outer(pos_h, omega)
    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)

    pos_w = grid_w.reshape(-1)
    out_w = np.outer(pos_w, omega)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)

    return np.concatenate([emb_h, emb_w], axis=1)


# ============================================================
# Embedding層 (DiTと同一)
# ============================================================

class TimestepEmbedder(nn.Module):
    """DiTと同一。ただしSiTでは連続時刻 t∈[0,1] が入力される"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        入力: t (B,) ← SiTでは連続値 t ∈ [0, 1] (DiTでは離散 {0,...,999})
        出力: emb (B, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """DiTと完全同一"""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# ============================================================
# SiTBlock (= DiTBlock, 完全同一)
# ============================================================

class SiTBlock(nn.Module):
    """
    SiT Transformer Block = DiTBlock (adaLN-Zero)

    構造、初期化、forward全てDiTBlockと同一。
    名前のみ SiTBlock に変更。

    入力: x (B, N, D), c (B, D)
    出力: x (B, N, D)

    対応: 公式 models.py L98-L119 (DiTBlock L101-L122 と同一コード)
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


# ============================================================
# FinalLayer (DiTと同一)
# ============================================================

class FinalLayer(nn.Module):
    """DiTと完全同一: adaLN(2パラメータ) + Linear射影"""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ============================================================
# SiT 全体モデル
# ============================================================

class SiT(nn.Module):
    """
    Scalable Interpolant Transformer

    ★ DiTとの唯一の違い: forward()の最後でlearn_sigma=Trueの場合に前半のみ返す

    ========================================
    Shape (SiT-XL/2 の場合)
    ========================================
    入力:
      x: (B, 4, 32, 32)   VAEの潜在表現 (ノイズ混合)
      t: (B,)              連続時刻 t ∈ [0, 1]
      y: (B,)              クラスラベル {0,...,999}

    出力:
      out: (B, 4, 32, 32)  速度場予測 (velocity)
           ← DiTは (B, 8, 32, 32) を返す

    ========================================
    DiTとの差異
    ========================================
    DiT:
      return self.unpatchify(x)  → (B, 8, 32, 32)  ε+σ全ch
    SiT:
      x = self.unpatchify(x)    → (B, 8, 32, 32)
      if self.learn_sigma:
          x, _ = x.chunk(2, dim=1)  → (B, 4, 32, 32)  前半のみ
      return x

    理由: SiTではvelocity/score/noiseを直接予測するため、
          learned varianceが不要。8chで学習するが使うのは前半4chのみ。

    対応: 公式 models.py L142-L265
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,       # SiTでは後半を破棄するが、構造上はTrue
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True
        )
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """DiTと完全同一の初期化 (adaLN-Zeroの0初期化含む)"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        grid_size = int(self.pos_embed.shape[1] ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """DiTと同一"""
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        SiTのforward pass

        入力:
          x: (B, 4, 32, 32)   ← x_t = α_t×x_1 + σ_t×x_0 (interpolant)
          t: (B,)              ← 連続時刻 t ∈ [0, 1]
          y: (B,)              ← クラスラベル

        出力:
          out: (B, 4, 32, 32)  ← velocity/score/noise予測
               ★ learn_sigma=True でも 4ch (DiTは8ch)
        """
        # === DiTと同一の処理 ===
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed                      # (B, 256, 1152)

        t_emb = self.t_embedder(t)                   # (B, 1152)
        y_emb = self.y_embedder(y, self.training)    # (B, 1152)
        c = t_emb + y_emb                            # (B, 1152)

        for block in self.blocks:
            x = block(x, c)                          # (B, 256, 1152)

        x = self.final_layer(x, c)                   # (B, 256, 32)
        x = self.unpatchify(x)                       # (B, 8, 32, 32)

        # === ★ SiT固有: 前半4chのみ返す ===
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
            # x: (B, 4, 32, 32) ← velocity/score/noise 予測
            # _: (B, 4, 32, 32) ← 破棄 (learned variance不使用)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        CFG付きforward (DiTと同一のCFG処理)

        入力: x (2B, 4, 32, 32), t (2B,), y (2B,), cfg_scale
        出力: (2B, 4, 32, 32)  ← 4ch (DiTは8ch)

        ★ SiTではCFGを全4チャンネルに適用 (DiTは3chのみ)
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # model_out: (2B, 4, 32, 32)

        # 公式コードでは互換性のため3chのみにCFG適用しているが、
        # 概念的には全4chに適用するのが自然
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ============================================================
# モデル設定 (DiTと同一のサイズバリエーション)
# ============================================================

SiT_configs = {
    'SiT-S/2':  dict(depth=12, hidden_size=384,  patch_size=2, num_heads=6),
    'SiT-B/2':  dict(depth=12, hidden_size=768,  patch_size=2, num_heads=12),
    'SiT-L/2':  dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    'SiT-XL/2': dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),  # ★ 675M
    # /4, /8 バリエーションも存在 (省略)
}


if __name__ == "__main__":
    print("=== SiT Model Architecture ===")
    print("(DiTと同一構造。forwardの出力チャンネル数のみ異なる)")
    print()
    for name, cfg in SiT_configs.items():
        model = SiT(**cfg, input_size=32, num_classes=1000)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params/1e6:.1f}M params")

    print()
    print("DiTとの出力差異:")
    print("  DiT forward:  (B, 8, 32, 32)  → ε(4ch) + σ(4ch)")
    print("  SiT forward:  (B, 4, 32, 32)  → velocity(4ch) のみ")
