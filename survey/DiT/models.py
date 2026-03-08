"""
DiT (Scalable Diffusion Models with Transformers) - モデルアーキテクチャ

対応: https://github.com/facebookresearch/DiT/blob/main/models.py

このファイルは DiT のモデル全体を疑似コードとして記述しています。
主要コンポーネント:
1. TimestepEmbedder - 時刻埋め込み (正弦波 + MLP)
2. LabelEmbedder - クラスラベル埋め込み (CFGドロップアウト付き)
3. DiTBlock - adaLN-Zero条件付きTransformerブロック
4. FinalLayer - 最終出力射影
5. DiT - 全体モデル

公式実装のアーキテクチャを忠実に再現しつつ、各処理のshapeを明記しています。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# ユーティリティ
# ============================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Layer Normalization の変調

    入力:
      x:     (B, N, D) - LayerNorm済みのテンソル
      shift: (B, D)    - シフトパラメータ
      scale: (B, D)    - スケールパラメータ

    出力:
      x:     (B, N, D) - 変調済みテンソル

    scale/shiftを unsqueeze(1) して (B, 1, D) にし、Nの次元でブロードキャスト
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================
# 位置符号化 (2D Sin-Cos)
# ============================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    2D正弦波余弦波位置符号化 (MAEと同じ方式)

    入力:
      embed_dim: int   - 埋め込み次元 (例: 1152)
      grid_size: int   - グリッドサイズ (例: 16 = 32/2)

    出力:
      pos_embed: (grid_size*grid_size, embed_dim)
                 例: (256, 1152)

    対応: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
    """
    grid_h = np.arange(grid_size, dtype=np.float32)  # (grid_size,)
    grid_w = np.arange(grid_size, dtype=np.float32)  # (grid_size,)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)     # 各 (grid_size, grid_size)
    # grid: (2, grid_size, grid_size)

    # 各次元の半分をH用、半分をW用に使用
    half_dim = embed_dim // 2
    omega = np.arange(half_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (half_dim / 2.0)))
    # omega: (half_dim//2,) = (288,) for D=1152

    # H方向
    pos_h = grid_h.reshape(-1)  # (grid_size²,) = (256,)
    out_h = np.outer(pos_h, omega)  # (256, 288)
    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)  # (256, 576)

    # W方向
    pos_w = grid_w.reshape(-1)  # (256,)
    out_w = np.outer(pos_w, omega)  # (256, 288)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)  # (256, 576)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (256, 1152)
    return pos_embed


# ============================================================
# TimestepEmbedder
# ============================================================

class TimestepEmbedder(nn.Module):
    """
    時刻ステップ t → ベクトル表現

    ========================================
    Shape
    ========================================
    入力: t (B,)         - 離散時刻 {0, ..., 999}
    出力: t_emb (B, D)   - 時刻埋め込みベクトル
          例: D=1152 (DiT-XL)

    ========================================
    処理詳細
    ========================================
    1. 正弦波位置符号化 (Transformerと同じ方式)
       t → sinusoidal(256) → (B, 256)
    2. MLP: Linear(256, D) → SiLU → Linear(D, D) → (B, D)

    対応: 公式 models.py L27-L64
    """

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
        正弦波時刻埋め込み (GLIDE/DDPM方式)

        入力: t (B,)     - 時刻スカラー (整数 or 小数)
        出力: emb (B, dim) - 正弦波埋め込み

        周波数: freq_k = exp(-log(10000) × k / (dim/2))  k=0,...,dim/2-1
        → 低周波 (大きな時間スケールの変化) から高周波 (細かい変化) まで
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # freqs: (dim/2,) = (128,) 幾何的に減少する周波数

        args = t[:, None].float() * freqs[None]
        # args: (B, dim/2) 各時刻×各周波数

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # embedding: (B, dim) = (B, 256)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        入力: t (B,)
        出力: t_emb (B, hidden_size)  例: (B, 1152)
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # t_freq: (B, 256)
        t_emb = self.mlp(t_freq)
        # t_emb: (B, hidden_size)
        return t_emb


# ============================================================
# LabelEmbedder
# ============================================================

class LabelEmbedder(nn.Module):
    """
    クラスラベル y → ベクトル表現 (CFGドロップアウト付き)

    ========================================
    Shape
    ========================================
    入力: labels (B,)     - クラスラベル {0, ..., 999}
    出力: y_emb (B, D)    - ラベル埋め込み

    ========================================
    処理詳細
    ========================================
    1. 学習時: 確率 dropout_prob (=0.1) でラベルを num_classes (=1000) に置換
       → num_classes がCFG用の「無条件」トークンとして機能
    2. Embedding(1001, D) でベクトル化
       → 語彙サイズ = 1000クラス + 1 (無条件トークン)

    対応: 公式 models.py L67-L94
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        # 1001個のembedding: 0~999=クラス, 1000=無条件
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids=None) -> torch.Tensor:
        """
        CFGドロップアウト: ランダムにラベルを無条件トークンに置換

        入力: labels (B,)     例: [207, 360, 387, ...]
        出力: labels (B,)     例: [207, 1000, 387, ...]  ← 360 → 1000 にドロップ
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # drop_ids: (B,) bool
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids=None) -> torch.Tensor:
        """
        入力: labels (B,)
        出力: embeddings (B, hidden_size)
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        # embeddings: (B, hidden_size)
        return embeddings


# ============================================================
# DiTBlock (adaLN-Zero)
# ============================================================

class DiTBlock(nn.Module):
    """
    DiT Transformer Block with adaLN-Zero conditioning

    ========================================
    Shape
    ========================================
    入力:
      x: (B, N, D)    例: (B, 256, 1152)  - トークン列
      c: (B, D)        例: (B, 1152)       - 条件ベクトル (t_emb + y_emb)

    出力:
      x: (B, N, D)    例: (B, 256, 1152)

    ========================================
    処理詳細
    ========================================
    1. adaLN変調パラメータ生成 (6つ)
       c → SiLU → Linear(D, 6D) → chunk(6) → shift/scale/gate × MSA/FFN
    2. Attention:
       x_norm = LN(x) × (1+scale_msa) + shift_msa
       x = x + gate_msa × MHSA(x_norm)
    3. FFN:
       x_norm = LN(x) × (1+scale_mlp) + shift_mlp
       x = x + gate_mlp × FFN(x_norm)

    ゼロ初期化: adaLN_modulation の Linear を weight=0, bias=0 で初期化
    → 学習初期は gate=0 のため、ブロックは恒等写像

    対応: 公式 models.py L101-L122
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Multi-Head Self-Attention (timmのAttentionクラス)
        # DiT-XL: 16ヘッド, head_dim=1152/16=72, qkv_bias=True
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        # 公式実装ではtimm.models.vision_transformer.Attentionを使用
        # QKV結合行列 + 出力射影 + qkv_bias=True

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # FFN: D → 4D → D (approximate GELU)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # DiT-XL: 1152 → 4608 → 1152

        # adaLN変調: c → 6つのパラメータ
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # ★ ゼロ初期化: initialize_weights() で weight=0, bias=0 にする

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        入力: x (B, N, D), c (B, D)
        出力: x (B, N, D)
        """
        # --- adaLN変調パラメータ生成 ---
        modulation_output = self.adaLN_modulation(c)
        # modulation_output: (B, 6D) = (B, 6912)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation_output.chunk(6, dim=1)
        # 各: (B, D) = (B, 1152)

        # --- Attention ---
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        # norm1(x): (B, N, D) → modulate: ×(1+scale) + shift → (B, N, D)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        # attn_out: (B, N, D)
        x = x + gate_msa.unsqueeze(1) * attn_out
        # gate_msa.unsqueeze(1): (B, D) → (B, 1, D) でトークン次元にブロードキャスト
        # 学習初期: gate_msa=0 → x = x + 0 (恒等写像)

        # --- FFN ---
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        ff_out = self.mlp(x_norm)
        # ff_out: (B, N, D)
        x = x + gate_mlp.unsqueeze(1) * ff_out

        return x


# ============================================================
# FinalLayer
# ============================================================

class FinalLayer(nn.Module):
    """
    DiTの最終出力層

    ========================================
    Shape
    ========================================
    入力:
      x: (B, N, D)     例: (B, 256, 1152)
      c: (B, D)         例: (B, 1152)

    出力:
      x: (B, N, P²×C)  例: (B, 256, 32)
         P=patch_size=2, C=out_channels=8

    ========================================
    処理詳細
    ========================================
    1. adaLN (2パラメータのみ: shift, scale。ゲートなし)
       c → SiLU → Linear(D, 2D) → shift, scale
    2. 正規化 + 変調
       x = LN(x) × (1+scale) + shift
    3. 線形射影
       x = Linear(D, P²×C)

    ゼロ初期化:
    - adaLN Linear: weight=0, bias=0
    - 出力 Linear: weight=0, bias=0
    → 学習初期の出力は全てゼロ

    対応: 公式 models.py L125-L142
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # DiT-XL: Linear(1152, 2×2×8) = Linear(1152, 32)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        入力: x (B, N, D), c (B, D)
        出力: x (B, N, P²×C)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # 各: (B, D)
        x = modulate(self.norm_final(x), shift, scale)
        # x: (B, N, D)
        x = self.linear(x)
        # x: (B, N, P²×C) = (B, 256, 32)
        return x


# ============================================================
# DiT 全体モデル
# ============================================================

class DiT(nn.Module):
    """
    Diffusion Transformer 全体モデル

    ========================================
    Shape (DiT-XL/2 の場合)
    ========================================
    入力:
      x: (B, 4, 32, 32)   VAEの潜在表現 (ノイズ付き)
      t: (B,)              離散時刻ステップ {0,...,999}
      y: (B,)              クラスラベル {0,...,999}

    出力:
      out: (B, 8, 32, 32)  ε予測(4ch) + 学習分散(4ch)
           learn_sigma=True → out_channels = in_channels × 2

    ========================================
    アーキテクチャ
    ========================================
    1. PatchEmbed: Conv2d(4, 1152, kernel=2, stride=2) → (B, 256, 1152)
    2. + 固定sin-cos 2D位置符号化
    3. 条件: t_emb + y_emb → c: (B, 1152)
    4. 28 × DiTBlock(adaLN-Zero)
    5. FinalLayer: (B, 256, 1152) → (B, 256, 32)
    6. Unpatchify: (B, 256, 32) → (B, 8, 32, 32)

    ========================================
    初期化の詳細
    ========================================
    - 全Linear: Xavier uniform
    - PatchEmbed Conv2d: Xavier uniform (Conv2dではなくLinearとして扱う)
    - pos_embed: 固定sin-cos (requires_grad=False)
    - LabelEmbedder: Normal(std=0.02)
    - TimestepEmbedder MLP: Normal(std=0.02)
    - adaLN Linear: weight=0, bias=0 ★
    - FinalLayer Linear: weight=0, bias=0 ★

    対応: 公式 models.py L145-L267
    """

    def __init__(
        self,
        input_size: int = 32,          # VAE潜在空間のサイズ (256px/8=32)
        patch_size: int = 2,            # パッチサイズ (/2, /4, /8)
        in_channels: int = 4,           # VAE潜在チャンネル数
        hidden_size: int = 1152,        # Transformer隠れ次元
        depth: int = 28,                # Transformerブロック数
        num_heads: int = 16,            # アテンションヘッド数
        mlp_ratio: float = 4.0,         # FFN拡張率
        class_dropout_prob: float = 0.1, # CFGドロップアウト確率
        num_classes: int = 1000,        # ImageNetクラス数
        learn_sigma: bool = True,       # 分散も学習するか
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # learn_sigma=True: 4→8ch (ε予測4ch + 分散予測4ch)
        self.patch_size = patch_size
        self.num_heads = num_heads

        # --- パッチ埋め込み ---
        # Conv2d(4, 1152, kernel_size=2, stride=2)
        # 入力 (B, 4, 32, 32) → (B, 1152, 16, 16) → flatten → (B, 256, 1152)
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True
        )
        num_patches = (input_size // patch_size) ** 2
        # 32/2 = 16, 16×16 = 256 パッチ

        # --- 位置符号化 (固定, 学習しない) ---
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        # (1, 256, 1152) ← 学習時に初期化される

        # --- 条件埋め込み ---
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # --- DiTブロック列 ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        # DiT-XL: 28ブロック

        # --- 最終層 ---
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """
        DiTの重み初期化

        ★ 重要: adaLNとFinalLayerのゼロ初期化
        これにより学習初期はネットワーク全体が恒等写像として振る舞う
        """
        # --- 全Linearモジュール: Xavier uniform ---
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # --- 位置符号化: 固定sin-cos ---
        grid_size = int(self.pos_embed.shape[1] ** 0.5)
        # grid_size = 16 (256パッチ = 16×16)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # --- パッチ埋め込み: Xavier uniform (Linearとして扱う) ---
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # --- ラベル埋め込み: Normal(std=0.02) ---
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # --- 時刻MLP: Normal(std=0.02) ---
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # ★ adaLN変調: ゼロ初期化 ★
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # ★ 最終層: ゼロ初期化 ★
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        パッチ列 → 画像テンソルに復元

        入力: x (B, N, P²×C)     例: (B, 256, 32)
        出力: imgs (B, C, H, W)   例: (B, 8, 32, 32)

        256 = 16×16パッチ, 各パッチ = 2×2ピクセル × 8ch
        → 16×2 = 32, 8ch → (B, 8, 32, 32)
        """
        c = self.out_channels  # 8
        p = self.patch_size    # 2
        h = w = int(x.shape[1] ** 0.5)  # 16
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        # (B, 16, 16, 2, 2, 8)
        x = torch.einsum('nhwpqc->nchpwq', x)
        # (B, 8, 16, 2, 16, 2)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        # (B, 8, 32, 32)
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        DiTのforward pass

        入力:
          x: (B, 4, 32, 32)   ノイズ付きVAE潜在表現
          t: (B,)              離散時刻ステップ
          y: (B,)              クラスラベル

        出力:
          out: (B, 8, 32, 32)  ε予測(4ch) + 分散予測(4ch)
        """
        # 1. パッチ埋め込み + 位置符号化
        x = self.x_embedder(x)  # Conv2d: (B, 4, 32, 32) → (B, 1152, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, 1152, 16, 16) → (B, 256, 1152)
        x = x + self.pos_embed  # (B, 256, 1152) + (1, 256, 1152)
        # 公式実装では timm.PatchEmbed がflatten+transposeを内部で行う

        # 2. 条件ベクトル
        t_emb = self.t_embedder(t)           # (B,) → (B, 1152)
        y_emb = self.y_embedder(y, self.training)  # (B,) → (B, 1152)
        c = t_emb + y_emb                    # (B, 1152) ← 単純加算

        # 3. DiTブロック列
        for block in self.blocks:
            x = block(x, c)                  # (B, 256, 1152) → (B, 256, 1152)

        # 4. 最終層
        x = self.final_layer(x, c)           # (B, 256, 1152) → (B, 256, 32)

        # 5. アンパッチ化
        x = self.unpatchify(x)               # (B, 256, 32) → (B, 8, 32, 32)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance付きforward

        入力:
          x: (2B, 4, 32, 32)  ← [conditional, unconditional] を結合
          t: (2B,)
          y: (2B,)              ← [class_labels, null_labels(=1000)] を結合
          cfg_scale: float      ← ガイダンス強度 (デフォルト4.0)

        出力:
          out: (2B, 8, 32, 32)

        ========================================
        CFGの仕組み
        ========================================
        推論時、同じノイズ入力に対して:
        - 条件付き出力: ε_cond = DiT(x, t, y)
        - 無条件出力:  ε_uncond = DiT(x, t, ∅)
        を並列計算し、ε = ε_uncond + s × (ε_cond - ε_uncond) で結合

        効率化: 入力を (2B) バッチに結合して1回のforwardで計算

        対応: 公式 models.py L250-L266
        """
        # 入力は [cond_input, uncond_input] が結合されている
        # 実際には同じノイズを複製
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        # combined: (2B, 4, 32, 32) ← 同じ入力を2回

        model_out = self.forward(combined, t, y)
        # model_out: (2B, 8, 32, 32)

        # ε部分にのみCFGを適用 (分散部分は条件付きのものをそのまま使用)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        # eps: (2B, 3, 32, 32)  ← 公式実装では3chのみにCFG適用 (再現性のため)
        # rest: (2B, 5, 32, 32)

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # 各: (B, 3, 32, 32)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # CFG式: ε = ε_uncond + s × (ε_cond - ε_uncond)
        eps = torch.cat([half_eps, half_eps], dim=0)
        # (2B, 3, 32, 32)

        return torch.cat([eps, rest], dim=1)
        # (2B, 8, 32, 32)


# ============================================================
# モデル設定
# ============================================================

DiT_configs = {
    # name:         depth, hidden_size, patch_size, num_heads, params
    'DiT-S/2':  dict(depth=12, hidden_size=384,  patch_size=2, num_heads=6),   # 33M
    'DiT-S/4':  dict(depth=12, hidden_size=384,  patch_size=4, num_heads=6),
    'DiT-S/8':  dict(depth=12, hidden_size=384,  patch_size=8, num_heads=6),
    'DiT-B/2':  dict(depth=12, hidden_size=768,  patch_size=2, num_heads=12),  # 130M
    'DiT-B/4':  dict(depth=12, hidden_size=768,  patch_size=4, num_heads=12),
    'DiT-B/8':  dict(depth=12, hidden_size=768,  patch_size=8, num_heads=12),
    'DiT-L/2':  dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),  # 458M
    'DiT-L/4':  dict(depth=24, hidden_size=1024, patch_size=4, num_heads=16),
    'DiT-L/8':  dict(depth=24, hidden_size=1024, patch_size=8, num_heads=16),
    'DiT-XL/2': dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),  # 675M ★
    'DiT-XL/4': dict(depth=28, hidden_size=1152, patch_size=4, num_heads=16),
    'DiT-XL/8': dict(depth=28, hidden_size=1152, patch_size=8, num_heads=16),
}


if __name__ == "__main__":
    print("=== DiT Model Architecture ===")
    print()
    for name, cfg in DiT_configs.items():
        model = DiT(**cfg, input_size=32, num_classes=1000)
        params = sum(p.numel() for p in model.parameters())
        num_patches = (32 // cfg['patch_size']) ** 2
        print(f"  {name}: {params/1e6:.1f}M params, {num_patches} patches, "
              f"depth={cfg['depth']}, dim={cfg['hidden_size']}, heads={cfg['num_heads']}")
