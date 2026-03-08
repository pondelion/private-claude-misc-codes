"""
Kronos Understanding - K-line Tokenizer (簡略化疑似コード)

K線(ローソク足)データの離散化を行うトークナイザ。
Transformer-based Autoencoder + Binary Spherical Quantization (BSQ) で
連続的な OHLCVA 値を階層的離散トークンに変換する。

対応する公式実装:
  - model/kronos.py: KronosTokenizer クラス
  - model/module.py: BSQuantizer, BinarySphericalQuantizer クラス

論文参照: Section 3 "K-line Tokenization"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Binary Spherical Quantizer (BSQ)
# 論文: https://arxiv.org/abs/2406.07548
# ============================================================

class BinarySphericalQuantizer(nn.Module):
    """
    連続ベクトルをバイナリコードに量子化する。

    特徴:
    - sign関数で二値化 {-1, +1} → Straight-Through Estimator (STE) で勾配伝播
    - L2正規化 + スケーリングで量子化誤差の上界を保証
    - エントロピー正則化でコードブック使用率を最大化

    量子化誤差の上界:
        E_a ||u - û|| < √(2 - 2/√L) < √2
        (L: コードブック次元。次元↑ → 誤差上界↓)
    """

    def __init__(self, embed_dim, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=5):
        """
        Args:
            embed_dim: コードブック次元 (例: 20)
            beta: commit loss の重み
            gamma0: per-sample entropy penalty の重み (最大化 → コードブック均等使用)
            gamma: codebook entropy penalty の重み (最大化 → 全コード使用)
            zeta: 全体のエントロピーペナルティスケール
            group_size: エントロピー近似用のグループサイズ (例: 5)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.group_size = group_size
        self.num_groups = embed_dim // group_size

        # ビット→インデックス変換用マスク
        # basis: [2^(K-1), 2^(K-2), ..., 2^1, 2^0]
        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))

        # グループ単位のコードブック (2^group_size エントリ)
        # エントロピー近似に使用 (全コードブック 2^embed_dim は巨大すぎるため)
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))

    def quantize(self, z):
        """
        sign関数による量子化 + STE (Straight-Through Estimator)

        入力: z (B, T, embed_dim) - 連続値
        出力: zhat (B, T, embed_dim) - 二値 {-1, +1}

        STE: 順伝播は sign(z)、逆伝播は z をそのまま通す
        """
        # sign関数: z > 0 → +1, z ≤ 0 → -1
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))

        # Straight-Through Estimator: 勾配は z に対してそのまま伝播
        # z + (zhat - z).detach() = zhat (順伝播), grad = ∂z/∂z = 1 (逆伝播)
        return z + (zhat - z).detach()

    def forward(self, z, collect_metrics=True):
        """
        入力: z (B, T, embed_dim) - エンコーダ出力
        出力:
            - zq (B, T, embed_dim): 量子化済みベクトル (スケーリング済み)
            - loss: commit_loss + entropy_penalty
            - metrics: {コードブックエントロピー, 使用コード数, ...}
        """
        # Step 1: 二値化
        zq = self.quantize(z)
        # zq: (B, T, embed_dim), 値 ∈ {-1, +1}

        # Step 2: L2正規化スケーリング
        # 量子化ベクトルを単位球面上に射影
        q_scale = 1.0 / (self.embed_dim ** 0.5)
        zq = zq * q_scale
        # zq: (B, T, embed_dim), 値 ∈ {-1/√K, +1/√K}

        if not collect_metrics:
            return zq, torch.tensor(0.0), {}

        # Step 3: Commit Loss (コミットメント損失)
        # エンコーダ出力を量子化結果に近づける
        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        # Step 4: Entropy Penalty (エントロピーペナルティ)
        # - per_sample_entropy: 最大化 → 各サンプルが多様なコードを使用
        # - codebook_entropy: 最大化 → コードブック全体の使用率向上
        per_sample_entropy, codebook_entropy = self._compute_entropy(z)
        entropy_penalty = self.gamma0 * per_sample_entropy - self.gamma * codebook_entropy

        total_loss = commit_loss + self.zeta * entropy_penalty

        return zq, total_loss, {"H": codebook_entropy}

    def _compute_entropy(self, z):
        """
        ソフトエントロピー近似 (グループ単位)

        全コードブック (2^20) は巨大すぎるため、
        group_size (=5) ごとのサブコードブック (2^5=32) で近似
        """
        # z をグループに分割
        # z: (B, T, embed_dim) → (B, T, num_groups, group_size)
        divided_z = z.reshape(*z.shape[:-1], self.num_groups, self.group_size)

        # Per-sample entropy (解析的計算)
        # p = sigmoid(-4z / √K) → 各ビットの「0になる確率」
        p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5))
        prob = torch.stack([p, 1 - p], dim=-1)  # (B, T, K, 2)
        per_sample_entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1).sum(dim=-1).mean()

        # Codebook entropy (グループ単位の近似)
        # 各グループのサブコードブックに対するソフト割り当て確率を計算
        # → マクロ平均 → エントロピー
        # 省略: 実装詳細は BinarySphericalQuantizer.soft_entropy_loss() を参照
        codebook_entropy = per_sample_entropy  # 簡略化

        return per_sample_entropy, codebook_entropy

    def codes_to_indexes(self, zhat):
        """
        バイナリコード → 整数インデックス

        入力: zhat (B, T, embed_dim) ∈ {-1/√K, +1/√K}
        出力: indices (B, T) ∈ [0, 2^embed_dim - 1]

        例: [-1, +1, -1, +1] → [0, 1, 0, 1] → 0*8 + 1*4 + 0*2 + 1*1 = 5
        """
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)


class BSQuantizer(nn.Module):
    """
    BSQのラッパー: s1 (coarse) + s2 (fine) の2分割を管理

    20ビット全体をBSQで量子化した後、
    前半10ビット → s1 (coarseサブトークン)
    後半10ビット → s2 (fineサブトークン)
    に分割してインデックス化
    """

    def __init__(self, s1_bits=10, s2_bits=10, **bsq_kwargs):
        """
        Args:
            s1_bits: Coarseサブトークンのビット数 (例: 10 → 語彙1024)
            s2_bits: Fineサブトークンのビット数 (例: 10 → 語彙1024)
        """
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        codebook_dim = s1_bits + s2_bits  # 合計20ビット
        self.bsq = BinarySphericalQuantizer(codebook_dim, **bsq_kwargs)

    def bits_to_indices(self, bits):
        """
        バイナリビット列 → 整数インデックス (LSB first)

        入力: bits (B, T, n_bits)
        出力: indices (B, T) ∈ [0, 2^n_bits - 1]
        """
        bits = (bits >= 0).to(torch.long)  # {-1,+1} → {0,1}
        indices = 2 ** torch.arange(0, bits.shape[-1], 1, device=bits.device, dtype=torch.long)
        return (bits * indices).sum(-1)

    def forward(self, z, half=False, collect_metrics=True):
        """
        入力: z (B, T, s1_bits + s2_bits) - エンコーダ出力
        出力:
            - bsq_loss: BSQ損失
            - quantized (B, T, s1_bits + s2_bits): 量子化済みベクトル
            - z_indices:
                half=True → [s1_indices (B,T), s2_indices (B,T)]  ← 推論時
                half=False → full_indices (B, T)                   ← 学習時

        処理:
            1. L2正規化 (単位球面上に射影)
            2. BSQ量子化 ({-1,+1}^20 → スケーリング)
            3. half=True なら前半/後半に分割してインデックス化
        """
        # L2正規化: 球面上に射影
        z = F.normalize(z, dim=-1)
        # z: (B, T, 20), ||z||_2 = 1

        # BSQ量子化
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        # quantized: (B, T, 20), 値 ∈ {-1/√20, +1/√20}

        if half:
            # s1/s2に分割してそれぞれインデックス化
            q_s1 = quantized[:, :, :self.s1_bits]     # (B, T, 10)
            q_s2 = quantized[:, :, self.s1_bits:]     # (B, T, 10)
            z_indices = [
                self.bits_to_indices(q_s1),  # (B, T) ∈ [0, 1023]
                self.bits_to_indices(q_s2),  # (B, T) ∈ [0, 1023]
            ]
        else:
            z_indices = self.bits_to_indices(quantized)  # (B, T)

        return bsq_loss, quantized, z_indices


# ============================================================
# KronosTokenizer
# ============================================================

class KronosTokenizer(nn.Module):
    """
    K線トークナイザ: 連続OHLCVA → 階層的離散トークン

    アーキテクチャ:
        Encoder: Linear(6, 256) → 3x TransformerBlock → Linear(256, 20)
        BSQ: 量子化 20ビット → s1(10bit) + s2(10bit)
        Decoder:
            Coarse: Linear(10, 256) → 3x TransformerBlock → Linear(256, 6)
            Fine:   Linear(20, 256) → 3x TransformerBlock → Linear(256, 6)

    学習損失:
        L = L_coarse + L_fine + λ * L_quant
        L_coarse = MSE(x, Decoder_coarse(BSQ_s1(Encoder(x))))
        L_fine   = MSE(x, Decoder_fine(BSQ_all(Encoder(x))))
        L_quant  = commit_loss + entropy_penalty
    """

    def __init__(
        self,
        d_in=6,           # 入力次元: OHLCVA
        d_model=256,      # Transformer次元
        n_heads=4,        # Attention ヘッド数
        ff_dim=512,       # FFN中間次元
        n_enc_layers=3,   # Encoderレイヤー数 (実際はn-1個のTransformerBlock)
        n_dec_layers=3,   # Decoderレイヤー数
        s1_bits=10,       # Coarseサブトークン ビット数
        s2_bits=10,       # Fineサブトークン ビット数
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        codebook_dim = s1_bits + s2_bits  # 20

        # --- Encoder ---
        self.embed = nn.Linear(d_in, d_model)  # (B,T,6) → (B,T,256)
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim)
            for _ in range(n_enc_layers - 1)  # 2ブロック (n-1)
        ])
        self.quant_embed = nn.Linear(d_model, codebook_dim)  # (B,T,256) → (B,T,20)

        # --- BSQ ---
        self.tokenizer = BSQuantizer(s1_bits, s2_bits)

        # --- Decoder (Coarse: s1のみ) ---
        self.post_quant_embed_pre = nn.Linear(s1_bits, d_model)  # (B,T,10) → (B,T,256)

        # --- Decoder (Fine: 全体) ---
        self.post_quant_embed = nn.Linear(codebook_dim, d_model)  # (B,T,20) → (B,T,256)

        # Encoder/Decoderでパラメータ共有
        self.decoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim)
            for _ in range(n_dec_layers - 1)
        ])
        self.head = nn.Linear(d_model, d_in)  # (B,T,256) → (B,T,6)

    def forward(self, x):
        """
        入力:
            x: (B, T, 6) - 正規化済みK線データ [Open, High, Low, Close, Vol, Amt]

        出力:
            (z_pre, z): 再構成結果
                z_pre: (B, T, 6) - s1のみからの粗い再構成
                z:     (B, T, 6) - s1+s2全体からの精密再構成
            bsq_loss: BSQ損失 (commit + entropy)
            quantized: (B, T, 20) - 量子化済みベクトル
            z_indices: インデックス

        処理フロー:
            x (B,T,6)
            → embed (B,T,256)
            → encoder (B,T,256)
            → quant_embed (B,T,20)
            → BSQ量子化 (B,T,20) ∈ {-1/√20, +1/√20}
            ├→ [:,:,:10] (s1) → post_quant_pre (B,T,256) → decoder → head → z_pre (B,T,6)
            └→ [:,:,:20] (all) → post_quant (B,T,256)    → decoder → head → z (B,T,6)
        """
        # === Encoder ===
        z = self.embed(x)                    # (B, T, 6) → (B, T, 256)
        for layer in self.encoder:
            z = layer(z)                     # (B, T, 256)
        z = self.quant_embed(z)              # (B, T, 256) → (B, T, 20)

        # === BSQ量子化 ===
        bsq_loss, quantized, z_indices = self.tokenizer(z)
        # quantized: (B, T, 20)

        # === Decoder (Coarse: s1のみ) ===
        quantized_pre = quantized[:, :, :self.s1_bits]  # (B, T, 10) - s1部分のみ
        z_pre = self.post_quant_embed_pre(quantized_pre) # (B, T, 10) → (B, T, 256)
        for layer in self.decoder:
            z_pre = layer(z_pre)                         # (B, T, 256)
        z_pre = self.head(z_pre)                         # (B, T, 256) → (B, T, 6)

        # === Decoder (Fine: 全体) ===
        z_full = self.post_quant_embed(quantized)        # (B, T, 20) → (B, T, 256)
        for layer in self.decoder:
            z_full = layer(z_full)                       # (B, T, 256)
        z_full = self.head(z_full)                       # (B, T, 256) → (B, T, 6)

        return (z_pre, z_full), bsq_loss, quantized, z_indices

    def encode(self, x, half=False):
        """
        エンコード: K線データ → トークンインデックス

        入力: x (B, T, 6) - 正規化済みK線
        出力:
            half=True  → [s1_ids (B,T), s2_ids (B,T)]  各 ∈ [0, 1023]
            half=False → full_ids (B, T) ∈ [0, 2^20 - 1]
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        _, _, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        """
        デコード: トークンインデックス → K線データ

        入力:
            half=True  → x = [s1_ids (B,T), s2_ids (B,T)]
            half=False → x = full_ids (B, T)
        出力: (B, T, 6) - 再構成K線

        処理:
            1. インデックス → バイナリビット列
            2. ビット列 → {-1/√K, +1/√K} にスケーリング
            3. post_quant_embed → decoder → head
        """
        quantized = self._indices_to_bits(x, half)       # (B, T, 20)
        z = self.post_quant_embed(quantized)             # (B, T, 20) → (B, T, 256)
        for layer in self.decoder:
            z = layer(z)                                 # (B, T, 256)
        z = self.head(z)                                 # (B, T, 256) → (B, T, 6)
        return z

    def _indices_to_bits(self, x, half=False):
        """
        整数インデックス → スケーリング済みバイナリベクトル

        例 (codebook_dim=4):
            index=5 → bits=[1,0,1,0] → scaled=[-1,+1,-1,+1] * (1/√4) = [-0.5, 0.5, -0.5, 0.5]
        """
        codebook_dim = self.s1_bits + self.s2_bits

        if half:
            # s1とs2の各インデックスを別々にビット展開し結合
            s1_idx, s2_idx = x[0], x[1]
            half_dim = codebook_dim // 2
            mask = 2 ** torch.arange(half_dim, device=s1_idx.device, dtype=torch.long)
            s1_bits = ((s1_idx.unsqueeze(-1) & mask) != 0)  # (B, T, 10)
            s2_bits = ((s2_idx.unsqueeze(-1) & mask) != 0)  # (B, T, 10)
            bits = torch.cat([s1_bits, s2_bits], dim=-1)     # (B, T, 20)
        else:
            mask = 2 ** torch.arange(codebook_dim, device=x.device, dtype=torch.long)
            bits = ((x.unsqueeze(-1) & mask) != 0)           # (B, T, 20)

        # {False, True} → {-1, +1} → スケーリング
        bits = bits.float() * 2 - 1       # (B, T, 20) ∈ {-1, +1}
        q_scale = 1.0 / (codebook_dim ** 0.5)
        return bits * q_scale              # (B, T, 20) ∈ {-1/√20, +1/√20}


# ============================================================
# 共通コンポーネント (トークナイザ用)
# ============================================================

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block (RMSNorm)

    構造: x → RMSNorm → Self-Attention → Residual → RMSNorm → SwiGLU FFN → Residual

    注意: トークナイザ内ではcausal maskなし (双方向Attention)
          ※ メインモデル(Kronos)ではcausal mask使用

    入力/出力: (B, T, d_model)
    """

    def __init__(self, d_model, n_heads, ff_dim, dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU_FFN(d_model, ff_dim, dropout_p)

    def forward(self, x):
        """
        入力: x (B, T, d_model)
        出力: x (B, T, d_model)
        """
        # Self-Attention + Residual
        residual = x
        x = self.norm1(x)
        x = residual + self.self_attn(x)

        # FFN + Residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    LayerNormと異なり平均の引き算をしない → 計算効率が良い

    x_norm = x / √(mean(x²) + ε) * weight

    入力/出力: (*, dim)
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MultiHeadAttention(nn.Module):
    """Self-Attention (RoPEなし、簡略化版 - トークナイザ用)"""

    def __init__(self, d_model, n_heads, dropout_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, n_heads, T, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)
        # out: (B, n_heads, T, head_dim)

        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class SwiGLU_FFN(nn.Module):
    """
    SwiGLU Feed-Forward Network

    LLaMA等で使用される効率的なFFN

    FFN(x) = W2 * (SiLU(W1*x) ⊙ W3*x)

    通常のFFN: 2 * d_model * ff_dim パラメータ
    SwiGLU:    3 * d_model * ff_dim パラメータ (ゲート機構のW3追加)

    入力/出力: (B, T, d_model)
    """

    def __init__(self, d_model, ff_dim, dropout_p=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)   # Gate
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)   # Up projection
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)   # Down projection
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        入力: (B, T, d_model)
        中間: (B, T, ff_dim)   ← SiLU(W1*x) ⊙ W3*x
        出力: (B, T, d_model)  ← W2 * 中間
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ============================================================
# 使用例
# ============================================================

if __name__ == "__main__":
    # トークナイザの初期化
    tokenizer = KronosTokenizer(
        d_in=6,        # OHLCVA
        d_model=256,
        n_heads=4,
        ff_dim=512,
        n_enc_layers=3,
        n_dec_layers=3,
        s1_bits=10,    # Coarse: 語彙1024
        s2_bits=10,    # Fine: 語彙1024
    )

    # サンプル入力 (正規化済みK線データ)
    B, T = 4, 128
    x = torch.randn(B, T, 6)  # (4, 128, 6)

    # === 学習時 ===
    (z_pre, z_fine), bsq_loss, quantized, z_indices = tokenizer(x)
    # z_pre:     (4, 128, 6)  - Coarse再構成
    # z_fine:    (4, 128, 6)  - Fine再構成
    # bsq_loss:  scalar
    # quantized: (4, 128, 20) - 量子化済み

    L_coarse = F.mse_loss(x, z_pre)
    L_fine = F.mse_loss(x, z_fine)
    L_total = L_coarse + L_fine + bsq_loss
    print(f"L_coarse={L_coarse:.4f}, L_fine={L_fine:.4f}, BSQ={bsq_loss:.4f}")

    # === 推論時: エンコード ===
    s1_ids, s2_ids = tokenizer.encode(x, half=True)
    # s1_ids: (4, 128) ∈ [0, 1023]
    # s2_ids: (4, 128) ∈ [0, 1023]
    print(f"s1 range: [{s1_ids.min()}, {s1_ids.max()}]")
    print(f"s2 range: [{s2_ids.min()}, {s2_ids.max()}]")

    # === 推論時: デコード ===
    x_recon = tokenizer.decode([s1_ids, s2_ids], half=True)
    # x_recon: (4, 128, 6)
    print(f"Reconstruction MSE: {F.mse_loss(x, x_recon):.4f}")
