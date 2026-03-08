"""
BitNet b1.58 量子化の核心ロジック

1.58-bit (三値 {-1, 0, 1}) 量子化とSTE (Straight-Through Estimator) の実装。
公式実装 (ggml-bitnet-mad.cpp) のPython簡略化版。

参考: BitNet Distillation (arXiv:2510.13998)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 基本量子化関数
# ==============================================================================

def round_clip(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    RoundClip関数 (Eq. 2)

    RoundClip(Y, a, b) = min(max(round(Y), a), b)

    三値量子化: a=-1, b=1 → {-1, 0, 1}
    INT8量子化: a=-128, b=127
    """
    return torch.clamp(torch.round(x), min_val, max_val)


def quantize_weights_ternary(weight: torch.Tensor, eps: float = 1e-8):
    """
    重みの三値量子化 (Eq. 1-2)

    Q_w(W) = Δ × RoundClip(W_FP16 / (Δ + ε), -1, 1)
    Δ = mean(|W|)   ← per-tensor absmean

    Args:
        weight: FP16重み (out_features, in_features)

    Returns:
        w_quant: 三値量子化済み重み (out_features, in_features)
        scale: スケールファクター Δ (スカラー)

    処理の流れ:
        W_FP16 → |W|の平均(Δ)を計算
                → W/Δ に正規化
                → round して {-1, 0, 1} に離散化
                → Δ を掛けて元のスケールに復元
    """
    # Per-tensor absmean スケールファクター
    scale = weight.abs().mean()  # Δ = mean(|W|)

    # 正規化 → 四捨五入 → クリッピング
    w_normalized = weight / (scale + eps)
    w_quant = round_clip(w_normalized, -1.0, 1.0)  # {-1, 0, 1}

    # スケール復元
    return w_quant * scale, scale


def quantize_activations_int8(x: torch.Tensor, eps: float = 1e-8):
    """
    活性化の8-bit量子化 (Eq. 3)

    Q_INT8(X) = (γ/127) × RoundClip(127·X / (γ+ε), -128, 127)
    γ = max(|X|)   ← per-token absmax

    Args:
        x: FP16活性化 (B, T, D)

    Returns:
        x_quant: INT8量子化済み活性化 (B, T, D)
        scale: per-tokenスケール (B, T, 1)

    注意: per-token量子化 (各トークン独立にスケーリング)
    """
    # Per-token absmax
    gamma = x.abs().amax(dim=-1, keepdim=True)  # (B, T, 1)

    # INT8にスケーリング
    x_scaled = 127.0 * x / (gamma + eps)
    x_quant = round_clip(x_scaled, -128.0, 127.0)

    # 復元用スケール
    scale = gamma / 127.0  # (B, T, 1)

    return x_quant, scale


# ==============================================================================
# STE (Straight-Through Estimator)
# ==============================================================================

def ste_quantize(weight: torch.Tensor, scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Straight-Through Estimator (STE) による三値量子化

    問題:
      RoundClip は不連続関数 → 勾配がほぼ0
      → 通常の逆伝播では重みが更新されない

    解決策:
      順伝播: 量子化を適用 (RoundClip)
      逆伝播: 量子化をスキップ (恒等関数として扱う)

    数学的表現:
      順伝播: w_q = RoundClip(w/Δ, -1, 1) × Δ
      逆伝播: ∂L/∂w ≈ ∂L/∂w_q  (量子化を無視)

    実装トリック (.detach() 方式):
      w_q = w + (quantize(w) - w).detach()

      なぜこれでSTEになるのか:
        - .detach() は (quantize(w) - w) の勾配計算を切断する
        - 値としては w + (quantize(w) - w) = quantize(w) → 量子化された値
        - 勾配としては ∂w_q/∂w = 1 (detach部分の勾配は0)
        - つまり順伝播では量子化が適用され、逆伝播ではwへ勾配がそのまま通る

    Args:
        weight: FP16マスター重み (out_features, in_features)
        scale: スケールファクター Δ (スカラー)

    Returns:
        w_quant: STE付き三値量子化重み
    """
    w_normalized = weight / (scale + eps)
    w_quant = torch.clamp(torch.round(w_normalized), -1.0, 1.0) * scale
    # STE: 値は量子化済み、勾配はweight方向にそのまま通す
    return weight + (w_quant - weight).detach()


# ==============================================================================
# BitLinear: 1.58-bit 線形層
# ==============================================================================

class BitLinear(nn.Module):
    """
    BitNet b1.58 の線形層

    通常のnn.Linearとの違い:
      - 重み: FP16 → {-1, 0, 1} (1.58-bit)
      - 活性化: FP16 → INT8 (8-bit)
      - 乗算 → 加減算 (重みが三値のため)

    学習時:
      1. FP16のマスター重みを保持
      2. 順伝播で三値に量子化 (STE)
      3. 逆伝播でFP16マスター重みを更新
      → 量子化を意識した学習 (QAT)

    推論時:
      1. 重みは事前に三値化・パック済み
      2. 活性化をINT8に量子化
      3. 三値×INT8の積和演算 (SIMD)

    数学的表現:
      Y = Q_a(X) @ Q_w(W)ᵀ
      Q_w(W) = Δ × RoundClip(W/(Δ+ε), -1, 1)
      Q_a(X) = (γ/127) × RoundClip(127·X/(γ+ε), -128, 127)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP16マスター重み (学習時に更新)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # 初期化
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力テンソル (B, T, in_features)

        Returns:
            output: 出力テンソル (B, T, out_features)

        処理フロー:
            x (FP16) → INT8量子化
            W (FP16) → 三値量子化 (STE)
            output = x_q @ w_q^T (スケール補正付き)
        """
        # 重み量子化 (1.58-bit, STE付き)
        scale_w = self.weight.abs().mean()
        if self.training:
            # 学習時: STEで勾配を通す
            w_quant = ste_quantize(self.weight, scale_w)
        else:
            # 推論時: 単純に量子化
            w_quant = scale_w * round_clip(self.weight / (scale_w + 1e-8), -1.0, 1.0)

        # 活性化量子化 (INT8)
        x_quant, scale_x = quantize_activations_int8(x)

        # 行列積 (量子化済み)
        # 実際のハードウェアでは三値×INT8は加減算のみ
        output = F.linear(x_quant, w_quant, self.bias)

        # スケール補正
        output = output * scale_x

        return output


# ==============================================================================
# SubLN (Sub-Layer Normalization)
# ==============================================================================

class SubLayerNorm(nn.Module):
    """
    SubLN: Sub-Layer Normalization

    BitDistillで導入された安定化手法。
    MHSA出力投影前とFFN出力投影前にLayerNormを追加。

    問題:
      1.58-bit量子化 → 活性化の分散が爆発
      → 次の量子化層で情報損失
      → 学習不安定

    解決策:
      量子化層に入る前にLayerNormで分散を安定化

    配置:
      Y = X + SubLN(Concat(heads)) × W_out     [MHSA内]
      X' = Y + SubLN(gate ⊙ up) × W_down       [FFN内]

    効果:
      SubLNなし: 74.09% → SubLNあり: 76.30% (MNLI, +2.2pt)
    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# ==============================================================================
# BitNet Transformer Layer
# ==============================================================================

class BitNetTransformerLayer(nn.Module):
    """
    SubLN付き BitNet Transformer Layer

    Qwen3アーキテクチャベース:
      Y_l = X_l + SubLN(Concat(heads)) × W^MHSA_out     (Eq. 4)
      X_{l+1} = Y_l + SubLN(gate ⊙ up) × W^FFN_down     (Eq. 5)

    全てのLinear層が BitLinear (1.58-bit)
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Input LayerNorm
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)

        # Multi-Head Self-Attention (全て BitLinear)
        self.q_proj = BitLinear(hidden_dim, hidden_dim)
        self.k_proj = BitLinear(hidden_dim, hidden_dim)
        self.v_proj = BitLinear(hidden_dim, hidden_dim)
        self.o_proj = BitLinear(hidden_dim, hidden_dim)

        # SubLN for MHSA
        self.sub_ln_attn = SubLayerNorm(hidden_dim)

        # FFN (SwiGLU, 全て BitLinear)
        self.gate_proj = BitLinear(hidden_dim, ffn_dim)
        self.up_proj = BitLinear(hidden_dim, ffn_dim)
        self.down_proj = BitLinear(ffn_dim, hidden_dim)

        # SubLN for FFN
        self.sub_ln_ffn = SubLayerNorm(ffn_dim)

    def forward(self, x: torch.Tensor, attention_mask=None):
        """
        Args:
            x: (B, T, D) 入力隠れ状態
            attention_mask: (B, 1, T, T) or None

        Returns:
            output: (B, T, D) 出力隠れ状態
            attn_weights: (B, H, T, T) Attention重み (蒸留用)

        処理フロー:
            1. LayerNorm → Q,K,V投影 (BitLinear)
            2. Scaled Dot-Product Attention
            3. SubLN → 出力投影 (BitLinear) → 残差接続
            4. LayerNorm → Gate,Up投影 (BitLinear)
            5. SwiGLU活性化
            6. SubLN → Down投影 (BitLinear) → 残差接続
        """
        B, T, D = x.shape
        residual = x

        # === Multi-Head Self-Attention ===
        x_norm = self.input_layernorm(x)

        # Q, K, V 投影 (BitLinear: 1.58-bit)
        q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, d_head)

        # Concat + SubLN + 出力投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.sub_ln_attn(attn_output)  # ← SubLN
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # === Feed-Forward Network (SwiGLU) ===
        residual = x
        x_norm = self.post_attention_layernorm(x)

        gate = torch.sigmoid(self.gate_proj(x_norm))  # σ(x × W_gate)
        up = self.up_proj(x_norm)                       # x × W_up
        ffn_hidden = gate * up                          # SwiGLU: gate ⊙ up

        ffn_hidden = self.sub_ln_ffn(ffn_hidden)        # ← SubLN
        output = self.down_proj(ffn_hidden)
        x = residual + output

        return x, attn_weights


# ==============================================================================
# 重みパッキング (推論用)
# ==============================================================================

def pack_ternary_weights(w_ternary: torch.Tensor) -> torch.Tensor:
    """
    三値重みをバイト列にパッキング

    4つの三値 (2 bits each) → 1バイト
    エンコーディング: -1→0b00, 0→0b01, 1→0b10

    公式実装 (ggml-bitnet-mad.cpp) のquantize_i2_s関数に対応

    Args:
        w_ternary: 三値テンソル {-1, 0, 1}  (rows, cols)

    Returns:
        packed: パック済み uint8 テンソル (rows, cols // 4)

    例:
        [1, -1, 0, 1] → [0b10_00_01_10] = 1バイト
    """
    assert w_ternary.shape[-1] % 4 == 0, "列数は4の倍数が必要"

    # {-1, 0, 1} → {0, 1, 2} にマッピング
    w_mapped = (w_ternary + 1).to(torch.uint8)  # -1→0, 0→1, 1→2

    # 4つずつパック
    rows, cols = w_ternary.shape
    w_reshaped = w_mapped.view(rows, cols // 4, 4)

    packed = (
        (w_reshaped[:, :, 0] << 6) |
        (w_reshaped[:, :, 1] << 4) |
        (w_reshaped[:, :, 2] << 2) |
        (w_reshaped[:, :, 3] << 0)
    )

    return packed.to(torch.uint8)


def unpack_ternary_weights(packed: torch.Tensor, cols: int) -> torch.Tensor:
    """
    パック済み重みを三値に展開

    Args:
        packed: パック済み uint8 テンソル (rows, cols // 4)
        cols: 元の列数

    Returns:
        w_ternary: 三値テンソル {-1, 0, 1} (rows, cols)
    """
    rows = packed.shape[0]

    w0 = (packed >> 6) & 0x03
    w1 = (packed >> 4) & 0x03
    w2 = (packed >> 2) & 0x03
    w3 = (packed >> 0) & 0x03

    w_mapped = torch.stack([w0, w1, w2, w3], dim=-1).view(rows, cols)

    # {0, 1, 2} → {-1, 0, 1}
    return w_mapped.to(torch.float32) - 1.0


# ==============================================================================
# 使用例
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("BitNet b1.58 量子化デモ")
    print("=" * 60)

    # --- 重み量子化 ---
    print("\n--- 重みの三値量子化 ---")
    weight = torch.randn(4, 8)
    print(f"元の重み:\n{weight}")

    w_quant, scale = quantize_weights_ternary(weight)
    w_ternary = round_clip(weight / (scale + 1e-8), -1.0, 1.0)
    print(f"\nスケール (Δ): {scale:.4f}")
    print(f"三値重み:\n{w_ternary}")
    print(f"量子化後重み (三値×Δ):\n{w_quant}")

    # 三値の分布
    unique, counts = torch.unique(w_ternary, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  値 {int(u.item()):+d}: {c.item()} 個 ({c.item()/w_ternary.numel()*100:.1f}%)")

    # --- 活性化量子化 ---
    print("\n--- 活性化のINT8量子化 ---")
    x = torch.randn(1, 2, 8)
    x_quant, scale_x = quantize_activations_int8(x)
    print(f"元の活性化: {x[0, 0, :4]}")
    print(f"INT8量子化: {x_quant[0, 0, :4]}")
    print(f"復元: {(x_quant * scale_x)[0, 0, :4]}")
    print(f"量子化誤差: {(x - x_quant * scale_x).abs().mean():.6f}")

    # --- BitLinear ---
    print("\n--- BitLinear 線形層 ---")
    layer = BitLinear(64, 32)
    x = torch.randn(2, 4, 64)
    output = layer(x)
    print(f"入力: {x.shape} → 出力: {output.shape}")

    # --- パッキング ---
    print("\n--- 三値重みのパッキング ---")
    w = torch.tensor([[-1, 0, 1, 1, -1, -1, 0, 0],
                       [1, 1, 0, -1, 0, 1, -1, 0]], dtype=torch.float32)
    packed = pack_ternary_weights(w)
    unpacked = unpack_ternary_weights(packed, 8)
    print(f"元の重み: {w[0].tolist()}")
    print(f"パック済み: {packed[0].tolist()} (2バイト)")
    print(f"展開後: {unpacked[0].tolist()}")
    print(f"一致: {torch.allclose(w, unpacked)}")

    # --- メモリ比較 ---
    print("\n--- メモリ使用量比較 ---")
    rows, cols = 4096, 4096
    fp16_bytes = rows * cols * 2
    packed_bytes = rows * (cols // 4) + rows * 4  # パック + スケール
    print(f"FP16: {fp16_bytes / 1024 / 1024:.1f} MB")
    print(f"1.58-bit (packed): {packed_bytes / 1024 / 1024:.1f} MB")
    print(f"圧縮率: {fp16_bytes / packed_bytes:.1f}x")
