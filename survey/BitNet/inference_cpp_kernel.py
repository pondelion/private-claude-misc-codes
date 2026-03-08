"""
C++推論カーネルの疑似コード (Python表現)

公式実装 (src/ggml-bitnet-mad.cpp) のSIMD最適化カーネルの動作を
Pythonで表現した疑似コード。

公式実装の言語: C++ (AVX2/SSSE3/NEON intrinsics)
この疑似コード: Python (動作の理解用、実用速度ではない)

参考: https://github.com/microsoft/BitNet
"""

import numpy as np
from typing import Tuple


# ==============================================================================
# 定数 (公式実装 gemm-config.h より)
# ==============================================================================

# 量子化ブロックサイズ
QK_I2_S_X86 = 128   # x86/SSE3
QK_I2_S_ARM = 64    # ARM NEON

# 並列化設定 (x86 AVX2 デフォルト)
ROW_BLOCK_SIZE = 4       # 行タイルサイズ
COL_BLOCK_SIZE = 128     # 列タイルサイズ
PARALLEL_SIZE = 4        # 並列処理行数

# ARM NEON (DOTPROD) デフォルト
# ROW_BLOCK_SIZE = 8
# COL_BLOCK_SIZE = 256
# PARALLEL_SIZE = 8


# ==============================================================================
# 三値量子化 & パッキング
# ==============================================================================

def quantize_i2_s(src: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    FP32 → I2_S (2-bit ternary) 量子化
    公式実装: quantize_i2_s() in ggml-bitnet-mad.cpp

    エンコーディング:
      f32 > 0  → 2 (0b10)  = +1
      f32 ≈ 0  → 1 (0b01)  =  0
      f32 < 0  → 0 (0b00)  = -1

    パッキング: 4つの2-bit値 → 1バイト
      [w3|w2|w1|w0] = byte (MSBから順)

    Args:
        src: FP32重みベクトル (n,)

    Returns:
        packed: パック済みバイト列 (n//4,)
        scale: 最大絶対値 (復元用)

    対応する公式実装:
    ```c
    size_t quantize_i2_s(const float * src, void * dst,
                          int64_t nrow, int64_t n_per_row)
    ```
    """
    n = len(src)
    assert n % 4 == 0

    # Step 1: 最大絶対値を計算 (スケールファクター)
    max_val = np.max(np.abs(src))
    scale = max_val if max_val > 0 else 1.0

    # Step 2: 三値にマッピング
    # f32 > 0 → 2, f32 ≈ 0 → 1, f32 < 0 → 0
    q = np.ones(n, dtype=np.uint8)  # デフォルト: 0 (encoded as 1)
    q[src > 0] = 2   # +1
    q[src < 0] = 0   # -1

    # Step 3: 4値を1バイトにパッキング
    packed = np.zeros(n // 4, dtype=np.uint8)
    for i in range(n // 4):
        packed[i] = (
            (q[4 * i + 0] << 6) |
            (q[4 * i + 1] << 4) |
            (q[4 * i + 2] << 2) |
            (q[4 * i + 3] << 0)
        )

    return packed, scale


def dequantize_i2_s(packed: np.ndarray, scale: float, n: int) -> np.ndarray:
    """
    I2_S → FP32 復元 (推論時)

    2-bitエンコーディング → {-1, 0, 1} → ×scale

    Args:
        packed: パック済みバイト列 (n//4,)
        scale: スケールファクター
        n: 元のベクトル長

    Returns:
        dst: 復元された値 (n,)  ※三値×scale
    """
    dst = np.zeros(n, dtype=np.float32)

    for i in range(n // 4):
        byte = packed[i]
        for j in range(4):
            # 2-bit抽出 (MSBから)
            val = (byte >> (6 - 2 * j)) & 0x03
            # {0, 1, 2} → {-1, 0, 1}
            dst[4 * i + j] = (val - 1) * scale

    return dst


# ==============================================================================
# 活性化量子化 (INT8)
# ==============================================================================

def quantize_activation_int8(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    活性化のINT8量子化 (per-token absmax)

    公式実装: BitLinearKernel.quant_input() in gpu/model.py
    および推論時のINT8量子化

    γ = max(|X|)
    Q(X) = RoundClip(127 × X / γ, -128, 127)
    scale = γ / 127

    Args:
        x: FP32活性化ベクトル (n,)

    Returns:
        x_q: INT8量子化済み (n,)
        scale: 復元用スケール
    """
    gamma = np.max(np.abs(x))
    if gamma < 1e-8:
        return np.zeros_like(x, dtype=np.int8), 0.0

    scale = gamma / 127.0
    x_scaled = x / scale
    x_q = np.clip(np.round(x_scaled), -128, 127).astype(np.int8)

    return x_q, scale


# ==============================================================================
# 行列ベクトル積 (GEMV) カーネル
# ==============================================================================

def ternary_dot_product_naive(
    weights_packed: np.ndarray,
    activations_int8: np.ndarray,
    weight_scale: float,
    act_scale: float,
    n: int,
) -> float:
    """
    三値重み × INT8活性化 のドット積 (ナイーブ実装)

    対応する公式実装:
      ggml_vec_dot_i2_i8_s_1x1() in ggml-bitnet-mad.cpp

    実際には {-1,0,1} × INT8 なので:
      dot = Σ_{w=1} x_i - Σ_{w=-1} x_i

    加減算のみで計算可能!

    Args:
        weights_packed: パック済み三値重み (n//4,)
        activations_int8: INT8活性化 (n,)
        weight_scale: 重みスケール
        act_scale: 活性化スケール
        n: ベクトル長

    Returns:
        result: ドット積結果 (FP32)
    """
    accumulator = 0

    for i in range(n // 4):
        byte = weights_packed[i]

        for j in range(4):
            # 2-bit抽出
            w_encoded = (byte >> (6 - 2 * j)) & 0x03
            # {0, 1, 2} → {-1, 0, 1}
            w = w_encoded - 1

            x = int(activations_int8[4 * i + j])

            # 三値なので: if w==1: acc+=x, elif w==-1: acc-=x
            accumulator += w * x

    return float(accumulator) * weight_scale * act_scale


def ternary_matvec_parallel(
    weights_packed: np.ndarray,
    activations_int8: np.ndarray,
    weight_scale: float,
    act_scale: float,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    並列化された行列ベクトル積 (公式実装の簡略版)

    対応する公式実装:
      ggml_vec_dot_i2_i8_s_1xN() in ggml-bitnet-mad.cpp

    並列化戦略 (ACT_PARALLEL):
      - 複数の出力行を同時に処理
      - 活性化ベクトルの読み込みを共有
      - PARALLEL_SIZE行を一度に計算

    タイリング:
      ┌─────────────────────────────────┐
      │ ROW_BLOCK_SIZE × COL_BLOCK_SIZE │  ← 1タイル
      │ (4行 × 128列)                   │
      └─────────────────────────────────┘
      → PARALLEL_SIZE行を同時処理

    パフォーマンス向上:
      - 重みの展開コストを複数行で分散
      - キャッシュライン効率の最適化
      - 1.15x-2.1x の追加高速化

    Args:
        weights_packed: (rows, cols//4) パック済み重み
        activations_int8: (cols,) INT8活性化
        weight_scale: 重みスケール
        act_scale: 活性化スケール
        rows: 出力次元
        cols: 入力次元

    Returns:
        output: (rows,) 行列ベクトル積の結果
    """
    output = np.zeros(rows, dtype=np.float32)

    # PARALLEL_SIZE行を同時に処理
    for row_start in range(0, rows, PARALLEL_SIZE):
        row_end = min(row_start + PARALLEL_SIZE, rows)
        num_parallel = row_end - row_start

        # アキュムレータ (各行用)
        accumulators = np.zeros(num_parallel, dtype=np.int32)

        # COL_BLOCK_SIZE列ずつ処理 (タイリング)
        for col_start in range(0, cols, COL_BLOCK_SIZE):
            col_end = min(col_start + COL_BLOCK_SIZE, cols)

            # 活性化ベクトルの読み込み (全行で共有)
            act_block = activations_int8[col_start:col_end]

            for row_offset in range(num_parallel):
                row = row_start + row_offset
                w_start = col_start // 4
                w_end = col_end // 4

                # 三値展開 + 積和演算
                for i in range(w_start, w_end):
                    byte = weights_packed[row, i]
                    for j in range(4):
                        w = ((byte >> (6 - 2 * j)) & 0x03) - 1
                        idx = (i - w_start) * 4 + j
                        if idx < len(act_block):
                            accumulators[row_offset] += w * int(act_block[idx])

        # スケール適用
        for i in range(num_parallel):
            output[row_start + i] = float(accumulators[i]) * weight_scale * act_scale

    return output


# ==============================================================================
# AVX2 SIMD の動作シミュレーション
# ==============================================================================

def avx2_unpack_ternary_simulate(packed_bytes: np.ndarray) -> np.ndarray:
    """
    AVX2での2-bit展開をシミュレーション

    公式実装での処理:
    ```c
    // 32バイトロード → 128個の三値を展開
    __m256i packed = _mm256_loadu_si256(weights);

    // 各2-bitを抽出
    __m256i q0 = _mm256_and_si256(packed, 0x03);           // bits [0:2]
    __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), 0x03);
    __m256i q2 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), 0x03);
    __m256i q3 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), 0x03);
    ```

    1バイト (8 bits) → 4つの2-bit値に展開

    Args:
        packed_bytes: パック済みバイト列 (32,) = 128三値分

    Returns:
        unpacked: 展開されたINT8値 (128,)  {0, 1, 2}
    """
    unpacked = np.zeros(len(packed_bytes) * 4, dtype=np.int8)

    for i, byte in enumerate(packed_bytes):
        # LSBから展開 (AVX2の_mm256_and_si256に対応)
        unpacked[4 * i + 3] = (byte >> 0) & 0x03
        unpacked[4 * i + 2] = (byte >> 2) & 0x03
        unpacked[4 * i + 1] = (byte >> 4) & 0x03
        unpacked[4 * i + 0] = (byte >> 6) & 0x03

    return unpacked


def avx2_maddubs_simulate(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    AVX2 _mm256_maddubs_epi16 のシミュレーション

    この命令は:
      - aを符号なしint8 (0-255) として扱う
      - bを符号ありint8 (-128~127) として扱う
      - 隣接する2つのペアの積和を計算
      - 結果はint16

    result[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]

    BitNetでの使い方:
      a = 三値(0,1,2として) → unsigned
      b = 活性化(INT8) → signed
      → 積和で三値×INT8のドット積を効率的に計算

    Args:
        a: 符号なしint8 (32,)  ← 三値エンコーディング
        b: 符号ありint8 (32,)  ← 活性化

    Returns:
        result: int16 (16,)  ← ペアワイズ積和
    """
    assert len(a) == len(b) == 32
    result = np.zeros(16, dtype=np.int16)

    for i in range(16):
        # 隣接2ペアの積和
        val = int(a[2 * i]) * int(b[2 * i]) + int(a[2 * i + 1]) * int(b[2 * i + 1])
        # INT16に飽和クリップ
        result[i] = np.clip(val, -32768, 32767)

    return result


# ==============================================================================
# GPU CUDA カーネルの疑似コード
# ==============================================================================

def cuda_w2a8_gemv_simulate(
    weights_packed: np.ndarray,
    activations_int8: np.ndarray,
    weight_scale: float,
    act_scale: float,
    M: int,  # 出力次元
    K: int,  # 入力次元
) -> np.ndarray:
    """
    GPU CUDAカーネル (W2A8 GEMV) のシミュレーション

    公式実装: bitlinear_int8xint2() in gpu/bitnet_kernels/bitnet_kernels.cu

    CUDA実装の特徴:
      1. dp4a命令: 4つのint8ペアのドット積を1命令で計算
      2. 重みオンザフライ復号: 2-bit → int8 変換
      3. LOP3命令: 高速ビット操作
      4. 共有メモリ: 活性化のキャッシュ

    復号ロジック (decode_i2s_to_i8s):
      パック済み2-bit → int8
      mapping: 0→-1, 1→0, 2→1

    パフォーマンス (A100):
      W2A8カーネル: 13-30μs
      BF16比: 2.89-3.27倍高速

    Args:
        weights_packed: (M, K//4) パック済み2-bit重み
        activations_int8: (K,) INT8活性化
        weight_scale: 重みスケール
        act_scale: 活性化スケール
        M: 出力次元
        K: 入力次元

    Returns:
        output: (M,) 出力ベクトル
    """
    output = np.zeros(M, dtype=np.float32)

    # 各出力行を処理 (実際にはCUDAスレッドで並列)
    for m in range(M):
        acc = 0

        # dp4a: 4要素ずつ処理
        for k in range(0, K, 4):
            byte_idx = k // 4
            byte = weights_packed[m, byte_idx]

            # decode_i2s_to_i8s: 2-bit → int8
            # 0→-1, 1→0, 2→1
            for j in range(4):
                w_encoded = (byte >> (6 - 2 * j)) & 0x03
                w_int8 = w_encoded - 1  # {-1, 0, 1}
                x_int8 = int(activations_int8[k + j])

                # dp4a の一部: 積和演算
                acc += w_int8 * x_int8

        output[m] = float(acc) * weight_scale * act_scale

    return output


# ==============================================================================
# Embedding量子化 (Q6_K)
# ==============================================================================

def quantize_embedding_q6k(embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embedding層のQ6_K量子化

    公式実装: utils/quantize_embeddings.py

    Q6_K: 6-bit量子化 (64レベル)
      - Embeddingは1.58-bitには量子化しない
      - Q6_K がパープレキシティと速度のバランスが最良

    精度への影響:
      F32: ベースライン
      Q8_0: PPL増加 <0.1%
      Q6_K: PPL増加 <0.5%  ← 推奨
      Q4_0: PPL増加 ~3-5%

    Args:
        embedding: (vocab_size, embed_dim) FP32 embedding

    Returns:
        quantized: (vocab_size, embed_dim) INT8
        scales: (vocab_size, embed_dim // block_size) スケール
    """
    block_size = 16  # Q6_K block size
    vocab_size, embed_dim = embedding.shape

    assert embed_dim % block_size == 0
    num_blocks = embed_dim // block_size

    quantized = np.zeros_like(embedding, dtype=np.int8)
    scales = np.zeros((vocab_size, num_blocks), dtype=np.float32)

    for v in range(vocab_size):
        for b in range(num_blocks):
            start = b * block_size
            end = start + block_size
            block = embedding[v, start:end]

            # ブロック内の最大絶対値
            max_val = np.max(np.abs(block))
            scale = max_val / 31.0 if max_val > 0 else 1.0  # 6-bit: [-31, 31]

            # 量子化
            quantized[v, start:end] = np.clip(
                np.round(block / scale), -32, 31
            ).astype(np.int8)
            scales[v, b] = scale

    return quantized, scales


# ==============================================================================
# 完全な推論フロー
# ==============================================================================

def bitnet_inference_flow(
    weights_packed: list,    # 各レイヤーのパック済み重み
    weight_scales: list,     # 各レイヤーのスケール
    embedding: np.ndarray,   # Embedding行列
    input_token: int,        # 入力トークンID
    num_layers: int,
    hidden_dim: int,
) -> np.ndarray:
    """
    BitNet推論の完全フロー (1トークン生成)

    処理フロー:
      1. Embedding lookup (FP32 or Q6_K)
      2. 各レイヤー:
         a. LayerNorm
         b. 活性化INT8量子化
         c. 三値重み × INT8活性化 (SIMD/CUDA)
         d. SubLN + 残差接続
      3. LM Head → logits

    公式実装:
      CPU: run_inference.py → llama.cpp + ggml-bitnet-mad.cpp
      GPU: gpu/generate.py → model.py + bitnet_kernels.cu

    Args:
        weights_packed: 各レイヤーのパック済み重み
        weight_scales: 各レイヤーの重みスケール
        embedding: (V, D) Embedding行列
        input_token: トークンID
        num_layers: レイヤー数
        hidden_dim: 隠れ層次元

    Returns:
        logits: (V,) 語彙上のスコア
    """
    # Step 1: Embedding
    x = embedding[input_token].copy()  # (D,)

    # Step 2: Transformer Layers
    for layer_idx in range(num_layers):
        residual = x.copy()

        # LayerNorm (簡略化)
        x = (x - x.mean()) / (x.std() + 1e-5)

        # 活性化のINT8量子化
        x_q, act_scale = quantize_activation_int8(x)

        # 三値重み × INT8活性化 (メイン計算)
        # 実際にはQ,K,V,O,Gate,Up,Down の7つのMatMul
        w_packed = weights_packed[layer_idx]
        w_scale = weight_scales[layer_idx]

        y = ternary_matvec_parallel(
            w_packed, x_q, w_scale, act_scale,
            rows=hidden_dim, cols=hidden_dim,
        )

        # 残差接続
        x = residual + y

    # Step 3: LM Head (簡略化)
    logits = embedding @ x  # (V,)

    return logits


# ==============================================================================
# 使用例
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("C++推論カーネル 疑似コードデモ")
    print("=" * 60)

    # --- 三値量子化 ---
    print("\n--- 三値量子化 & パッキング ---")
    src = np.array([0.5, -0.3, 0.0, 0.8, -0.1, 0.0, 0.2, -0.7], dtype=np.float32)
    packed, scale = quantize_i2_s(src)
    restored = dequantize_i2_s(packed, scale, len(src))
    print(f"元の値:   {src}")
    print(f"パック済み: {packed} (2バイト)")
    print(f"復元:     {restored}")
    print(f"スケール:  {scale:.4f}")

    # --- ドット積 ---
    print("\n--- 三値×INT8 ドット積 ---")
    n = 16
    weights = np.random.randn(n).astype(np.float32)
    activations = np.random.randn(n).astype(np.float32)

    w_packed, w_scale = quantize_i2_s(weights)
    a_q, a_scale = quantize_activation_int8(activations)

    result = ternary_dot_product_naive(w_packed, a_q, w_scale, a_scale, n)
    expected = np.dot(weights, activations)
    print(f"三値×INT8: {result:.4f}")
    print(f"FP32:      {expected:.4f}")
    print(f"誤差:      {abs(result - expected):.4f}")

    # --- AVX2シミュレーション ---
    print("\n--- AVX2 SIMD シミュレーション ---")
    packed_32bytes = np.random.randint(0, 256, size=32, dtype=np.uint8)
    unpacked = avx2_unpack_ternary_simulate(packed_32bytes)
    print(f"パック済み: {packed_32bytes[:4]}... (32バイト)")
    print(f"展開後:    {unpacked[:16]}... (128値)")
    print(f"展開後の値域: {unpacked.min()} - {unpacked.max()}")

    # --- 行列ベクトル積 ---
    print("\n--- 並列行列ベクトル積 ---")
    rows, cols = 32, 64
    W = np.random.randn(rows, cols).astype(np.float32)

    # 行ごとにパック
    W_packed = np.zeros((rows, cols // 4), dtype=np.uint8)
    W_scale = 0.0
    for r in range(rows):
        packed_row, s = quantize_i2_s(W[r])
        W_packed[r] = packed_row
        W_scale = max(W_scale, s)

    x = np.random.randn(cols).astype(np.float32)
    x_q, x_scale = quantize_activation_int8(x)

    output = ternary_matvec_parallel(W_packed, x_q, W_scale, x_scale, rows, cols)
    expected = W @ x

    print(f"出力形状: {output.shape}")
    print(f"出力[0:4]: {output[:4]}")
    print(f"期待[0:4]: {expected[:4]}")
    print(f"平均絶対誤差: {np.mean(np.abs(output - expected)):.4f}")

    # --- メモリ比較 ---
    print("\n--- メモリ使用量 (4096×4096行列) ---")
    n = 4096
    fp32_mb = n * n * 4 / 1024 / 1024
    fp16_mb = n * n * 2 / 1024 / 1024
    i2_mb = n * (n // 4 + 4) / 1024 / 1024  # パック + スケール
    print(f"FP32:     {fp32_mb:.1f} MB")
    print(f"FP16:     {fp16_mb:.1f} MB")
    print(f"I2_S:     {i2_mb:.1f} MB")
    print(f"圧縮率 (vs FP16): {fp16_mb / i2_mb:.1f}x")
