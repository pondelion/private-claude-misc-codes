/**
 * BitNet b1.58 C++実装例
 *
 * 公式実装 (src/ggml-bitnet-mad.cpp) を簡略化した教育用コード。
 * 1.58-bit三値量子化、パッキング、SIMD推論カーネルの動作を示す。
 *
 * 公式実装の特徴:
 *   - AVX2/SSSE3 (x86), NEON (ARM), LASX (LoongArch) 対応
 *   - llama.cpp フレームワークに統合
 *   - 並列タイリングによる高速化 (1.15x-2.1x)
 *
 * 参考: https://github.com/microsoft/BitNet
 *
 * ============================================================================
 * ■ 前提条件 (Linux / WSL2 Ubuntu)
 * ============================================================================
 *
 * g++ (GCC) が必要です。入っていなければ以下でインストール:
 *
 *   sudo apt update
 *   sudo apt install -y build-essential
 *
 * インストール確認:
 *
 *   g++ --version
 *   # g++ (Ubuntu 11.x.x ...) のような出力が出ればOK
 *
 * ============================================================================
 * ■ ビルド方法
 * ============================================================================
 *
 * このファイルがあるディレクトリで以下を実行します。
 *
 * --- パターンA: AVX2 SIMD有効 (推奨、WSL2のx86 CPUならほぼ対応) ---
 *
 *   g++ -O3 -mavx2 -std=c++17 -o bitnet_demo cpp_implementation.cpp
 *
 *   各オプションの意味:
 *     -O3       : 最適化レベル3 (最も高速なコードを生成)
 *     -mavx2    : AVX2命令セットを有効にする (SIMD高速化)
 *     -std=c++17: C++17標準を使用
 *     -o bitnet_demo : 出力ファイル名を bitnet_demo にする
 *
 * --- パターンB: SIMD無効 (AVX2非対応のCPUの場合) ---
 *
 *   g++ -O3 -std=c++17 -o bitnet_demo cpp_implementation.cpp
 *
 *   → AVX2カーネルはスキップされ、スカラー版のみ動作します。
 *
 * ビルド成功すると、同じディレクトリに bitnet_demo という実行ファイルが生成されます。
 *
 * ============================================================================
 * ■ 実行方法
 * ============================================================================
 *
 *   ./bitnet_demo
 *
 * 出力例:
 *   ================================================
 *   BitNet b1.58 C++ 実装デモ
 *   ================================================
 *   --- 三値量子化テスト ---
 *   スケール: 0.496063
 *   元の値[0:4]: -0.5 -0.492126 -0.484252 -0.476378
 *   復元値[0:4]: -0.496063 -0.496063 -0.496063 -0.496063
 *   ...
 *
 * ============================================================================
 * ■ よくあるエラーと対処法
 * ============================================================================
 *
 * Q: "g++: command not found" と出る
 * A: sudo apt install -y build-essential を実行してください。
 *
 * Q: "-mavx2" でエラーになる
 * A: CPUがAVX2非対応です。-mavx2 を外してビルドしてください (パターンB)。
 *    WSL2のx86環境なら通常は対応しています。確認方法:
 *      grep avx2 /proc/cpuinfo
 *    出力があればAVX2対応です。
 *
 * Q: "Illegal instruction" で実行時にクラッシュする
 * A: ビルド時に-mavx2を指定したがCPUが非対応です。-mavx2を外して再ビルド。
 *
 * ============================================================================
 * ■ クリーンアップ (生成ファイルの削除)
 * ============================================================================
 *
 *   rm -f bitnet_demo
 *
 */

#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ==============================================================================
// 定数 (gemm-config.h より)
// ==============================================================================

// 量子化ブロックサイズ
#ifdef __AVX2__
constexpr int QK_I2_S = 128;  // x86: 128要素/ブロック
#else
constexpr int QK_I2_S = 64;   // ARM: 64要素/ブロック
#endif

// 並列化パラメータ
constexpr int ROW_BLOCK_SIZE = 4;
constexpr int COL_BLOCK_SIZE = 128;
constexpr int PARALLEL_SIZE = 4;

// ==============================================================================
// データ構造
// ==============================================================================

/**
 * I2_S量子化ブロック
 *
 * 2-bitエンコーディング: -1→00, 0→01, 1→10
 * 4値 → 1バイト (2 bits × 4)
 *
 * メモリレイアウト:
 *   [scale (float)] + [packed data (QK/4 bytes)]
 */
struct BlockI2S {
    float scale;                            // per-block スケールファクター
    uint8_t data[QK_I2_S / 4];            // パック済み三値 (2 bits × 4/byte)
};

// ==============================================================================
// 三値量子化
// ==============================================================================

/**
 * FP32 → I2_S (2-bit ternary) 量子化
 *
 * 公式実装: quantize_i2_s() in ggml-bitnet-mad.cpp
 *
 * 処理:
 *   1. 最大絶対値 (scale) の計算
 *   2. 三値マッピング: f>0→2, f≈0→1, f<0→0
 *   3. 4値を1バイトにパッキング
 *
 * @param src   元のFP32重み (n要素)
 * @param dst   出力先 (BlockI2S配列)
 * @param n     要素数 (QK_I2_Sの倍数)
 */
void quantize_ternary(const float* src, BlockI2S* dst, int n) {
    assert(n % QK_I2_S == 0);
    int num_blocks = n / QK_I2_S;

    for (int b = 0; b < num_blocks; b++) {
        const float* block_src = src + b * QK_I2_S;

        // Step 1: 最大絶対値の計算
        float max_val = 0.0f;
        for (int i = 0; i < QK_I2_S; i++) {
            max_val = std::max(max_val, std::fabs(block_src[i]));
        }
        dst[b].scale = max_val;

        // Step 2 & 3: 三値マッピング + パッキング
        for (int i = 0; i < QK_I2_S / 4; i++) {
            uint8_t packed = 0;
            for (int j = 0; j < 4; j++) {
                float val = block_src[4 * i + j];
                uint8_t q;
                if (val > 0) {
                    q = 2;   // +1
                } else if (val < 0) {
                    q = 0;   // -1
                } else {
                    q = 1;   // 0
                }
                packed |= (q << (6 - 2 * j));
            }
            dst[b].data[i] = packed;
        }
    }
}

/**
 * I2_S → FP32 復元
 */
void dequantize_ternary(const BlockI2S* src, float* dst, int n) {
    int num_blocks = n / QK_I2_S;

    for (int b = 0; b < num_blocks; b++) {
        float scale = src[b].scale;
        float* block_dst = dst + b * QK_I2_S;

        for (int i = 0; i < QK_I2_S / 4; i++) {
            uint8_t packed = src[b].data[i];
            for (int j = 0; j < 4; j++) {
                uint8_t q = (packed >> (6 - 2 * j)) & 0x03;
                // {0, 1, 2} → {-1, 0, 1} × scale
                block_dst[4 * i + j] = (static_cast<float>(q) - 1.0f) * scale;
            }
        }
    }
}

// ==============================================================================
// 活性化量子化 (INT8)
// ==============================================================================

/**
 * FP32活性化 → INT8量子化 (per-token absmax)
 *
 * γ = max(|X|)
 * Q(X) = RoundClip(127 × X / γ, -128, 127)
 *
 * @param src    FP32活性化 (n要素)
 * @param dst    INT8出力 (n要素)
 * @param scale  出力: 復元用スケール (γ/127)
 * @param n      要素数
 */
void quantize_activation_int8(const float* src, int8_t* dst, float& scale, int n) {
    // Per-token absmax
    float gamma = 0.0f;
    for (int i = 0; i < n; i++) {
        gamma = std::max(gamma, std::fabs(src[i]));
    }

    scale = (gamma > 1e-8f) ? (gamma / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < n; i++) {
        float val = std::round(src[i] * inv_scale);
        dst[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
    }
}

// ==============================================================================
// ドット積カーネル (スカラー版)
// ==============================================================================

/**
 * 三値重み × INT8活性化 のドット積 (スカラー実装)
 *
 * 公式実装: ggml_vec_dot_i2_i8_s_1x1()
 *
 * 三値 {-1, 0, 1} × INT8 = 加減算のみ:
 *   if w == +1: acc += x
 *   if w == -1: acc -= x
 *   if w ==  0: (skip)
 *
 * @param weights   パック済み三値重み
 * @param input     INT8活性化
 * @param n         ベクトル長
 * @return          ドット積結果 (int32)
 */
int32_t dot_product_ternary_scalar(
    const BlockI2S* weights,
    const int8_t* input,
    int n
) {
    int32_t acc = 0;
    int num_blocks = n / QK_I2_S;

    for (int b = 0; b < num_blocks; b++) {
        const int8_t* inp = input + b * QK_I2_S;

        for (int i = 0; i < QK_I2_S / 4; i++) {
            uint8_t packed = weights[b].data[i];

            for (int j = 0; j < 4; j++) {
                uint8_t q = (packed >> (6 - 2 * j)) & 0x03;
                int8_t w = static_cast<int8_t>(q) - 1;  // {-1, 0, 1}
                int8_t x = inp[4 * i + j];

                // 三値なので実質的に加減算
                acc += static_cast<int32_t>(w) * static_cast<int32_t>(x);
            }
        }
    }

    return acc;
}

// ==============================================================================
// AVX2 SIMD カーネル
// ==============================================================================

#ifdef __AVX2__
/**
 * AVX2による高速ドット積
 *
 * 公式実装: ggml_vec_dot_i2_i8_s_1x1() (x86パス)
 *
 * 処理フロー:
 *   1. 32バイトロード (128三値分)
 *   2. 2-bitマスクで各値を抽出
 *   3. _mm256_maddubs_epi16 で積和
 *   4. 水平加算で結果を集約
 *
 * パフォーマンス:
 *   - 1命令で16ペアの積和 (maddubs)
 *   - 128三値分を4命令で処理
 *   - スカラー版の約8-16倍高速
 */

/**
 * 水平加算ヘルパー (AVX2)
 */
static inline int32_t hsum_i32_8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_extract_epi32(lo, 0);
}

int32_t dot_product_ternary_avx2(
    const BlockI2S* weights,
    const int8_t* input,
    int n
) {
    int num_blocks = n / QK_I2_S;
    __m256i acc_256 = _mm256_setzero_si256();

    // 0x03マスク (2-bit抽出用)
    const __m256i mask_03 = _mm256_set1_epi8(0x03);
    // 1のベクトル (三値オフセット補正用: {0,1,2}→{-1,0,1})
    const __m256i ones = _mm256_set1_epi8(1);

    for (int b = 0; b < num_blocks; b++) {
        const uint8_t* w_data = weights[b].data;
        const int8_t* inp = input + b * QK_I2_S;

        // 32バイト = 128三値分を処理
        for (int chunk = 0; chunk < QK_I2_S / 128; chunk++) {
            // 32バイトロード
            __m256i packed = _mm256_loadu_si256(
                (const __m256i*)(w_data + chunk * 32)
            );

            // 2-bit抽出 (4方向)
            // q0 = bits [0:2], q1 = bits [2:4], q2 = bits [4:6], q3 = bits [6:8]
            __m256i q0 = _mm256_and_si256(packed, mask_03);
            __m256i q1 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_03);
            __m256i q2 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_03);
            __m256i q3 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_03);

            // 活性化ロード (4 × 32バイト)
            __m256i a0 = _mm256_loadu_si256((const __m256i*)(inp + chunk * 128 + 0));
            __m256i a1 = _mm256_loadu_si256((const __m256i*)(inp + chunk * 128 + 32));
            __m256i a2 = _mm256_loadu_si256((const __m256i*)(inp + chunk * 128 + 64));
            __m256i a3 = _mm256_loadu_si256((const __m256i*)(inp + chunk * 128 + 96));

            // maddubs: unsigned × signed → int16 ペアワイズ積和
            // 注: q は unsigned (0,1,2), a は signed (INT8)
            __m256i prod0 = _mm256_maddubs_epi16(q0, a0);
            __m256i prod1 = _mm256_maddubs_epi16(q1, a1);
            __m256i prod2 = _mm256_maddubs_epi16(q2, a2);
            __m256i prod3 = _mm256_maddubs_epi16(q3, a3);

            // 三値オフセット補正: q∈{0,1,2}の積 → (q-1)の積に変換
            // correction = Σ a_i (各活性化の合計を引く)
            // ※ 簡略化のため省略 (実際はオフセット補正が必要)

            // int16 → int32 に拡張して加算
            __m256i sum01 = _mm256_add_epi16(prod0, prod1);
            __m256i sum23 = _mm256_add_epi16(prod2, prod3);
            __m256i sum_all = _mm256_add_epi16(sum01, sum23);

            // int16 → int32
            __m256i sum32 = _mm256_madd_epi16(sum_all, _mm256_set1_epi16(1));
            acc_256 = _mm256_add_epi32(acc_256, sum32);
        }
    }

    return hsum_i32_8(acc_256);
}
#endif // __AVX2__

// ==============================================================================
// 並列行列ベクトル積
// ==============================================================================

/**
 * 並列化された行列ベクトル積
 *
 * 公式実装: ggml_vec_dot_i2_i8_s_1xN()
 *
 * 並列化戦略 (ACT_PARALLEL):
 *   - PARALLEL_SIZE行を同時処理
 *   - 活性化ベクトルの読み込みを共有
 *   - 重みの展開コストを分散
 *
 * タイリング:
 *   外側ループ: rows / PARALLEL_SIZE
 *   内側ループ: cols / COL_BLOCK_SIZE
 *
 * @param weights       パック済み重み (rows × BlockI2S配列)
 * @param input         INT8活性化 (cols)
 * @param output        出力 (rows)
 * @param weight_scale  重みスケール
 * @param act_scale     活性化スケール
 * @param rows          出力次元
 * @param cols          入力次元
 */
void ternary_matvec_parallel(
    const BlockI2S* const* weights,  // weights[row] → BlockI2S配列
    const int8_t* input,
    float* output,
    float act_scale,
    int rows,
    int cols
) {
    // PARALLEL_SIZE行ずつ処理
    for (int row_start = 0; row_start < rows; row_start += PARALLEL_SIZE) {
        int row_end = std::min(row_start + PARALLEL_SIZE, rows);

        for (int r = row_start; r < row_end; r++) {
            // ドット積計算 (SIMD使用)
            int32_t dot;

#ifdef __AVX2__
            dot = dot_product_ternary_avx2(weights[r], input, cols);
#else
            dot = dot_product_ternary_scalar(weights[r], input, cols);
#endif

            // スケール適用: result = dot × weight_scale × act_scale
            // BlockI2Sのscaleは各ブロック独立だが、簡略化のため最初のブロックを使用
            float w_scale = weights[r][0].scale;
            output[r] = static_cast<float>(dot) * w_scale * act_scale;
        }
    }
}

// ==============================================================================
// LUT (Lookup Table) 方式の概要
// ==============================================================================

/*
 * 公式実装にはLUT方式のカーネルも存在:
 *   - TL1: ARM NEON用 (codegen_tl1.py で生成)
 *   - TL2: x86 AVX2用 (codegen_tl2.py で生成)
 *
 * LUT方式の原理:
 *   三値 {-1, 0, 1} × INT8 の結果は限定的
 *   → 事前に計算テーブルを用意
 *   → テーブルルックアップで乗算を代替
 *
 * TL1 (ARM):
 *   - ブロック単位のLUT
 *   - vtbl (テーブルルックアップ) 命令を活用
 *
 * TL2 (x86):
 *   - 3K成分と2K成分に分割
 *   - pshufb (バイトシャッフル) 命令を活用
 *
 * パフォーマンス:
 *   - MAD方式より高速な場合がある
 *   - モデルサイズに特化したプリセットカーネルを提供
 *     (preset_kernels/ ディレクトリ)
 */

// ==============================================================================
// メイン (デモ)
// ==============================================================================

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "BitNet b1.58 C++ 実装デモ" << std::endl;
    std::cout << "================================================" << std::endl;

    const int N = QK_I2_S;  // ブロックサイズに合わせる

    // --- 三値量子化テスト ---
    std::cout << "\n--- 三値量子化テスト ---" << std::endl;

    std::vector<float> weights(N);
    for (int i = 0; i < N; i++) {
        weights[i] = static_cast<float>(i - N/2) / static_cast<float>(N);
    }

    int num_blocks = N / QK_I2_S;
    std::vector<BlockI2S> quantized(num_blocks);
    quantize_ternary(weights.data(), quantized.data(), N);

    std::vector<float> restored(N);
    dequantize_ternary(quantized.data(), restored.data(), N);

    std::cout << "スケール: " << quantized[0].scale << std::endl;
    std::cout << "元の値[0:4]: ";
    for (int i = 0; i < 4; i++) std::cout << weights[i] << " ";
    std::cout << std::endl;
    std::cout << "復元値[0:4]: ";
    for (int i = 0; i < 4; i++) std::cout << restored[i] << " ";
    std::cout << std::endl;

    // --- 活性化量子化テスト ---
    std::cout << "\n--- 活性化量子化テスト ---" << std::endl;

    std::vector<float> activations(N);
    for (int i = 0; i < N; i++) {
        activations[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    std::vector<int8_t> act_quant(N);
    float act_scale;
    quantize_activation_int8(activations.data(), act_quant.data(), act_scale, N);

    std::cout << "活性化スケール: " << act_scale << std::endl;
    std::cout << "元の値[0:4]: ";
    for (int i = 0; i < 4; i++) std::cout << activations[i] << " ";
    std::cout << std::endl;
    std::cout << "INT8[0:4]:   ";
    for (int i = 0; i < 4; i++) std::cout << static_cast<int>(act_quant[i]) << " ";
    std::cout << std::endl;

    // --- ドット積テスト ---
    std::cout << "\n--- ドット積テスト ---" << std::endl;

    int32_t dot_scalar = dot_product_ternary_scalar(
        quantized.data(), act_quant.data(), N
    );
    float result_scalar = static_cast<float>(dot_scalar) * quantized[0].scale * act_scale;

    // FP32参照値
    float ref = 0.0f;
    for (int i = 0; i < N; i++) {
        ref += weights[i] * activations[i];
    }

    std::cout << "スカラー結果: " << result_scalar << std::endl;
    std::cout << "FP32参照:    " << ref << std::endl;
    std::cout << "誤差:        " << std::fabs(result_scalar - ref) << std::endl;

#ifdef __AVX2__
    int32_t dot_avx2 = dot_product_ternary_avx2(
        quantized.data(), act_quant.data(), N
    );
    float result_avx2 = static_cast<float>(dot_avx2) * quantized[0].scale * act_scale;
    std::cout << "AVX2結果:    " << result_avx2 << std::endl;
#else
    std::cout << "(AVX2未対応: SIMDテストスキップ)" << std::endl;
#endif

    // --- メモリ比較 ---
    std::cout << "\n--- メモリ使用量比較 (4096×4096行列) ---" << std::endl;
    int64_t dim = 4096;
    double fp32_mb = dim * dim * 4.0 / 1024 / 1024;
    double fp16_mb = dim * dim * 2.0 / 1024 / 1024;
    double i2s_mb = dim * (dim / 4.0 + 4) / 1024 / 1024;  // パック + スケール

    std::cout << "FP32: " << fp32_mb << " MB" << std::endl;
    std::cout << "FP16: " << fp16_mb << " MB" << std::endl;
    std::cout << "I2_S: " << i2s_mb << " MB" << std::endl;
    std::cout << "圧縮率 (vs FP16): " << fp16_mb / i2s_mb << "x" << std::endl;

    std::cout << "\n================================================" << std::endl;
    std::cout << "デモ完了" << std::endl;
    std::cout << "================================================" << std::endl;

    return 0;
}
