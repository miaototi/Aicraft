/**
 * @file simd_math.h
 * @brief SIMD-optimized math kernels for the Aicraft inference engine.
 * @ingroup simdmath
 *
 * Hand-tuned vectorized operations for maximum throughput.
 *
 * @par GEMM — BLIS-style packed matrix multiply with register-level micro-kernel
 *   - AVX-512: MR=6, NR=32  (6 rows x 2 ZMM = 32 cols, 192 FMAs/k)
 *   - AVX-2:   MR=6, NR=16  (6 rows x 2 YMM = 16 cols, 96 FMAs/k)
 *   - Multi-threaded via thread pool (parallel over M blocks)
 *   - L2/L3 cache blocking: MC*KC panel fits in L2, KC*NC in L3
 *
 * @par Element-wise
 *   AVX-512 -> AVX2 -> SSE -> NEON cascade (auto width selection).
 *
 * @see fast_math.h, thread_pool.h
 */

#ifndef AICRAFT_SIMD_MATH_H
#define AICRAFT_SIMD_MATH_H

#include "aicraft/platform.h"
#include "aicraft/fast_math.h"
#include "aicraft/thread_pool.h"
#include <math.h>
#include <string.h>

#ifdef AC_SIMD_AVX512
    #include <immintrin.h>
#elif defined(AC_SIMD_AVX2)
    #include <immintrin.h>
#elif defined(AC_SIMD_SSE)
    #include <emmintrin.h>
    #include <smmintrin.h>
#elif defined(AC_SIMD_NEON)
    #include <arm_neon.h>
#endif

/* Aligned allocation for packing buffers */
#ifdef _WIN32
    #include <malloc.h>
    #define AC_ALLOC_ALIGNED(sz, align) _aligned_malloc((sz), (align))
    #define AC_FREE_ALIGNED(ptr)        _aligned_free(ptr)
#else
    #include <stdlib.h>
    #define AC_ALLOC_ALIGNED(sz, align) aligned_alloc((align), (sz))
    #define AC_FREE_ALIGNED(ptr)        free(ptr)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup simdmath SIMD Math Kernels */
/** @{ */

/** @name Element-wise Operations
 *  Cascading AVX-512 -> AVX2 -> SSE -> NEON -> scalar. */
/** @{ */

/** @brief Element-wise vector addition: out[i] = a[i] + b[i].
 *  @param[in]  a   First input array.
 *  @param[in]  b   Second input array.
 *  @param[out] out Output array (may alias @p a or @p b).
 *  @param[in]  n   Number of elements.
 *  @note Uses AVX-512 (16-wide), AVX2 (8-wide), SSE/NEON (4-wide), then scalar tail.
 *  @see ac_simd_scale, ac_simd_fma */

AC_INLINE void ac_simd_add(const float* AC_RESTRICT a,
                           const float* AC_RESTRICT b,
                           float* AC_RESTRICT out,
                           ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#endif
#if defined(AC_SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#elif defined(AC_SIMD_SSE)
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(out + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
#elif defined(AC_SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
#endif
    for (; i < n; i++) out[i] = a[i] + b[i];
}

/** @brief Element-wise vector multiplication: out[i] = a[i] * b[i].
 *  @param[in]  a   First input array.
 *  @param[in]  b   Second input array.
 *  @param[out] out Output array.
 *  @param[in]  n   Number of elements.
 *  @note Uses AVX-512/AVX2/SSE/NEON with scalar tail.
 *  @see ac_simd_add, ac_simd_fma */

AC_INLINE void ac_simd_mul(const float* AC_RESTRICT a,
                           const float* AC_RESTRICT b,
                           float* AC_RESTRICT out,
                           ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#endif
#if defined(AC_SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#elif defined(AC_SIMD_SSE)
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
#elif defined(AC_SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
#endif
    for (; i < n; i++) out[i] = a[i] * b[i];
}

/** @brief Scalar-vector multiply: out[i] = a[i] * scalar.
 *  @param[in]  a      Input array.
 *  @param[in]  scalar Scalar multiplier broadcast to all lanes.
 *  @param[out] out    Output array.
 *  @param[in]  n      Number of elements.
 *  @note Broadcasts scalar into SIMD register, then cascades widths.
 *  @see ac_simd_mul, ac_simd_fma */

AC_INLINE void ac_simd_scale(const float* AC_RESTRICT a, float scalar,
                             float* AC_RESTRICT out, ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    __m512 vs512 = _mm512_set1_ps(scalar);
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), vs512));
    }
#endif
#if defined(AC_SIMD_AVX2)
    __m256 vs256 = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vs256));
    }
#elif defined(AC_SIMD_SSE)
    __m128 vs128 = _mm_set1_ps(scalar);
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(a + i), vs128));
    }
#elif defined(AC_SIMD_NEON)
    float32x4_t vs = vdupq_n_f32(scalar);
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(a + i), vs));
    }
#endif
    for (; i < n; i++) out[i] = a[i] * scalar;
}

/** @brief Fused multiply-add: out[i] = a[i] * b[i] + c[i].
 *  @param[in]  a   Multiplicand array.
 *  @param[in]  b   Multiplier array.
 *  @param[in]  c   Addend array.
 *  @param[out] out Output array.
 *  @param[in]  n   Number of elements.
 *  @note Uses hardware FMA where available (AVX-512, AVX2+FMA, NEON vmlaq).
 *  @see ac_simd_mul, ac_simd_add */

AC_INLINE void ac_simd_fma(const float* AC_RESTRICT a,
                           const float* AC_RESTRICT b,
                           const float* AC_RESTRICT c,
                           float* AC_RESTRICT out, ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_fmadd_ps(
            _mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), _mm512_loadu_ps(c + i)));
    }
#endif
#if defined(AC_SIMD_AVX2) && defined(__FMA__)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(
            _mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_loadu_ps(c + i)));
    }
#elif defined(AC_SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_mul_ps(va, vb), vc));
    }
#elif defined(AC_SIMD_SSE)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_loadu_ps(c + i);
        _mm_storeu_ps(out + i, _mm_add_ps(_mm_mul_ps(va, vb), vc));
    }
#elif defined(AC_SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        vst1q_f32(out + i, vmlaq_f32(vc, va, vb));
    }
#endif
    for (; i < n; i++) out[i] = a[i] * b[i] + c[i];
}

/** @brief Dot product: sum(a[i] * b[i]).
 *  @param[in] a First input array.
 *  @param[in] b Second input array.
 *  @param[in] n Number of elements.
 *  @return    Scalar dot product.
 *  @note Accumulates in SIMD, then horizontal reduce. Uses FMA where available.
 *  @see ac_simd_sum */

AC_INLINE float ac_simd_dot(const float* AC_RESTRICT a,
                            const float* AC_RESTRICT b, ac_size n) {
    float sum = 0.0f;
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    __m512 vsum512 = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        vsum512 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), vsum512);
    }
    sum += _mm512_reduce_add_ps(vsum512);
#endif
#if defined(AC_SIMD_AVX2)
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#ifdef __FMA__
        vsum = _mm256_fmadd_ps(va, vb, vsum);
#else
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
#endif
    }
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum += _mm_cvtss_f32(sum128);
#elif defined(AC_SIMD_SSE)
    __m128 vsum = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        vsum = _mm_add_ps(vsum, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    sum += _mm_cvtss_f32(vsum);
#elif defined(AC_SIMD_NEON)
    float32x4_t vsum4 = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        vsum4 = vmlaq_f32(vsum4, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    sum += vaddvq_f32(vsum4);
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/** @brief Sum reduction: returns sum of all elements.
 *  @param[in] a Input array.
 *  @param[in] n Number of elements.
 *  @return    Scalar sum of @p a.
 *  @note Uses SIMD horizontal add with cascading widths.
 *  @see ac_simd_dot, ac_simd_max */

AC_INLINE float ac_simd_sum(const float* AC_RESTRICT a, ac_size n) {
    float sum = 0.0f;
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    __m512 vsum512 = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        vsum512 = _mm512_add_ps(vsum512, _mm512_loadu_ps(a + i));
    }
    sum += _mm512_reduce_add_ps(vsum512);
#endif
#if defined(AC_SIMD_AVX2)
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(a + i));
    }
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum += _mm_cvtss_f32(sum128);
#elif defined(AC_SIMD_SSE)
    __m128 vsum = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        vsum = _mm_add_ps(vsum, _mm_loadu_ps(a + i));
    }
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    sum += _mm_cvtss_f32(vsum);
#elif defined(AC_SIMD_NEON)
    float32x4_t vsum4 = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        vsum4 = vaddq_f32(vsum4, vld1q_f32(a + i));
    }
    sum += vaddvq_f32(vsum4);
#endif
    for (; i < n; i++) sum += a[i];
    return sum;
}

/** @brief Max reduction: returns maximum element value.
 *  @param[in] a Input array.
 *  @param[in] n Number of elements.
 *  @return    Maximum value in @p a, or 0.0f if @p n == 0.
 *  @note Uses SIMD max with horizontal reduce.
 *  @see ac_simd_sum */

AC_INLINE float ac_simd_max(const float* AC_RESTRICT a, ac_size n) {
    if (n == 0) return 0.0f;
    float max_val = a[0];
    ac_size i = 1;
#if defined(AC_SIMD_AVX512)
    if (n >= 16) {
        __m512 vmax = _mm512_loadu_ps(a);
        for (i = 16; i + 16 <= n; i += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(a + i));
        }
        max_val = _mm512_reduce_max_ps(vmax);
        i = (n / 16) * 16;
    }
#elif defined(AC_SIMD_AVX2)
    if (n >= 8) {
        __m256 vmax = _mm256_loadu_ps(a);
        for (i = 8; i + 8 <= n; i += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(a + i));
        }
        float temp[8]; _mm256_storeu_ps(temp, vmax);
        for (int j = 0; j < 8; j++) if (temp[j] > max_val) max_val = temp[j];
    }
#elif defined(AC_SIMD_NEON)
    if (n >= 4) {
        float32x4_t vmax4 = vld1q_f32(a);
        for (i = 4; i + 4 <= n; i += 4) {
            vmax4 = vmaxq_f32(vmax4, vld1q_f32(a + i));
        }
        max_val = vmaxvq_f32(vmax4);
    }
#endif
    for (; i < n; i++) if (a[i] > max_val) max_val = a[i];
    return max_val;
}

/** @brief Vectorized exponential: out[i] = exp(a[i]).
 *  @param[in]  a   Input array.
 *  @param[out] out Output array.
 *  @param[in]  n   Number of elements.
 *  @note Delegates to ac_fast_exp() (Cephes polynomial, <1 ULP accuracy).
 *  @see ac_fast_exp */

AC_INLINE void ac_simd_exp(const float* AC_RESTRICT a, float* AC_RESTRICT out, ac_size n) {
    /* Delegates to fast_math.h polynomial approximation (<1 ULP accuracy) */
    ac_fast_exp(a, out, n);
}

/** @brief Vectorized ReLU: out[i] = max(a[i], 0).
 *  @param[in]  a   Input array.
 *  @param[out] out Output array.
 *  @param[in]  n   Number of elements.
 *  @note Uses SIMD max against zero register.
 *  @see ac_simd_relu_backward */

AC_INLINE void ac_simd_relu(const float* AC_RESTRICT a, float* AC_RESTRICT out, ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    __m512 z512 = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_max_ps(_mm512_loadu_ps(a + i), z512));
    }
#endif
#if defined(AC_SIMD_AVX2)
    __m256 z256 = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_max_ps(_mm256_loadu_ps(a + i), z256));
    }
#elif defined(AC_SIMD_SSE)
    __m128 z128 = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(out + i, _mm_max_ps(_mm_loadu_ps(a + i), z128));
    }
#elif defined(AC_SIMD_NEON)
    float32x4_t z4 = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmaxq_f32(vld1q_f32(a + i), z4));
    }
#endif
    for (; i < n; i++) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

/** @brief Vectorized ReLU backward pass (accumulates into grad_input).
 *  @param[in]     input       Forward-pass input (for sign test).
 *  @param[in]     grad_output Upstream gradient.
 *  @param[in,out] grad_input  Gradient accumulator; += grad_output where input > 0.
 *  @param[in]     n           Number of elements.
 *  @note Uses masked move / bitwise-AND for branch-free gradient selection.
 *  @see ac_simd_relu */

AC_INLINE void ac_simd_relu_backward(const float* AC_RESTRICT input,
                                     const float* AC_RESTRICT grad_output,
                                     float* AC_RESTRICT grad_input,
                                     ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    __m512 z512 = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 inp = _mm512_loadu_ps(input + i);
        __m512 grad = _mm512_loadu_ps(grad_output + i);
        __m512 cur = _mm512_loadu_ps(grad_input + i);
        __mmask16 mask = _mm512_cmp_ps_mask(inp, z512, _CMP_GT_OS);
        _mm512_storeu_ps(grad_input + i, _mm512_add_ps(cur, _mm512_maskz_mov_ps(mask, grad)));
    }
#endif
#if defined(AC_SIMD_AVX2)
    __m256 z256 = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 inp = _mm256_loadu_ps(input + i);
        __m256 grad = _mm256_loadu_ps(grad_output + i);
        __m256 cur = _mm256_loadu_ps(grad_input + i);
        __m256 mask = _mm256_cmp_ps(inp, z256, _CMP_GT_OS);
        _mm256_storeu_ps(grad_input + i, _mm256_add_ps(cur, _mm256_and_ps(grad, mask)));
    }
#elif defined(AC_SIMD_SSE)
    __m128 z128 = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 inp = _mm_loadu_ps(input + i);
        __m128 grad = _mm_loadu_ps(grad_output + i);
        __m128 cur = _mm_loadu_ps(grad_input + i);
        __m128 mask = _mm_cmpgt_ps(inp, z128);
        _mm_storeu_ps(grad_input + i, _mm_add_ps(cur, _mm_and_ps(grad, mask)));
    }
#elif defined(AC_SIMD_NEON)
    float32x4_t z4 = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t inp = vld1q_f32(input + i);
        float32x4_t grad = vld1q_f32(grad_output + i);
        float32x4_t cur = vld1q_f32(grad_input + i);
        uint32x4_t mask = vcgtq_f32(inp, z4);
        float32x4_t masked = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(grad), mask));
        vst1q_f32(grad_input + i, vaddq_f32(cur, masked));
    }
#endif
    for (; i < n; i++) grad_input[i] += input[i] > 0.0f ? grad_output[i] : 0.0f;
}


/** @} */

/** @name BLIS-style Packed GEMM
 *  C[M*N] = A[M*K] * B[K*N] using GotoBLAS/BLIS algorithm.
 *
 *  1. Block loops: jc over N (NC), pc over K (KC), ic over M (MC).
 *  2. Pack B[KC*NC] into contiguous NR-wide panels (L3 resident).
 *  3. Pack A[MC*KC] into contiguous MR-wide panels (L2 resident).
 *  4. Macro-kernel: iterate MR*NR register tiles.
 *  5. Micro-kernel: FMA accumulation in registers, MR*NR output tile. */
/** @{ */

/** @name GEMM Blocking Parameters */
/** @{ */

#define AC_GEMM_MR   6      /**< Register block rows (MR). */
#define AC_GEMM_MC   72     /**< Cache block rows (12 * MR, fits in L2). */
#define AC_GEMM_KC   256    /**< Cache block K dimension. */
#define AC_GEMM_NC   4096   /**< Cache block cols (fits in L3). */

#if defined(AC_SIMD_AVX512)
    #define AC_GEMM_NR 32   /**< Register block cols: 2 * ZMM (2 * 16). */
#elif defined(AC_SIMD_AVX2)
    #define AC_GEMM_NR 16   /**< Register block cols: 2 * YMM (2 * 8). */
#elif defined(AC_SIMD_NEON)
    #define AC_GEMM_NR 8    /**< Register block cols: 2 * Q (2 * 4). */
#else
    #define AC_GEMM_NR 8    /**< Register block cols (scalar/SSE fallback). */
#endif
/** @} */

/** @brief Pack A sub-block (mc*kc) into MR-contiguous panels.
 *
 *  Layout: panel[p] = { A[i,k] for k in [0,kc), i in [ir..ir+MR) }.
 *  Access: packed_A + p * MR * kc + k * MR + ii.
 *
 *  @param[in]  A      Source matrix (row-major).
 *  @param[out] packed Destination buffer (64-byte aligned).
 *  @param[in]  mc     Number of rows in the sub-block.
 *  @param[in]  kc     Number of columns (K dimension).
 *  @param[in]  lda    Leading dimension of @p A.
 *  @see ac_pack_B, ac_macro_kernel */
static AC_INLINE void ac_pack_A(const float* AC_RESTRICT A, float* AC_RESTRICT packed,
                                int mc, int kc, int lda) {
    for (int ir = 0; ir < mc; ir += AC_GEMM_MR) {
        int mr = (ir + AC_GEMM_MR <= mc) ? AC_GEMM_MR : (mc - ir);
        for (int k = 0; k < kc; k++) {
            int ii;
            for (ii = 0; ii < mr; ii++) {
                packed[k * AC_GEMM_MR + ii] = A[(ir + ii) * lda + k];
            }
            for (; ii < AC_GEMM_MR; ii++) {
                packed[k * AC_GEMM_MR + ii] = 0.0f;
            }
        }
        packed += AC_GEMM_MR * kc;
    }
}

/** @brief Pack B sub-block (kc*nc) into NR-contiguous panels.
 *
 *  Layout: panel[q] = { B[k,j] for k in [0,kc), j in [jr..jr+NR) }.
 *  Access: packed_B + q * NR * kc + k * NR + jj.
 *
 *  @param[in]  B      Source matrix (row-major).
 *  @param[out] packed Destination buffer (64-byte aligned).
 *  @param[in]  kc     Number of rows (K dimension).
 *  @param[in]  nc     Number of columns in the sub-block.
 *  @param[in]  ldb    Leading dimension of @p B.
 *  @see ac_pack_A, ac_macro_kernel */
static AC_INLINE void ac_pack_B(const float* AC_RESTRICT B, float* AC_RESTRICT packed,
                                int kc, int nc, int ldb) {
    for (int jr = 0; jr < nc; jr += AC_GEMM_NR) {
        int nr = (jr + AC_GEMM_NR <= nc) ? AC_GEMM_NR : (nc - jr);
        for (int k = 0; k < kc; k++) {
            int jj;
            for (jj = 0; jj < nr; jj++) {
                packed[k * AC_GEMM_NR + jj] = B[k * ldb + (jr + jj)];
            }
            for (; jj < AC_GEMM_NR; jj++) {
                packed[k * AC_GEMM_NR + jj] = 0.0f;
            }
        }
        packed += AC_GEMM_NR * kc;
    }
}

/** @brief Scalar fallback micro-kernel for edge tiles.
 *  @param[in]     kc       K-dimension block length.
 *  @param[in]     packed_A Packed A panel (MR-contiguous).
 *  @param[in]     packed_B Packed B panel (NR-contiguous).
 *  @param[in,out] C        Output tile pointer.
 *  @param[in]     ldc      Leading dimension of C.
 *  @param[in]     mr       Actual row count (<= AC_GEMM_MR).
 *  @param[in]     nr       Actual column count (<= AC_GEMM_NR).
 *  @note Used when MR or NR tile is not full-size.
 *  @see ac_macro_kernel */

static AC_INLINE void ac_micro_kernel_scalar(int kc,
                                             const float* AC_RESTRICT packed_A,
                                             const float* AC_RESTRICT packed_B,
                                             float* AC_RESTRICT C, int ldc,
                                             int mr, int nr) {
    for (int k = 0; k < kc; k++) {
        for (int ii = 0; ii < mr; ii++) {
            float a_val = packed_A[k * AC_GEMM_MR + ii];
            for (int jj = 0; jj < nr; jj++) {
                C[ii * ldc + jj] += a_val * packed_B[k * AC_GEMM_NR + jj];
            }
        }
    }
}

/** @brief AVX2 micro-kernel (6x16 tile).
 *
 *  12 YMM accumulators (6 rows * 2 YMM cols)
 *  + 2 YMM for B loads + 6 broadcasts from A = 15 registers total.
 *  Per k-step: 12 FMA ops * 8 floats = 96 multiply-adds.
 *
 *  @param[in]     kc       K-dimension block length.
 *  @param[in]     packed_A Packed A panel (MR-contiguous).
 *  @param[in]     packed_B Packed B panel (NR-contiguous, 16-wide).
 *  @param[in,out] C        Output tile pointer.
 *  @param[in]     ldc      Leading dimension of C.
 *  @simd AVX2 (+ optional FMA3).
 *  @see ac_micro_kernel_6x32, ac_macro_kernel */
#if defined(AC_SIMD_AVX2) && !defined(AC_SIMD_AVX512)

static AC_INLINE void ac_micro_kernel_6x16(int kc,
                                           const float* AC_RESTRICT packed_A,
                                           const float* AC_RESTRICT packed_B,
                                           float* AC_RESTRICT C, int ldc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int k = 0; k < kc; k++) {
        const float* bp = packed_B + k * 16;
        const float* ap = packed_A + k * 6;

        __m256 b0 = _mm256_load_ps(bp);
        __m256 b1 = _mm256_load_ps(bp + 8);

        __m256 a0 = _mm256_broadcast_ss(ap + 0);
        __m256 a1 = _mm256_broadcast_ss(ap + 1);
        __m256 a2 = _mm256_broadcast_ss(ap + 2);
        __m256 a3 = _mm256_broadcast_ss(ap + 3);
        __m256 a4 = _mm256_broadcast_ss(ap + 4);
        __m256 a5 = _mm256_broadcast_ss(ap + 5);

#ifdef __FMA__
        c00 = _mm256_fmadd_ps(a0, b0, c00);  c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);  c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);  c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);  c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);  c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);  c51 = _mm256_fmadd_ps(a5, b1, c51);
#else
        c00 = _mm256_add_ps(c00, _mm256_mul_ps(a0, b0));  c01 = _mm256_add_ps(c01, _mm256_mul_ps(a0, b1));
        c10 = _mm256_add_ps(c10, _mm256_mul_ps(a1, b0));  c11 = _mm256_add_ps(c11, _mm256_mul_ps(a1, b1));
        c20 = _mm256_add_ps(c20, _mm256_mul_ps(a2, b0));  c21 = _mm256_add_ps(c21, _mm256_mul_ps(a2, b1));
        c30 = _mm256_add_ps(c30, _mm256_mul_ps(a3, b0));  c31 = _mm256_add_ps(c31, _mm256_mul_ps(a3, b1));
        c40 = _mm256_add_ps(c40, _mm256_mul_ps(a4, b0));  c41 = _mm256_add_ps(c41, _mm256_mul_ps(a4, b1));
        c50 = _mm256_add_ps(c50, _mm256_mul_ps(a5, b0));  c51 = _mm256_add_ps(c51, _mm256_mul_ps(a5, b1));
#endif

        /* Prefetch next B panel for L1 */
        if ((k & 3) == 0 && k + 4 < kc) {
            _mm_prefetch((const char*)(packed_B + (k + 4) * 16), _MM_HINT_T0);
        }
    }

    /* Accumulate into C: C += accumulators */
    _mm256_storeu_ps(C + 0*ldc,     _mm256_add_ps(_mm256_loadu_ps(C + 0*ldc),     c00));
    _mm256_storeu_ps(C + 0*ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 0*ldc + 8), c01));
    _mm256_storeu_ps(C + 1*ldc,     _mm256_add_ps(_mm256_loadu_ps(C + 1*ldc),     c10));
    _mm256_storeu_ps(C + 1*ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 1*ldc + 8), c11));
    _mm256_storeu_ps(C + 2*ldc,     _mm256_add_ps(_mm256_loadu_ps(C + 2*ldc),     c20));
    _mm256_storeu_ps(C + 2*ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 2*ldc + 8), c21));
    _mm256_storeu_ps(C + 3*ldc,     _mm256_add_ps(_mm256_loadu_ps(C + 3*ldc),     c30));
    _mm256_storeu_ps(C + 3*ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 3*ldc + 8), c31));
    _mm256_storeu_ps(C + 4*ldc,     _mm256_add_ps(_mm256_loadu_ps(C + 4*ldc),     c40));
    _mm256_storeu_ps(C + 4*ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 4*ldc + 8), c41));
    _mm256_storeu_ps(C + 5*ldc,     _mm256_add_ps(_mm256_loadu_ps(C + 5*ldc),     c50));
    _mm256_storeu_ps(C + 5*ldc + 8, _mm256_add_ps(_mm256_loadu_ps(C + 5*ldc + 8), c51));
}

#endif /* AVX2-only micro-kernel */

#if defined(AC_SIMD_AVX512)
/** @brief AVX-512 micro-kernel (6x32 tile).
 *
 *  12 ZMM accumulators (6 rows * 2 ZMM cols)
 *  + 2 ZMM for B loads + broadcasts from A = 15 total.
 *  Per k-step: 12 FMA ops * 16 floats = 192 multiply-adds.
 *
 *  @param[in]     kc       K-dimension block length.
 *  @param[in]     packed_A Packed A panel (MR-contiguous).
 *  @param[in]     packed_B Packed B panel (NR-contiguous, 32-wide).
 *  @param[in,out] C        Output tile pointer.
 *  @param[in]     ldc      Leading dimension of C.
 *  @simd AVX-512 (FMA always available).
 *  @see ac_micro_kernel_6x16, ac_macro_kernel */

static AC_INLINE void ac_micro_kernel_6x32(int kc,
                                           const float* AC_RESTRICT packed_A,
                                           const float* AC_RESTRICT packed_B,
                                           float* AC_RESTRICT C, int ldc) {
    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();

    for (int k = 0; k < kc; k++) {
        const float* bp = packed_B + k * 32;
        const float* ap = packed_A + k * 6;

        __m512 b0 = _mm512_load_ps(bp);
        __m512 b1 = _mm512_load_ps(bp + 16);

        __m512 a0 = _mm512_set1_ps(ap[0]);
        __m512 a1 = _mm512_set1_ps(ap[1]);
        __m512 a2 = _mm512_set1_ps(ap[2]);
        __m512 a3 = _mm512_set1_ps(ap[3]);
        __m512 a4 = _mm512_set1_ps(ap[4]);
        __m512 a5 = _mm512_set1_ps(ap[5]);

        c00 = _mm512_fmadd_ps(a0, b0, c00);  c01 = _mm512_fmadd_ps(a0, b1, c01);
        c10 = _mm512_fmadd_ps(a1, b0, c10);  c11 = _mm512_fmadd_ps(a1, b1, c11);
        c20 = _mm512_fmadd_ps(a2, b0, c20);  c21 = _mm512_fmadd_ps(a2, b1, c21);
        c30 = _mm512_fmadd_ps(a3, b0, c30);  c31 = _mm512_fmadd_ps(a3, b1, c31);
        c40 = _mm512_fmadd_ps(a4, b0, c40);  c41 = _mm512_fmadd_ps(a4, b1, c41);
        c50 = _mm512_fmadd_ps(a5, b0, c50);  c51 = _mm512_fmadd_ps(a5, b1, c51);

        if ((k & 3) == 0 && k + 4 < kc) {
            _mm_prefetch((const char*)(packed_B + (k + 4) * 32), _MM_HINT_T0);
            _mm_prefetch((const char*)(packed_B + (k + 4) * 32 + 16), _MM_HINT_T0);
        }
    }

    /* Accumulate into C */
    _mm512_storeu_ps(C + 0*ldc,      _mm512_add_ps(_mm512_loadu_ps(C + 0*ldc),      c00));
    _mm512_storeu_ps(C + 0*ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 0*ldc + 16), c01));
    _mm512_storeu_ps(C + 1*ldc,      _mm512_add_ps(_mm512_loadu_ps(C + 1*ldc),      c10));
    _mm512_storeu_ps(C + 1*ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 1*ldc + 16), c11));
    _mm512_storeu_ps(C + 2*ldc,      _mm512_add_ps(_mm512_loadu_ps(C + 2*ldc),      c20));
    _mm512_storeu_ps(C + 2*ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 2*ldc + 16), c21));
    _mm512_storeu_ps(C + 3*ldc,      _mm512_add_ps(_mm512_loadu_ps(C + 3*ldc),      c30));
    _mm512_storeu_ps(C + 3*ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 3*ldc + 16), c31));
    _mm512_storeu_ps(C + 4*ldc,      _mm512_add_ps(_mm512_loadu_ps(C + 4*ldc),      c40));
    _mm512_storeu_ps(C + 4*ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 4*ldc + 16), c41));
    _mm512_storeu_ps(C + 5*ldc,      _mm512_add_ps(_mm512_loadu_ps(C + 5*ldc),      c50));
    _mm512_storeu_ps(C + 5*ldc + 16, _mm512_add_ps(_mm512_loadu_ps(C + 5*ldc + 16), c51));
}

#endif /* AVX-512 micro-kernel */

#if defined(AC_SIMD_NEON) && !defined(AC_SIMD_AVX2)
/** @brief NEON micro-kernel (6x8 tile).
 *
 *  12 Q-reg accumulators (6 rows * 2 float32x4_t cols = 8 cols).
 *  Per k-step: 12 FMA ops * 4 floats = 48 multiply-adds.
 *
 *  @param[in]     kc       K-dimension block length.
 *  @param[in]     packed_A Packed A panel (MR-contiguous).
 *  @param[in]     packed_B Packed B panel (NR-contiguous, 8-wide).
 *  @param[in,out] C        Output tile pointer.
 *  @param[in]     ldc      Leading dimension of C.
 *  @simd ARM NEON (vfmaq_n_f32).
 *  @see ac_micro_kernel_6x16, ac_macro_kernel */

static AC_INLINE void ac_micro_kernel_6x8_neon(int kc,
                                               const float* AC_RESTRICT packed_A,
                                               const float* AC_RESTRICT packed_B,
                                               float* AC_RESTRICT C, int ldc) {
    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
    float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
    float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);

    for (int k = 0; k < kc; k++) {
        const float* bp = packed_B + k * 8;
        const float* ap = packed_A + k * 6;

        float32x4_t b0 = vld1q_f32(bp);
        float32x4_t b1 = vld1q_f32(bp + 4);

        c00 = vfmaq_n_f32(c00, b0, ap[0]);  c01 = vfmaq_n_f32(c01, b1, ap[0]);
        c10 = vfmaq_n_f32(c10, b0, ap[1]);  c11 = vfmaq_n_f32(c11, b1, ap[1]);
        c20 = vfmaq_n_f32(c20, b0, ap[2]);  c21 = vfmaq_n_f32(c21, b1, ap[2]);
        c30 = vfmaq_n_f32(c30, b0, ap[3]);  c31 = vfmaq_n_f32(c31, b1, ap[3]);
        c40 = vfmaq_n_f32(c40, b0, ap[4]);  c41 = vfmaq_n_f32(c41, b1, ap[4]);
        c50 = vfmaq_n_f32(c50, b0, ap[5]);  c51 = vfmaq_n_f32(c51, b1, ap[5]);
    }

    /* Accumulate into C */
    vst1q_f32(C + 0*ldc,     vaddq_f32(vld1q_f32(C + 0*ldc),     c00));
    vst1q_f32(C + 0*ldc + 4, vaddq_f32(vld1q_f32(C + 0*ldc + 4), c01));
    vst1q_f32(C + 1*ldc,     vaddq_f32(vld1q_f32(C + 1*ldc),     c10));
    vst1q_f32(C + 1*ldc + 4, vaddq_f32(vld1q_f32(C + 1*ldc + 4), c11));
    vst1q_f32(C + 2*ldc,     vaddq_f32(vld1q_f32(C + 2*ldc),     c20));
    vst1q_f32(C + 2*ldc + 4, vaddq_f32(vld1q_f32(C + 2*ldc + 4), c21));
    vst1q_f32(C + 3*ldc,     vaddq_f32(vld1q_f32(C + 3*ldc),     c30));
    vst1q_f32(C + 3*ldc + 4, vaddq_f32(vld1q_f32(C + 3*ldc + 4), c31));
    vst1q_f32(C + 4*ldc,     vaddq_f32(vld1q_f32(C + 4*ldc),     c40));
    vst1q_f32(C + 4*ldc + 4, vaddq_f32(vld1q_f32(C + 4*ldc + 4), c41));
    vst1q_f32(C + 5*ldc,     vaddq_f32(vld1q_f32(C + 5*ldc),     c50));
    vst1q_f32(C + 5*ldc + 4, vaddq_f32(vld1q_f32(C + 5*ldc + 4), c51));
}

#endif /* NEON micro-kernel */

/** @brief Macro-kernel: iterate MR*NR tiles over packed A and B panels.
 *  @param[in]     packed_A Packed A buffer (mc rows, MR-contiguous).
 *  @param[in]     packed_B Packed B buffer (nc cols, NR-contiguous).
 *  @param[in,out] C        Output matrix pointer (sub-block).
 *  @param[in]     mc       Row count of the A panel.
 *  @param[in]     nc       Column count of the B panel.
 *  @param[in]     kc       Shared K dimension.
 *  @param[in]     ldc      Leading dimension of C.
 *  @note Dispatches to the best available SIMD micro-kernel; scalar for edge tiles.
 *  @see ac_gemm, ac_micro_kernel_6x16, ac_micro_kernel_6x32 */

static AC_INLINE void ac_macro_kernel(const float* AC_RESTRICT packed_A,
                                      const float* AC_RESTRICT packed_B,
                                      float* AC_RESTRICT C,
                                      int mc, int nc, int kc, int ldc) {
    for (int jr = 0; jr < nc; jr += AC_GEMM_NR) {
        int nr = (jr + AC_GEMM_NR <= nc) ? AC_GEMM_NR : (nc - jr);
        for (int ir = 0; ir < mc; ir += AC_GEMM_MR) {
            int mr = (ir + AC_GEMM_MR <= mc) ? AC_GEMM_MR : (mc - ir);

            const float* pa = packed_A + ir * kc;
            const float* pb = packed_B + jr * kc;
            float* cp = C + ir * ldc + jr;

            if (mr == AC_GEMM_MR && nr == AC_GEMM_NR) {
                /* Full SIMD micro-kernel */
#if defined(AC_SIMD_AVX512)
                ac_micro_kernel_6x32(kc, pa, pb, cp, ldc);
#elif defined(AC_SIMD_AVX2)
                ac_micro_kernel_6x16(kc, pa, pb, cp, ldc);
#elif defined(AC_SIMD_NEON)
                ac_micro_kernel_6x8_neon(kc, pa, pb, cp, ldc);
#else
                ac_micro_kernel_scalar(kc, pa, pb, cp, ldc, mr, nr);
#endif
            } else {
                /* Edge tile: scalar fallback */
                ac_micro_kernel_scalar(kc, pa, pb, cp, ldc, mr, nr);
            }
        }
    }
}

/** @brief Thread-worker context for parallel GEMM.
 *  Carries pointers and blocking parameters shared across threads. */

typedef struct {
    const float*  A;            /**< Full A matrix pointer. */
    const float*  packed_B;     /**< Shared packed B panel (read-only). */
    float*        C;
    int           K, N;
    int           pc, kc, jc, nc;
} ac_gemm_ctx;

/** @brief GEMM worker function invoked per thread.
 *  @param[in] ctx_ptr   Pointer to an ac_gemm_ctx.
 *  @param[in] thread_id Thread index (unused).
 *  @param[in] start_row First row index (inclusive).
 *  @param[in] end_row   Last row index (exclusive).
 *  @note Allocates a thread-local packed_A buffer; freed on return.
 *  @see ac_gemm, ac_gemm_ctx */
static void ac_gemm_worker(void* ctx_ptr, int thread_id, int start_row, int end_row) {
    ac_gemm_ctx* ctx = (ac_gemm_ctx*)ctx_ptr;
    (void)thread_id;

    /* Thread-local pack_A buffer (MC is already a multiple of MR) */
    int mc_padded = ((AC_GEMM_MC + AC_GEMM_MR - 1) / AC_GEMM_MR) * AC_GEMM_MR;
    float* packed_A = (float*)AC_ALLOC_ALIGNED(
        (ac_size)mc_padded * (ac_size)ctx->kc * sizeof(float), 64);
    if (!packed_A) return;

    for (int ic = start_row; ic < end_row; ic += AC_GEMM_MC) {
        int mc = (ic + AC_GEMM_MC <= end_row) ? AC_GEMM_MC : (end_row - ic);
        ac_pack_A(ctx->A + ic * ctx->K + ctx->pc, packed_A, mc, ctx->kc, ctx->K);
        ac_macro_kernel(packed_A, ctx->packed_B,
                        ctx->C + ic * ctx->N + ctx->jc,
                        mc, ctx->nc, ctx->kc, ctx->N);
    }

    AC_FREE_ALIGNED(packed_A);
}

/** @brief Top-level GEMM: C[M*N] = A[M*K] * B[K*N].
 *
 *  BLIS-style cache-blocked, multi-threaded matrix multiply.
 *  Clears C to zero, then accumulates the product.
 *
 *  @param[in]  A  Input matrix A (M x K, row-major).
 *  @param[in]  B  Input matrix B (K x N, row-major).
 *  @param[out] C  Output matrix C (M x N, row-major), overwritten.
 *  @param[in]  M  Number of rows in A / C.
 *  @param[in]  N  Number of columns in B / C.
 *  @param[in]  K  Shared inner dimension.
 *  @note Uses thread pool for M >= 2*MC. Falls back to single-threaded otherwise.
 *  @see ac_gemm_bias, ac_macro_kernel */

AC_INLINE void ac_gemm(const float* AC_RESTRICT A,
                       const float* AC_RESTRICT B,
                       float* AC_RESTRICT C,
                       ac_size M, ac_size N, ac_size K) {
    memset(C, 0, M * N * sizeof(float));

    int iM = (int)M, iN = (int)N, iK = (int)K;

    /* Allocate packed B buffer (shared across threads, read-only)
     * Size must account for NR-padding: each panel is NR-wide even for edge cols */
    int max_kc = (iK < AC_GEMM_KC) ? iK : AC_GEMM_KC;
    int max_nc = (iN < AC_GEMM_NC) ? iN : AC_GEMM_NC;
    int max_nc_padded = ((max_nc + AC_GEMM_NR - 1) / AC_GEMM_NR) * AC_GEMM_NR;
    float* packed_B = (float*)AC_ALLOC_ALIGNED(
        (ac_size)max_kc * (ac_size)max_nc_padded * sizeof(float), 64);
    if (!packed_B) return;

    for (int jc = 0; jc < iN; jc += AC_GEMM_NC) {
        int nc = (jc + AC_GEMM_NC <= iN) ? AC_GEMM_NC : (iN - jc);

        for (int pc = 0; pc < iK; pc += AC_GEMM_KC) {
            int kc = (pc + AC_GEMM_KC <= iK) ? AC_GEMM_KC : (iK - pc);

            /* Pack B panel: B[pc:pc+kc, jc:jc+nc] -> packed_B */
            ac_pack_B(B + pc * iN + jc, packed_B, kc, nc, iN);

            /* Dispatch row blocks across threads */
            ac_gemm_ctx ctx;
            ctx.A = A;
            ctx.packed_B = packed_B;
            ctx.C = C;
            ctx.K = iK;
            ctx.N = iN;
            ctx.pc = pc;
            ctx.kc = kc;
            ctx.jc = jc;
            ctx.nc = nc;

            /* Use threading only for large matrices */
            if (iM >= AC_GEMM_MC * 2 && g_thread_pool_initialized) {
                ac_parallel_for(&g_thread_pool, 0, iM,
                                ac_gemm_worker, &ctx);
            } else {
                ac_gemm_worker(&ctx, 0, 0, iM);
            }
        }
    }

    AC_FREE_ALIGNED(packed_B);
}

/** @brief GEMM with per-row bias addition: C = A * B + bias.
 *  @param[in]  A    Input matrix A (M x K, row-major).
 *  @param[in]  B    Input matrix B (K x N, row-major).
 *  @param[in]  bias Bias vector (length N), added to every row.
 *  @param[out] C    Output matrix C (M x N, row-major), overwritten.
 *  @param[in]  M    Number of rows in A / C.
 *  @param[in]  N    Number of columns in B / C.
 *  @param[in]  K    Shared inner dimension.
 *  @see ac_gemm, ac_simd_add */

AC_INLINE void ac_gemm_bias(const float* AC_RESTRICT A,
                            const float* AC_RESTRICT B,
                            const float* AC_RESTRICT bias,
                            float* AC_RESTRICT C,
                            ac_size M, ac_size N, ac_size K) {
    ac_gemm(A, B, C, M, N, K);
    for (ac_size i = 0; i < M; i++) {
        ac_simd_add(C + i * N, bias, C + i * N, N);
    }
}


/** @} */

/** @name Transpose */
/** @{ */

/** @brief Out-of-place matrix transpose: B = A^T.
 *
 *  Uses 8x8 block tiling for cache efficiency.
 *
 *  @param[in]  A    Input matrix (rows x cols, row-major).
 *  @param[out] B    Output matrix (cols x rows, row-major).
 *  @param[in]  rows Number of rows in A.
 *  @param[in]  cols Number of columns in A.
 *  @note Handles remainder rows/cols outside the 8x8 blocks. */

AC_INLINE void ac_transpose(const float* AC_RESTRICT A, float* AC_RESTRICT B,
                            ac_size rows, ac_size cols) {
    /* 8x8 block transpose for cache efficiency */
    ac_size bi, bj;
    const ac_size BS = 8;
    for (bi = 0; bi + BS <= rows; bi += BS) {
        for (bj = 0; bj + BS <= cols; bj += BS) {
            for (ac_size i = bi; i < bi + BS; i++) {
                for (ac_size j = bj; j < bj + BS; j++) {
                    B[j * rows + i] = A[i * cols + j];
                }
            }
        }
    }
    /* Handle row remainder */
    for (ac_size i = bi; i < rows; i++) {
        for (ac_size j = 0; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
    /* Handle col remainder for main rows */
    for (ac_size i = 0; i < bi; i++) {
        for (ac_size j = bj; j < cols; j++) {
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

/** @} */
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_SIMD_MATH_H */
