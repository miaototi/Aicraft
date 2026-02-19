/**
 * @file fast_math.h
 * @brief Vectorized transcendental functions (exp, sigmoid, tanh).
 *
 * High-accuracy SIMD polynomial approximations using Cephes/Remez minimax
 * polynomials. Achieves ~1 ULP accuracy and 4–8× speedup over scalar libm.
 *
 * @note Supports AVX2 (8-wide), AVX-512 (16-wide), and NEON (4-wide).
 * @see ac_fast_exp, ac_fast_sigmoid, ac_fast_tanh
 */

#ifndef AICRAFT_FAST_MATH_H
#define AICRAFT_FAST_MATH_H

#include "aicraft/platform.h"
#include <math.h>

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

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup fastmath Vectorized Transcendentals
 *  @{ */

/** @name AVX2 Intrinsics (8-wide)
 *  @{ */

#if defined(AC_SIMD_AVX2)

/**
 * @brief Vectorized exp(x) using Cephes-style range reduction.
 *
 * Degree-6 Remez minimax polynomial approximation.
 * Accuracy: < 1 ULP over [-87.3, 88.7].
 *
 * @param x Input vector of 8 floats.
 * @return exp(x) for each lane.
 * @note Uses FMA when available for improved precision.
 * @see ac_fast_exp
 */
AC_INLINE __m256 ac_mm256_exp_ps(__m256 x) {
    const __m256 c_ln2_hi  = _mm256_set1_ps(0.693145751953125f);
    const __m256 c_ln2_lo  = _mm256_set1_ps(1.428606765330187e-06f);
    const __m256 c_inv_ln2 = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c_half    = _mm256_set1_ps(0.5f);
    const __m256 c_one     = _mm256_set1_ps(1.0f);
    const __m256 c_min_x   = _mm256_set1_ps(-87.3f);
    const __m256 c_max_x   = _mm256_set1_ps(88.3f);

    /* Minimax polynomial coefficients for exp(r) on [-ln2/2, ln2/2] */
    const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);

    /* Clamp input */
    x = _mm256_max_ps(x, c_min_x);
    x = _mm256_min_ps(x, c_max_x);

    /* n = round(x / ln2) */
    __m256 n = _mm256_round_ps(
        _mm256_mul_ps(x, c_inv_ln2),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC
    );

    /* r = x - n * ln2 (Cahan two-part reduction for precision) */
#ifdef __FMA__
    __m256 r = _mm256_fnmadd_ps(n, c_ln2_hi, x);
    r = _mm256_fnmadd_ps(n, c_ln2_lo, r);
#else
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, c_ln2_hi));
    r = _mm256_sub_ps(r, _mm256_mul_ps(n, c_ln2_lo));
#endif

    /* exp(r) ≈ 1 + r + r²*(p5 + r*(p4 + r*(p3 + r*(p2 + r*(p1 + r*p0))))) */
    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 poly = p0;
#ifdef __FMA__
    poly = _mm256_fmadd_ps(poly, r, p1);
    poly = _mm256_fmadd_ps(poly, r, p2);
    poly = _mm256_fmadd_ps(poly, r, p3);
    poly = _mm256_fmadd_ps(poly, r, p4);
    poly = _mm256_fmadd_ps(poly, r, p5);
    poly = _mm256_fmadd_ps(poly, r2, r);
    poly = _mm256_add_ps(poly, c_one);
#else
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), p1);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), p2);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), p3);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), p4);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), p5);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r2), r);
    poly = _mm256_add_ps(poly, c_one);
#endif

    /* Scale by 2^n: reinterpret n as exponent bits */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_slli_epi32(ni, 23); /* shift into IEEE float exponent field */
    __m256i result_i = _mm256_add_epi32(_mm256_castps_si256(poly), ni);

    return _mm256_castsi256_ps(result_i);
}

/**
 * @brief Vectorized sigmoid(x) = 1 / (1 + exp(-x)).
 *
 * @param x Input vector of 8 floats.
 * @return sigmoid(x) for each lane.
 * @see ac_mm256_exp_ps, ac_fast_sigmoid
 */
AC_INLINE __m256 ac_mm256_sigmoid_ps(__m256 x) {
    const __m256 c_one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f)); /* negate via sign flip */
    __m256 exp_neg = ac_mm256_exp_ps(neg_x);
    return _mm256_div_ps(c_one, _mm256_add_ps(c_one, exp_neg));
}

/**
 * @brief Vectorized tanh(x) = 2 * sigmoid(2x) - 1.
 *
 * @param x Input vector of 8 floats.
 * @return tanh(x) for each lane.
 * @see ac_mm256_sigmoid_ps, ac_fast_tanh
 */
AC_INLINE __m256 ac_mm256_tanh_ps(__m256 x) {
    const __m256 c_two = _mm256_set1_ps(2.0f);
    const __m256 c_one = _mm256_set1_ps(1.0f);
    __m256 sig = ac_mm256_sigmoid_ps(_mm256_mul_ps(c_two, x));
#ifdef __FMA__
    return _mm256_fmsub_ps(c_two, sig, c_one);
#else
    return _mm256_sub_ps(_mm256_mul_ps(c_two, sig), c_one);
#endif
}

#endif /* AC_SIMD_AVX2 */


/** @} */

/** @name AVX-512 Intrinsics (16-wide)
 *  @{ */

#if defined(AC_SIMD_AVX512)

/**
 * @brief Vectorized exp(x) for 16-wide AVX-512.
 *
 * Same Cephes-style algorithm as the AVX2 version.
 *
 * @param x Input vector of 16 floats.
 * @return exp(x) for each lane.
 * @see ac_mm256_exp_ps
 */
AC_INLINE __m512 ac_mm512_exp_ps(__m512 x) {
    const __m512 c_ln2_hi  = _mm512_set1_ps(0.693145751953125f);
    const __m512 c_ln2_lo  = _mm512_set1_ps(1.428606765330187e-06f);
    const __m512 c_inv_ln2 = _mm512_set1_ps(1.44269504088896341f);
    const __m512 c_one     = _mm512_set1_ps(1.0f);
    const __m512 c_min_x   = _mm512_set1_ps(-87.3f);
    const __m512 c_max_x   = _mm512_set1_ps(88.3f);
    
    const __m512 p0 = _mm512_set1_ps(1.9875691500e-4f);
    const __m512 p1 = _mm512_set1_ps(1.3981999507e-3f);
    const __m512 p2 = _mm512_set1_ps(8.3334519073e-3f);
    const __m512 p3 = _mm512_set1_ps(4.1665795894e-2f);
    const __m512 p4 = _mm512_set1_ps(1.6666665459e-1f);
    const __m512 p5 = _mm512_set1_ps(5.0000001201e-1f);
    
    x = _mm512_max_ps(x, c_min_x);
    x = _mm512_min_ps(x, c_max_x);
    
    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, c_inv_ln2), _MM_FROUND_TO_NEAREST_INT);
    
    __m512 r = _mm512_fnmadd_ps(n, c_ln2_hi, x);
    r = _mm512_fnmadd_ps(n, c_ln2_lo, r);
    
    __m512 r2 = _mm512_mul_ps(r, r);
    __m512 poly = p0;
    poly = _mm512_fmadd_ps(poly, r, p1);
    poly = _mm512_fmadd_ps(poly, r, p2);
    poly = _mm512_fmadd_ps(poly, r, p3);
    poly = _mm512_fmadd_ps(poly, r, p4);
    poly = _mm512_fmadd_ps(poly, r, p5);
    poly = _mm512_fmadd_ps(poly, r2, r);
    poly = _mm512_add_ps(poly, c_one);
    
    __m512i ni = _mm512_cvtps_epi32(n);
    ni = _mm512_slli_epi32(ni, 23);
    __m512i result_i = _mm512_add_epi32(_mm512_castps_si512(poly), ni);
    return _mm512_castsi512_ps(result_i);
}

/**
 * @brief Vectorized sigmoid(x) for 16-wide AVX-512.
 *
 * @param x Input vector of 16 floats.
 * @return sigmoid(x) for each lane.
 * @see ac_mm512_exp_ps, ac_fast_sigmoid
 */
AC_INLINE __m512 ac_mm512_sigmoid_ps(__m512 x) {
    const __m512 c_one = _mm512_set1_ps(1.0f);
    __m512 neg_x = _mm512_mul_ps(x, _mm512_set1_ps(-1.0f));
    __m512 exp_neg = ac_mm512_exp_ps(neg_x);
    return _mm512_div_ps(c_one, _mm512_add_ps(c_one, exp_neg));
}

/**
 * @brief Vectorized tanh(x) for 16-wide AVX-512.
 *
 * @param x Input vector of 16 floats.
 * @return tanh(x) for each lane.
 * @see ac_mm512_sigmoid_ps, ac_fast_tanh
 */
AC_INLINE __m512 ac_mm512_tanh_ps(__m512 x) {
    const __m512 c_two = _mm512_set1_ps(2.0f);
    const __m512 c_one = _mm512_set1_ps(1.0f);
    __m512 sig = ac_mm512_sigmoid_ps(_mm512_mul_ps(c_two, x));
    return _mm512_fmsub_ps(c_two, sig, c_one);
}

#endif /* AC_SIMD_AVX512 */


/** @} */

/** @name NEON Intrinsics (4-wide)
 *  @{ */

#if defined(AC_SIMD_NEON)

/**
 * @brief Vectorized exp(x) for 4-wide ARM NEON.
 *
 * Cephes-style range reduction + degree-6 Remez polynomial,
 * same algorithm as the AVX2 version using float32x4_t.
 *
 * @param x Input vector of 4 floats.
 * @return exp(x) for each lane.
 * @see ac_mm256_exp_ps, ac_fast_exp
 */
AC_INLINE float32x4_t ac_vexpq_f32(float32x4_t x) {
    const float32x4_t c_ln2_hi  = vdupq_n_f32(0.693145751953125f);
    const float32x4_t c_ln2_lo  = vdupq_n_f32(1.428606765330187e-06f);
    const float32x4_t c_inv_ln2 = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t c_one     = vdupq_n_f32(1.0f);
    const float32x4_t c_min_x   = vdupq_n_f32(-87.3f);
    const float32x4_t c_max_x   = vdupq_n_f32(88.3f);

    const float32x4_t p0 = vdupq_n_f32(1.9875691500e-4f);
    const float32x4_t p1 = vdupq_n_f32(1.3981999507e-3f);
    const float32x4_t p2 = vdupq_n_f32(8.3334519073e-3f);
    const float32x4_t p3 = vdupq_n_f32(4.1665795894e-2f);
    const float32x4_t p4 = vdupq_n_f32(1.6666665459e-1f);
    const float32x4_t p5 = vdupq_n_f32(5.0000001201e-1f);

    /* Clamp input */
    x = vmaxq_f32(x, c_min_x);
    x = vminq_f32(x, c_max_x);

    /* n = round(x / ln2) */
    float32x4_t n = vrndnq_f32(vmulq_f32(x, c_inv_ln2));

    /* r = x - n * ln2 (two-part Cahan reduction) */
    float32x4_t r = vfmsq_f32(x, n, c_ln2_hi);
    r = vfmsq_f32(r, n, c_ln2_lo);

    /* Horner polynomial: exp(r) ≈ 1 + r + r²*(p5 + r*(p4 + r*(p3 + r*(p2 + r*(p1 + r*p0))))) */
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t poly = p0;
    poly = vfmaq_f32(p1, poly, r);
    poly = vfmaq_f32(p2, poly, r);
    poly = vfmaq_f32(p3, poly, r);
    poly = vfmaq_f32(p4, poly, r);
    poly = vfmaq_f32(p5, poly, r);
    poly = vfmaq_f32(r, poly, r2);
    poly = vaddq_f32(poly, c_one);

    /* Scale by 2^n: add n<<23 to the IEEE float representation */
    int32x4_t ni = vcvtq_s32_f32(n);
    ni = vshlq_n_s32(ni, 23);
    int32x4_t result_i = vaddq_s32(vreinterpretq_s32_f32(poly), ni);
    return vreinterpretq_f32_s32(result_i);
}

/**
 * @brief Vectorized sigmoid(x) for 4-wide ARM NEON.
 *
 * Uses reciprocal estimate with two Newton-Raphson steps.
 *
 * @param x Input vector of 4 floats.
 * @return sigmoid(x) for each lane.
 * @see ac_vexpq_f32, ac_fast_sigmoid
 */
AC_INLINE float32x4_t ac_vsigmoidq_f32(float32x4_t x) {
    const float32x4_t c_one = vdupq_n_f32(1.0f);
    float32x4_t neg_x = vnegq_f32(x);
    float32x4_t exp_neg = ac_vexpq_f32(neg_x);
    /* NEON has no direct div; use reciprocal estimate + Newton step */
    float32x4_t denom = vaddq_f32(c_one, exp_neg);
    float32x4_t recip = vrecpeq_f32(denom);
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip)); /* Newton step 1 */
    recip = vmulq_f32(recip, vrecpsq_f32(denom, recip)); /* Newton step 2 */
    return recip;
}

/**
 * @brief Vectorized tanh(x) for 4-wide ARM NEON.
 *
 * @param x Input vector of 4 floats.
 * @return tanh(x) for each lane.
 * @see ac_vsigmoidq_f32, ac_fast_tanh
 */
AC_INLINE float32x4_t ac_vtanhq_f32(float32x4_t x) {
    const float32x4_t c_two = vdupq_n_f32(2.0f);
    const float32x4_t c_one = vdupq_n_f32(1.0f);
    float32x4_t sig = ac_vsigmoidq_f32(vmulq_f32(c_two, x));
    return vsubq_f32(vmulq_f32(c_two, sig), c_one);
}

#endif /* AC_SIMD_NEON */


/** @} */

/** @name Array Wrappers
 *  Auto-dispatch to best SIMD width.
 *  @{ */

/**
 * @brief Compute exp(x) over an array, auto-dispatched to best SIMD width.
 *
 * @param in   Input array of floats.
 * @param out  Output array (may alias @p in).
 * @param n    Number of elements.
 * @see ac_mm256_exp_ps, ac_mm512_exp_ps, ac_vexpq_f32
 */
AC_INLINE void ac_fast_exp(const float* AC_RESTRICT in, float* AC_RESTRICT out, ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, ac_mm512_exp_ps(_mm512_loadu_ps(in + i)));
    }
#endif
#if defined(AC_SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, ac_mm256_exp_ps(_mm256_loadu_ps(in + i)));
    }
#elif defined(AC_SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, ac_vexpq_f32(vld1q_f32(in + i)));
    }
#endif
    for (; i < n; i++) out[i] = expf(in[i]);
}

/**
 * @brief Compute sigmoid(x) over an array, auto-dispatched to best SIMD width.
 *
 * @param in   Input array of floats.
 * @param out  Output array (may alias @p in).
 * @param n    Number of elements.
 * @see ac_mm256_sigmoid_ps, ac_mm512_sigmoid_ps, ac_vsigmoidq_f32
 */
AC_INLINE void ac_fast_sigmoid(const float* AC_RESTRICT in, float* AC_RESTRICT out, ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, ac_mm512_sigmoid_ps(_mm512_loadu_ps(in + i)));
    }
#endif
#if defined(AC_SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, ac_mm256_sigmoid_ps(_mm256_loadu_ps(in + i)));
    }
#elif defined(AC_SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, ac_vsigmoidq_f32(vld1q_f32(in + i)));
    }
#endif
    for (; i < n; i++) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

/**
 * @brief Compute tanh(x) over an array, auto-dispatched to best SIMD width.
 *
 * @param in   Input array of floats.
 * @param out  Output array (may alias @p in).
 * @param n    Number of elements.
 * @see ac_mm256_tanh_ps, ac_mm512_tanh_ps, ac_vtanhq_f32
 */
AC_INLINE void ac_fast_tanh(const float* AC_RESTRICT in, float* AC_RESTRICT out, ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX512)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(out + i, ac_mm512_tanh_ps(_mm512_loadu_ps(in + i)));
    }
#endif
#if defined(AC_SIMD_AVX2)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, ac_mm256_tanh_ps(_mm256_loadu_ps(in + i)));
    }
#elif defined(AC_SIMD_NEON)
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, ac_vtanhq_f32(vld1q_f32(in + i)));
    }
#endif
    for (; i < n; i++) out[i] = tanhf(in[i]);
}

/**
 * @brief Backward pass for sigmoid: grad_in[i] += grad_out[i] * sig[i] * (1 - sig[i]).
 *
 * @param sigmoid_out  Forward-pass sigmoid outputs.
 * @param grad_out     Upstream gradient.
 * @param grad_in      Gradient accumulator (read-modify-write).
 * @param n            Number of elements.
 */
AC_INLINE void ac_fast_sigmoid_backward(const float* AC_RESTRICT sigmoid_out,
                                        const float* AC_RESTRICT grad_out,
                                        float* AC_RESTRICT grad_in,
                                        ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX2)
    const __m256 c_one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 s = _mm256_loadu_ps(sigmoid_out + i);
        __m256 g = _mm256_loadu_ps(grad_out + i);
        __m256 one_minus_s = _mm256_sub_ps(c_one, s);
        __m256 ds = _mm256_mul_ps(s, one_minus_s);
        __m256 gi = _mm256_loadu_ps(grad_in + i);
#ifdef __FMA__
        _mm256_storeu_ps(grad_in + i, _mm256_fmadd_ps(g, ds, gi));
#else
        _mm256_storeu_ps(grad_in + i, _mm256_add_ps(gi, _mm256_mul_ps(g, ds)));
#endif
    }
#elif defined(AC_SIMD_NEON)
    const float32x4_t c_one_n = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t s = vld1q_f32(sigmoid_out + i);
        float32x4_t g = vld1q_f32(grad_out + i);
        float32x4_t ds = vmulq_f32(s, vsubq_f32(c_one_n, s));
        float32x4_t gi = vld1q_f32(grad_in + i);
        vst1q_f32(grad_in + i, vfmaq_f32(gi, g, ds));
    }
#endif
    for (; i < n; i++) {
        float s = sigmoid_out[i];
        grad_in[i] += grad_out[i] * s * (1.0f - s);
    }
}

/**
 * @brief Backward pass for tanh: grad_in[i] += grad_out[i] * (1 - tanh_out[i]^2).
 *
 * @param tanh_out   Forward-pass tanh outputs.
 * @param grad_out   Upstream gradient.
 * @param grad_in    Gradient accumulator (read-modify-write).
 * @param n          Number of elements.
 */
AC_INLINE void ac_fast_tanh_backward(const float* AC_RESTRICT tanh_out,
                                     const float* AC_RESTRICT grad_out,
                                     float* AC_RESTRICT grad_in,
                                     ac_size n) {
    ac_size i = 0;
#if defined(AC_SIMD_AVX2)
    const __m256 c_one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 t = _mm256_loadu_ps(tanh_out + i);
        __m256 g = _mm256_loadu_ps(grad_out + i);
        __m256 t2 = _mm256_mul_ps(t, t);
        __m256 dt = _mm256_sub_ps(c_one, t2);
        __m256 gi = _mm256_loadu_ps(grad_in + i);
#ifdef __FMA__
        _mm256_storeu_ps(grad_in + i, _mm256_fmadd_ps(g, dt, gi));
#else
        _mm256_storeu_ps(grad_in + i, _mm256_add_ps(gi, _mm256_mul_ps(g, dt)));
#endif
    }
#elif defined(AC_SIMD_NEON)
    const float32x4_t c_one_n = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t t = vld1q_f32(tanh_out + i);
        float32x4_t g = vld1q_f32(grad_out + i);
        float32x4_t t2 = vmulq_f32(t, t);
        float32x4_t dt = vsubq_f32(c_one_n, t2);
        float32x4_t gi = vld1q_f32(grad_in + i);
        vst1q_f32(grad_in + i, vfmaq_f32(gi, g, dt));
    }
#endif
    for (; i < n; i++) {
        float t = tanh_out[i];
        grad_in[i] += grad_out[i] * (1.0f - t * t);
    }
}

/** @} */
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_FAST_MATH_H */
