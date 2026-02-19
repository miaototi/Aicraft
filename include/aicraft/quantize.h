/**
 * @file quantize.h
 * @brief INT8 Quantization Engine for edge/embedded inference.
 *
 * Provides per-tensor affine quantization using an asymmetric scheme:
 *
 *   Quantize:   @code q = clamp(round(x / scale) + zero_point, 0, 255) @endcode
 *   Dequantize: @code x = (q - zero_point) * scale @endcode
 *
 * Features:
 *   - Float32 to UINT8 quantization (calibrated from min/max)
 *   - UINT8 to Float32 dequantization
 *   - Quantized dense layer (INT8 matmul with INT32 accumulation)
 *   - Per-tensor and per-channel quantization parameters
 *   - Model size estimation for deployment
 *
 * @see ac_quant_params, ac_qtensor, ac_qdense
 */

#ifndef AICRAFT_QUANTIZE_H
#define AICRAFT_QUANTIZE_H

#include "aicraft/platform.h"
#include "aicraft/tensor.h"
#include "aicraft/memory.h"
#include <math.h>
#include <string.h>

#ifdef AC_SIMD_NEON
    #include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup quantize INT8 Quantization */
/** @{ */

/**
 * @brief Per-tensor affine quantization parameters.
 *
 * Holds the scale, zero-point, and calibrated range used to convert
 * between float32 and uint8 representations.
 */
typedef struct {
    float   scale;       /**< Scale factor: (max - min) / 255. */
    int32_t zero_point;  /**< Zero-point: maps real 0.0 to [0, 255]. */
    float   min_val;     /**< Calibrated minimum value. */
    float   max_val;     /**< Calibrated maximum value. */
} ac_quant_params;

/**
 * @brief A tensor stored in quantized UINT8 format.
 *
 * Wraps a raw uint8 buffer together with its shape and the
 * quantization parameters needed for dequantization.
 */
typedef struct {
    uint8_t*        data;       /**< Quantized UINT8 values. */
    ac_shape        shape;      /**< Tensor dimensions. */
    ac_quant_params qparams;    /**< Associated quantization parameters. */
} ac_qtensor;

/**
 * @brief Calibrate quantization parameters from a float tensor.
 *
 * Scans the data to find the min/max range and computes the
 * scale and zero-point for asymmetric UINT8 quantization.
 * The range is extended to include zero so that the zero-point
 * maps exactly to 0.0.
 *
 * @param data  Pointer to the float32 source data.
 * @param n     Number of elements in @p data.
 * @return Calibrated ac_quant_params ready for quantize/dequantize.
 *
 * @see ac_quantize, ac_dequantize
 */
AC_INLINE ac_quant_params ac_calibrate(const float* data, ac_size n) {
    ac_quant_params qp;
    float min_val = data[0], max_val = data[0];
    for (ac_size i = 1; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    /* Ensure range includes zero for correct zero_point */
    if (min_val > 0.0f) min_val = 0.0f;
    if (max_val < 0.0f) max_val = 0.0f;

    qp.min_val = min_val;
    qp.max_val = max_val;
    float range = max_val - min_val;
    if (range < 1e-10f) range = 1e-10f;

    qp.scale = range / 255.0f;
    qp.zero_point = (int32_t)roundf(-min_val / qp.scale);
    if (qp.zero_point < 0) qp.zero_point = 0;
    if (qp.zero_point > 255) qp.zero_point = 255;
    return qp;
}

/**
 * @brief Quantize a float32 array to UINT8.
 *
 * Applies the affine transform
 * @code q = clamp(round(src[i] / scale) + zero_point, 0, 255) @endcode
 *
 * @param src  Pointer to the float32 source buffer.
 * @param dst  Pointer to the uint8 destination buffer (must hold @p n bytes).
 * @param n    Number of elements to quantize.
 * @param qp   Quantization parameters (scale, zero-point).
 *
 * @simd NEON (4-wide) and AVX2 (8-wide) fast paths.
 * @see ac_calibrate, ac_dequantize
 */
AC_INLINE void ac_quantize(const float* AC_RESTRICT src, uint8_t* AC_RESTRICT dst,
                           ac_size n, const ac_quant_params* qp) {
    float inv_scale = 1.0f / qp->scale;
    int32_t zp = qp->zero_point;
    ac_size i = 0;

#if defined(AC_SIMD_NEON)
    float32x4_t v_inv_scale = vdupq_n_f32(inv_scale);
    float32x4_t v_zp = vcvtq_f32_s32(vdupq_n_s32(zp));
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    float32x4_t v_255 = vdupq_n_f32(255.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t val = vld1q_f32(src + i);
        /* q = clamp(round(val * inv_scale + zp), 0, 255) */
        float32x4_t q = vfmaq_f32(v_zp, val, v_inv_scale);
        q = vrndnq_f32(q);
        q = vmaxq_f32(q, v_zero);
        q = vminq_f32(q, v_255);
        uint32x4_t qi = vcvtq_u32_f32(q);
        /* Narrow 32→16→8 */
        uint16x4_t qi16 = vmovn_u32(qi);
        uint16x8_t qi16_full = vcombine_u16(qi16, qi16);
        uint8x8_t qi8 = vmovn_u16(qi16_full);
        /* Store first 4 bytes */
        vst1_lane_u32((uint32_t*)(dst + i), vreinterpret_u32_u8(qi8), 0);
    }
#elif defined(AC_SIMD_AVX2)
    __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
    __m256 v_zp = _mm256_set1_ps((float)zp);
    __m256 v_zero = _mm256_setzero_ps();
    __m256 v_255 = _mm256_set1_ps(255.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 val = _mm256_loadu_ps(src + i);
#ifdef __FMA__
        __m256 q = _mm256_fmadd_ps(val, v_inv_scale, v_zp);
#else
        __m256 q = _mm256_add_ps(_mm256_mul_ps(val, v_inv_scale), v_zp);
#endif
        q = _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q = _mm256_max_ps(q, v_zero);
        q = _mm256_min_ps(q, v_255);
        __m256i qi = _mm256_cvtps_epi32(q);
        /* Pack 32→16→8 */
        __m128i lo = _mm256_castsi256_si128(qi);
        __m128i hi = _mm256_extracti128_si256(qi, 1);
        __m128i packed16 = _mm_packs_epi32(lo, hi);
        __m128i packed8 = _mm_packus_epi16(packed16, packed16);
        /* Store 8 bytes */
        *(int64_t*)(dst + i) = _mm_cvtsi128_si64(packed8);
    }
#endif

    for (; i < n; i++) {
        float q = roundf(src[i] * inv_scale) + (float)zp;
        if (q < 0.0f) q = 0.0f;
        if (q > 255.0f) q = 255.0f;
        dst[i] = (uint8_t)q;
    }
}

/**
 * @brief Dequantize a UINT8 array back to float32.
 *
 * Applies the inverse affine transform
 * @code dst[i] = (src[i] - zero_point) * scale @endcode
 *
 * @param src  Pointer to the uint8 source buffer.
 * @param dst  Pointer to the float32 destination buffer (must hold @p n floats).
 * @param n    Number of elements to dequantize.
 * @param qp   Quantization parameters (scale, zero-point).
 *
 * @simd NEON (4-wide) and AVX2 (8-wide) fast paths.
 * @see ac_calibrate, ac_quantize
 */
AC_INLINE void ac_dequantize(const uint8_t* AC_RESTRICT src, float* AC_RESTRICT dst,
                             ac_size n, const ac_quant_params* qp) {
    float scale = qp->scale;
    int32_t zp = qp->zero_point;
    ac_size i = 0;

#if defined(AC_SIMD_NEON)
    float32x4_t v_scale = vdupq_n_f32(scale);
    float32x4_t v_zp = vcvtq_f32_s32(vdupq_n_s32(zp));
    for (; i + 4 <= n; i += 4) {
        /* Load 4 uint8 → widen to uint32 → float */
        uint32_t raw = *(const uint32_t*)(src + i);
        uint8x8_t u8 = vcreate_u8((uint64_t)raw);
        uint16x8_t u16 = vmovl_u8(u8);
        uint32x4_t u32 = vmovl_u16(vget_low_u16(u16));
        float32x4_t val = vcvtq_f32_u32(u32);
        val = vmulq_f32(vsubq_f32(val, v_zp), v_scale);
        vst1q_f32(dst + i, val);
    }
#elif defined(AC_SIMD_AVX2)
    __m256 v_scale = _mm256_set1_ps(scale);
    __m256 v_zp = _mm256_set1_ps((float)zp);
    for (; i + 8 <= n; i += 8) {
        /* Load 8 uint8 → expand to int32 → float */
        __m128i raw = _mm_loadl_epi64((const __m128i*)(src + i));
        __m256i i32 = _mm256_cvtepu8_epi32(raw);
        __m256 val = _mm256_cvtepi32_ps(i32);
        val = _mm256_mul_ps(_mm256_sub_ps(val, v_zp), v_scale);
        _mm256_storeu_ps(dst + i, val);
    }
#endif

    for (; i < n; i++) {
        dst[i] = ((float)src[i] - (float)zp) * scale;
    }
}

/**
 * @brief Create a zero-initialized quantized tensor.
 *
 * Allocates data from the global tensor arena and zeroes
 * both the uint8 buffer and the quantization parameters.
 *
 * @param shape  Desired tensor shape.
 * @return Pointer to the newly created ac_qtensor.
 *
 * @note Memory is arena-allocated; no individual free is needed.
 * @see ac_tensor_quantize, ac_qtensor_dequantize
 */
AC_INLINE ac_qtensor* ac_qtensor_create(ac_shape shape) {
    ac_ensure_arena();
    ac_qtensor* qt = (ac_qtensor*)ac_arena_alloc(&g_tensor_arena, sizeof(ac_qtensor));
    qt->shape = shape;
    qt->data = (uint8_t*)ac_arena_alloc(&g_tensor_arena, shape.total_size);
    memset(qt->data, 0, shape.total_size);
    memset(&qt->qparams, 0, sizeof(ac_quant_params));
    return qt;
}

/**
 * @brief Quantize a float tensor to a quantized tensor.
 *
 * Calibrates quantization parameters from the tensor data and
 * converts all elements from float32 to UINT8.
 *
 * @param t  Source float tensor.
 * @return Pointer to a new ac_qtensor holding the quantized data.
 *
 * @see ac_qtensor_dequantize, ac_calibrate
 */
AC_INLINE ac_qtensor* ac_tensor_quantize(const ac_tensor* t) {
    ac_qtensor* qt = ac_qtensor_create(t->shape);
    qt->qparams = ac_calibrate(t->data, t->shape.total_size);
    ac_quantize(t->data, qt->data, t->shape.total_size, &qt->qparams);
    return qt;
}

/**
 * @brief Dequantize a quantized tensor back to float32.
 *
 * Creates a new float tensor and reconstructs the approximate
 * original values using the stored quantization parameters.
 *
 * @param qt  Source quantized tensor.
 * @return Pointer to a new ac_tensor holding the dequantized float data.
 *
 * @see ac_tensor_quantize, ac_dequantize
 */
AC_INLINE ac_tensor* ac_qtensor_dequantize(const ac_qtensor* qt) {
    ac_tensor* t = ac_tensor_create(qt->shape, 0);
    ac_dequantize(qt->data, t->data, qt->shape.total_size, &qt->qparams);
    return t;
}

/**
 * @brief Quantized matrix multiply: C = A_q x B_q.
 *
 * Performs INT8 matrix multiplication with INT32 accumulation,
 * then rescales the result to float32:
 * @code
 *   C_float[i,j] = scale_a * scale_b *
 *                  sum_k (A_q[i,k] - zp_a) * (B_q[k,j] - zp_b)
 * @endcode
 *
 * @param A  Quantized matrix [M x K].
 * @param B  Quantized matrix [K x N].
 * @param C  Output float32 buffer [M x N] (caller-allocated).
 * @param M  Number of rows of A / rows of C.
 * @param N  Number of columns of B / columns of C.
 * @param K  Shared inner dimension.
 *
 * @note Accumulates in INT32 to avoid overflow.
 * @simd NEON 8-wide widening multiply-accumulate path.
 * @see ac_qdense_forward
 */
AC_INLINE void ac_qgemm(const ac_qtensor* A, const ac_qtensor* B,
                        float* AC_RESTRICT C,
                        ac_size M, ac_size N, ac_size K) {
    int32_t zp_a = A->qparams.zero_point;
    int32_t zp_b = B->qparams.zero_point;
    float combined_scale = A->qparams.scale * B->qparams.scale;

    for (ac_size i = 0; i < M; i++) {
        for (ac_size j = 0; j < N; j++) {
            int32_t acc = 0;
            ac_size k = 0;

#if defined(AC_SIMD_NEON)
            int32x4_t vacc = vdupq_n_s32(0);
            int16x8_t v_zpa = vdupq_n_s16((int16_t)zp_a);
            int16x8_t v_zpb = vdupq_n_s16((int16_t)zp_b);
            for (; k + 8 <= K; k += 8) {
                /* Load 8 uint8 values from A and B */
                uint8x8_t a8 = vld1_u8(A->data + i * K + k);
                uint8x8_t b8 = vld1_u8(B->data + k * N + j); /* Note: this accesses B column-wise */
                /* Actually for column access, we need to gather. Fall back to simpler approach */
                /* Widen to int16, subtract zero points */
                int16x8_t a16 = vreinterpretq_s16_u16(vmovl_u8(a8));
                a16 = vsubq_s16(a16, v_zpa);
                /* For B we need column access - gather manually for now */
                int16_t b_vals[8];
                for (int kk = 0; kk < 8; kk++) {
                    b_vals[kk] = (int16_t)B->data[(k + kk) * N + j] - (int16_t)zp_b;
                }
                int16x8_t b16 = vld1q_s16(b_vals);
                /* Multiply-accumulate: widening to int32 */
                vacc = vmlal_s16(vacc, vget_low_s16(a16), vget_low_s16(b16));
                vacc = vmlal_s16(vacc, vget_high_s16(a16), vget_high_s16(b16));
            }
            acc = vaddvq_s32(vacc);
#endif

            for (; k < K; k++) {
                int32_t a_val = (int32_t)A->data[i * K + k] - zp_a;
                int32_t b_val = (int32_t)B->data[k * N + j] - zp_b;
                acc += a_val * b_val;
            }
            C[i * N + j] = (float)acc * combined_scale;
        }
    }
}

/**
 * @brief Quantized dense (fully-connected) layer for inference.
 *
 * Stores weights in UINT8 and bias in float32.  Created by
 * converting a trained floating-point dense layer via
 * ac_qdense_from_dense().
 *
 * @see ac_qdense_from_dense, ac_qdense_forward
 */
typedef struct {
    ac_qtensor* weight;      /**< Quantized weight matrix [out_features x in_features]. */
    float*      bias;        /**< Bias vector [out_features], kept in float32. */
    ac_size     in_features; /**< Number of input features. */
    ac_size     out_features;/**< Number of output features. */
} ac_qdense;

/**
 * @brief Convert a trained dense layer to quantized form.
 *
 * Quantizes the weight matrix to UINT8 and retains the bias
 * in float32 for inference.
 *
 * @param qd            Output quantized dense layer.
 * @param weight        Float32 weight tensor [out_features x in_features].
 * @param bias          Float32 bias tensor [out_features], or NULL.
 * @param in_features   Number of input features.
 * @param out_features  Number of output features.
 *
 * @see ac_qdense_forward
 */
AC_INLINE void ac_qdense_from_dense(ac_qdense* qd, const ac_tensor* weight,
                                     const ac_tensor* bias,
                                     ac_size in_features, ac_size out_features) {
    qd->in_features = in_features;
    qd->out_features = out_features;
    qd->weight = ac_tensor_quantize(weight);
    qd->bias = bias ? bias->data : NULL;
}

/**
 * @brief Forward pass through a quantized dense layer.
 *
 * Quantizes the input on-the-fly, performs INT8 matmul via
 * ac_qgemm(), then adds the float32 bias:
 * @code output = dequant(quant(input) x weight_q) + bias @endcode
 *
 * @param qd     Quantized dense layer.
 * @param input  Float32 input tensor [batch x in_features].
 * @return New float32 output tensor [batch x out_features].
 *
 * @see ac_qdense_from_dense, ac_qgemm
 */
AC_INLINE ac_tensor* ac_qdense_forward(const ac_qdense* qd, const ac_tensor* input) {
    ac_size batch = input->shape.dims[0];
    ac_tensor* output = ac_tensor_create(ac_shape_2d(batch, qd->out_features), 0);

    /* Quantize input on-the-fly */
    ac_qtensor* input_q = ac_tensor_quantize(input);

    /* INT8 matmul with INT32 accumulation → float32 output */
    ac_qgemm(input_q, qd->weight, output->data,
             batch, qd->out_features, qd->in_features);

    /* Add bias (float32) */
    if (qd->bias) {
        for (ac_size i = 0; i < batch; i++) {
            for (ac_size j = 0; j < qd->out_features; j++) {
                output->data[i * qd->out_features + j] += qd->bias[j];
            }
        }
    }
    return output;
}

/**
 * @brief Model size report for FP32 vs INT8 comparison.
 *
 * Returned by ac_estimate_model_size() to summarise storage
 * requirements before and after quantization.
 */
typedef struct {
    ac_size fp32_bytes;     /**< Original float32 model size in bytes. */
    ac_size int8_bytes;     /**< Quantized INT8 model size in bytes. */
    float   compression;    /**< Compression ratio (fp32 / int8). */
    ac_size num_params;     /**< Total number of parameters. */
} ac_model_size_info;

/**
 * @brief Estimate model size before and after INT8 quantization.
 *
 * Iterates over all parameter tensors in @p params, accumulates
 * their element counts, and computes the FP32 and INT8 storage
 * sizes including per-tensor quantization overhead.
 *
 * @param params  Parameter group containing all model tensors.
 * @return ac_model_size_info with size and compression statistics.
 *
 * @see ac_print_model_size
 */
AC_INLINE ac_model_size_info ac_estimate_model_size(ac_param_group* params) {
    ac_model_size_info info;
    info.num_params = 0;
    for (ac_size i = 0; i < params->num_params; i++) {
        info.num_params += params->params[i]->shape.total_size;
    }
    info.fp32_bytes = info.num_params * sizeof(float);
    /* INT8: 1 byte per param + 4 floats overhead per tensor (scale, zp, min, max) */
    info.int8_bytes = info.num_params * sizeof(uint8_t)
                    + params->num_params * sizeof(ac_quant_params);
    info.compression = (info.int8_bytes > 0)
                     ? (float)info.fp32_bytes / (float)info.int8_bytes
                     : 0.0f;
    return info;
}

/**
 * @brief Print a human-readable model size report to stdout.
 *
 * Displays parameter count, FP32 size, INT8 size, and the
 * compression ratio.
 *
 * @param info  Pointer to a previously computed ac_model_size_info.
 *
 * @see ac_estimate_model_size
 */
AC_INLINE void ac_print_model_size(const ac_model_size_info* info) {
    printf("  Model Size Report:\n");
    printf("  Parameters:    %zu\n", (size_t)info->num_params);
    printf("  FP32 size:     %zu bytes (%.2f KB)\n",
           (size_t)info->fp32_bytes, (float)info->fp32_bytes / 1024.0f);
    printf("  INT8 size:     %zu bytes (%.2f KB)\n",
           (size_t)info->int8_bytes, (float)info->int8_bytes / 1024.0f);
    printf("  Compression:   %.1fx\n", info->compression);
}

/** @} */ /* end of quantize group */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_QUANTIZE_H */
