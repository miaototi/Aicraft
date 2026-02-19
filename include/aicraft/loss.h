/**
 * @file loss.h
 * @brief Loss functions for neural-network training.
 *
 * Provides Mean Squared Error, Cross-Entropy, and Binary Cross-Entropy
 * losses, each with built-in autograd support for back-propagation.
 */

#ifndef AICRAFT_LOSS_H
#define AICRAFT_LOSS_H

#include "aicraft/tensor.h"
#include "aicraft/tensor_ops.h"
#include "aicraft/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup loss Loss Functions
 *  @{ */

/**
 * @brief Mean Squared Error loss: \f$\frac{1}{n}\sum(pred - target)^2\f$.
 * @param prediction  Predicted tensor.
 * @param target      Ground-truth tensor (same size).
 * @return Scalar (1-element) loss tensor with autograd support.
 * @see ac_backward()
 */
static AC_INLINE ac_tensor* ac_mse_loss(ac_tensor* prediction, ac_tensor* target) {
    AC_CHECK_NULL(prediction && target, AC_ERR_NULL_PTR, "mse_loss: NULL input");
    AC_CHECK_NULL(prediction->shape.total_size == target->shape.total_size,
                  AC_ERR_SHAPE_MISMATCH, "mse_loss: size mismatch %zu vs %zu",
                  prediction->shape.total_size, target->shape.total_size);
    ac_size n = prediction->shape.total_size;
    
    ac_tensor* loss = ac_tensor_create(ac_shape_1d(1), prediction->requires_grad);
    
    float sum = 0.0f;
    for (ac_size i = 0; i < n; i++) {
        float diff = prediction->data[i] - target->data[i];
        sum += diff * diff;
    }
    loss->data[0] = sum / (float)n;
    
    /* Store refs for backward */
    loss->op = AC_OP_MSE_LOSS;
    loss->parents[0] = prediction;
    loss->parents[1] = target;
    
    /* Pre-compute gradient: d(MSE)/d(pred) = 2*(pred - target)/n */
    if (prediction->requires_grad) {
        if (!prediction->grad) {
            ac_size aligned = (n + AC_SIMD_WIDTH - 1) & ~(AC_SIMD_WIDTH - 1);
            prediction->grad = (float*)ac_arena_alloc(&g_tensor_arena, aligned * sizeof(float));
            memset(prediction->grad, 0, aligned * sizeof(float));
        }
    }
    
    return loss;
}

/**
 * @brief Cross-Entropy loss with fused softmax (numerically stable).
 *
 * Expects raw logits @c [N,C] and integer labels @c [N] (stored as float).
 * Computes softmax internally and caches probabilities for the backward pass.
 *
 * @param logits          Raw scores [N, C].
 * @param target_indices  Class labels [N] (as float).
 * @return Scalar loss tensor.
 * @note Gradient is pre-computed and stored in @c aux for efficiency.
 */
static AC_INLINE ac_tensor* ac_cross_entropy_loss(ac_tensor* logits, ac_tensor* target_indices) {
    /* logits: [N, C] raw scores
     * target_indices: [N] integer class labels stored as float */
    AC_CHECK_NULL(logits && target_indices, AC_ERR_NULL_PTR, "ce_loss: NULL input");
    AC_CHECK_NULL(logits->shape.ndim == 2, AC_ERR_INVALID_DIM,
                  "ce_loss: logits must be 2D [N, C], got ndim=%zu", logits->shape.ndim);
    
    ac_size N = logits->shape.dims[0];
    ac_size C = logits->shape.dims[1];
    
    /* Compute softmax + cross-entropy in fused manner (numerically stable) */
    ac_tensor* loss = ac_tensor_create(ac_shape_1d(1), logits->requires_grad);
    
    /* Also allocate softmax probabilities for backward */
    ac_ensure_arena();
    float* probs = (float*)ac_arena_alloc(&g_tensor_arena, N * C * sizeof(float));
    
    float total_loss = 0.0f;
    
    for (ac_size n = 0; n < N; n++) {
        const float* logit_row = logits->data + n * C;
        float* prob_row = probs + n * C;
        ac_size label = (ac_size)target_indices->data[n];
        
        /* Softmax */
        float max_val = logit_row[0];
        for (ac_size c = 1; c < C; c++) {
            if (logit_row[c] > max_val) max_val = logit_row[c];
        }
        
        float sum_exp = 0.0f;
        for (ac_size c = 0; c < C; c++) {
            prob_row[c] = expf(logit_row[c] - max_val);
            sum_exp += prob_row[c];
        }
        
        float inv_sum = 1.0f / sum_exp;
        for (ac_size c = 0; c < C; c++) {
            prob_row[c] *= inv_sum;
        }
        
        /* Cross-entropy: -log(p[label]) */
        total_loss -= logf(prob_row[label] + 1e-10f);
    }
    
    loss->data[0] = total_loss / (float)N;
    loss->op = AC_OP_CE_LOSS;
    loss->parents[0] = logits;
    loss->parents[1] = target_indices;
    
    /* Pre-compute gradient for logits: p - one_hot(target) / N */
    if (logits->requires_grad) {
        if (!logits->grad) {
            ac_size aligned = (N * C + AC_SIMD_WIDTH - 1) & ~(AC_SIMD_WIDTH - 1);
            logits->grad = (float*)ac_arena_alloc(&g_tensor_arena, aligned * sizeof(float));
            memset(logits->grad, 0, aligned * sizeof(float));
        }
        /* NOTE: we do NOT memset here — gradients accumulate.
         * The pre-computed CE gradient is stored in aux for the backward pass. */
        float* ce_grad = (float*)ac_arena_alloc(&g_tensor_arena, N * C * sizeof(float));
        float inv_N = 1.0f / (float)N;
        for (ac_size n = 0; n < N; n++) {
            ac_size label = (ac_size)target_indices->data[n];
            for (ac_size c = 0; c < C; c++) {
                ce_grad[n * C + c] = (probs[n * C + c] - (c == label ? 1.0f : 0.0f)) * inv_N;
            }
        }
        loss->aux = ce_grad;
        loss->aux_size = N * C;
    }
    
    return loss;
}

/**
 * @brief Binary Cross-Entropy loss for sigmoid outputs.
 *
 * Inputs must be in (0, 1); internal clamping prevents log(0).
 *
 * @param prediction  Predicted probabilities.
 * @param target      Binary ground-truth labels (0 or 1).
 * @return Scalar loss tensor with autograd support.
 */
static AC_INLINE ac_tensor* ac_bce_loss(ac_tensor* prediction, ac_tensor* target) {
    AC_CHECK_NULL(prediction && target, AC_ERR_NULL_PTR, "bce_loss: NULL input");
    AC_CHECK_NULL(prediction->shape.total_size == target->shape.total_size,
                  AC_ERR_SHAPE_MISMATCH, "bce_loss: size mismatch %zu vs %zu",
                  prediction->shape.total_size, target->shape.total_size);
    ac_size n = prediction->shape.total_size;
    
    ac_tensor* loss = ac_tensor_create(ac_shape_1d(1), prediction->requires_grad);
    
    float sum = 0.0f;
    for (ac_size i = 0; i < n; i++) {
        float p = prediction->data[i];
        float t = target->data[i];
        /* Clamp for numerical stability */
        if (p < 1e-7f) p = 1e-7f;
        if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
        sum -= t * logf(p) + (1.0f - t) * logf(1.0f - p);
    }
    loss->data[0] = sum / (float)n;
    
    loss->op = AC_OP_BCE_LOSS;
    loss->parents[0] = prediction;
    loss->parents[1] = target;
    
    /* Pre-compute gradient: d(BCE)/d(pred) = (-t/p + (1-t)/(1-p)) / n */
    if (prediction->requires_grad) {
        if (!prediction->grad) {
            ac_size aligned = (n + AC_SIMD_WIDTH - 1) & ~(AC_SIMD_WIDTH - 1);
            prediction->grad = (float*)ac_arena_alloc(&g_tensor_arena, aligned * sizeof(float));
            memset(prediction->grad, 0, aligned * sizeof(float));
        }
    }
    
    return loss;
}

/** @} */ /* end of defgroup loss */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_LOSS_H */
