/**
 * @file tensor_ops.h
 * @brief SIMD-accelerated tensor operations with autograd support.
 *
 * Every operation records its parents and operation tag so that
 * ac_backward() can compute gradients automatically.
 */

#ifndef AICRAFT_TENSOR_OPS_H
#define AICRAFT_TENSOR_OPS_H

#include "aicraft/tensor.h"
#include "aicraft/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup tensorops Tensor Operations */
/** @{ */

/** @name Element-wise Arithmetic */
/** @{ */

/**
 * @brief Element-wise addition of two tensors (SIMD-accelerated).
 *
 * Computes @p out[i] = a[i] + b[i] using SIMD intrinsics.
 * Records parents for autograd.
 *
 * @param a  Left operand tensor.
 * @param b  Right operand tensor (must match @p a in size).
 * @return   New tensor holding the element-wise sum.
 * @see      ac_simd_add, ac_backward
 * @note     Both tensors must have identical total_size.
 */
static AC_INLINE ac_tensor* ac_tensor_add(ac_tensor* a, ac_tensor* b) {
    AC_CHECK_NULL(a && b, AC_ERR_NULL_PTR, "tensor_add: NULL input");
    AC_CHECK_NULL(a->shape.total_size == b->shape.total_size, AC_ERR_SHAPE_MISMATCH,
                  "tensor_add: size mismatch %zu vs %zu", a->shape.total_size, b->shape.total_size);
    int req_grad = a->requires_grad || b->requires_grad;
    ac_tensor* out = ac_tensor_create(a->shape, req_grad);
    ac_simd_add(a->data, b->data, out->data, a->shape.total_size);
    out->op = AC_OP_ADD;
    out->parents[0] = a;
    out->parents[1] = b;
    return out;
}

/**
 * @brief Element-wise subtraction of two tensors.
 *
 * Computes @p out[i] = a[i] - b[i].
 * Records parents for autograd.
 *
 * @param a  Left operand tensor.
 * @param b  Right operand tensor (must match @p a in size).
 * @return   New tensor holding the element-wise difference.
 * @see      ac_tensor_add
 * @note     Both tensors must have identical total_size.
 */
static AC_INLINE ac_tensor* ac_tensor_sub(ac_tensor* a, ac_tensor* b) {
    AC_CHECK_NULL(a && b, AC_ERR_NULL_PTR, "tensor_sub: NULL input");
    AC_CHECK_NULL(a->shape.total_size == b->shape.total_size, AC_ERR_SHAPE_MISMATCH,
                  "tensor_sub: size mismatch %zu vs %zu", a->shape.total_size, b->shape.total_size);
    int req_grad = a->requires_grad || b->requires_grad;
    ac_tensor* out = ac_tensor_create(a->shape, req_grad);
    for (ac_size i = 0; i < a->shape.total_size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    out->op = AC_OP_SUB;
    out->parents[0] = a;
    out->parents[1] = b;
    return out;
}

/**
 * @brief Element-wise multiplication of two tensors (SIMD-accelerated).
 *
 * Computes @p out[i] = a[i] * b[i] using SIMD intrinsics.
 * Records parents for autograd.
 *
 * @param a  Left operand tensor.
 * @param b  Right operand tensor (must match @p a in size).
 * @return   New tensor holding the element-wise product.
 * @see      ac_simd_mul, ac_backward
 * @note     Both tensors must have identical total_size.
 */
static AC_INLINE ac_tensor* ac_tensor_mul(ac_tensor* a, ac_tensor* b) {
    AC_CHECK_NULL(a && b, AC_ERR_NULL_PTR, "tensor_mul: NULL input");
    AC_CHECK_NULL(a->shape.total_size == b->shape.total_size, AC_ERR_SHAPE_MISMATCH,
                  "tensor_mul: size mismatch %zu vs %zu", a->shape.total_size, b->shape.total_size);
    int req_grad = a->requires_grad || b->requires_grad;
    ac_tensor* out = ac_tensor_create(a->shape, req_grad);
    ac_simd_mul(a->data, b->data, out->data, a->shape.total_size);
    out->op = AC_OP_MUL;
    out->parents[0] = a;
    out->parents[1] = b;
    return out;
}

/**
 * @brief Element-wise division of two tensors (safe, checks for zero).
 *
 * Computes @p out[i] = a[i] / b[i].  If @p b[i] == 0 the result
 * is clamped to 0.0f to avoid undefined behaviour.
 *
 * @param a  Numerator tensor.
 * @param b  Denominator tensor (must match @p a in size).
 * @return   New tensor holding the element-wise quotient.
 * @see      ac_tensor_mul
 * @note     Division by zero produces 0.0f, not NaN/Inf.
 */
static AC_INLINE ac_tensor* ac_tensor_div(ac_tensor* a, ac_tensor* b) {
    AC_CHECK_NULL(a && b, AC_ERR_NULL_PTR, "tensor_div: NULL input");
    AC_CHECK_NULL(a->shape.total_size == b->shape.total_size, AC_ERR_SHAPE_MISMATCH,
                  "tensor_div: size mismatch %zu vs %zu", a->shape.total_size, b->shape.total_size);
    int req_grad = a->requires_grad || b->requires_grad;
    ac_tensor* out = ac_tensor_create(a->shape, req_grad);
    for (ac_size i = 0; i < a->shape.total_size; i++) {
        float denom = b->data[i];
        out->data[i] = (denom != 0.0f) ? (a->data[i] / denom) : 0.0f;
    }
    out->op = AC_OP_DIV;
    out->parents[0] = a;
    out->parents[1] = b;
    return out;
}

/** @} */  /* Element-wise Arithmetic */

/** @name Reshape / Scale */
/** @{ */

/**
 * @brief Reshape a tensor (zero-copy view, shares data pointer).
 *
 * Returns a new tensor header whose @c data pointer aliases the
 * original.  The view does @b not own the memory, so freeing it
 * will not release the underlying buffer.
 *
 * @param a          Source tensor.
 * @param new_shape  Desired shape (total_size must match).
 * @return           A view tensor with the new shape.
 * @see              ac_tensor_alloc
 * @note             The returned tensor has @c owns_data == 0.
 */
static AC_INLINE ac_tensor* ac_tensor_reshape(ac_tensor* a, ac_shape new_shape) {
    AC_CHECK_NULL(a != NULL, AC_ERR_NULL_PTR, "tensor_reshape: NULL input");
    AC_CHECK_NULL(a->shape.total_size == new_shape.total_size, AC_ERR_SHAPE_MISMATCH,
                  "tensor_reshape: size mismatch %zu vs %zu", a->shape.total_size, new_shape.total_size);
    ac_tensor* out = ac_tensor_alloc();
    out->shape = new_shape;
    out->data = a->data;           /* share data — true view */
    out->requires_grad = a->requires_grad;
    if (a->requires_grad && a->grad) {
        out->grad = a->grad;       /* share grad buffer too */
    }
    out->owns_data = 0;            /* view does NOT own the data */
    out->op = AC_OP_RESHAPE;
    out->parents[0] = a;
    return out;
}

/**
 * @brief Multiply every element by a scalar (SIMD-accelerated).
 *
 * Computes @p out[i] = a[i] * scalar.  The scalar value is stored
 * in @c out->scalar for use during the backward pass.
 *
 * @param a       Input tensor.
 * @param scalar  The scalar multiplier.
 * @return        New tensor with each element scaled.
 * @see           ac_simd_scale, ac_backward
 */
static AC_INLINE ac_tensor* ac_tensor_scale(ac_tensor* a, float scalar) {
    ac_tensor* out = ac_tensor_create(a->shape, a->requires_grad);
    ac_simd_scale(a->data, scalar, out->data, a->shape.total_size);
    out->op = AC_OP_SCALE;
    out->scalar = scalar; /* store for backward */
    out->parents[0] = a;
    return out;
}

/** @} */  /* Reshape / Scale */

/** @name Matrix Operations */
/** @{ */

/**
 * @brief General matrix multiplication (GEMM): C[M×N] = A[M×K] × B[K×N].
 *
 * Both operands must be 2-D tensors whose inner dimensions agree.
 * The heavy lifting is delegated to ac_gemm().
 *
 * @param a  Left matrix (M × K).
 * @param b  Right matrix (K × N).
 * @return   New (M × N) tensor holding the product.
 * @see      ac_gemm, ac_shape_2d
 * @note     Only 2-D tensors are accepted; higher ranks are rejected.
 */
static AC_INLINE ac_tensor* ac_tensor_matmul(ac_tensor* a, ac_tensor* b) {
    AC_CHECK_NULL(a && b, AC_ERR_NULL_PTR, "tensor_matmul: NULL input");
    AC_CHECK_NULL(a->shape.ndim == 2 && b->shape.ndim == 2, AC_ERR_INVALID_DIM,
                  "tensor_matmul: requires 2D tensors, got %zuD and %zuD", a->shape.ndim, b->shape.ndim);
    AC_CHECK_NULL(a->shape.dims[1] == b->shape.dims[0], AC_ERR_SHAPE_MISMATCH,
                  "tensor_matmul: inner dims mismatch %zu vs %zu", a->shape.dims[1], b->shape.dims[0]);
    
    ac_size M = a->shape.dims[0];
    ac_size K = a->shape.dims[1];
    ac_size N = b->shape.dims[1];
    
    int req_grad = a->requires_grad || b->requires_grad;
    ac_tensor* out = ac_tensor_create(ac_shape_2d(M, N), req_grad);
    
    ac_gemm(a->data, b->data, out->data, M, N, K);
    
    out->op = AC_OP_MATMUL;
    out->parents[0] = a;
    out->parents[1] = b;
    return out;
}

/** @} */  /* Matrix Operations */

/** @name Activations */
/** @{ */

/**
 * @brief Rectified Linear Unit activation (SIMD-accelerated).
 *
 * Computes @p out[i] = max(0, a[i]).
 *
 * @param a  Input tensor.
 * @return   New tensor with ReLU applied element-wise.
 * @see      ac_simd_relu, ac_backward
 */
static AC_INLINE ac_tensor* ac_tensor_relu(ac_tensor* a) {
    ac_tensor* out = ac_tensor_create(a->shape, a->requires_grad);
    ac_simd_relu(a->data, out->data, a->shape.total_size);
    out->op = AC_OP_RELU;
    out->parents[0] = a;
    return out;
}

/**
 * @brief Sigmoid activation (SIMD-vectorized via fast_math.h).
 *
 * Computes @p out[i] = 1 / (1 + exp(-a[i])) using a fast
 * polynomial approximation.
 *
 * @param a  Input tensor.
 * @return   New tensor with sigmoid applied element-wise.
 * @see      ac_fast_sigmoid
 * @simd     Uses vectorized exp approximation from fast_math.h.
 */
static AC_INLINE ac_tensor* ac_tensor_sigmoid(ac_tensor* a) {
    ac_tensor* out = ac_tensor_create(a->shape, a->requires_grad);
    ac_fast_sigmoid(a->data, out->data, a->shape.total_size);
    out->op = AC_OP_SIGMOID;
    out->parents[0] = a;
    return out;
}

/**
 * @brief Hyperbolic tangent activation (SIMD-vectorized via fast_math.h).
 *
 * Computes @p out[i] = tanh(a[i]) using a fast polynomial
 * approximation.
 *
 * @param a  Input tensor.
 * @return   New tensor with tanh applied element-wise.
 * @see      ac_fast_tanh
 * @simd     Uses vectorized tanh approximation from fast_math.h.
 */
static AC_INLINE ac_tensor* ac_tensor_tanh(ac_tensor* a) {
    ac_tensor* out = ac_tensor_create(a->shape, a->requires_grad);
    ac_fast_tanh(a->data, out->data, a->shape.total_size);
    out->op = AC_OP_TANH;
    out->parents[0] = a;
    return out;
}

/**
 * @brief Row-wise softmax for 2-D tensors (SIMD-accelerated).
 *
 * For each row: shift by max for numerical stability, compute
 * exp via Cephes polynomial, then normalise.
 *
 * @param a  Input 2-D tensor (rows × cols).
 * @return   New tensor with softmax applied per row.
 * @see      ac_fast_exp, ac_simd_max, ac_simd_sum
 * @note     Requires a 2-D tensor; asserts on other ranks.
 * @simd     Max, exp, sum, and scale steps are all SIMD-vectorized.
 */
static AC_INLINE ac_tensor* ac_tensor_softmax(ac_tensor* a) {
    AC_CHECK_NULL(a != NULL, AC_ERR_NULL_PTR, "tensor_softmax: NULL input");
    AC_CHECK_NULL(a->shape.ndim == 2, AC_ERR_INVALID_DIM,
                  "tensor_softmax: requires 2D tensor, got %zuD", a->shape.ndim);
    ac_size rows = a->shape.dims[0];
    ac_size cols = a->shape.dims[1];
    
    ac_tensor* out = ac_tensor_create(a->shape, a->requires_grad);
    
    for (ac_size i = 0; i < rows; i++) {
        const float* row_in = a->data + i * cols;
        float* row_out = out->data + i * cols;
        
        /* SIMD max for numerical stability */
        float max_val = ac_simd_max(row_in, cols);
        
        /* Shift input: tmp = x - max (SIMD) */
        ac_simd_scale(row_in, 1.0f, row_out, cols); /* copy */
        for (ac_size j = 0; j < cols; j++) row_out[j] -= max_val;
        
        /* Vectorized exp (Cephes polynomial from fast_math.h) */
        ac_fast_exp(row_out, row_out, cols);
        
        /* SIMD sum + normalize (guard against zero sum) */
        float sum = ac_simd_sum(row_out, cols);
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        ac_simd_scale(row_out, inv_sum, row_out, cols);
    }
    
    out->op = AC_OP_SOFTMAX;
    out->parents[0] = a;
    return out;
}

/** @} */  /* Activations */

/** @name Reductions */
/** @{ */

/**
 * @brief Sum all elements into a scalar tensor (SIMD-accelerated).
 *
 * Returns a 1-element tensor whose value is the sum of every
 * element in @p a.
 *
 * @param a  Input tensor of any shape.
 * @return   Scalar (1-D, size 1) tensor with the sum.
 * @see      ac_simd_sum
 */
static AC_INLINE ac_tensor* ac_tensor_sum(ac_tensor* a) {
    ac_tensor* out = ac_tensor_create(ac_shape_1d(1), a->requires_grad);
    out->data[0] = ac_simd_sum(a->data, a->shape.total_size);
    out->op = AC_OP_SUM;
    out->parents[0] = a;
    return out;
}

/**
 * @brief Mean of all elements (SIMD sum / count).
 *
 * Returns a 1-element tensor whose value is the arithmetic mean
 * of every element in @p a.
 *
 * @param a  Input tensor of any shape.
 * @return   Scalar (1-D, size 1) tensor with the mean.
 * @see      ac_tensor_sum
 */
static AC_INLINE ac_tensor* ac_tensor_mean(ac_tensor* a) {
    ac_tensor* out = ac_tensor_create(ac_shape_1d(1), a->requires_grad);
    out->data[0] = ac_simd_sum(a->data, a->shape.total_size) / (float)a->shape.total_size;
    out->op = AC_OP_MEAN;
    out->parents[0] = a;
    return out;
}

/** @} */  /* Reductions */

/** @name Broadcasting */
/** @{ */

/**
 * @brief Broadcast-add a 1-D bias to every row of a 2-D tensor.
 *
 * @p bias must have the same number of elements as columns in @p a.
 * Each row of @p a is added element-wise to @p bias using SIMD.
 *
 * @param a     Input 2-D tensor (rows × cols).
 * @param bias  1-D bias vector (length == cols).
 * @return      New 2-D tensor with the bias added to each row.
 * @see         ac_simd_add
 * @note        @p a must be 2-D and @p bias must be 1-D.
 */
static AC_INLINE ac_tensor* ac_tensor_bias_add(ac_tensor* a, ac_tensor* bias) {
    AC_CHECK_NULL(a && bias, AC_ERR_NULL_PTR, "tensor_bias_add: NULL input");
    AC_CHECK_NULL(a->shape.ndim == 2, AC_ERR_INVALID_DIM, "bias_add: requires 2D tensor");
    AC_CHECK_NULL(bias->shape.ndim == 1, AC_ERR_INVALID_DIM, "bias_add: bias must be 1D");
    AC_CHECK_NULL(a->shape.dims[1] == bias->shape.dims[0], AC_ERR_SHAPE_MISMATCH,
                  "bias_add: cols %zu != bias %zu", a->shape.dims[1], bias->shape.dims[0]);
    
    int req_grad = a->requires_grad || bias->requires_grad;
    ac_tensor* out = ac_tensor_create(a->shape, req_grad);
    ac_size rows = a->shape.dims[0];
    ac_size cols = a->shape.dims[1];
    
    for (ac_size i = 0; i < rows; i++) {
        ac_simd_add(a->data + i * cols, bias->data, out->data + i * cols, cols);
    }
    
    out->op = AC_OP_BIAS_ADD;
    out->parents[0] = a;
    out->parents[1] = bias;
    return out;
}

/** @} */  /* Broadcasting */
/** @} */  /* tensorops */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_TENSOR_OPS_H */
