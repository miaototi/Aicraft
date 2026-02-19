/**
 * @file autograd.h
 * @brief Automatic Differentiation Engine — reverse-mode autodiff (backpropagation).
 *
 * Implements a reverse-mode automatic differentiation engine with topological
 * sorting for correct gradient propagation. The design targets zero overhead:
 * no virtual dispatch and no per-operation heap allocation.
 *
 * Key optimizations:
 *  - O(1) visited check using an epoch counter (no linear scan)
 *  - SIMD-vectorized sigmoid/tanh backward passes (via fast_math.h)
 *
 * @see tensor.h, tensor_ops.h, fast_math.h, layers.h
 */

#ifndef AICRAFT_AUTOGRAD_H
#define AICRAFT_AUTOGRAD_H

#include "aicraft/tensor.h"
#include "aicraft/tensor_ops.h"
#include "aicraft/fast_math.h"
#include "aicraft/layers.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup autograd Automatic Differentiation
 *  @brief Reverse-mode autodiff with topological sorting.
 *  @{
 */

/**
 * @brief Global autograd epoch counter for O(1) visited checks.
 *
 * Incremented once per backward pass. Each tensor stores the epoch at which
 * it was last visited, so the visited test is a simple integer comparison
 * rather than an O(n) linear scan or hash-set lookup.
 *
 * @note Declared in core.c; must be linked exactly once.
 */
extern int g_autograd_epoch;

/**
 * @brief Initial capacity (number of pointers) for the topological-sort buffer.
 *
 * The buffer is grown dynamically if the computation graph exceeds this size.
 */
#define AC_GRAPH_INITIAL_CAPACITY 4096

/**
 * @brief Recursively visit nodes in DFS order to build a topological sort.
 *
 * Traverses the computation graph starting from @p node, appending each
 * visited node to the @p sorted array.  The array is reallocated when
 * @p count reaches @p capacity.
 *
 * @param node     Current tensor node to visit (may be NULL).
 * @param sorted   Pointer to the dynamically-allocated sorted-node array.
 * @param count    Pointer to the current number of nodes in @p sorted.
 * @param capacity Pointer to the current allocated capacity of @p sorted.
 *
 * @note Uses the epoch-based O(1) visited check: a node is skipped when
 *       its @c visited_epoch equals the global @c g_autograd_epoch.
 * @see  g_autograd_epoch, ac_backward
 */
static void ac_topo_sort_visit(ac_tensor* node,
                               ac_tensor*** sorted,
                               int* count,
                               int* capacity) {
    if (!node) return;

    /* O(1) visited check: compare tensor's epoch to current epoch */
    if (node->visited_epoch == g_autograd_epoch) return;
    node->visited_epoch = g_autograd_epoch;

    /* Visit parents first (DFS) */
    if (node->parents[0]) ac_topo_sort_visit(node->parents[0], sorted, count, capacity);
    if (node->parents[1]) ac_topo_sort_visit(node->parents[1], sorted, count, capacity);

    /* Grow if needed */
    if (*count >= *capacity) {
        int new_cap = *capacity * 2;
        ac_tensor** new_sorted = (ac_tensor**)realloc(*sorted, (ac_size)new_cap * sizeof(ac_tensor*));
        if (!new_sorted) return; /* keep old allocation and capacity on failure */
        *sorted = new_sorted;
        *capacity = new_cap;
    }

    /* Add to sorted list */
    (*sorted)[(*count)++] = node;
}

/**
 * @brief Compute and accumulate gradients for a single computation-graph node.
 *
 * Dispatches on @c node->op to apply the correct backward rule, covering
 * all 22 @c AC_OP_* operation types (arithmetic, activation, loss, layer, etc.).
 * Gradients are accumulated into each parent's @c grad buffer.
 *
 * @param node The tensor whose incoming gradient (@c node->grad) is back-propagated
 *             to its parents.
 *
 * @note Assumes @c node->grad has already been filled by a downstream caller.
 * @see  ac_backward, ac_topo_sort_visit
 */
static void ac_backward_node(ac_tensor* node) {
    if (!node->grad || node->op == AC_OP_NONE) return;

    ac_tensor* p0 = node->parents[0];
    ac_tensor* p1 = node->parents[1];
    ac_size n = node->shape.total_size;

    switch (node->op) {

    case AC_OP_ADD:
        if (p0 && p0->grad) {
            ac_simd_add(p0->grad, node->grad, p0->grad, n);
        }
        if (p1 && p1->grad) {
            ac_simd_add(p1->grad, node->grad, p1->grad, n);
        }
        break;

    case AC_OP_SUB:
        /* d(a-b)/da = 1, d(a-b)/db = -1 */
        if (p0 && p0->grad) {
            ac_simd_add(p0->grad, node->grad, p0->grad, n);
        }
        if (p1 && p1->grad) {
            for (ac_size i = 0; i < n; i++) {
                p1->grad[i] -= node->grad[i];
            }
        }
        break;

    case AC_OP_MUL:
        if (p0 && p0->grad) {
            ac_simd_fma(node->grad, p1->data, p0->grad, p0->grad, n);
        }
        if (p1 && p1->grad) {
            ac_simd_fma(node->grad, p0->data, p1->grad, p1->grad, n);
        }
        break;

    case AC_OP_DIV:
        /* d(a/b)/da = 1/b, d(a/b)/db = -a/b^2 */
        if (p0 && p0->grad) {
            for (ac_size i = 0; i < n; i++) {
                float denom = p1->data[i];
                p0->grad[i] += (denom != 0.0f) ? (node->grad[i] / denom) : 0.0f;
            }
        }
        if (p1 && p1->grad) {
            for (ac_size i = 0; i < n; i++) {
                float denom = p1->data[i] * p1->data[i];
                p1->grad[i] -= (denom != 0.0f) 
                    ? (node->grad[i] * p0->data[i] / denom) : 0.0f;
            }
        }
        break;

    case AC_OP_MATMUL: {
        ac_size M = p0->shape.dims[0];
        ac_size K = p0->shape.dims[1];
        ac_size N = p1->shape.dims[1];
        ac_ensure_arena();
        if (p0 && p0->grad) {
            float* BT = (float*)ac_arena_alloc(&g_tensor_arena, K * N * sizeof(float));
            ac_transpose(p1->data, BT, K, N);
            float* dA_temp = (float*)ac_arena_alloc(&g_tensor_arena, M * K * sizeof(float));
            ac_gemm(node->grad, BT, dA_temp, M, K, N);
            ac_simd_add(p0->grad, dA_temp, p0->grad, M * K);
        }
        if (p1 && p1->grad) {
            float* AT = (float*)ac_arena_alloc(&g_tensor_arena, M * K * sizeof(float));
            ac_transpose(p0->data, AT, M, K);
            float* dB_temp = (float*)ac_arena_alloc(&g_tensor_arena, K * N * sizeof(float));
            ac_gemm(AT, node->grad, dB_temp, K, N, M);
            ac_simd_add(p1->grad, dB_temp, p1->grad, K * N);
        }
        break;
    }

    case AC_OP_RELU:
        if (p0 && p0->grad) {
            ac_simd_relu_backward(p0->data, node->grad, p0->grad, n);
        }
        break;

    case AC_OP_SIGMOID:
        /* SIMD-vectorized: grad_in += grad_out * sig * (1 - sig) */
        if (p0 && p0->grad) {
            ac_fast_sigmoid_backward(node->data, node->grad, p0->grad, n);
        }
        break;

    case AC_OP_TANH:
        /* SIMD-vectorized: grad_in += grad_out * (1 - tanh^2) */
        if (p0 && p0->grad) {
            ac_fast_tanh_backward(node->data, node->grad, p0->grad, n);
        }
        break;

    case AC_OP_SOFTMAX:
        if (p0 && p0->grad) {
            ac_size rows = node->shape.dims[0];
            ac_size cols = node->shape.dims[1];
            for (ac_size r = 0; r < rows; r++) {
                float* grad_in = p0->grad + r * cols;
                const float* grad_out = node->grad + r * cols;
                const float* y = node->data + r * cols;
                float dot = ac_simd_dot(grad_out, y, cols);
                for (ac_size j = 0; j < cols; j++) {
                    grad_in[j] += y[j] * (grad_out[j] - dot);
                }
            }
        }
        break;

    case AC_OP_SCALE:
        if (p0 && p0->grad) {
            float scalar = node->scalar;
            for (ac_size i = 0; i < p0->shape.total_size; i++) {
                p0->grad[i] += node->grad[i] * scalar;
            }
        }
        break;

    case AC_OP_BIAS_ADD:
        if (p0 && p0->grad) {
            ac_simd_add(p0->grad, node->grad, p0->grad, p0->shape.total_size);
        }
        if (p1 && p1->grad) {
            ac_size rows = node->shape.dims[0];
            ac_size cols = node->shape.dims[1];
            for (ac_size i = 0; i < rows; i++) {
                ac_simd_add(p1->grad, node->grad + i * cols, p1->grad, cols);
            }
        }
        break;

    case AC_OP_SUM:
        if (p0 && p0->grad) {
            float g = node->grad[0];
            ac_size sz = p0->shape.total_size;
            ac_size i = 0;
#if defined(AC_SIMD_AVX512)
            __m512 vg512 = _mm512_set1_ps(g);
            for (; i + 16 <= sz; i += 16) {
                __m512 cur = _mm512_loadu_ps(p0->grad + i);
                _mm512_storeu_ps(p0->grad + i, _mm512_add_ps(cur, vg512));
            }
#elif defined(AC_SIMD_AVX2)
            __m256 vg = _mm256_set1_ps(g);
            for (; i + 8 <= sz; i += 8) {
                __m256 cur = _mm256_loadu_ps(p0->grad + i);
                _mm256_storeu_ps(p0->grad + i, _mm256_add_ps(cur, vg));
            }
#endif
            for (; i < sz; i++) p0->grad[i] += g;
        }
        break;

    case AC_OP_MEAN:
        if (p0 && p0->grad) {
            float g = node->grad[0] / (float)p0->shape.total_size;
            ac_size sz = p0->shape.total_size;
            ac_size i = 0;
#if defined(AC_SIMD_AVX512)
            __m512 vg512 = _mm512_set1_ps(g);
            for (; i + 16 <= sz; i += 16) {
                __m512 cur = _mm512_loadu_ps(p0->grad + i);
                _mm512_storeu_ps(p0->grad + i, _mm512_add_ps(cur, vg512));
            }
#elif defined(AC_SIMD_AVX2)
            __m256 vg = _mm256_set1_ps(g);
            for (; i + 8 <= sz; i += 8) {
                __m256 cur = _mm256_loadu_ps(p0->grad + i);
                _mm256_storeu_ps(p0->grad + i, _mm256_add_ps(cur, vg));
            }
#endif
            for (; i < sz; i++) p0->grad[i] += g;
        }
        break;

    case AC_OP_MSE_LOSS:
        if (p0 && p0->grad && p1) {
            ac_size nn = p0->shape.total_size;
            float scale_factor = 2.0f * node->grad[0] / (float)nn;
            ac_size i = 0;
#if defined(AC_SIMD_AVX512)
            __m512 vs512 = _mm512_set1_ps(scale_factor);
            for (; i + 16 <= nn; i += 16) {
                __m512 pred = _mm512_loadu_ps(p0->data + i);
                __m512 targ = _mm512_loadu_ps(p1->data + i);
                __m512 diff = _mm512_sub_ps(pred, targ);
                __m512 cur = _mm512_loadu_ps(p0->grad + i);
                _mm512_storeu_ps(p0->grad + i, _mm512_fmadd_ps(vs512, diff, cur));
            }
#elif defined(AC_SIMD_AVX2)
            __m256 vs = _mm256_set1_ps(scale_factor);
            for (; i + 8 <= nn; i += 8) {
                __m256 pred = _mm256_loadu_ps(p0->data + i);
                __m256 targ = _mm256_loadu_ps(p1->data + i);
                __m256 diff = _mm256_sub_ps(pred, targ);
                __m256 cur = _mm256_loadu_ps(p0->grad + i);
#ifdef __FMA__
                _mm256_storeu_ps(p0->grad + i, _mm256_fmadd_ps(vs, diff, cur));
#else
                _mm256_storeu_ps(p0->grad + i, _mm256_add_ps(cur, _mm256_mul_ps(vs, diff)));
#endif
            }
#endif
            for (; i < nn; i++) {
                p0->grad[i] += scale_factor * (p0->data[i] - p1->data[i]);
            }
        }
        break;

    case AC_OP_CE_LOSS:
        if (p0 && p0->grad && node->aux) {
            /* Accumulate pre-computed CE gradients from aux buffer */
            float* ce_grad = node->aux;
            ac_size nn = p0->shape.total_size;
            for (ac_size i = 0; i < nn; i++) {
                p0->grad[i] += node->grad[0] * ce_grad[i];
            }
        }
        break;

    case AC_OP_BCE_LOSS:
        if (p0 && p0->grad && p1) {
            ac_size nn = p0->shape.total_size;
            float inv_n = node->grad[0] / (float)nn;
            for (ac_size i = 0; i < nn; i++) {
                float p = p0->data[i];
                float t = p1->data[i];
                /* Clamp for numerical stability */
                if (p < 1e-7f) p = 1e-7f;
                if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
                /* d(BCE)/d(p) = (-t/p + (1-t)/(1-p)) / n */
                p0->grad[i] += (-t / p + (1.0f - t) / (1.0f - p)) * inv_n;
            }
        }
        break;

    /* ── Layer backward passes ──────────────────────────────────────────── */

    case AC_OP_FLATTEN:
        /* Flatten is a zero-copy view — if grad buffers are shared, skip.
         * Otherwise accumulate gradient back to parent's shape. */
        if (p0 && p0->grad && p0->grad != node->grad) {
            ac_simd_add(p0->grad, node->grad, p0->grad, p0->shape.total_size);
        }
        break;

    case AC_OP_RESHAPE:
        /* Reshape is a view — if grad buffers are shared, skip.
         * Otherwise accumulate gradient. */
        if (p0 && p0->grad && p0->grad != node->grad) {
            ac_simd_add(p0->grad, node->grad, p0->grad, p0->shape.total_size);
        }
        break;

    case AC_OP_DROPOUT:
        /* grad_input += grad_output * mask (mask stored in aux) */
        if (p0 && p0->grad && node->aux) {
            float* mask = node->aux;
            ac_size nn = p0->shape.total_size;
            ac_size i = 0;
#if defined(AC_SIMD_AVX2)
            for (; i + 8 <= nn; i += 8) {
                __m256 g = _mm256_loadu_ps(node->grad + i);
                __m256 m = _mm256_loadu_ps(mask + i);
                __m256 cur = _mm256_loadu_ps(p0->grad + i);
#ifdef __FMA__
                _mm256_storeu_ps(p0->grad + i, _mm256_fmadd_ps(g, m, cur));
#else
                _mm256_storeu_ps(p0->grad + i, _mm256_add_ps(cur, _mm256_mul_ps(g, m)));
#endif
            }
#endif
            for (; i < nn; i++) {
                p0->grad[i] += node->grad[i] * mask[i];
            }
        }
        break;

    case AC_OP_MAXPOOL:
        /* Route gradient only to the max element (indices stored in aux) */
        if (p0 && p0->grad && node->aux) {
            float* max_indices = node->aux;
            ac_size N_pool = node->shape.dims[0];
            ac_size C_pool = node->shape.dims[1];
            ac_size outH_pool = node->shape.dims[2];
            ac_size outW_pool = node->shape.dims[3];
            ac_size inH = p0->shape.dims[2];
            ac_size inW = p0->shape.dims[3];
            
            for (ac_size np = 0; np < N_pool; np++) {
                for (ac_size cp = 0; cp < C_pool; cp++) {
                    ac_size out_off = np * C_pool * outH_pool * outW_pool + cp * outH_pool * outW_pool;
                    ac_size in_off  = np * C_pool * inH * inW + cp * inH * inW;
                    
                    for (ac_size oh = 0; oh < outH_pool; oh++) {
                        for (ac_size ow = 0; ow < outW_pool; ow++) {
                            ac_size out_idx = out_off + oh * outW_pool + ow;
                            ac_size max_idx = (ac_size)max_indices[out_idx];
                            p0->grad[in_off + max_idx] += node->grad[out_idx];
                        }
                    }
                }
            }
        }
        break;

    case AC_OP_BATCHNORM:
        /* BatchNorm backward: aux = [x_norm, inv_std, mean], p1 = gamma */
        if (p0 && p0->grad && node->aux) {
            ac_size N_bn = node->shape.dims[0];
            ac_size F_bn = (ac_size)node->scalar;
            float* x_norm = node->aux;
            float* inv_stds = node->aux + N_bn * F_bn;
            
            /* Gradient w.r.t. input using simplified batchnorm backward */
            for (ac_size f = 0; f < F_bn; f++) {
                float gamma_f = p1 ? p1->data[f] : 1.0f;
                float inv_std_f = inv_stds[f];
                
                /* Compute intermediate sums */
                float sum_dy = 0.0f, sum_dy_xn = 0.0f;
                for (ac_size nb = 0; nb < N_bn; nb++) {
                    float dy = node->grad[nb * F_bn + f];
                    sum_dy += dy;
                    sum_dy_xn += dy * x_norm[nb * F_bn + f];
                }
                
                /* Gradient w.r.t input */
                float inv_N = 1.0f / (float)N_bn;
                for (ac_size nb = 0; nb < N_bn; nb++) {
                    float dy = node->grad[nb * F_bn + f];
                    float xn = x_norm[nb * F_bn + f];
                    p0->grad[nb * F_bn + f] += gamma_f * inv_std_f * inv_N 
                        * ((float)N_bn * dy - sum_dy - xn * sum_dy_xn);
                }
                
                /* Gradient w.r.t gamma */
                if (p1 && p1->grad) {
                    float dg = 0.0f;
                    for (ac_size nb = 0; nb < N_bn; nb++) {
                        dg += node->grad[nb * F_bn + f] * x_norm[nb * F_bn + f];
                    }
                    p1->grad[f] += dg;
                }
            }
            
            /* Note: beta gradient would need beta stored as parents[2] (not supported).
             * For now beta grad = sum of grad_output per feature, applied via bias_add pattern. */
        }
        break;

    case AC_OP_CONV2D:
        /* Conv2D backward using im2col cached in layer struct pointed by aux */
        if (node->aux && p0) {
            /* aux stores the ac_conv2d layer pointer (set in conv2d_forward) */
            ac_conv2d* lyr = (ac_conv2d*)node->aux;
            
            ac_size N_cv = lyr->last_N;
            ac_size C_cv = lyr->last_C;
            ac_size H_cv = lyr->last_H;
            ac_size W_cv = lyr->last_W;
            ac_size outH_cv = lyr->last_outH;
            ac_size outW_cv = lyr->last_outW;
            ac_size OC = lyr->out_channels;
            ac_size col_rows = lyr->last_col_rows;
            ac_size col_cols = lyr->last_col_cols;
            ac_size kH = lyr->kernel_h;
            ac_size kW = lyr->kernel_w;
            ac_size pad = lyr->padding;
            ac_size stride_cv = lyr->stride;
            
            ac_ensure_arena();
            
            for (ac_size nv = 0; nv < N_cv; nv++) {
                float* grad_out_n = node->grad + nv * OC * outH_cv * outW_cv;
                float* col_buf_n = lyr->last_col_buf + nv * col_rows * col_cols;
                
                /* Gradient w.r.t weight: dW += grad_out @ col_buf^T */
                if (p1 && p1->grad) {
                    /* dW[OC, col_rows] += grad_out_n[OC, col_cols] @ col_buf_n^T[col_cols, col_rows] */
                    for (ac_size oc = 0; oc < OC; oc++) {
                        for (ac_size cr = 0; cr < col_rows; cr++) {
                            float sum = 0.0f;
                            for (ac_size cc = 0; cc < col_cols; cc++) {
                                sum += grad_out_n[oc * col_cols + cc] * col_buf_n[cr * col_cols + cc];
                            }
                            p1->grad[oc * col_rows + cr] += sum;
                        }
                    }
                }
                
                /* Gradient w.r.t input: col2im(weight^T @ grad_out) */
                if (p0->grad) {
                    float* d_col = (float*)ac_arena_alloc(&g_tensor_arena, col_rows * col_cols * sizeof(float));
                    
                    /* d_col[col_rows, col_cols] = weight^T[col_rows, OC] @ grad_out_n[OC, col_cols] */
                    for (ac_size cr = 0; cr < col_rows; cr++) {
                        for (ac_size cc = 0; cc < col_cols; cc++) {
                            float sum = 0.0f;
                            for (ac_size oc = 0; oc < OC; oc++) {
                                sum += p1->data[oc * col_rows + cr] * grad_out_n[oc * col_cols + cc];
                            }
                            d_col[cr * col_cols + cc] = sum;
                        }
                    }
                    
                    /* col2im: scatter d_col back to input grad */
                    for (ac_size c = 0; c < C_cv; c++) {
                        for (ac_size kh = 0; kh < kH; kh++) {
                            for (ac_size kw = 0; kw < kW; kw++) {
                                ac_size row = c * kH * kW + kh * kW + kw;
                                for (ac_size oh = 0; oh < outH_cv; oh++) {
                                    for (ac_size ow = 0; ow < outW_cv; ow++) {
                                        ac_int64 ih = (ac_int64)(oh * stride_cv + kh) - (ac_int64)pad;
                                        ac_int64 iw = (ac_int64)(ow * stride_cv + kw) - (ac_int64)pad;
                                        if (ih >= 0 && ih < (ac_int64)H_cv && iw >= 0 && iw < (ac_int64)W_cv) {
                                            ac_size col = oh * outW_cv + ow;
                                            p0->grad[nv * C_cv * H_cv * W_cv + c * H_cv * W_cv + ih * W_cv + iw]
                                                += d_col[row * col_cols + col];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            /* Gradient w.r.t bias: sum of grad_output over batch and spatial dims */
            if (lyr->bias && lyr->bias->grad) {
                for (ac_size nv = 0; nv < N_cv; nv++) {
                    float* grad_out_n = node->grad + nv * OC * outH_cv * outW_cv;
                    for (ac_size oc = 0; oc < OC; oc++) {
                        float sum_b = 0.0f;
                        for (ac_size s = 0; s < outH_cv * outW_cv; s++) {
                            sum_b += grad_out_n[oc * outH_cv * outW_cv + s];
                        }
                        lyr->bias->grad[oc] += sum_b;
                    }
                }
            }
        }
        break;

    default:
        break;
    }
}

/**
 * @brief Main entry point for reverse-mode automatic differentiation.
 *
 * Seeds the gradient of @p loss to 1.0, increments the global autograd
 * epoch for O(1) visited tracking, performs a topological sort of the
 * computation graph, and then executes the backward pass in reverse
 * topological order.
 *
 * @param loss  Scalar loss tensor from which gradients are propagated.
 *
 * @note The caller must ensure that every tensor requiring gradients has
 *       a pre-allocated @c grad buffer (see @c ac_tensor_require_grad).
 * @see  ac_topo_sort_visit, ac_backward_node, g_autograd_epoch
 */
static AC_INLINE void ac_backward(ac_tensor* loss) {
    /* Seed gradient */
    if (!loss->grad) {
        ac_size aligned = (loss->shape.total_size + AC_SIMD_WIDTH - 1) & ~(AC_SIMD_WIDTH - 1);
        loss->grad = (float*)ac_arena_alloc(&g_tensor_arena, aligned * sizeof(float));
    }
    loss->grad[0] = 1.0f;

    /* Increment epoch for fresh O(1) visited tracking */
    g_autograd_epoch++;

    /* Topological sort with dynamic allocation */
    int capacity = AC_GRAPH_INITIAL_CAPACITY;
    ac_tensor** sorted = (ac_tensor**)malloc((ac_size)capacity * sizeof(ac_tensor*));
    int sort_count = 0;

    ac_topo_sort_visit(loss, &sorted, &sort_count, &capacity);

    /* Backward in reverse topological order */
    for (int i = sort_count - 1; i >= 0; i--) {
        ac_backward_node(sorted[i]);
    }

    free(sorted);
}

/** @} */ /* end of autograd group */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_AUTOGRAD_H */
