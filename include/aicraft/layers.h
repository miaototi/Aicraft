/**
 * @file layers.h
 * @brief Neural network layers: Dense, Conv2D, MaxPool2D, BatchNorm, Dropout, Flatten.
 *
 * All layers are plain C structs initialised with `_init()` functions and
 * executed via `_forward()`.  No virtual dispatch — zero vtable overhead.
 */

#ifndef AICRAFT_LAYERS_H
#define AICRAFT_LAYERS_H

#include "aicraft/tensor.h"
#include "aicraft/tensor_ops.h"
#include "aicraft/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup layers Neural Network Layers
 *  @{ */

/** @name Dense (Fully Connected) Layer
 *  @{ */

/**
 * @brief Dense (fully connected) layer.
 *
 * Stores a weight matrix and bias vector for an affine transform
 * `y = x @ W + b`.
 */
typedef struct {
    ac_tensor* weight;       /**< Weight matrix [in_features, out_features]. */
    ac_tensor* bias;         /**< Bias vector   [out_features]. */
    ac_size    in_features;  /**< Number of input  features. */
    ac_size    out_features; /**< Number of output features. */
} ac_dense;

/**
 * @brief Initialise a Dense layer.
 *
 * Allocates weight and bias tensors and applies He initialisation to the
 * weights and zero-fills the bias.
 *
 * @param layer        Pointer to the layer struct to initialise.
 * @param in_features  Number of input  features.
 * @param out_features Number of output features.
 *
 * @see ac_dense_forward
 */
static AC_INLINE void ac_dense_init(ac_dense* layer, ac_size in_features, ac_size out_features) {
    layer->in_features = in_features;
    layer->out_features = out_features;
    
    layer->weight = ac_tensor_2d(in_features, out_features, 1);
    layer->bias = ac_tensor_1d(out_features, 1);
    
    /* He initialization for weights, zeros for bias */
    ac_tensor_he_init(layer->weight, in_features);
    ac_tensor_zeros(layer->bias);
}

/**
 * @brief Forward pass for a Dense layer.
 *
 * Computes `output = input @ weight + bias`.
 *
 * @param layer  Initialised Dense layer.
 * @param input  Input tensor – 2-D `[batch, in_features]`.
 * @return       Output tensor – 2-D `[batch, out_features]`.
 *
 * @see ac_dense_init
 */
static AC_INLINE ac_tensor* ac_dense_forward(ac_dense* layer, ac_tensor* input) {
    /* output = input @ weight + bias */
    ac_tensor* mm = ac_tensor_matmul(input, layer->weight);
    ac_tensor* out = ac_tensor_bias_add(mm, layer->bias);
    return out;
}

/** @} */

/** @name Conv2D Layer
 *  im2col-based convolution.
 *  @{ */

/**
 * @brief 2-D convolutional layer.
 *
 * Uses an im2col strategy followed by GEMM for maximum throughput.
 * Intermediate buffers are cached in the struct for the backward pass.
 */
typedef struct ac_conv2d_s {
    ac_tensor* weight;       /**< Filter weights [out_ch, in_ch, kH, kW]. */
    ac_tensor* bias;         /**< Bias vector    [out_ch]. */
    ac_size    in_channels;  /**< Number of input  channels. */
    ac_size    out_channels; /**< Number of output channels (filters). */
    ac_size    kernel_h;     /**< Kernel height. */
    ac_size    kernel_w;     /**< Kernel width. */
    ac_size    stride;       /**< Convolution stride. */
    ac_size    padding;      /**< Zero-padding added to each spatial side. */
    /* Cached for backward */
    float*     last_col_buf; /**< im2col buffer from the last forward pass. */
    ac_size    last_col_rows;/**< Rows in the im2col matrix. */
    ac_size    last_col_cols;/**< Columns in the im2col matrix. */
    ac_size    last_N;       /**< Cached batch size. */
    ac_size    last_C;       /**< Cached input channels. */
    ac_size    last_H;       /**< Cached input height. */
    ac_size    last_W;       /**< Cached input width. */
    ac_size    last_outH;    /**< Cached output height. */
    ac_size    last_outW;    /**< Cached output width. */
} ac_conv2d;

/**
 * @brief Initialise a Conv2D layer.
 *
 * @param layer       Pointer to the layer struct to initialise.
 * @param in_ch       Number of input  channels.
 * @param out_ch      Number of output channels (filters).
 * @param kernel_size Square kernel side length.
 * @param stride      Convolution stride.
 * @param padding     Zero-padding added to each spatial side.
 *
 * @see ac_conv2d_forward
 */
static AC_INLINE void ac_conv2d_init(ac_conv2d* layer, ac_size in_ch, ac_size out_ch,
                              ac_size kernel_size, ac_size stride, ac_size padding) {
    layer->in_channels = in_ch;
    layer->out_channels = out_ch;
    layer->kernel_h = kernel_size;
    layer->kernel_w = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->last_col_buf = NULL;
    layer->last_col_rows = 0;
    layer->last_col_cols = 0;
    
    layer->weight = ac_tensor_create(ac_shape_4d(out_ch, in_ch, kernel_size, kernel_size), 1);
    layer->bias = ac_tensor_1d(out_ch, 1);
    
    ac_tensor_he_init(layer->weight, in_ch * kernel_size * kernel_size);
    ac_tensor_zeros(layer->bias);
}

/**
 * @brief Forward pass for a Conv2D layer (im2col + GEMM).
 *
 * @param layer  Initialised Conv2D layer.
 * @param input  4-D tensor `[N, C, H, W]`.
 * @return       4-D tensor `[N, out_channels, outH, outW]`.
 *
 * @note The im2col buffer is kept in `layer` for the backward pass.
 * @see  ac_conv2d_init
 */
static AC_INLINE ac_tensor* ac_conv2d_forward(ac_conv2d* layer, ac_tensor* input) {
    AC_CHECK_NULL(input && input->shape.ndim == 4, AC_ERR_INVALID_DIM,
                  "conv2d_forward: requires 4D input [N,C,H,W]"); /* [N, C, H, W] */
    
    ac_size N = input->shape.dims[0];
    ac_size C = input->shape.dims[1];
    ac_size H = input->shape.dims[2];
    ac_size W = input->shape.dims[3];
    ac_size kH = layer->kernel_h;
    ac_size kW = layer->kernel_w;
    ac_size pad = layer->padding;
    ac_size stride = layer->stride;
    
    ac_size outH = (H + 2 * pad - kH) / stride + 1;
    ac_size outW = (W + 2 * pad - kW) / stride + 1;
    ac_size col_rows = C * kH * kW;
    ac_size col_cols = outH * outW;
    
    ac_tensor* output = ac_tensor_create(ac_shape_4d(N, layer->out_channels, outH, outW), 
                                         input->requires_grad || layer->weight->requires_grad);
    
    /* Allocate im2col buffer (persisted for backward) */
    ac_ensure_arena();
    float* col_buf = (float*)ac_arena_alloc(&g_tensor_arena, N * col_rows * col_cols * sizeof(float));
    
    /* Cache for backward */
    layer->last_col_buf = col_buf;
    layer->last_col_rows = col_rows;
    layer->last_col_cols = col_cols;
    layer->last_N = N; layer->last_C = C;
    layer->last_H = H; layer->last_W = W;
    layer->last_outH = outH; layer->last_outW = outW;
    
    /* Reshape weight to [out_ch, C*kH*kW] */
    float* weight_2d = layer->weight->data;
    
    for (ac_size n = 0; n < N; n++) {
        /* im2col: unfold input patches into columns */
        float* sample_col = col_buf + n * col_rows * col_cols;
        memset(sample_col, 0, col_rows * col_cols * sizeof(float));
        
        for (ac_size c = 0; c < C; c++) {
            for (ac_size kh = 0; kh < kH; kh++) {
                for (ac_size kw = 0; kw < kW; kw++) {
                    ac_size row = c * kH * kW + kh * kW + kw;
                    for (ac_size oh = 0; oh < outH; oh++) {
                        for (ac_size ow = 0; ow < outW; ow++) {
                            ac_int64 ih = (ac_int64)(oh * stride + kh) - (ac_int64)pad;
                            ac_int64 iw = (ac_int64)(ow * stride + kw) - (ac_int64)pad;
                            ac_size col = oh * outW + ow;
                            if (ih >= 0 && ih < (ac_int64)H && iw >= 0 && iw < (ac_int64)W) {
                                sample_col[row * col_cols + col] = 
                                    input->data[n * C * H * W + c * H * W + ih * W + iw];
                            }
                        }
                    }
                }
            }
        }
        
        /* GEMM: output[n] = weight @ col_buf */
        float* out_n = output->data + n * layer->out_channels * outH * outW;
        ac_gemm(weight_2d, sample_col, out_n, layer->out_channels, col_cols, col_rows);
        
        /* Add bias */
        for (ac_size oc = 0; oc < layer->out_channels; oc++) {
            float b = layer->bias->data[oc];
            float* out_ch = out_n + oc * outH * outW;
            for (ac_size j = 0; j < outH * outW; j++) {
                out_ch[j] += b;
            }
        }
    }
    
    output->op = AC_OP_CONV2D;
    output->parents[0] = input;
    output->parents[1] = layer->weight;
    output->aux = (float*)((void*)layer);  /* store layer pointer for backward */
    return output;
}

/** @} */

/** @name MaxPool2D
 *  @{ */

/**
 * @brief 2-D max-pooling layer.
 */
typedef struct {
    ac_size pool_size; /**< Square pooling window side length. */
    ac_size stride;    /**< Pooling stride. */
} ac_maxpool2d;

/**
 * @brief Initialise a MaxPool2D layer.
 *
 * @param layer     Pointer to the layer struct to initialise.
 * @param pool_size Square pooling window side length.
 * @param stride    Pooling stride.
 *
 * @see ac_maxpool2d_forward
 */
AC_INLINE void ac_maxpool2d_init(ac_maxpool2d* layer, ac_size pool_size, ac_size stride) {
    layer->pool_size = pool_size;
    layer->stride = stride;
}

/**
 * @brief Forward pass for a MaxPool2D layer.
 *
 * Stores the indices of the maximum values for the backward pass.
 *
 * @param layer  Initialised MaxPool2D layer.
 * @param input  4-D tensor `[N, C, H, W]`.
 * @return       4-D tensor `[N, C, outH, outW]`.
 *
 * @see ac_maxpool2d_init
 */
static AC_INLINE ac_tensor* ac_maxpool2d_forward(ac_maxpool2d* layer, ac_tensor* input) {
    AC_CHECK_NULL(input && input->shape.ndim == 4, AC_ERR_INVALID_DIM,
                  "maxpool2d_forward: requires 4D input [N,C,H,W]");
    
    ac_size N = input->shape.dims[0];
    ac_size C = input->shape.dims[1];
    ac_size H = input->shape.dims[2];
    ac_size W = input->shape.dims[3];
    ac_size ps = layer->pool_size;
    ac_size stride = layer->stride;
    
    ac_size outH = (H - ps) / stride + 1;
    ac_size outW = (W - ps) / stride + 1;
    
    ac_tensor* output = ac_tensor_create(ac_shape_4d(N, C, outH, outW), input->requires_grad);
    
    /* Allocate max indices buffer for backward */
    ac_size out_total = N * C * outH * outW;
    ac_ensure_arena();
    float* max_indices = (float*)ac_arena_alloc(&g_tensor_arena, out_total * sizeof(float));
    
    for (ac_size n = 0; n < N; n++) {
        for (ac_size c = 0; c < C; c++) {
            const float* in_ch = input->data + n * C * H * W + c * H * W;
            float* out_ch = output->data + n * C * outH * outW + c * outH * outW;
            float* idx_ch = max_indices + n * C * outH * outW + c * outH * outW;
            
            for (ac_size oh = 0; oh < outH; oh++) {
                for (ac_size ow = 0; ow < outW; ow++) {
                    float max_val = -1e30f;
                    ac_size max_idx = 0;
                    for (ac_size ph = 0; ph < ps; ph++) {
                        for (ac_size pw = 0; pw < ps; pw++) {
                            ac_size in_idx = (oh * stride + ph) * W + ow * stride + pw;
                            float val = in_ch[in_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = in_idx;
                            }
                        }
                    }
                    out_ch[oh * outW + ow] = max_val;
                    idx_ch[oh * outW + ow] = (float)max_idx;
                }
            }
        }
    }
    
    output->op = AC_OP_MAXPOOL;
    output->parents[0] = input;
    output->aux = max_indices;
    output->aux_size = out_total;
    return output;
}

/** @} */

/** @name Batch Normalization
 *  @{ */

/**
 * @brief Batch-normalisation layer.
 *
 * During training, normalises each feature over the mini-batch and
 * maintains exponential running statistics for inference.
 */
typedef struct {
    ac_tensor* gamma;        /**< Learnable scale  [num_features]. */
    ac_tensor* beta;         /**< Learnable shift  [num_features]. */
    ac_tensor* running_mean; /**< Running mean     [num_features]. */
    ac_tensor* running_var;  /**< Running variance [num_features]. */
    ac_size    num_features; /**< Number of features (channels). */
    float      momentum;     /**< Momentum for running-stat update (default 0.1). */
    float      eps;          /**< Small constant for numerical stability (default 1e-5). */
    int        training;     /**< Non-zero while in training mode. */
} ac_batchnorm;

/**
 * @brief Initialise a Batch Normalization layer.
 *
 * Sets gamma to ones, beta and running_mean to zeros, running_var to ones.
 *
 * @param layer        Pointer to the layer struct to initialise.
 * @param num_features Number of features (channels) to normalise.
 *
 * @see ac_batchnorm_forward
 */
static AC_INLINE void ac_batchnorm_init(ac_batchnorm* layer, ac_size num_features) {
    layer->num_features = num_features;
    layer->momentum = 0.1f;
    layer->eps = 1e-5f;
    layer->training = 1;
    
    layer->gamma = ac_tensor_1d(num_features, 1);
    layer->beta = ac_tensor_1d(num_features, 1);
    layer->running_mean = ac_tensor_1d(num_features, 0);
    layer->running_var = ac_tensor_1d(num_features, 0);
    
    ac_tensor_ones(layer->gamma);
    ac_tensor_zeros(layer->beta);
    ac_tensor_zeros(layer->running_mean);
    ac_tensor_ones(layer->running_var);
}

/**
 * @brief Forward pass for Batch Normalization.
 *
 * In training mode computes batch statistics; during inference uses the
 * stored running statistics.
 *
 * @param layer  Initialised BatchNorm layer.
 * @param input  2-D tensor `[N, features]`.
 * @return       Normalised tensor of the same shape.
 *
 * @see ac_batchnorm_init
 */
static AC_INLINE ac_tensor* ac_batchnorm_forward(ac_batchnorm* layer, ac_tensor* input) {
    AC_CHECK_NULL(input && input->shape.ndim == 2, AC_ERR_INVALID_DIM,
                  "batchnorm_forward: requires 2D input [N, features]");
    ac_size N = input->shape.dims[0];
    ac_size F = input->shape.dims[1];
    AC_CHECK_NULL(F == layer->num_features, AC_ERR_SHAPE_MISMATCH,
                  "batchnorm: input features %zu != expected %zu", F, layer->num_features);
    
    ac_tensor* output = ac_tensor_create(input->shape, input->requires_grad);
    
    /* Allocate aux buffer: x_norm [N*F] + inv_std [F] + mean [F] */
    ac_ensure_arena();
    float* aux_buf = (float*)ac_arena_alloc(&g_tensor_arena, (N * F + F + F) * sizeof(float));
    float* x_norm = aux_buf;
    float* inv_stds = aux_buf + N * F;
    float* means = inv_stds + F;
    
    for (ac_size f = 0; f < F; f++) {
        float mean = 0, var = 0;
        
        if (layer->training) {
            /* Compute batch mean */
            for (ac_size n = 0; n < N; n++) mean += input->data[n * F + f];
            mean /= (float)N;
            
            /* Compute batch variance */
            for (ac_size n = 0; n < N; n++) {
                float diff = input->data[n * F + f] - mean;
                var += diff * diff;
            }
            var /= (float)N;
            
            /* Update running stats */
            layer->running_mean->data[f] = (1.0f - layer->momentum) * layer->running_mean->data[f] 
                                            + layer->momentum * mean;
            layer->running_var->data[f] = (1.0f - layer->momentum) * layer->running_var->data[f] 
                                           + layer->momentum * var;
        } else {
            mean = layer->running_mean->data[f];
            var = layer->running_var->data[f];
        }
        
        float inv_std = 1.0f / sqrtf(var + layer->eps);
        float gamma = layer->gamma->data[f];
        float beta = layer->beta->data[f];
        
        inv_stds[f] = inv_std;
        means[f] = mean;
        
        for (ac_size n = 0; n < N; n++) {
            float xn = (input->data[n * F + f] - mean) * inv_std;
            x_norm[n * F + f] = xn;
            output->data[n * F + f] = gamma * xn + beta;
        }
    }
    
    output->op = AC_OP_BATCHNORM;
    output->parents[0] = input;
    output->parents[1] = layer->gamma;
    output->aux = aux_buf;
    output->aux_size = N * F + F + F;
    output->scalar = (float)F; /* store num_features for backward */
    return output;
}

/** @} */

/** @name Dropout
 *  @{ */

/**
 * @brief Dropout regularisation layer.
 *
 * During training, randomly zeroes elements with probability `rate` and
 * scales the remaining elements by `1 / (1 - rate)` (inverted dropout).
 */
typedef struct {
    float rate;     /**< Drop probability in [0, 1). */
    int   training; /**< Non-zero while in training mode. */
} ac_dropout;

/**
 * @brief Initialise a Dropout layer.
 *
 * @param layer  Pointer to the layer struct to initialise.
 * @param rate   Drop probability in [0, 1).
 *
 * @see ac_dropout_forward
 */
AC_INLINE void ac_dropout_init(ac_dropout* layer, float rate) {
    layer->rate = rate;
    layer->training = 1;
}

/**
 * @brief Forward pass for Dropout.
 *
 * Returns the input unchanged during inference.  In training mode each
 * element is zeroed with probability `rate` and the rest are scaled by
 * `1 / (1 - rate)`.
 *
 * @param layer  Initialised Dropout layer.
 * @param input  Tensor of any shape.
 * @return       Tensor of the same shape (may alias @p input at inference).
 *
 * @see ac_dropout_init
 */
static AC_INLINE ac_tensor* ac_dropout_forward(ac_dropout* layer, ac_tensor* input) {
    if (!layer->training || layer->rate <= 0.0f) {
        return input; /* No-op during inference */
    }
    
    ac_tensor* output = ac_tensor_create(input->shape, input->requires_grad);
    float scale = 1.0f / (1.0f - layer->rate);
    
    /* Allocate mask for backward */
    ac_ensure_arena();
    ac_size n = input->shape.total_size;
    float* mask = (float*)ac_arena_alloc(&g_tensor_arena, n * sizeof(float));
    
    for (ac_size i = 0; i < n; i++) {
        if (ac_randf() >= layer->rate) {
            mask[i] = scale;
            output->data[i] = input->data[i] * scale;
        } else {
            mask[i] = 0.0f;
            output->data[i] = 0.0f;
        }
    }
    
    output->op = AC_OP_DROPOUT;
    output->parents[0] = input;
    output->aux = mask;       /* store mask for backward */
    output->aux_size = n;
    return output;
}

/** @} */

/** @name Flatten
 *  @{ */

/**
 * @brief Flatten all dimensions except the batch dimension (zero-copy view).
 *
 * Reshapes `[N, ...]` into `[N, flat]` where `flat = total_size / N`.
 * The returned tensor shares its data with the input (no allocation).
 *
 * @param input  Tensor with at least 2 dimensions.
 * @return       2-D view `[N, flat]` sharing data with @p input.
 *
 * @note The returned tensor does **not** own its data; do not free it
 *       independently of @p input.
 */
static AC_INLINE ac_tensor* ac_flatten(ac_tensor* input) {
    /* Flatten all dims except batch dim into single dim (zero-copy view) */
    ac_size batch = input->shape.dims[0];
    ac_size flat = input->shape.total_size / batch;
    
    ac_tensor* out = ac_tensor_alloc();
    out->shape = ac_shape_2d(batch, flat);
    out->data = input->data;  /* Share data — zero copy */
    out->requires_grad = input->requires_grad;
    if (input->requires_grad && input->grad) {
        out->grad = input->grad;  /* Share grad buffer too */
    }
    out->owns_data = 0;  /* View — do not free */
    out->op = AC_OP_FLATTEN;
    out->parents[0] = input;
    /* Original shape is accessible via parents[0]->shape for backward */
    return out;
}

/** @} */
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_LAYERS_H */
