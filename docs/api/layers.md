---
sidebar_position: 4
title: Layers
---

# Layers API

`#include <aicraft/layers.h>`

## Dense (Fully Connected)

```c
typedef struct {
    ac_tensor* weight;  // [out_features, in_features]
    ac_tensor* bias;    // [out_features]
    ac_size in_features;
    ac_size out_features;
} ac_dense;
```

```c
ac_dense_init(&layer, in_features, out_features);  // He initialization
ac_tensor* out = ac_dense_forward(&layer, input);   // out = input @ W^T + b
```

**Backward**: Autograd computes `dW`, `dB`, `dInput` automatically via MATMUL + BIAS_ADD ops.

## Conv2D

```c
typedef struct {
    ac_tensor* weight;     // [out_ch, in_ch, kH, kW]
    ac_tensor* bias;       // [out_ch]
    int in_channels, out_channels;
    int kernel_size, stride, padding;
    // Cached for backward:
    float* im2col_buf;
    int batch, in_h, in_w, out_h, out_w;
} ac_conv2d;
```

```c
ac_conv2d_init(&conv, in_channels, out_channels, kernel_size, stride, padding);
ac_tensor* out = ac_conv2d_forward(&conv, input);
// input: [batch, in_ch, H, W]
// output: [batch, out_ch, outH, outW]
```

**Implementation**: Uses **im2col + GEMM** approach (same as Caffe/cuDNN):
1. Unfold input patches into columns
2. GEMM: filters × columns → output
3. Cache im2col buffer for backward

## MaxPool2D

```c
typedef struct {
    int pool_size, stride;
    int* indices;  // Argmax indices for backward
} ac_maxpool2d;
```

```c
ac_maxpool2d_init(&pool, pool_size, stride);
ac_tensor* out = ac_maxpool2d_forward(&pool, input);
```

**Backward**: Gradient routed only to max elements via stored indices.

## BatchNorm

```c
typedef struct {
    ac_tensor* gamma;        // Scale parameter [channels]
    ac_tensor* beta;         // Shift parameter [channels]
    float* running_mean;     // Running mean (inference)
    float* running_var;      // Running variance (inference)
    int num_features;
    float momentum, epsilon;
} ac_batchnorm;
```

```c
ac_batchnorm_init(&bn, num_features);
ac_tensor* out = ac_batchnorm_forward(&bn, input, is_training);
// is_training=1: use batch stats, update running stats
// is_training=0: use running stats
```

## Dropout

```c
typedef struct {
    float p;         // Drop probability
    uint8_t* mask;   // Binary mask for backward
} ac_dropout;
```

```c
ac_dropout_init(&drop, 0.5f);  // 50% dropout
ac_tensor* out = ac_dropout_forward(&drop, input, is_training);
// Uses inverted dropout: scale by 1/(1-p) during training
// No-op during inference
```

## Flatten

```c
ac_tensor* out = ac_flatten_forward(input);
// Zero-copy view: shares data pointer, no allocation
// Reshapes [batch, c, h, w] → [batch, c*h*w]
```

## Layer Summary

| Layer | Forward | Backward | Init |
|---|---|---|---|
| Dense | GEMM + bias | dW, dB, dX via autograd | He |
| Conv2D | im2col + GEMM | Full im2col-based grads | He |
| MaxPool2D | Max selection + index storage | Gradient routing | — |
| BatchNorm | Normalize + scale/shift | Stabilized formula | γ=1, β=0 |
| Dropout | Random mask × scale | Mask-gated gradient | — |
| Flatten | Zero-copy reshape | Shape restore | — |
