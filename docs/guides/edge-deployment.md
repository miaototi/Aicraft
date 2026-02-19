---
sidebar_position: 2
title: Edge Deployment
---

# Edge Deployment Guide

Deploy Aicraft models on resource-constrained devices using INT8 quantization and ARM NEON optimization.

## Why Edge Deployment?

| Metric | Float32 | INT8 Quantized |
|---|---|---|
| Model size | Baseline | **~4× smaller** |
| Memory bandwidth | Baseline | **~4× less** |
| Data type | 32-bit float | 8-bit unsigned int |
| Accumulation | FP32 | INT32 (overflow-safe) |

## Quantization Pipeline

### 1. Train in FP32

Train your model normally using standard floating-point precision:

```c
ac_init();

ac_dense fc1, fc2;
ac_dense_init(&fc1, 784, 256);
ac_dense_init(&fc2, 256, 10);

// ... training loop ...
```

### 2. Quantize Layers

Convert trained layers to INT8:

```c
// Create quantized versions of each layer
ac_qdense qfc1, qfc2;
ac_qdense_from_dense(&qfc1, fc1.weight, fc1.bias, 784, 256);
ac_qdense_from_dense(&qfc2, fc2.weight, fc2.bias, 256, 10);
```

This performs per-tensor **asymmetric affine quantization**:

> `q = round((x - min) / scale) + zero_point`

Where:
- `scale = (max - min) / 255`
- `zero_point = round(-min / scale)`

### 3. Run Quantized Inference

```c
ac_tensor* input = ac_tensor_2d(1, 784, 0);
ac_tensor_uniform(input, -1.0f, 1.0f);

// Quantized forward pass
ac_tensor* hidden = ac_qdense_forward(&qfc1, input);

// Apply activation on dequantized output
for (ac_size i = 0; i < hidden->shape.total_size; i++)
    hidden->data[i] = hidden->data[i] > 0 ? hidden->data[i] : 0;  // relu

ac_tensor* output = ac_qdense_forward(&qfc2, hidden);
ac_tensor_print(output, "quantized prediction");
```

### 4. Check Model Size

```c
ac_param_group params;
ac_param_group_init(&params);
ac_param_group_add(&params, fc1.weight);
ac_param_group_add(&params, fc1.bias);
ac_param_group_add(&params, fc2.weight);
ac_param_group_add(&params, fc2.bias);

ac_model_size_info info = ac_estimate_model_size(&params);
ac_print_model_size(&info);
// Output:
//   FP32 model size:   804.00 KB
//   INT8 model size:   201.00 KB
//   Compression ratio: 4.00×

ac_param_group_destroy(&params);
```

## How Quantization Works

### Calibration

```c
ac_quant_params qp;
ac_calibrate(float_data, size, &qp);
// Scans data for min/max range
// Ensures zero is representable (important for ReLU)
```

### Quantize / Dequantize

```c
// Float32 → UINT8
ac_qtensor qt;
ac_quantize(float_tensor, &qt);

// UINT8 → Float32
ac_tensor* recovered = ac_dequantize(&qt);
```

Both operations are **SIMD-accelerated**:
- **AVX2**: 8-wide vectorized pack/unpack (`_mm256_cvtps_epi32` → `_mm_packs_epi32` → `_mm_packus_epi16`)
- **NEON**: 4-wide vectorized with `vcvtnq_s32_f32` → `vmovn` narrowing

### Quantized GEMM

```c
// INT8 × INT8 → INT32 accumulation → Float32 output
ac_qgemm(&qA, &qB, output, M, N, K);
```

Uses **INT32 accumulation** to prevent overflow during matrix multiplication (a 256×256 matmul could overflow INT16).

## ARM NEON Optimizations

On ARM platforms, Aicraft provides real NEON intrinsics for:

| Operation | NEON Implementation |
|---|---|
| Element-wise add/mul/scale | `vaddq_f32`, `vmulq_f32`, `vmulq_n_f32` |
| FMA | `vfmaq_f32` (fused multiply-add) |
| Dot product | 4-wide multiply-accumulate |
| ReLU | `vmaxq_f32(x, zero)` |
| Exp | Cephes polynomial approximation |
| Sigmoid | Exp + Newton-Raphson reciprocal |
| Tanh | `2 × sigmoid(2x) - 1` |
| GEMM | 6×8 micro-kernel with 12 Q-register accumulators |

### GEMM Micro-kernel (NEON 6×8)

```c
// 12 accumulator registers (4 floats each)
float32x4_t c[6][2];  // 6 rows × 2 columns of 4-wide vectors

for (int k = 0; k < K; k++) {
    float32x4_t b0 = vld1q_f32(&B[k * N + j]);
    float32x4_t b1 = vld1q_f32(&B[k * N + j + 4]);
    
    for (int i = 0; i < 6; i++) {
        c[i][0] = vfmaq_n_f32(c[i][0], b0, A[i * K + k]);
        c[i][1] = vfmaq_n_f32(c[i][1], b1, A[i * K + k]);
    }
}
```

## Deployment Targets

### Raspberry Pi 4 (ARM Cortex-A72)

```bash
# Cross-compile for ARM
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
         -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
cmake --build . --config Release
```

### Android (NDK)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_NATIVE_API_LEVEL=21
cmake --build .
```

### Bare Metal / RTOS

Since Aicraft has zero dependencies and uses only standard C, it can be compiled for any target with a C99 compiler. Disable the thread pool for single-core systems:

```c
// In your build:
// Don't compile thread_pool functionality
// Use arena allocator with a fixed-size buffer
```

## Best Practices

:::tip Optimization Checklist
1. **Train in FP32** — Don't quantize during training
2. **Quantize weights only** — Input is quantized on-the-fly per inference
3. **Check accuracy drop** — Compare FP32 vs INT8 output on your validation set
4. **Use arena allocator** — Eliminates `malloc`/`free` overhead on embedded
5. **Profile on target** — Measure actual latency, not just model size
:::
