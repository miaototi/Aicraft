---
sidebar_position: 9
title: Quantization
---

# Quantization API

`#include <aicraft/quantize.h>`

INT8 asymmetric per-tensor affine quantization for ~4× model compression and faster inference on edge devices.

## Structures

### `ac_quant_params`

```c
typedef struct {
    float scale;        // (max - min) / 255
    int zero_point;     // round(-min / scale)
    float min, max;     // Calibrated range
} ac_quant_params;
```

### `ac_qtensor`

```c
typedef struct {
    uint8_t* data;      // Quantized values
    ac_shape shape;
    ac_quant_params qp;
} ac_qtensor;
```

### `ac_qdense`

```c
typedef struct {
    ac_qtensor qweight;    // Quantized weight matrix
    float* bias;           // Bias remains in FP32
    ac_size in_features, out_features;
} ac_qdense;
```

## Calibration

```c
ac_quant_params qp;
ac_calibrate(float_data, num_elements, &qp);
// Scans for min/max, ensures zero is representable
```

## Quantize / Dequantize

```c
// Float32 → UINT8
ac_qtensor qt;
ac_quantize(float_tensor, &qt);

// UINT8 → Float32
ac_tensor* recovered = ac_dequantize(&qt);
```

SIMD-accelerated:
- **AVX2**: 8-wide `_mm256_cvtps_epi32` → `_mm_packs_epi32` → `_mm_packus_epi16`
- **NEON**: 4-wide `vcvtnq_s32_f32` → `vmovn` narrowing

## Quantized GEMM

```c
void ac_qgemm(const ac_qtensor* A, const ac_qtensor* B,
              float* C, ac_size M, ac_size N, ac_size K);
```

- INT8 multiplication with **INT32 accumulation** (prevents overflow)
- NEON: widening multiply-accumulate (`vmull_u8` + `vaddw`)
- Output is dequantized to FP32

## Quantized Dense Layer

```c
// Create from trained FP32 layer
ac_qdense qlayer;
ac_qdense_from_dense(&qlayer, dense.weight, dense.bias,
                     in_features, out_features);

// Run quantized inference
ac_tensor* output = ac_qdense_forward(&qlayer, float_input);
// Input is quantized on-the-fly → QGEMM → dequantize → add bias → float output
```

## Model Size Estimation

```c
ac_model_size_info info = ac_estimate_model_size(&params);
ac_print_model_size(&info);
```

Output:
```
FP32 model size:   804.00 KB (206,346 params)
INT8 model size:   201.00 KB
Compression ratio: 4.00×
```
