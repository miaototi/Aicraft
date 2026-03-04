---
sidebar_position: 8
title: SIMD API
---

# SIMD API

`#include "aicraft/simd.h"`

## Overview

Aicraft includes hand-tuned SIMD kernels for all hot paths. The backend is selected at compile time based on compiler flags.

## Backend Selection

| Flag | Backend | Width |
|------|---------|-------|
| (default) | Scalar | 1 |
| `-msse4.2` | SSE 4.2 | 128-bit |
| `-mavx2 -mfma` | AVX2 + FMA | 256-bit |
| `-mavx512f` | AVX-512 | 512-bit |
| `-mfpu=neon` | ARM NEON | 128-bit |

## Key Kernels

### ac_simd_gemm

```c
void ac_simd_gemm(float *C, const float *A, const float *B,
                  int M, int N, int K);
```

General matrix multiply using BLIS-style micro-kernels. This is the hottest path in the entire framework.

### ac_simd_relu / ac_simd_sigmoid

```c
void ac_simd_relu(float *out, const float *in, int n);
void ac_simd_sigmoid(float *out, const float *in, int n);
```

Vectorised activation functions.

### ac_simd_dot

```c
float ac_simd_dot(const float *a, const float *b, int n);
```

Vectorised dot product.

### ac_simd_add / ac_simd_mul

```c
void ac_simd_add(float *out, const float *a, const float *b, int n);
void ac_simd_mul(float *out, const float *a, const float *b, int n);
```

Element-wise vectorised operations.

## Performance

Typical speedups over scalar baseline on an Intel i7-12700K:

| Operation | Scalar | AVX2 | AVX-512 | Speedup |
|-----------|--------|------|---------|---------|
| GEMM 128×128 | 2.1 ms | 0.31 ms | 0.18 ms | 6.8-11.7× |
| ReLU 10K | 12 μs | 1.8 μs | 1.1 μs | 6.7-10.9× |
| Dot 10K | 9.5 μs | 1.4 μs | 0.9 μs | 6.8-10.6× |
