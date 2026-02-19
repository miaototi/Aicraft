---
sidebar_position: 8
title: SIMD Kernels
---

# SIMD Math API

`#include <aicraft/simd_math.h>` and `#include <aicraft/fast_math.h>`

## Architecture Detection

Aicraft automatically selects the best SIMD backend at compile time:

```
AVX-512  →  AVX2  →  SSE4.2  →  ARM NEON  →  Scalar
(best)                                        (fallback)
```

Detection uses compiler-defined macros: `__AVX512F__`, `__AVX2__`, `__SSE4_2__`, `__ARM_NEON`.

## Element-wise Operations

All functions process arrays of `n` floats with automatic SIMD vectorization:

```c
void ac_simd_add(const float* a, const float* b, float* c, ac_size n);    // c = a + b
void ac_simd_mul(const float* a, const float* b, float* c, ac_size n);    // c = a * b
void ac_simd_scale(const float* a, float s, float* c, ac_size n);         // c = a * s
void ac_simd_fma(const float* a, const float* b, float s, float* c, ac_size n); // c = a*b + s
void ac_simd_relu(const float* a, float* c, ac_size n);                   // c = max(0, a)
void ac_simd_relu_backward(const float* a, const float* grad, float* c, ac_size n);
void ac_simd_exp(const float* a, float* c, ac_size n);                    // c = exp(a)
```

## Reductions

```c
float ac_simd_dot(const float* a, const float* b, ac_size n);  // dot product
float ac_simd_sum(const float* a, ac_size n);                   // sum
float ac_simd_max(const float* a, ac_size n);                   // max
```

## GEMM (General Matrix Multiply)

```c
void ac_gemm(const float* A, const float* B, float* C,
             ac_size M, ac_size N, ac_size K);
// C[M×N] = A[M×K] × B[K×N]
```

### BLIS-style Algorithm

```
5-loop blocking structure:
Loop 1: N blocks (NC = 4096)
  Loop 2: K blocks (KC = 256)
    Pack B → B̃ [KC × NC, column-contiguous]
    Loop 3: M blocks (MC = 72)
      Pack A → Ã [MC × KC, row-contiguous]
      Loop 4: NR tiles
        Loop 5: MR tiles → micro-kernel
```

### Micro-kernels

| Arch | Function | Tile | Registers | Peak FMAs |
|---|---|---|---|---|
| AVX-512 | `ac_micro_kernel_6x32` | 6×32 | 12 ZMM | 192/k-step |
| AVX2 | `ac_micro_kernel_6x16` | 6×16 | 12 YMM | 96/k-step |
| NEON | `ac_micro_kernel_6x8_neon` | 6×8 | 12 Q-reg | 48/k-step |
| Scalar | `ac_micro_kernel_scalar` | 6×8 | — | Baseline |

### Threading

For large matrices (M ≥ 2×MC), the M-loop is parallelized across the thread pool:

```c
// Automatic: single-thread for small, multi-thread for large
if (M >= 2 * MC && g_thread_pool_initialized) {
    // Parallel dispatch over M blocks
}
```

## Transpose

```c
void ac_transpose(const float* src, float* dst, ac_size rows, ac_size cols);
// 8×8 block-tiled for cache efficiency
```

## Fast Math (Approximations)

```c
float ac_fast_exp(float x);     // Polynomial approximation (< 0.1% error)
float ac_fast_tanh(float x);    // Based on fast_exp
float ac_fast_sigmoid(float x); // 1 / (1 + fast_exp(-x))
```

### NEON Transcendentals

On ARM platforms, vectorized versions:

```c
float32x4_t ac_neon_exp(float32x4_t x);      // Cephes polynomial
float32x4_t ac_neon_sigmoid(float32x4_t x);   // Exp + Newton-Raphson reciprocal
float32x4_t ac_neon_tanh(float32x4_t x);      // 2*sigmoid(2x) - 1
```
