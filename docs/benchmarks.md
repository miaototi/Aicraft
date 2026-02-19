---
sidebar_position: 11
title: Benchmarks
---

# Benchmarks

Performance measurements of Aicraft's core operations. Run benchmarks with:

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./aicraft_bench
```

## Benchmark Suite

### 1. GEMM Throughput

Matrix multiplication (512×512 × 512×512):

| Backend | Target | Metric |
|---|---|---|
| AVX2 (BLIS-style) | Desktop x86_64 | GFLOPS |
| AVX-512 | Server x86_64 | GFLOPS |
| ARM NEON | Raspberry Pi / Mobile | GFLOPS |
| Vulkan GPU | Discrete/Integrated GPU | GFLOPS |

The BLIS-style GEMM uses:
- Panel packing for cache-aligned access
- Architecture-specific micro-kernels (6×32, 6×16, 6×8)
- Thread-parallel outer loop for large matrices
- L1 prefetch hints (AVX2)

### 2. MLP Forward Pass

4-layer network: 784 → 256 → 256 → 128 → 10, batch=32

Measures end-to-end inference latency including:
- Dense GEMM × 4
- ReLU activations × 3
- Softmax output

### 3. Full Training Step

Complete training iteration including:
- Forward pass (GEMM + activations)
- Loss computation (cross-entropy)
- Backward pass (autograd, 22 op types)
- Gradient clipping (L2-norm)
- Adam optimizer step (SIMD-vectorized)

### 4. Element-wise Operations

1M elements: add + mul + relu + scale

| Architecture | Throughput |
|---|---|
| AVX2 | M elements/sec |
| NEON | M elements/sec |
| Scalar | M elements/sec |

### 5. Memory Allocation

1000× 4KB block allocations:

| Allocator | Time | Speedup |
|---|---|---|
| `malloc` | Baseline | 1× |
| Arena (`ac_arena_alloc`) | Faster | **N×** |

The arena allocator uses bump-pointer allocation — O(1) with no free-list traversal or system calls after the initial block allocation.

### 6. Dot Product

1M element dot product:

| Architecture | GFLOPS |
|---|---|
| AVX2 | Measured |
| NEON | Measured |
| Scalar | Measured |

## Running Your Own Benchmarks

The benchmark source is in [benchmarks/bench_main.cpp](https://github.com/your-github-username/Aicraft/blob/main/benchmarks/bench_main.cpp). Each benchmark:

1. Runs a warm-up iteration
2. Times N iterations with `ac_timer`
3. Reports throughput in appropriate units (GFLOPS, M elements/sec, μs)

```c
ac_timer timer;
ac_timer_start(&timer);

for (int i = 0; i < iterations; i++) {
    ac_gemm(A, B, C, M, N, K);
}

double elapsed = ac_timer_stop(&timer);
double gflops = (2.0 * M * N * K * iterations) / (elapsed * 1e9);
```

## Comparison Targets

For a fair comparison, benchmark against:

| Library | Command |
|---|---|
| OpenBLAS | `OPENBLAS_NUM_THREADS=1 ./bench_openblas` |
| Intel MKL | `MKL_NUM_THREADS=1 ./bench_mkl` |
| Eigen | Single-threaded GEMM |
| PyTorch (CPU) | `torch.mm()` with `OMP_NUM_THREADS=1` |

:::info
Benchmark results vary significantly by hardware. Always run on your target platform for meaningful comparisons.
:::
