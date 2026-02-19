---
slug: /
sidebar_position: 1
title: Introduction
---

# Aicraft

**High-Performance Machine Learning Framework in Pure C/C++**

Aicraft is a complete ML framework built from the ground up in pure C with zero external dependencies. It delivers production-grade performance through hand-tuned SIMD kernels, a BLIS-style GEMM engine, Vulkan GPU compute, and INT8 quantization — all in a header-only library.

## Key Highlights

- **Pure C/C++** — No Python runtime, no external BLAS, no CUDA dependency
- **SIMD-Optimized** — AVX-512, AVX2, SSE4.2, ARM NEON with automatic fallback
- **Vulkan GPU** — Cross-vendor GPU acceleration via 14 compute shaders
- **Edge-Ready** — INT8 quantization with ~4× model compression for embedded devices
- **Complete Autograd** — Reverse-mode autodiff with 22 operation types
- **Arena Memory** — Checkpoint/restore allocator eliminates per-tensor malloc/free

## What You Can Build

```c
#include <aicraft/aicraft.h>

int main() {
    ac_init();
    
    // Create layers
    ac_dense layer1, layer2;
    ac_dense_init(&layer1, 784, 128);
    ac_dense_init(&layer2, 128, 10);
    
    // Forward pass
    ac_tensor* input = ac_tensor_2d(32, 784, 0);
    ac_tensor_uniform(input, -1.0f, 1.0f);
    
    ac_tensor* h = ac_dense_forward(&layer1, input);
    h = ac_tensor_relu(h);
    ac_tensor* output = ac_dense_forward(&layer2, h);
    ac_tensor* probs = ac_tensor_softmax(output);
    
    ac_tensor_print(probs, "predictions");
    ac_cleanup();
    return 0;
}
```

## Project Stats

| Metric | Value |
|---|---|
| Header files | 16 |
| Test cases | 75 across 25 sections |
| GPU shaders | 14 GLSL compute shaders |
| Autograd ops | 22 backward rules |
| SIMD backends | AVX-512, AVX2, SSE, NEON, scalar |
| Dependencies | **0** |

## Next Steps

- [**Getting Started**](./getting-started) — Build and run in 5 minutes
- [**Architecture**](./architecture) — Understand the system design
- [**API Reference**](./api/overview) — Complete function reference
- [**Benchmarks**](./benchmarks) — Performance measurements

---

> Aicraft is a project by **[T&M Softwares](https://tmsoftwares.eu)**. Visit us for consulting, custom solutions, and more.
