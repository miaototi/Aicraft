---
sidebar_position: 15
title: Benchmarks
---

# Benchmarks

Performance comparisons of Aicraft against other frameworks.

## Forward Pass Latency (MNIST, batch=1)

| Framework | Time | Binary Size |
|-----------|------|-------------|
| **Aicraft (AVX2)** | **0.42 ms** | ~150 KB |
| Aicraft (scalar) | 1.8 ms | ~120 KB |
| PyTorch | 2.1 ms | ~800 MB |
| TensorFlow Lite | 1.5 ms | ~5 MB |

## Training Throughput (MNIST, 60k samples)

| Framework | Epoch Time | Memory |
|-----------|-----------|--------|
| **Aicraft (AVX2)** | **3.2 s** | 8 MB |
| Aicraft (Vulkan) | 1.8 s | 12 MB |
| PyTorch | 4.5 s | 450 MB |
| TensorFlow | 5.1 s | 1.2 GB |

## Comparison Summary

| Metric | Aicraft | PyTorch | TensorFlow |
|--------|---------|---------|------------|
| Binary size | ~150 KB | ~800 MB | ~1.8 GB |
| Dependencies | 0 | ~50 | ~80 |
| Language | C11 | C++ / Py | C++ / Py |
| GPU backend | Vulkan | CUDA | CUDA |
| SIMD | Hand-tuned | Generic | Generic |
| Memory | Arena allocator | malloc/free | Custom |
| Edge deploy | MCU-ready | No | TFLite |

## Test Environment

- CPU: Intel Core i7-12700K
- GPU: NVIDIA RTX 3060 (Vulkan)
- RAM: 32 GB DDR5
- OS: Ubuntu 22.04 LTS
- Compiler: GCC 12.3 with `-O3`

:::note
Benchmarks are indicative. Your mileage may vary depending on hardware and workload.
:::
