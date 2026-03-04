---
slug: /
sidebar_position: 1
title: Introduction
---

# Aicraft

**Pure C machine-learning framework. No dependencies, no runtime.**

Aicraft is a complete deep-learning framework written entirely in pure C11. It is SIMD-optimised, Vulkan-accelerated, and header-only — from training to edge inference in a single `#include`.

## Why Aicraft?

- **Zero dependencies** — no pip, no conda, no CMake, no vcpkg
- **Header-only** — drop the folder into your project, pass `-I./include`, done
- **SIMD vectorised** — AVX2, AVX-512, ARM NEON hand-tuned kernels
- **Vulkan GPU** — 14 GLSL compute shaders for cross-vendor acceleration
- **Autograd engine** — 22 differentiable ops with reverse-mode autodiff
- **INT8 quantisation** — post-training quantisation for edge deployment
- **Arena allocator** — checkpoint/restore memory, zero per-tensor malloc

## Quick Start

```bash
git clone https://github.com/TobiasTesauri/Aicraft.git
cd Aicraft
gcc -O3 demo.c -I./include -o demo
./demo
```

## Project Structure

```
Aicraft/
├── include/aicraft/
│   ├── aicraft.h          # Single include entry point
│   ├── tensor.h           # N-dimensional tensor
│   ├── autograd.h         # Reverse-mode autodiff
│   ├── layers.h           # Dense, Conv, etc.
│   ├── loss.h             # Cross-entropy, MSE, Huber
│   ├── optimizer.h        # SGD, Adam, AdamW
│   ├── memory.h           # Arena allocator
│   ├── simd.h             # SIMD intrinsics
│   ├── quantize.h         # INT8 quantisation
│   └── vulkan.h           # Vulkan compute backend
├── shaders/               # 14 GLSL compute shaders
├── tests/                 # 75+ test cases
└── demo.c                 # Example program
```

## License

Aicraft is licensed under the **MIT License**. A project by [Tobias Tesauri](https://tmsoftwares.eu) — T&M Softwares.
