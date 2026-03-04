---
sidebar_position: 16
title: Design Decisions
---

# Design Decisions

The key design choices behind Aicraft and the reasoning behind them.

## Why Pure C?

- **Portability** — C compilers exist for virtually every platform, from x86 desktops to ARM Cortex-M microcontrollers
- **Control** — No hidden allocations, no garbage collector, no runtime surprises
- **Embeddability** — Any language with a C FFI can call Aicraft
- **Simplicity** — The entire framework is ~5,000 lines of C11

## Why Header-Only?

- **Zero build friction** — No CMake, no Meson, no vcpkg, no conan
- **Single translation unit** — One `#include` and compile
- **Easy vendoring** — Copy the folder, done

## Why Arena Allocator?

Traditional `malloc`/`free` has several issues for ML workloads:

1. **Fragmentation** over long training runs
2. **Overhead** of per-tensor allocation tracking
3. **Complexity** of ownership semantics in computational graphs

The arena allocator solves all three: pre-allocate a block, bump-allocate within it, and restore to a checkpoint when done.

## Why Vulkan (not CUDA)?

- **Vendor-neutral** — Works on NVIDIA, AMD, Intel, and even mobile GPUs
- **Open standard** — No proprietary SDK lock-in
- **Compute shaders** — GLSL compute is well-suited for tensor operations
- **Cross-platform** — Windows, Linux, macOS (via MoltenVK), Android

## Why Not C++?

C++ would add:
- Longer compile times
- Template complexity
- ABI incompatibilities
- Harder FFI integration

Aicraft proves that modern ML primitives can be expressed cleanly in C11.

## Trade-offs

| Decision | Pro | Con |
|----------|-----|-----|
| Pure C | Maximum portability | Manual memory management |
| Header-only | Zero build friction | Recompiles on every change |
| Arena allocator | O(1) alloc/free | Fixed memory budget |
| Vulkan | Vendor-neutral GPU | More setup than CUDA |
| Hand-tuned SIMD | Peak performance | Per-platform maintenance |
