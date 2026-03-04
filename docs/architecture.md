---
sidebar_position: 3
title: Architecture
---

# Architecture

Aicraft is a vertically integrated stack. No external libraries sit between your code and the hardware.

## Stack Overview

```
┌─────────────────────────────────┐
│       Your Application          │  main.c
├─────────────────────────────────┤
│          aicraft.h              │  Single include
├─────────────────────────────────┤
│   Layers / Loss / Optimizer     │  High-level API
├─────────────────────────────────┤
│       Autograd Engine           │  22 ops, DAG-based
├─────────────────────────────────┤
│        Tensor Core              │  N-dim, broadcasting
├──────────────┬──────────────────┤
│ SIMD Kernels │  Vulkan Compute  │  Backends
│ AVX-512/NEON │  14 GLSL shaders │
├──────────────┴──────────────────┤
│       Arena Allocator           │  Checkpoint/restore
└─────────────────────────────────┘
```

## Design Principles

1. **Single header** — `#include "aicraft/aicraft.h"` pulls in everything
2. **Zero allocations in the hot path** — the arena allocator pre-reserves memory
3. **Compile-time dispatch** — SIMD backend is selected at compile time via macros
4. **DAG-based autograd** — reverse-mode autodiff with topological sorting
5. **Vulkan compute** — optional GPU path with zero CUDA dependency

## Memory Model

Aicraft uses a custom **arena allocator** with checkpoint/restore semantics:

```c
ac_mem_checkpoint();    // Save current memory state
// ... allocate tensors, run forward/backward ...
ac_mem_restore();       // Free everything since checkpoint
```

This gives constant-memory training with zero per-tensor `malloc`/`free` calls.

## Tensor Layout

Tensors are stored in **row-major** (C-contiguous) order with support for:

- Arbitrary dimensions (N-dim)
- Broadcasting
- Strides and views (zero-copy slicing)
- Optional gradient tracking for autograd

## Backend Selection

| Backend | Flag | Platforms |
|---------|------|-----------|
| Scalar (fallback) | default | All |
| SSE4.2 | `-msse4.2` | x86/x86_64 |
| AVX2 | `-mavx2 -mfma` | x86_64 |
| AVX-512 | `-mavx512f` | x86_64 (Skylake+) |
| ARM NEON | `-mfpu=neon` | ARM/AArch64 |
| Vulkan | `-lvulkan` | Any GPU vendor |
