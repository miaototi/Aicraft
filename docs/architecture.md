---
sidebar_position: 3
title: Architecture
---

# Architecture

Aicraft follows a layered architecture with zero-abstraction-cost design. Every component is a plain C struct with inline functions — no virtual dispatch, no heap allocation in hot paths.

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                     │
├────────────┬────────────┬──────────────┬────────────────┤
│   Layers   │    Loss    │  Optimizer   │  Serialize     │
│  Dense     │  MSE       │  SGD/Adam    │  Save/Load     │
│  Conv2D    │  CE/BCE    │  LR Schedule │  .acml format  │
│  BatchNorm │            │  Grad Clip   │                │
├────────────┴────────────┴──────────────┴────────────────┤
│                   Autograd Engine                        │
│        Reverse-mode AD · 22 ops · Dynamic graph          │
├──────────────────────┬──────────────────────────────────┤
│    Tensor Core       │         Quantization             │
│  N-D tensors (≤8D)   │   INT8 affine · QGEMM           │
│  SIMD-aligned data   │   ~4× compression               │
├──────────────────────┴──────────────────────────────────┤
│                  Compute Backends                        │
│  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │   CPU / SIMD     │  │      Vulkan GPU              │  │
│  │  AVX-512/AVX2    │  │  14 compute shaders          │  │
│  │  SSE / NEON      │  │  Shared-mem tiled GEMM       │  │
│  │  BLIS-style GEMM │  │  Auto-dispatch (size-based)  │  │
│  └──────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   Systems Layer                          │
│   Arena/Pool Memory  ·  Thread Pool  ·  Error Handling   │
│   Platform Detection ·  PRNG (xoshiro128**)              │
└─────────────────────────────────────────────────────────┘
```

## Header Map

```
include/aicraft/
├── platform.h       # Platform detection, SIMD macros, compiler hints
├── error.h          # Error codes, handlers, reporting macros
├── memory.h         # Arena & pool allocators, checkpoint/restore
├── simd_math.h      # SIMD kernels: GEMM, dot, relu, exp, etc.
├── fast_math.h      # Approximated exp, tanh (polynomial fits)
├── tensor.h         # Tensor structure with autograd metadata
├── tensor_ops.h     # Tensor operations (add, sub, mul, matmul, reshape)
├── autograd.h       # Reverse-mode autodiff engine (dynamic graph)
├── layers.h         # Dense, Conv2D, BatchNorm, Dropout, MaxPool
├── loss.h           # MSE, CrossEntropy, BCE losses
├── optimizer.h      # SGD/Adam/AdamW + grad clipping + LR schedulers
├── serialize.h      # Binary model save/load
├── quantize.h       # INT8 quantization engine
├── vulkan.h         # Vulkan compute backend
├── thread_pool.h    # Thread pool for parallel GEMM
└── aicraft.h        # Single-header include + lifecycle
```

## Memory Management

Aicraft uses a two-tier memory system designed to eliminate per-tensor allocation overhead:

### Arena Allocator

```
Training loop with checkpoint/restore:

ac_arena_save()     ←── save pointer position
  │
  ├── forward pass allocations (intermediates)
  ├── backward pass allocations (gradients)
  ├── optimizer temporaries
  │
ac_arena_restore()  ←── reset pointer, all intermediates freed instantly
```

**Key properties:**
- Single `malloc` for 64 MB blocks, bump-pointer sub-allocation
- SIMD-aligned (64-byte) by default
- O(1) bulk deallocation via `restore()`
- Model parameters allocated *before* the checkpoint survive restore

### Pool Allocator

Fixed-size free-list allocator for uniform objects (e.g., autograd nodes). O(1) alloc and free with bounds checking.

## GEMM Engine

The GEMM implementation follows the BLIS/GotoBLAS algorithm:

```
For each block of N (NC=4096):
  Pack B panel → B̃  (KC × NC, column-contiguous)
  For each block of M (MC=72):
    Pack A panel → Ã  (MC × KC, row-contiguous)
    For each micro-tile:
      Micro-kernel: Ã[MR×KC] × B̃[KC×NR] → C[MR×NR]
```

### Micro-kernels by Architecture

| Architecture | Tile Size | Registers | FMAs/cycle |
|---|---|---|---|
| AVX-512 | 6×32 | 12 ZMM + 2 load | 192/k-step |
| AVX2 | 6×16 | 12 YMM + 2 load | 96/k-step |
| ARM NEON | 6×8 | 12 Q-reg | 48/k-step |
| Scalar | 6×8 | — | Baseline |

**Optimizations:**
- L1 prefetch hints every 4 K iterations (AVX2)
- Thread-parallel outer loop for M ≥ 2×MC
- Panel packing for cache-line-aligned access
- FMA instructions when available (`__FMA__` detection)

## Autograd Engine

The autograd engine implements reverse-mode automatic differentiation:

1. **Forward pass**: Each operation records its inputs and type in the tensor's `op` field
2. **Topological sort**: DFS from the loss tensor, collecting all dependencies
3. **Backward pass**: Walk the sorted list in reverse, computing gradients

```
Loss tensor (seed grad = 1.0)
    │
    ▼
Topological Sort (DFS)
    │
    ▼
[loss, softmax, matmul, relu, matmul, ...]  ← reverse order
    │
    ▼
For each op: dispatch backward rule (22 op types)
    │
    ▼
All parameter .grad fields populated
```

**O(1) visited check**: A global epoch counter increments per backward call. Each tensor stores the epoch it was last visited — no hash set needed.

## Vulkan Compute Pipeline

```
Host                          GPU
─────                         ────
ac_vk_init()            →     Create instance, device, queues
ac_vk_create_buffer()   →     Allocate STORAGE_BUFFER
ac_vk_upload()          →     Staging buffer → GPU transfer
ac_vk_gemm()            →     Dispatch GEMM shader (16×16 tiles)
ac_vk_download()        →     GPU → staging buffer → host
```

- **14 compute shaders**: GEMM, add, mul, scale, FMA, ReLU (fwd+bwd), sigmoid (fwd+bwd), tanh (fwd+bwd), softmax, sum
- **Auto-dispatch**: Operations on tensors ≥4096 elements go to GPU; smaller to CPU SIMD
- **Dynamic loading**: Vulkan functions loaded at runtime (no compile-time SDK dependency)

## Data Flow: Training Step

```
1. ac_zero_grad()           Clear all .grad buffers
2. ac_dense_forward()       GEMM → bias add → store autograd info
3. ac_tensor_relu()         Element-wise ReLU → record op
4. ac_cross_entropy_loss()  Fused softmax+CE → scalar loss
5. ac_backward()            Topo-sort → reverse backward rules
6. ac_clip_grad_norm()      SIMD L2-norm compute → scale gradients
7. ac_adam_step()            SIMD-vectorized parameter update
8. ac_arena_restore()       Bulk-free all intermediates
```
