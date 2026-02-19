---
sidebar_position: 12
title: Design Decisions
---

# Design Decisions

The reasoning behind Aicraft's key architectural choices.

## 1. Header-Only Core

All hot paths are `static inline` in header files — zero function call overhead for tensor operations, SIMD kernels, and layer forward passes.

**Trade-off**: Longer compile times for the first build, but zero-cost abstractions at runtime. Only `core.c` and `vulkan.c` are separate compilation units (for global state definitions and Vulkan dynamic loading).

## 2. Arena Allocation

One large allocation (64 MB blocks), bump-pointer sub-allocation. Checkpoint/restore for training loops.

**Why not malloc/free?** In a training loop, thousands of intermediate tensors are created and destroyed per epoch. Each `malloc`/`free` is a syscall with overhead. The arena allocator:
- Allocates in O(1) — bump a pointer
- Frees all intermediates in O(1) — reset pointer position
- Zero fragmentation — linear allocation pattern
- SIMD-aligned by default

## 3. SIMD Everywhere

Every numerical kernel uses hand-tuned intrinsics with cascading fallback:

```
AVX-512 → AVX2 → SSE4.2 → ARM NEON → Scalar
```

**Why not auto-vectorization?** Compilers miss opportunities for:
- Register tiling (12-register accumulators in GEMM micro-kernels)
- Prefetch hints for L1 cache
- FMA instruction selection
- Optimal unroll factors per architecture

## 4. BLIS-style GEMM

Panel packing + cache-blocked 5-loop tiling with architecture-specific micro-kernels.

**Why not just call OpenBLAS?** Zero dependencies is a project goal. The BLIS algorithm is well-documented and achieves near-peak FLOPS when micro-kernels are properly tuned. Our implementation uses the same algorithm as OpenBLAS/BLIS themselves.

**Cache blocking parameters**:

| Parameter | Value | Rationale |
|---|---|---|
| MC | 72 | ~36 KB packed A panel → fits L1 (32-48 KB) |
| KC | 256 | ~72 KB packed B slice → fits L2 |
| NC | 4096 | Full N dimension → L3 streaming |
| MR | 6 | 6 rows × NR columns = 12 accumulators ≈ max FP regs |
| NR | 32/16/8 | AVX-512/AVX2/NEON SIMD width |

## 5. Fused Operations

Combined softmax + cross-entropy eliminates an intermediate buffer and improves numerical stability:

```c
// Instead of:
softmax = exp(x) / sum(exp(x))  // intermediate buffer
loss = -sum(y * log(softmax))    // second pass

// Fused:
loss = -sum(y * log_softmax(x))  // single pass, log-sum-exp trick
// Gradient: softmax(x) - y       // closed-form, no log needed
```

## 6. No Abstraction Tax

Direct C structs + functions. No virtual dispatch, no RTTI, no vtables.

```c
// Layer is a plain struct
ac_dense layer;
ac_dense_init(&layer, 784, 10);

// Forward is a direct function call — inlined by compiler
ac_tensor* out = ac_dense_forward(&layer, input);
```

**Comparison**: PyTorch uses `Module.forward()` via Python virtual dispatch → C++ virtual dispatch → kernel launch. Aicraft inlines directly to SIMD instructions.

## 7. Vulkan over CUDA

Vulkan compute works across all GPU vendors (NVIDIA, AMD, Intel, Qualcomm). CUDA is NVIDIA-only.

**Trade-offs**:
- Vulkan has more boilerplate (pipeline creation, descriptor sets)
- CUDA has more mature ML ecosystem (cuDNN, cuBLAS)
- Vulkan enables deployment on mobile GPUs and integrated graphics
- Dynamic loading means no compile-time SDK dependency

## 8. Dynamic Autograd Graph

No static limits on graph size. Param groups and topology buffers grow dynamically.

**Why dynamic?** Static limits force users to predict graph size. Dynamic allocation with the arena allocator is essentially free (bump pointer), so there's no reason to limit.

## 9. Production Error Handling

Error codes + user callbacks replace raw `assert()`.

```c
// Instead of:
assert(shape.ndim > 0);  // crashes in release, no recovery

// We use:
if (shape.ndim == 0) {
    AC_RETURN_ERROR(AC_ERROR_SHAPE, "empty shape");
    // User callback notified, error state set, function returns error code
}
```

## 10. xoshiro128** PRNG

Passes BigCrush statistical tests (unlike basic LCGs). Fast, small state (16 bytes), good quality.

| PRNG | BigCrush | Speed | State Size |
|---|---|---|---|
| LCG | **Fails** | Fast | 4 bytes |
| Mersenne Twister | Passes | Medium | 2.5 KB |
| **xoshiro128**** | **Passes** | **Fast** | **16 bytes** |
