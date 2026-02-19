---
sidebar_position: 1
title: API Overview
---

# API Reference

Aicraft's API is organized into focused modules, each in its own header file. Include everything with a single header:

```c
#include <aicraft/aicraft.h>
```

Or include individual modules:

```c
#include <aicraft/tensor.h>
#include <aicraft/layers.h>
#include <aicraft/optimizer.h>
```

## Module Map

| Module | Header | Description |
|---|---|---|
| [**Tensor**](./tensor) | `tensor.h` + `tensor_ops.h` | N-D tensors, creation, operations, activations |
| [**Autograd**](./autograd) | `autograd.h` | Reverse-mode automatic differentiation |
| [**Layers**](./layers) | `layers.h` | Dense, Conv2D, BatchNorm, Dropout, MaxPool, Flatten |
| [**Loss**](./loss) | `loss.h` | MSE, Cross-Entropy, Binary Cross-Entropy |
| [**Optimizer**](./optimizer) | `optimizer.h` | SGD, Adam/AdamW, LR schedulers, gradient clipping |
| [**Memory**](./memory) | `memory.h` | Arena & pool allocators |
| [**SIMD**](./simd) | `simd_math.h` + `fast_math.h` | SIMD kernels, GEMM, approximated math |
| [**Quantization**](./quantize) | `quantize.h` | INT8 quantization, QGEMM, quantized layers |
| [**Vulkan**](./vulkan) | `vulkan.h` | GPU compute backend |

## Lifecycle

```c
ac_init();       // Initialize arena, thread pool, PRNG, Vulkan (if enabled)
// ... your code ...
ac_cleanup();    // Free all resources
```

## Naming Conventions

- **Prefix**: All public symbols use `ac_` prefix
- **Structs**: `ac_tensor`, `ac_dense`, `ac_adam`, etc.
- **Functions**: `ac_<module>_<action>()` — e.g., `ac_tensor_relu()`, `ac_dense_forward()`
- **Constants**: `AC_UPPER_CASE` — e.g., `AC_MAX_DIMS`, `AC_OK`
- **Types**: `ac_size` (size type), `ac_error_code` (enum)

## Thread Safety

- **Global state**: `g_tensor_arena`, `g_rng`, `g_thread_pool` are global. Not thread-safe for concurrent `ac_init`/`ac_cleanup`.
- **Inference**: Thread-safe if each thread uses its own tensors
- **Training**: Single-threaded (autograd uses global epoch counter)
- **GEMM**: Internally parallelized via thread pool
