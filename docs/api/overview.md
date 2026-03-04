---
sidebar_position: 1
title: API Overview
---

# API Overview

The complete Aicraft API at a glance.

## Single Include

```c
#include "aicraft/aicraft.h"
```

This includes all sub-modules. You can also include individual headers:

```c
#include "aicraft/tensor.h"
#include "aicraft/layers.h"
#include "aicraft/autograd.h"
```

## Module Map

| Module | Header | Description |
|--------|--------|-------------|
| [Tensor](tensor) | `tensor.h` | N-dimensional tensor creation and operations |
| [Autograd](autograd) | `autograd.h` | Reverse-mode automatic differentiation |
| [Layers](layers) | `layers.h` | Neural network layers (Dense, etc.) |
| [Loss](loss) | `loss.h` | Loss functions (CE, MSE, Huber) |
| [Optimizer](optimizer) | `optimizer.h` | Optimisers (SGD, Adam, AdamW) |
| [Memory](memory) | `memory.h` | Arena allocator with checkpoint/restore |
| [SIMD](simd) | `simd.h` | Platform-specific vectorised kernels |
| [Quantize](quantize) | `quantize.h` | INT8 post-training quantisation |
| [Vulkan](vulkan) | `vulkan.h` | GPU compute backend |

## Lifecycle

```c
ac_init();              // Initialise framework
// ... use API ...
ac_cleanup();           // Release all resources
```

## Conventions

- All public symbols are prefixed with `ac_` or `Ac`
- Types use PascalCase: `AcTensor`, `AcLayer`, `AcOptimizer`
- Functions use snake_case: `ac_tensor_new`, `ac_forward_seq`
- Constants use SCREAMING_CASE: `AC_RELU`, `AC_QUANT_INT8`
