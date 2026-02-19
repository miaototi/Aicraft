---
sidebar_position: 10
title: Vulkan Backend
---

# Vulkan Compute API

`#include <aicraft/vulkan.h>`

Optional GPU compute backend using Vulkan. Cross-vendor (NVIDIA, AMD, Intel, Qualcomm). Enabled with `-DAICRAFT_ENABLE_VULKAN=ON`.

## Initialization

```c
ac_vk_init();   // Called automatically by ac_init() if Vulkan enabled
ac_vk_cleanup();
```

Creates: Vulkan instance, physical/logical device, command pool, descriptor pool, staging buffer (16 MB), pipeline cache.

## GPU Buffers

```c
ac_vk_buffer buf;
ac_vk_create_buffer(&buf, size_bytes);     // STORAGE_BUFFER
ac_vk_upload(&buf, host_ptr, size);         // Host → GPU
ac_vk_download(&buf, host_ptr, size);       // GPU → Host
ac_vk_destroy_buffer(&buf);
```

## Auto-Dispatch

```c
int ac_vk_should_use_gpu(ac_size num_elements);
// Returns 1 when num_elements >= 4096
```

## GPU Operations

### Matrix Multiply

```c
ac_vk_gemm(A, B, C, M, N, K, alpha, beta);
// C = alpha * A @ B + beta * C
// 16×16 shared-memory tiled kernel
```

### Element-wise

```c
ac_vk_add(X, Y, Out, n);
ac_vk_mul(X, Y, Out, n);
ac_vk_scale(X, Out, n, scalar);
ac_vk_fma(X, Y, Out, n, scalar);
```

### Activations

```c
ac_vk_relu(X, Out, n);
ac_vk_relu_backward(X, Grad, Out, n);
ac_vk_sigmoid(X, Out, n);
ac_vk_sigmoid_backward(Out, Grad, DX, n);
ac_vk_tanh_act(X, Out, n);
ac_vk_tanh_backward(Out, Grad, DX, n);
```

### Reductions

```c
ac_vk_softmax(X, Out, n);
ac_vk_sum(X, Out, n);     // Two-stage parallel reduction
```

## Compute Shaders

14 GLSL 4.50 compute shaders in `src/shaders/`:

| Shader | Workgroup | Shared Memory |
|---|---|---|
| `gemm.comp` | 16×16 | 2 × TILE×TILE floats |
| `add.comp` | 256 | — |
| `mul.comp` | 256 | — |
| `scale.comp` | 256 | — |
| `fma.comp` | 256 | — |
| `relu.comp` | 256 | — |
| `relu_backward.comp` | 256 | — |
| `sigmoid.comp` | 256 | — |
| `sigmoid_backward.comp` | 256 | — |
| `tanh_act.comp` | 256 | — |
| `tanh_backward.comp` | 256 | — |
| `softmax.comp` | 256 | 256 floats |
| `sum.comp` | 256 | 256 floats |
| `max.comp` | 256 | 256 floats |

All shaders use **push constants** for runtime parameters (n, scalar values), avoiding descriptor set updates per dispatch.
