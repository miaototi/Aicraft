---
sidebar_position: 3
title: Vulkan GPU
---

# Vulkan GPU Compute

Aicraft includes an optional Vulkan compute backend for GPU-accelerated tensor operations. Unlike CUDA, Vulkan works across all GPU vendors (NVIDIA, AMD, Intel, Qualcomm).

## Setup

### Prerequisites

- [Vulkan SDK](https://vulkan.lunarg.com/) (headers + `glslc` shader compiler)
- A Vulkan-capable GPU with up-to-date drivers

### Build with Vulkan

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DAICRAFT_ENABLE_VULKAN=ON
cmake --build . --config Release
```

## How It Works

### Initialization

```c
ac_init();  // also calls ac_vk_init() if Vulkan is enabled
```

`ac_vk_init()` creates:
- Vulkan instance and device
- Command pool and command buffers
- Descriptor pool for shader bindings
- 16 MB staging buffer for host↔GPU transfers
- Pipeline cache for compiled shaders

### Auto-Dispatch

Aicraft automatically decides whether to use GPU or CPU for each operation:

```c
// Automatic: GPU for large tensors, CPU SIMD for small ones
int use_gpu = ac_vk_should_use_gpu(tensor_size);
// Returns true when tensor_size >= 4096 elements
```

This threshold can be tuned. The overhead of GPU dispatch (buffer upload, kernel launch, download) only pays off for larger tensors.

## Available GPU Operations

### GEMM (Matrix Multiplication)

```c
ac_vk_gemm(A_gpu, B_gpu, C_gpu, M, N, K, alpha, beta);
// C = alpha * A @ B + beta * C
// Uses 16×16 shared-memory tiling
```

The GEMM shader uses shared-memory blocking:
```glsl
// gemm.comp — 16×16 tile with shared memory
shared float tileA[TILE][TILE];
shared float tileB[TILE][TILE];

// Load tiles → barrier → accumulate → barrier → next tile
```

### Element-wise Operations

```c
ac_vk_add(X, Y, Out, n);       // Out = X + Y
ac_vk_mul(X, Y, Out, n);       // Out = X * Y
ac_vk_scale(X, Out, n, alpha); // Out = X * alpha
ac_vk_fma(X, Y, Out, n, a);   // Out = X * Y + a
```

### Activations (forward + backward)

```c
ac_vk_relu(X, Out, n);                 // Out = max(0, X)
ac_vk_relu_backward(X, Grad, Out, n);  // dX = Grad * (X > 0)
ac_vk_sigmoid(X, Out, n);              // Out = 1/(1+exp(-X))
ac_vk_sigmoid_backward(Out, Grad, DX, n);
ac_vk_tanh_act(X, Out, n);             // Out = tanh(X)
ac_vk_tanh_backward(Out, Grad, DX, n);
```

### Reductions

```c
ac_vk_softmax(X, Out, n);  // Row-wise softmax
ac_vk_sum(X, Out, n);      // Parallel reduction sum
```

The sum shader uses a **two-stage parallel reduction**: each workgroup of 256 threads reduces a chunk using shared memory, writing partial sums. The host runs a second pass if needed.

## GPU Memory Management

```c
// Allocate GPU buffer
ac_vk_buffer buf;
ac_vk_create_buffer(&buf, size_in_bytes);

// Upload data to GPU
ac_vk_upload(&buf, host_data, size_in_bytes);

// Download results
ac_vk_download(&buf, host_data, size_in_bytes);

// Free GPU buffer
ac_vk_destroy_buffer(&buf);
```

Transfers use a 16 MB staging buffer with `vkMapMemory` for efficient host↔device copies.

## Shader Pipeline

Each operation has a dedicated GLSL compute shader:

| Shader | Workgroup Size | Description |
|---|---|---|
| `gemm.comp` | 16×16 | Tiled GEMM with shared memory |
| `add.comp` | 256 | Element-wise addition |
| `mul.comp` | 256 | Element-wise multiplication |
| `scale.comp` | 256 | Scalar multiplication |
| `fma.comp` | 256 | Fused multiply-add |
| `relu.comp` | 256 | ReLU forward |
| `relu_backward.comp` | 256 | ReLU backward |
| `sigmoid.comp` | 256 | Sigmoid forward |
| `sigmoid_backward.comp` | 256 | Sigmoid backward |
| `tanh_act.comp` | 256 | Tanh forward |
| `tanh_backward.comp` | 256 | Tanh backward |
| `softmax.comp` | 256 | Row-wise softmax |
| `sum.comp` | 256 | Parallel reduction sum |
| `max.comp` | 256 | Parallel reduction max |

All shaders use **push constants** for runtime parameters (tensor size, scalar values) — avoiding descriptor set updates for each dispatch.

## Performance Considerations

:::tip When to Use GPU
- **Use GPU**: Large GEMM (≥512×512), batch inference, large element-wise ops (≥4096 elements)
- **Use CPU**: Small tensors, single-sample inference, operations where transfer overhead dominates
- **Auto-dispatch**: Let `ac_vk_should_use_gpu()` decide automatically
:::

:::caution Transfer Overhead
Each GPU operation requires upload → compute → download. For small tensors, this overhead exceeds the compute savings. The auto-dispatch handles this, but be aware when writing custom pipelines.
:::
