---
sidebar_position: 3
title: Vulkan GPU Guide
---

# Vulkan GPU Guide

Accelerate Aicraft with Vulkan compute shaders.

## Prerequisites

- Vulkan SDK installed (or system Vulkan drivers)
- A GPU that supports Vulkan 1.0+ compute

## Enabling Vulkan

```bash
gcc -O3 demo.c -I./include -lvulkan -o demo
```

At runtime, Aicraft automatically detects available GPU devices and selects the best one.

## Device Selection

```c
ac_init();
ac_vulkan_init();  // Auto-select best GPU

// Or manually select a device
ac_vulkan_select_device(0);  // First GPU

// Query device info
AcVkDeviceInfo info = ac_vulkan_device_info();
printf("GPU: %s (%d MB VRAM)\n", info.name, info.vram_mb);
```

## Compute Shaders

Aicraft includes 14 GLSL compute shaders:

| Shader | Operation |
|--------|-----------|
| `gemm.comp` | General matrix multiply |
| `relu.comp` | ReLU activation |
| `sigmoid.comp` | Sigmoid activation |
| `softmax.comp` | Softmax activation |
| `add.comp` | Element-wise add |
| `mul.comp` | Element-wise multiply |
| `reduce_sum.comp` | Sum reduction |
| `reduce_max.comp` | Max reduction |
| `transpose.comp` | Matrix transpose |
| `cross_entropy.comp` | Cross-entropy loss |
| `mse.comp` | MSE loss |
| `adam_step.comp` | Adam optimiser step |
| `quantize.comp` | INT8 quantisation |
| `broadcast.comp` | Broadcasting |

## CPU ↔ GPU Transfer

```c
// Move tensor to GPU
ac_vulkan_upload(tensor);

// Run operations on GPU (automatic)
AcTensor *y = ac_forward_seq(net, 2, x);

// Download result back to CPU
ac_vulkan_download(y);
```

## Performance Tips

- Batch operations to minimise CPU↔GPU transfers
- Use `ac_vulkan_sync()` explicitly only when needed
- Larger batch sizes benefit more from GPU acceleration
