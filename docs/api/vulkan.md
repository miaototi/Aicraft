---
sidebar_position: 10
title: Vulkan API
---

# Vulkan API

`#include "aicraft/vulkan.h"`

## Overview

Optional Vulkan compute backend for GPU-accelerated operations. Link with `-lvulkan` to enable.

## Initialisation

```c
ac_vulkan_init();                  // Auto-select best GPU
ac_vulkan_select_device(int idx);  // Select specific GPU
ac_vulkan_shutdown();              // Release GPU resources
```

## Device Info

```c
typedef struct AcVkDeviceInfo {
    char  name[256];
    int   vram_mb;
    int   compute_units;
    int   max_workgroup_size;
} AcVkDeviceInfo;

AcVkDeviceInfo info = ac_vulkan_device_info();
```

## Data Transfer

```c
void ac_vulkan_upload(AcTensor *t);     // CPU → GPU
void ac_vulkan_download(AcTensor *t);   // GPU → CPU
void ac_vulkan_sync();                   // Wait for GPU
```

## Compute Shaders

14 GLSL compute shaders are compiled at initialisation:

| Shader | Operation |
|--------|-----------|
| `gemm.comp` | General matrix multiply |
| `relu.comp` | ReLU activation |
| `sigmoid.comp` | Sigmoid activation |
| `softmax.comp` | Softmax |
| `add.comp` | Element-wise add |
| `mul.comp` | Element-wise mul |
| `reduce_sum.comp` | Sum reduction |
| `reduce_max.comp` | Max reduction |
| `transpose.comp` | Matrix transpose |
| `cross_entropy.comp` | CE loss |
| `mse.comp` | MSE loss |
| `adam_step.comp` | Adam update |
| `quantize.comp` | INT8 quantisation |
| `broadcast.comp` | Broadcasting |

## Backend Dispatch

When Vulkan is initialised, operations on uploaded tensors automatically dispatch to GPU shaders. No code changes needed:

```c
ac_vulkan_init();
ac_vulkan_upload(x);

// This runs on GPU automatically
AcTensor *y = ac_forward_seq(net, 2, x);

ac_vulkan_download(y);
```

## Fallback

If Vulkan initialisation fails (e.g., no GPU), operations fall back to the CPU (SIMD) backend transparently.
