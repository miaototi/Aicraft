---
sidebar_position: 10
title: FAQ
---

# Frequently Asked Questions

## General

### What is Aicraft?

Aicraft is a complete deep-learning framework written entirely in pure C (C11). It features SIMD-optimised kernels, Vulkan GPU acceleration, automatic differentiation, and requires zero external dependencies.

### Why C instead of C++ or Python?

- **Portability**: C compiles everywhere — from x86 servers to ARM microcontrollers
- **Control**: No hidden allocations, no garbage collection, deterministic memory
- **Integration**: Easy to embed in any language via FFI
- **Simplicity**: One file, one compiler, done

### Is it production-ready?

Aicraft is suitable for edge inference and embedded systems. For large-scale training (millions of parameters, huge datasets), consider using GPU clusters with Python frameworks. Aicraft excels in:
- Embedded/IoT inference
- Research prototyping
- Educational purposes
- Latency-critical applications

---

## Installation

### Do I need to install anything?

No. Aicraft is header-only. Just clone and include:

```c
#include "aicraft/aicraft.h"
```

### What compilers are supported?

- **GCC** 7+ (recommended)
- **Clang** 6+
- **MSVC** 2019+

### Does it work on Windows?

Yes. Use MSVC or MinGW-w64. For Vulkan, install the Vulkan SDK.

---

## Performance

### How do I enable SIMD?

Pass the appropriate compiler flag:

```bash
# AVX2 (most modern x86)
gcc -O3 -mavx2 ...

# AVX-512 (Intel Xeon, some i9)
gcc -O3 -mavx512f ...

# ARM NEON (Raspberry Pi, Apple Silicon via Rosetta)
gcc -O3 -mfpu=neon ...
```

Aicraft auto-detects available SIMD at compile time.

### How do I use the GPU?

```c
ac_init();
ac_vulkan_init();  // Enable Vulkan backend

// ... your code ...

ac_vulkan_cleanup();
ac_cleanup();
```

Requires Vulkan SDK installed on your system.

### Why is my model slow?

Common issues:
1. **No SIMD flags**: Add `-mavx2` or `-mfpu=neon`
2. **Debug build**: Use `-O3`, not `-O0`
3. **Small batch size**: SIMD shines with batch ≥ 32
4. **Memory allocation in loop**: Use `ac_mem_checkpoint()` / `ac_mem_restore()`

---

## Training

### How do I train a model?

```c
AcLayer *net[] = {
    ac_dense(784, 128, AC_RELU),
    ac_dense(128, 10,  AC_SOFTMAX)
};
AcOptimizer *opt = ac_adam(net, 2, 0.001f);

for (int i = 0; i < epochs; i++) {
    AcTensor *pred = ac_forward_seq(net, 2, input);
    AcTensor *loss = ac_cross_entropy(pred, target);
    ac_backward(loss);
    ac_optimizer_step(opt);
}
```

See the [Training Guide](/docs/guides/training) for details.

### What optimisers are available?

- `ac_sgd(layers, n, lr)` — Stochastic Gradient Descent
- `ac_sgd_momentum(layers, n, lr, momentum)` — SGD with momentum
- `ac_adam(layers, n, lr)` — Adam (recommended)
- `ac_adamw(layers, n, lr, weight_decay)` — AdamW with decoupled weight decay

### What loss functions exist?

- `ac_mse(pred, target)` — Mean Squared Error
- `ac_cross_entropy(pred, target)` — Categorical Cross-Entropy
- `ac_binary_cross_entropy(pred, target)` — Binary Cross-Entropy
- `ac_huber(pred, target, delta)` — Huber loss

---

## Memory

### How does memory management work?

Aicraft uses an arena allocator. All tensors are allocated from a single memory pool.

```c
ac_mem_checkpoint();   // Mark current position
// ... allocate tensors ...
ac_mem_restore();      // Free all tensors since checkpoint
```

This prevents memory leaks in training loops.

### My program runs out of memory

- Use `ac_mem_checkpoint()` / `ac_mem_restore()` in your training loop
- Reduce batch size
- For very large models, increase arena size: `ac_init_with_arena(1024 * 1024 * 512)` // 512 MB

---

## Deployment

### How do I save/load a model?

```c
// Save
ac_save_weights(net, num_layers, "model.bin");

// Load
ac_load_weights(net, num_layers, "model.bin");
```

### How do I quantise to INT8?

```c
ac_quantize_model(net, num_layers, AC_QUANT_INT8);
ac_save_weights(net, num_layers, "model_int8.bin");
```

See [Edge Deployment](/docs/guides/edge-deployment) for details.

### Can I run on Raspberry Pi / ARM?

Yes. Compile with:

```bash
arm-linux-gnueabihf-gcc -O3 -mfpu=neon your_code.c -I./include -o program
```

---

## Troubleshooting

### I get NaN values during training

Common causes:
1. **Learning rate too high**: Try `0.0001` instead of `0.01`
2. **Exploding gradients**: Add gradient clipping
3. **Bad data**: Check for NaN/Inf in your input

Debug with:
```c
ac_tensor_print(tensor);  // Print tensor values
ac_tensor_has_nan(tensor); // Returns true if NaN present
```

### Compilation errors about missing headers

Make sure you pass `-I./include` (path to Aicraft's include folder).

### Vulkan not found

1. Install Vulkan SDK from https://vulkan.lunarg.com/
2. Ensure `VULKAN_SDK` environment variable is set
3. Add `-lvulkan` to your linker flags

---

## Contributing

### Where's the source code?

https://github.com/TobiasTesauri/Aicraft

### How can I contribute?

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

See the README for coding guidelines.
