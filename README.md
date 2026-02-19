# Aicraft

**High-Performance Machine Learning Framework in Pure C/C++**

Zero dependencies. SIMD-optimized. Edge/Embedded ready with ARM NEON and INT8 quantization.
Vulkan compute backend for GPU-accelerated operations.
Production-ready error handling, serialization, and memory management.

```
  ╔══════════════════════════════════════════════════════════╗
  ║     AICRAFT v1.0.0 - ML Without Compromises            ║
  ║     No Python. No Dependencies. Pure Performance.       ║
  ╚══════════════════════════════════════════════════════════╝
```

## Why Aicraft

| Aspect | PyTorch/TF | Aicraft |
|---|---|---|
| **Language** | Python + C++ backend | Pure C/C++ |
| **Memory** | malloc/free per op | Arena allocator + checkpoint/restore |
| **Dispatch** | Virtual functions + Python overhead | Static dispatch, inlined |
| **SIMD** | General-purpose kernels | Hand-tuned AVX-512/AVX2/SSE/NEON kernels |
| **GPU** | CUDA-only | Vulkan compute (cross-vendor, auto-fallback) |
| **GEMM** | External BLAS (MKL/OpenBLAS) | BLIS-style tiled GEMM with panel packing |
| **Fused Ops** | Separate kernel launches | Fused softmax+CE, FMA |
| **Allocations** | Per-tensor heap alloc | Zero-alloc arena + pool |
| **Error handling** | Python exceptions | Error codes + callbacks |

## Architecture

```
include/aicraft/
├── platform.h       # Platform detection, SIMD, compiler hints
├── error.h          # Error codes, handlers, reporting macros
├── memory.h         # Arena & pool allocators, checkpoint/restore
├── simd_math.h      # SIMD kernels: GEMM, dot, relu, exp, etc.
├── fast_math.h      # Approximated exp, tanh (polynomial fits)
├── tensor.h         # Tensor structure with autograd metadata
├── tensor_ops.h     # Tensor operations (add, sub, mul, div, matmul, reshape)
├── autograd.h       # Reverse-mode autodiff engine (dynamic graph)
├── layers.h         # Dense, Conv2D, BatchNorm, Dropout, MaxPool
├── loss.h           # MSE, CrossEntropy, BCE losses
├── optimizer.h      # SGD/Adam/AdamW + grad clipping + LR schedulers
├── serialize.h      # Binary model save/load
├── quantize.h       # INT8 quantization engine (calibrate, quantize, qgemm)├── vulkan.h         # Vulkan compute backend (GPU GEMM, activations, ops)├── thread_pool.h    # Thread pool for parallel GEMM
└── aicraft.h        # Single-header include + lifecycle
```

## Features

### Core
- **Tensor Core**: N-dimensional tensors (up to 8D) with SIMD-aligned storage
- **Autograd Engine**: Reverse-mode autodiff with dynamic topological sorting (no static graph limit)
- **PRNG**: xoshiro128** (passes BigCrush statistical tests)

### Neural Network
- **Layers**: Dense, Conv2D (im2col), MaxPool2D, BatchNorm, Dropout, Flatten
- **Activations**: ReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy
- **Optimizers**: SGD (with momentum + weight decay), Adam, AdamW — all SIMD-accelerated
- **Gradient Clipping**: Global L2-norm and per-element value clipping
- **LR Schedulers**: Step decay, cosine annealing, exponential decay
- **Initialization**: Xavier/Glorot, He, Uniform

### Edge / Embedded
- **ARM NEON**: Real NEON intrinsics for all element-wise ops (add, mul, scale, fma, dot, sum, max, relu) + GEMM 6×8 micro-kernel
- **NEON Transcendentals**: Vectorized exp (Cephes polynomial), sigmoid (Newton-Raphson reciprocal), tanh — forward and backward
- **INT8 Quantization**: Asymmetric per-tensor affine quantization (float32 → uint8) with ~4× model compression
- **Quantized Inference**: INT8 GEMM with INT32 accumulation for overflow-safe matmul, quantized dense layer
- **Model Size Estimation**: Utilities to compare float32 vs INT8 model footprint

### Systems
- **Memory**: Arena allocator with checkpoint/restore for training loops + pool allocator
- **SIMD**: Cascading AVX-512 → AVX2 → SSE → NEON → scalar fallback (with real hand-tuned intrinsics per arch)
- **Vulkan Compute**: GPU-accelerated GEMM, element-wise ops, activations, softmax, reductions — auto-dispatch (GPU for large tensors, CPU SIMD for small)
- **GEMM**: BLIS-style with panel packing — AVX-512: 6×32, AVX2: 6×16, NEON: 6×8, scalar fallback + tiled GPU GEMM with shared-memory blocking
- **Threading**: Thread pool for parallel GEMM
- **Error Handling**: Error codes, user callbacks, debug-mode stderr reporting
- **Serialization**: Binary model save/load with versioned format and magic header
- **Gradient Clipping**: Global L2-norm clipping and per-element value clipping
- **LR Scheduling**: Step decay, cosine annealing, exponential decay

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `AICRAFT_BUILD_TESTS` | ON | Build test suite |
| `AICRAFT_BUILD_BENCH` | ON | Build benchmarks |
| `AICRAFT_BUILD_DEMO` | ON | Build demo |
| `AICRAFT_ENABLE_AVX512` | OFF | Enable AVX-512 (requires CPU support) |
| `AICRAFT_ENABLE_VULKAN` | OFF | Enable Vulkan compute backend (requires Vulkan SDK) |

### Debug Build (with sanitizers)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

### Vulkan Build (GPU acceleration)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DAICRAFT_ENABLE_VULKAN=ON
cmake --build . --config Release
```

Requires:
- [Vulkan SDK](https://vulkan.lunarg.com/) (headers + `glslc` shader compiler)
- A Vulkan-capable GPU driver

## Run

```bash
# Run tests (75 tests across 25 sections)
./aicraft_test

# Run XOR demo
./aicraft_demo

# Run benchmarks
./aicraft_bench
```

## Quick Start

```c
#include <aicraft/aicraft.h>

int main() {
    ac_init();
    
    // Create a simple neural network
    ac_dense layer1, layer2;
    ac_dense_init(&layer1, 784, 128);
    ac_dense_init(&layer2, 128, 10);
    
    // Create input tensor [batch=32, features=784]
    ac_tensor* input = ac_tensor_2d(32, 784, 0);
    ac_tensor_uniform(input, -1.0f, 1.0f);
    
    // Forward pass
    ac_tensor* h = ac_dense_forward(&layer1, input);
    h = ac_tensor_relu(h);
    ac_tensor* output = ac_dense_forward(&layer2, h);
    ac_tensor* probs = ac_tensor_softmax(output);
    
    ac_tensor_print(probs, "predictions");
    
    ac_cleanup();
    return 0;
}
```

## Training Example

```c
#include <aicraft/aicraft.h>

int main() {
    ac_init();
    
    // Model
    ac_dense fc1, fc2;
    ac_dense_init(&fc1, 784, 256);
    ac_dense_init(&fc2, 256, 10);
    
    // Register parameters
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, fc1.weight);
    ac_param_group_add(&params, fc1.bias);
    ac_param_group_add(&params, fc2.weight);
    ac_param_group_add(&params, fc2.bias);
    
    // Adam optimizer
    ac_adam opt;
    ac_adam_init(&opt, &params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, 0);
    
    // Training loop with arena checkpointing (prevents memory growth)
    for (int epoch = 0; epoch < 100; epoch++) {
        ac_arena_checkpoint cp;
        ac_arena_save(&g_tensor_arena, &cp);
        
        ac_zero_grad(&params);
        
        ac_tensor* h = ac_dense_forward(&fc1, input);
        h = ac_tensor_relu(h);
        ac_tensor* logits = ac_dense_forward(&fc2, h);
        
        ac_tensor* loss = ac_cross_entropy_loss(logits, labels);
        ac_backward(loss);
        ac_clip_grad_norm(&params, 1.0f);  // gradient clipping
        ac_adam_step(&opt);
        
        ac_arena_restore(&g_tensor_arena, &cp);  // free intermediates
    }
    
    // Save trained model
    ac_model_save("model.acml", &params);
    
    ac_param_group_destroy(&params);
    ac_cleanup();
    return 0;
}
```

## INT8 Quantized Inference

```c
#include <aicraft/aicraft.h>

int main() {
    ac_init();
    
    // Train your model (or load from file)
    ac_dense fc1, fc2;
    ac_dense_init(&fc1, 784, 256);
    ac_dense_init(&fc2, 256, 10);
    // ... training ...
    
    // Quantize layers for edge deployment (~4x smaller)
    ac_qdense qfc1, qfc2;
    ac_qdense_from_dense(&qfc1, fc1.weight, fc1.bias, 784, 256);
    ac_qdense_from_dense(&qfc2, fc2.weight, fc2.bias, 256, 10);
    
    // Print model size comparison
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, fc1.weight);
    ac_param_group_add(&params, fc1.bias);
    ac_param_group_add(&params, fc2.weight);
    ac_param_group_add(&params, fc2.bias);
    ac_model_size_info info = ac_estimate_model_size(&params);
    ac_print_model_size(&info);  // shows FP32 vs INT8 sizes
    ac_param_group_destroy(&params);
    
    // Quantized inference (INT8 matmul + INT32 accumulation)
    ac_tensor* input = ac_tensor_2d(1, 784, 0);
    ac_tensor_uniform(input, -1.0f, 1.0f);
    
    ac_tensor* hidden = ac_qdense_forward(&qfc1, input);
    // Apply activation on float output
    for (ac_size i = 0; i < hidden->shape.total_size; i++)
        hidden->data[i] = hidden->data[i] > 0 ? hidden->data[i] : 0;  // relu
    ac_tensor* output = ac_qdense_forward(&qfc2, hidden);
    
    ac_tensor_print(output, "quantized prediction");
    
    ac_cleanup();
    return 0;
}
```

## Model Save/Load

```c
// Save
ac_model_save("weights.acml", &params);

// Load
ac_error_code err = ac_model_load("weights.acml", &params);
if (err != AC_OK) {
    printf("Load failed: %s\n", ac_get_last_error_message());
}
```

## Error Handling

```c
// Set a custom error handler
void my_handler(const ac_error* err, void* user_data) {
    fprintf(stderr, "Error at %s:%d: %s\n", err->file, err->line, err->message);
}
ac_set_error_handler(my_handler, NULL);

// Check errors
ac_error_code err = ac_model_load("missing.acml", &params);
if (err != AC_OK) {
    printf("Error: %s\n", ac_error_string(err));
    ac_clear_error();
}
```

## Key Design Decisions

1. **Header-only core**: All hot paths are `inline` — zero function call overhead
2. **Arena allocation**: One large allocation, sub-allocate within. Checkpoint/restore for training loops
3. **SIMD everywhere**: Every numerical kernel uses AVX-512/AVX2/SSE/NEON with real hand-tuned intrinsics
4. **BLIS-style GEMM**: Panel packing + cache-blocked tiling for L1/L2/L3 efficiency (architecture-specific micro-kernels)
5. **Fused operations**: Combined softmax+cross-entropy eliminates intermediate buffers
6. **No abstraction tax**: Direct C structs + functions, no virtual dispatch
7. **Vulkan GPU compute**: Optional GPU offload for large tensor operations with automatic CPU fallback
8. **Dynamic graph**: Autograd graph and param groups grow dynamically (no static limits)
8. **Production error handling**: Error codes + callbacks replace raw asserts
9. **xoshiro128\*\* PRNG**: Passes BigCrush, fast, small state (vs weak LCG)

## Test Suite

75 tests across 25 sections:

- Tensor core, SIMD ops, matrix ops, activations
- Layers (Dense, Conv2D, MaxPool2D, BatchNorm, Dropout, Flatten)
- Loss functions (MSE, BCE, Cross-Entropy)
- Autograd backward (add, mul, matmul, relu, MSE, BCE)
- New tensor ops (sub, div, reshape — forward + backward)
- Autograd regression tests (scale backward, relu accumulation)
- Gradient clipping (L2 norm, value)
- LR schedulers (step, cosine, exponential)
- Layer backward passes (Dense, Flatten, Dropout)
- Conv2D / MaxPool / BatchNorm backward passes
- Autograd: sigmoid, tanh, softmax, mean backward
- Optimizers (SGD, Adam)
- Memory management (arena, checkpoint/restore, pool)
- Serialization (save/load roundtrip)
- PRNG statistical tests
- Integration tests (XOR training, arena checkpointing during training)
- Error handling and dynamic limits
- INT8 quantization (roundtrip accuracy, calibration, quantized dense, model size)

## License

MIT License.
