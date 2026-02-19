---
sidebar_position: 2
title: Getting Started
---

# Getting Started

Build and run Aicraft in under 5 minutes. All you need is a C99 compiler and CMake.

## Prerequisites

- **C Compiler**: GCC 7+, Clang 6+, or MSVC 2019+
- **CMake**: 3.15+
- **Optional**: Vulkan SDK (for GPU acceleration)

## Build

### Release Build

```bash
git clone https://github.com/your-github-username/Aicraft.git
cd Aicraft
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

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

## CMake Options

| Option | Default | Description |
|---|---|---|
| `AICRAFT_BUILD_TESTS` | `ON` | Build test suite |
| `AICRAFT_BUILD_BENCH` | `ON` | Build benchmarks |
| `AICRAFT_BUILD_DEMO` | `ON` | Build demo |
| `AICRAFT_ENABLE_AVX512` | `OFF` | Enable AVX-512 (requires CPU support) |
| `AICRAFT_ENABLE_VULKAN` | `OFF` | Enable Vulkan compute backend |

## Run

```bash
# Run tests (75 tests across 25 sections)
./aicraft_test

# Run XOR demo
./aicraft_demo

# Run benchmarks
./aicraft_bench
```

## Quick Start: Inference

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

## Quick Start: Training

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
    
    // Training loop with arena checkpointing
    for (int epoch = 0; epoch < 100; epoch++) {
        ac_arena_checkpoint cp;
        ac_arena_save(&g_tensor_arena, &cp);
        
        ac_zero_grad(&params);
        
        ac_tensor* h = ac_dense_forward(&fc1, input);
        h = ac_tensor_relu(h);
        ac_tensor* logits = ac_dense_forward(&fc2, h);
        
        ac_tensor* loss = ac_cross_entropy_loss(logits, labels);
        ac_backward(loss);
        ac_clip_grad_norm(&params, 1.0f);
        ac_adam_step(&opt);
        
        ac_arena_restore(&g_tensor_arena, &cp);
    }
    
    ac_model_save("model.acml", &params);
    ac_param_group_destroy(&params);
    ac_cleanup();
    return 0;
}
```

## Integration

Aicraft is header-only. To use it in your project:

1. Copy the `include/aicraft/` directory to your project
2. Add `#include <aicraft/aicraft.h>` to your source files
3. Compile `src/core.c` (defines global state)
4. Optionally compile `src/vulkan.c` with `-DAICRAFT_ENABLE_VULKAN`

Or use CMake:

```cmake
add_subdirectory(Aicraft)
target_link_libraries(your_app PRIVATE aicraft)
```

## Next Steps

- [Architecture](./architecture) — System design overview
- [Training Guide](./guides/training) — In-depth training walkthrough
- [Edge Deployment](./guides/edge-deployment) — INT8 quantization for embedded
- [API Reference](./api/overview) — Complete function documentation
