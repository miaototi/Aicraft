---
sidebar_position: 2
title: Tensor
---

# Tensor API

`#include <aicraft/tensor.h>` and `#include <aicraft/tensor_ops.h>`

## Structures

### `ac_shape`

```c
typedef struct {
    ac_size dims[AC_MAX_DIMS];  // Dimension sizes (max 8D)
    int ndim;                    // Number of dimensions
    ac_size total_size;          // Product of all dims
} ac_shape;
```

### `ac_tensor`

```c
typedef struct ac_tensor {
    float* data;              // SIMD-aligned float array
    ac_shape shape;           // Tensor dimensions
    float* grad;              // Gradient buffer (NULL if not tracking)
    int requires_grad;        // Whether to track gradients
    ac_op op;                 // Operation that created this tensor
    struct ac_tensor* src[2]; // Input tensors (for autograd)
    void* cache;              // Cached data for backward pass
    int autograd_epoch;       // Epoch for O(1) visited check
} ac_tensor;
```

## Creation

```c
// Create N-dimensional tensor (up to 8D)
ac_tensor* ac_tensor_alloc(ac_shape shape, int requires_grad);

// Convenience constructors
ac_tensor* ac_tensor_1d(ac_size n, int requires_grad);
ac_tensor* ac_tensor_2d(ac_size rows, ac_size cols, int requires_grad);
ac_tensor* ac_tensor_3d(ac_size d0, ac_size d1, ac_size d2, int requires_grad);
ac_tensor* ac_tensor_4d(ac_size d0, ac_size d1, ac_size d2, ac_size d3, int requires_grad);
```

## Initialization

```c
ac_tensor_fill(t, 0.0f);                  // Fill with constant
ac_tensor_uniform(t, -1.0f, 1.0f);        // Uniform random
ac_tensor_xavier(t, fan_in, fan_out);      // Xavier/Glorot init
ac_tensor_he(t, fan_in);                   // He/Kaiming init
```

## Operations

### Arithmetic

```c
ac_tensor* ac_tensor_add(ac_tensor* a, ac_tensor* b);      // a + b
ac_tensor* ac_tensor_sub(ac_tensor* a, ac_tensor* b);      // a - b
ac_tensor* ac_tensor_mul(ac_tensor* a, ac_tensor* b);      // a * b (element-wise)
ac_tensor* ac_tensor_div(ac_tensor* a, ac_tensor* b);      // a / b
ac_tensor* ac_tensor_scale(ac_tensor* a, float s);          // a * s
ac_tensor* ac_tensor_matmul(ac_tensor* a, ac_tensor* b);   // a @ b (matrix multiply)
```

### Reductions

```c
ac_tensor* ac_tensor_sum(ac_tensor* a);    // Sum all elements → scalar
ac_tensor* ac_tensor_mean(ac_tensor* a);   // Mean of all elements → scalar
```

### Activations

```c
ac_tensor* ac_tensor_relu(ac_tensor* a);       // max(0, x)
ac_tensor* ac_tensor_sigmoid(ac_tensor* a);    // 1 / (1 + exp(-x))
ac_tensor* ac_tensor_tanh_act(ac_tensor* a);   // tanh(x)
ac_tensor* ac_tensor_softmax(ac_tensor* a);    // exp(x) / sum(exp(x))
```

### Shape

```c
ac_tensor* ac_tensor_reshape(ac_tensor* a, ac_shape new_shape);
// Reshape preserving data (total_size must match)
```

## Utilities

```c
ac_tensor_print(t, "label");          // Pretty-print tensor
ac_shape_eq(a->shape, b->shape);       // Check shape equality
```

## PRNG

```c
// Global xoshiro128** PRNG
float ac_random_float(float min, float max);
uint32_t ac_xoshiro128ss(ac_rng* rng);
```
