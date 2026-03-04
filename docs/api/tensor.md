---
sidebar_position: 2
title: Tensor API
---

# Tensor API

`#include "aicraft/tensor.h"`

## Types

### AcTensor

```c
typedef struct AcTensor {
    float *data;       // Raw data pointer
    int   *shape;      // Shape array
    int   *strides;    // Stride array
    int    ndim;       // Number of dimensions
    int    size;       // Total number of elements
    bool   requires_grad;
    struct AcGradNode *grad_node;  // Autograd node
} AcTensor;
```

## Creation

### ac_tensor_new

```c
AcTensor *ac_tensor_new(int *shape, int ndim);
```

Create a zero-initialised tensor with the given shape.

### ac_tensor_rand

```c
AcTensor *ac_tensor_rand(int *shape, int ndim);
```

Create a tensor filled with uniform random values in [0, 1).

### ac_tensor_ones / ac_tensor_zeros

```c
AcTensor *ac_tensor_ones(int *shape, int ndim);
AcTensor *ac_tensor_zeros(int *shape, int ndim);
```

### ac_tensor_from_data

```c
AcTensor *ac_tensor_from_data(float *data, int *shape, int ndim);
```

Create a tensor from existing data (copies the data).

## Operations

### ac_tensor_add / ac_tensor_mul

```c
AcTensor *ac_tensor_add(AcTensor *a, AcTensor *b);
AcTensor *ac_tensor_mul(AcTensor *a, AcTensor *b);
```

Element-wise addition and multiplication with broadcasting.

### ac_tensor_matmul

```c
AcTensor *ac_tensor_matmul(AcTensor *a, AcTensor *b);
```

Matrix multiplication. Dispatches to SIMD or Vulkan backend.

### ac_tensor_transpose

```c
AcTensor *ac_tensor_transpose(AcTensor *t);
```

### ac_scalar

```c
float ac_scalar(AcTensor *t);
```

Extract a scalar value from a 1-element tensor.

## Memory

Tensors are allocated from the arena. Use `ac_mem_checkpoint()` / `ac_mem_restore()` for bulk deallocation.
