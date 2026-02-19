---
sidebar_position: 3
title: Autograd
---

# Autograd API

`#include <aicraft/autograd.h>`

Reverse-mode automatic differentiation engine with dynamic graph construction and 22 supported operation types.

## Core Functions

### `ac_backward`

```c
void ac_backward(ac_tensor* loss);
```

Compute gradients of `loss` with respect to all tensors with `requires_grad=1`.

1. Seeds `loss->grad = 1.0`
2. Topological sort via DFS
3. Reverse-order backward dispatch

### `ac_zero_grad`

```c
void ac_zero_grad(ac_param_group* params);
```

Zero all `.grad` buffers in the parameter group. **Must be called before each forward pass.**

## Supported Operations

The backward pass handles 22 operation types:

| Category | Operations |
|---|---|
| **Arithmetic** | ADD, SUB, MUL, DIV, MATMUL, SCALE, BIAS_ADD |
| **Reductions** | SUM, MEAN |
| **Activations** | RELU, SIGMOID, TANH, SOFTMAX |
| **Losses** | MSE, Cross-Entropy, BCE |
| **Layers** | FLATTEN, RESHAPE, DROPOUT, MAXPOOL, BATCHNORM, CONV2D |

## How It Works

### Forward Pass

Each operation records autograd metadata:

```c
ac_tensor* c = ac_tensor_add(a, b);
// c->op = AC_OP_ADD
// c->src[0] = a
// c->src[1] = b
// c->requires_grad = (a->requires_grad || b->requires_grad)
```

### Backward Pass

```c
ac_backward(loss);
// After this, all param->grad fields are populated
```

### O(1) Visited Check

```c
// Global epoch increments each backward call
extern int g_autograd_epoch;

// Each tensor stores last visited epoch
// visited = (tensor->autograd_epoch == g_autograd_epoch)
// No hash set or linear scan needed
```

## Gradient Computation Examples

### ADD backward
```
dA += dC
dB += dC
```

### MATMUL backward (C = A @ B)
```
dA += dC @ B^T
dB += A^T @ dC
```

### ReLU backward
```
dA += dC * (A > 0)   // SIMD masked move
```

### Cross-Entropy backward
```
dLogits = softmax(logits) - labels   // Fused softmax+CE gradient
```

## Parameter Groups

```c
ac_param_group params;
ac_param_group_init(&params);             // Initialize (dynamic array)
ac_param_group_add(&params, tensor);       // Add a parameter
ac_zero_grad(&params);                     // Zero all gradients
ac_param_group_destroy(&params);           // Free
```

The parameter group grows dynamically — no static limit.
