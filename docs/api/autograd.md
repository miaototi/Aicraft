---
sidebar_position: 3
title: Autograd API
---

# Autograd API

`#include "aicraft/autograd.h"`

Reverse-mode automatic differentiation with 22 differentiable operations.

## Core Function

### ac_backward

```c
void ac_backward(AcTensor *loss);
```

Compute gradients of `loss` with respect to all tensors with `requires_grad = true`. Performs reverse-mode autodiff through the computational graph.

## How It Works

1. Each operation records itself in a DAG (directed acyclic graph)
2. `ac_backward()` topologically sorts the graph
3. Gradients are propagated backwards through each node
4. O(1) cycle detection prevents infinite loops

## Gradient Access

```c
AcTensor *x = ac_tensor_rand((int[]){1, 784}, 2);
x->requires_grad = true;

AcTensor *y = ac_forward_seq(net, 2, x);
ac_backward(y);

// Access gradient
AcTensor *grad = ac_grad(x);
```

## Supported Operations

| Op | Forward | Backward |
|----|---------|----------|
| Add | `a + b` | `∂L/∂a = 1, ∂L/∂b = 1` |
| Mul | `a * b` | `∂L/∂a = b, ∂L/∂b = a` |
| MatMul | `a @ b` | `∂L/∂a = ∂L/∂y @ bᵀ` |
| ReLU | `max(0, x)` | `∂L/∂x · (x > 0)` |
| Sigmoid | `σ(x)` | `∂L/∂x · σ(x)(1 - σ(x))` |
| Softmax | `softmax(x)` | Jacobian-vector product |
| Sum | `Σx` | `∂L/∂x = 1` |
| Mean | `mean(x)` | `∂L/∂x = 1/n` |
| Exp | `eˣ` | `∂L/∂x · eˣ` |
| Log | `ln(x)` | `∂L/∂x · 1/x` |
| Neg | `-x` | `∂L/∂x = -1` |
| Reshape | `reshape(x)` | `reshape(∂L/∂y)` |
| Transpose | `xᵀ` | `(∂L/∂y)ᵀ` |

## No-Grad Context

```c
ac_no_grad_begin();
// Operations here won't be tracked
AcTensor *pred = ac_forward_seq(net, 2, x);
ac_no_grad_end();
```
