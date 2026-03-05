---
sidebar_position: 1
title: Autograd Internals
---

# Autograd Internals

How automatic differentiation works in Aicraft.

## Overview

Aicraft implements **reverse-mode automatic differentiation** (backpropagation) using a **dynamic computational graph**. Every operation that produces a tensor can optionally record its computation, allowing gradients to flow backward.

```
Forward:  x → [Linear] → [ReLU] → [Linear] → [Softmax] → loss
Backward: ∂loss/∂x ← ∂/∂z₃ ← ∂/∂z₂ ← ∂/∂z₁ ← ∂/∂y ← 1.0
```

---

## Computational Graph

### Nodes

Each tensor with `requires_grad = true` has an associated `AcGradNode`:

```c
typedef struct AcGradNode {
    AcTensor *tensor;           // The tensor this node belongs to
    AcTensor *grad;             // Accumulated gradient
    
    AcTensor *inputs[AC_MAX_INPUTS];  // Input tensors
    int num_inputs;
    
    void (*backward)(struct AcGradNode *node, AcTensor *grad_output);
    void *context;              // Operation-specific data
    
    bool visited;               // For topological sort
} AcGradNode;
```

### Building the Graph

The graph is built dynamically during the forward pass:

```c
// When you compute: c = a + b
AcTensor *ac_add(AcTensor *a, AcTensor *b) {
    AcTensor *c = ac_tensor_new(a->shape, a->ndim);
    
    // Compute forward
    for (int i = 0; i < c->size; i++) {
        c->data[i] = a->data[i] + b->data[i];
    }
    
    // Build graph if needed
    if (a->requires_grad || b->requires_grad) {
        c->requires_grad = true;
        c->grad_node = ac_grad_node_new(c);
        c->grad_node->inputs[0] = a;
        c->grad_node->inputs[1] = b;
        c->grad_node->num_inputs = 2;
        c->grad_node->backward = add_backward;
    }
    
    return c;
}
```

---

## Backward Pass

### Topological Sort

Before computing gradients, we topologically sort the graph (output to input):

```c
void ac_backward(AcTensor *loss) {
    // Start with gradient of 1.0 w.r.t. loss
    loss->grad_node->grad = ac_tensor_ones(loss->shape, loss->ndim);
    
    // Collect all nodes in reverse order
    AcGradNode **sorted = topological_sort(loss->grad_node);
    
    // Backward through each node
    for (int i = 0; i < num_nodes; i++) {
        AcGradNode *node = sorted[i];
        if (node->backward && node->grad) {
            node->backward(node, node->grad);
        }
    }
}
```

### Topological Sort Algorithm

```c
static void topo_visit(AcGradNode *node, AcGradNode **stack, int *idx) {
    if (node->visited) return;
    node->visited = true;
    
    for (int i = 0; i < node->num_inputs; i++) {
        AcTensor *input = node->inputs[i];
        if (input && input->grad_node) {
            topo_visit(input->grad_node, stack, idx);
        }
    }
    
    stack[(*idx)++] = node;
}
```

---

## Backward Functions

Each operation defines how gradients flow through it.

### Addition

```c
// c = a + b
// ∂L/∂a = ∂L/∂c, ∂L/∂b = ∂L/∂c

static void add_backward(AcGradNode *node, AcTensor *grad_output) {
    AcTensor *a = node->inputs[0];
    AcTensor *b = node->inputs[1];
    
    if (a->grad_node) {
        ac_grad_accumulate(a->grad_node, grad_output);
    }
    if (b->grad_node) {
        ac_grad_accumulate(b->grad_node, grad_output);
    }
}
```

### Multiplication (Element-wise)

```c
// c = a * b
// ∂L/∂a = ∂L/∂c * b, ∂L/∂b = ∂L/∂c * a

static void mul_backward(AcGradNode *node, AcTensor *grad_output) {
    AcTensor *a = node->inputs[0];
    AcTensor *b = node->inputs[1];
    
    if (a->grad_node) {
        AcTensor *grad_a = ac_mul(grad_output, b);
        ac_grad_accumulate(a->grad_node, grad_a);
    }
    if (b->grad_node) {
        AcTensor *grad_b = ac_mul(grad_output, a);
        ac_grad_accumulate(b->grad_node, grad_b);
    }
}
```

### Matrix Multiplication

```c
// C = A @ B (A: [M, K], B: [K, N], C: [M, N])
// ∂L/∂A = ∂L/∂C @ Bᵀ
// ∂L/∂B = Aᵀ @ ∂L/∂C

static void matmul_backward(AcGradNode *node, AcTensor *grad_output) {
    AcTensor *A = node->inputs[0];
    AcTensor *B = node->inputs[1];
    
    if (A->grad_node) {
        AcTensor *B_T = ac_transpose(B);
        AcTensor *grad_A = ac_matmul(grad_output, B_T);
        ac_grad_accumulate(A->grad_node, grad_A);
    }
    if (B->grad_node) {
        AcTensor *A_T = ac_transpose(A);
        AcTensor *grad_B = ac_matmul(A_T, grad_output);
        ac_grad_accumulate(B->grad_node, grad_B);
    }
}
```

### ReLU

```c
// y = max(0, x)
// ∂L/∂x = ∂L/∂y * 1(x > 0)

static void relu_backward(AcGradNode *node, AcTensor *grad_output) {
    AcTensor *x = node->inputs[0];
    
    if (x->grad_node) {
        AcTensor *grad_x = ac_tensor_new(x->shape, x->ndim);
        for (int i = 0; i < x->size; i++) {
            grad_x->data[i] = (x->data[i] > 0) ? grad_output->data[i] : 0;
        }
        ac_grad_accumulate(x->grad_node, grad_x);
    }
}
```

### Softmax + Cross-Entropy

Combined for numerical stability:

```c
// L = -Σ yᵢ log(softmax(xᵢ))
// ∂L/∂x = softmax(x) - y

static void softmax_ce_backward(AcGradNode *node, AcTensor *grad_output) {
    AcTensor *logits = node->inputs[0];
    AcTensor *targets = node->inputs[1];  // One-hot
    AcTensor *softmax_out = (AcTensor *)node->context;
    
    if (logits->grad_node) {
        AcTensor *grad = ac_tensor_new(logits->shape, logits->ndim);
        for (int i = 0; i < logits->size; i++) {
            grad->data[i] = softmax_out->data[i] - targets->data[i];
        }
        // Scale by incoming gradient (usually 1.0 for scalar loss)
        ac_tensor_scale_inplace(grad, ac_scalar(grad_output));
        ac_grad_accumulate(logits->grad_node, grad);
    }
}
```

---

## Gradient Accumulation

Gradients are **accumulated** (summed), not replaced. This handles:

1. **Multiple uses** of the same tensor
2. **Batch dimension** gradients

```c
void ac_grad_accumulate(AcGradNode *node, AcTensor *grad) {
    if (!node->grad) {
        node->grad = ac_tensor_clone(grad);
    } else {
        for (int i = 0; i < node->grad->size; i++) {
            node->grad->data[i] += grad->data[i];
        }
    }
}
```

---

## Broadcasting in Backward

When forward involves broadcasting, backward must reduce:

```c
// Forward: C[M, N] = A[M, N] + B[1, N] (B is broadcast)
// Backward: ∂L/∂B = sum over M dimension of ∂L/∂C

static void add_broadcast_backward(AcGradNode *node, AcTensor *grad_output) {
    AcTensor *b = node->inputs[1];  // The broadcast operand
    
    if (b->grad_node) {
        // Sum along broadcast dimensions
        AcTensor *grad_b = ac_sum(grad_output, /*axis=*/0);
        ac_grad_accumulate(b->grad_node, grad_b);
    }
}
```

---

## Memory Management

### Gradient Cleanup

After `ac_optimizer_step`, gradients are zeroed:

```c
void ac_optimizer_zero_grad(AcOptimizer *opt) {
    for (int i = 0; i < opt->num_layers; i++) {
        AcTensor *w = opt->layers[i]->weights;
        if (w && w->grad_node && w->grad_node->grad) {
            ac_tensor_zero_inplace(w->grad_node->grad);
        }
        // Same for bias
    }
}
```

### Graph Cleanup

The computational graph is freed with `ac_mem_restore()`:

```c
ac_mem_checkpoint();
// Forward pass builds graph
AcTensor *loss = ...;
ac_backward(loss);
ac_optimizer_step(opt);
ac_mem_restore();  // Frees all tensors and grad nodes
```

---

## No-Grad Mode

Disable graph building for inference:

```c
// Option 1: Explicit flag
ac_set_grad_enabled(false);
AcTensor *y = ac_forward_seq(net, n, x);  // No graph built
ac_set_grad_enabled(true);

// Option 2: Use input without requires_grad
AcTensor *x = ac_tensor_from_data(data, shape, ndim);
x->requires_grad = false;  // Default
```

---

## Cycle Detection

Aicraft prevents infinite loops with O(1) cycle detection:

```c
// During backward, each node is visited exactly once
if (node->visited) {
    return;  // Already processed
}
node->visited = true;
```

---

## Supported Operations

The autograd engine supports 22 differentiable operations:

| Category | Operations |
|----------|------------|
| Arithmetic | add, sub, mul, div, neg |
| Matrix | matmul, transpose |
| Reductions | sum, mean, max |
| Activations | relu, sigmoid, tanh, softmax, gelu |
| Loss | mse, cross_entropy, bce |
| Shape | reshape, squeeze, unsqueeze |
| Other | log, exp, pow |

---

## Adding Custom Ops

See [Custom Layers Guide](/docs/guides/custom-layers) for implementing your own differentiable operations.

---

## Debugging

```c
// Print the computational graph
ac_print_graph(loss);

// Check specific gradients
ac_tensor_print(layer->weights->grad_node->grad);

// Numerical gradient check
ac_gradient_check(loss_fn, input, 1e-5, 1e-4);
```

---

## Performance Considerations

1. **In-place operations don't support autograd** — they break the graph
2. **Reuse checkpoints** — avoid graph explosion in training loops  
3. **Detach when needed** — `ac_tensor_detach(t)` creates a copy without grad connection

---

## Next Steps

- [Memory Internals](/docs/api/memory) — Arena allocator details
- [Custom Layers](/docs/guides/custom-layers) — Implement new layers
- [Debugging Guide](/docs/guides/debugging) — Troubleshoot autograd issues
