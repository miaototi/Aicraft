---
sidebar_position: 11
title: Activations API
---

# Activations API

`#include "aicraft/activations.h"`

Activation functions add non-linearity to neural networks, enabling them to learn complex patterns.

---

## Overview

| Activation | Formula | Range | Use Case |
|------------|---------|-------|----------|
| ReLU | max(0, x) | [0, ∞) | Hidden layers (default) |
| LeakyReLU | max(αx, x) | (-∞, ∞) | Prevents dying neurons |
| Sigmoid | 1 / (1 + e^-x) | (0, 1) | Binary classification |
| Tanh | (e^x - e^-x) / (e^x + e^-x) | (-1, 1) | Hidden layers, RNNs |
| Softmax | e^xi / Σe^xj | (0, 1) | Multi-class output |
| GELU | x · Φ(x) | (-∞, ∞) | Transformers |
| SiLU/Swish | x · σ(x) | (-∞, ∞) | Modern architectures |

---

## ReLU

**Rectified Linear Unit** — the most common activation.

```c
AcTensor *ac_relu(AcTensor *x);
```

**Formula:** `ReLU(x) = max(0, x)`

### Example

```c
AcTensor *x = ac_tensor_from_data(
    (float[]){-2, -1, 0, 1, 2}, (int[]){5}, 1
);
AcTensor *y = ac_relu(x);
// y = [0, 0, 0, 1, 2]
```

### Properties

- **Pros**: Fast, sparse activations, no vanishing gradient for positive values
- **Cons**: "Dying ReLU" problem — neurons can get stuck at 0
- **Derivative**: 1 if x &gt; 0, else 0

---

## Leaky ReLU

Prevents dying neurons by allowing small negative values.

```c
AcTensor *ac_leaky_relu(AcTensor *x, float alpha);
```

**Formula:** `LeakyReLU(x) = max(αx, x)`

Where α is typically 0.01. If x &gt; 0, output is x. Otherwise, output is αx.

### Example

```c
AcTensor *y = ac_leaky_relu(x, 0.01f);
// For x = [-2, -1, 0, 1, 2]
// y = [-0.02, -0.01, 0, 1, 2]
```

### Typical α values

- **0.01**: Standard (default)
- **0.2**: Aggressive leak
- **α learnable**: Parametric ReLU (PReLU)

---

## Sigmoid

Maps values to range (0, 1).

```c
AcTensor *ac_sigmoid(AcTensor *x);
```

**Formula:** `σ(x) = 1 / (1 + e^-x)`

### Example

```c
AcTensor *y = ac_sigmoid(x);
// For x = [-2, 0, 2]
// y ≈ [0.119, 0.5, 0.881]
```

### Use Cases

- Binary classification output layer
- Gates in LSTMs and GRUs
- Attention weights

### Numerical Stability

Aicraft uses a stable implementation:

```c
// For large negative x, use: 1 - sigmoid(-x)
if (x < 0) {
    float ex = expf(x);
    return ex / (1.0f + ex);
} else {
    return 1.0f / (1.0f + expf(-x));
}
```

---

## Tanh

Hyperbolic tangent — similar to sigmoid but outputs in (-1, 1).

```c
AcTensor *ac_tanh(AcTensor *x);
```

**Formula:** `tanh(x) = (e^x - e^-x) / (e^x + e^-x)`

### Example

```c
AcTensor *y = ac_tanh(x);
// For x = [-2, 0, 2]
// y ≈ [-0.964, 0, 0.964]
```

### Relationship to Sigmoid

`tanh(x) = 2σ(2x) - 1`

---

## Softmax

Converts logits to probability distribution.

```c
AcTensor *ac_softmax(AcTensor *x);  // Last axis
AcTensor *ac_softmax_axis(AcTensor *x, int axis);
```

**Formula:** `Softmax(xi) = e^xi / Σe^xj`

### Example

```c
AcTensor *logits = ac_tensor_from_data(
    (float[]){2.0, 1.0, 0.1}, (int[]){3}, 1
);
AcTensor *probs = ac_softmax(logits);
// probs ≈ [0.659, 0.242, 0.099]
// Sum = 1.0
```

### Numerical Stability

Aicraft subtracts the max before exponentiating:

```c
// Stable softmax: subtract max to prevent overflow
float max_val = ac_tensor_max(x);
for (int i = 0; i < x->size; i++) {
    exp_vals[i] = expf(x->data[i] - max_val);
}
```

---

## GELU

**Gaussian Error Linear Unit** — used in BERT, GPT, and other transformers.

```c
AcTensor *ac_gelu(AcTensor *x);
```

**Formula:** `GELU(x) = x · Φ(x) = x · 0.5 · (1 + erf(x / √2))`

### Approximation

Aicraft uses the fast tanh approximation:

`GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])`

### Example

```c
AcTensor *y = ac_gelu(x);
// Smoother than ReLU, handles negative values better
```

---

## SiLU / Swish

Self-gated activation — x multiplied by its sigmoid.

```c
AcTensor *ac_silu(AcTensor *x);
// Alias: ac_swish(x)
```

**Formula:** `SiLU(x) = x · σ(x) = x / (1 + e^-x)`

### Properties

- Smooth, non-monotonic
- Learnable β version: Swish-β = x · σ(βx)
- Used in EfficientNet, YOLOv5

---

## In-Place Operations

For memory efficiency:

```c
// In-place versions (modify tensor directly)
void ac_relu_inplace(AcTensor *x);
void ac_sigmoid_inplace(AcTensor *x);
void ac_tanh_inplace(AcTensor *x);
```

### Example

```c
AcTensor *x = ac_tensor_rand((int[]){1024}, 1);
ac_relu_inplace(x);  // No new tensor allocated
```

---

## Using with Layers

When creating dense layers, specify activation:

```c
// Built-in activation constants
AcLayer *l1 = ac_dense(784, 256, AC_RELU);
AcLayer *l2 = ac_dense(256, 128, AC_LEAKY_RELU);
AcLayer *l3 = ac_dense(128, 10,  AC_SOFTMAX);
```

### Available Constants

```c
AC_NONE        // No activation (linear)
AC_RELU        // ReLU
AC_LEAKY_RELU  // LeakyReLU with α=0.01
AC_SIGMOID     // Sigmoid
AC_TANH        // Tanh
AC_SOFTMAX     // Softmax
AC_GELU        // GELU
AC_SILU        // SiLU/Swish
```

---

## Custom Activations

Register your own activation function:

```c
// Define function and derivative
float my_activation(float x) {
    return x * x;  // Example: square
}

float my_activation_grad(float x, float output) {
    return 2 * x;
}

// Register
ac_register_activation(
    "square",
    my_activation,
    my_activation_grad
);

// Use
AcLayer *layer = ac_dense_with_activation(784, 256, "square");
```

---

## Choosing Activations

| Task | Recommended |
|------|-------------|
| Hidden layers (default) | ReLU |
| Deep networks | LeakyReLU / GELU |
| Binary classification | Sigmoid (output) |
| Multi-class classification | Softmax (output) |
| Regression | None (linear) |
| Transformers | GELU |
| Object detection | SiLU / LeakyReLU |

---

## Next Steps

- [Layers API](/docs/api/layers) — Layer implementations
- [Loss API](/docs/api/loss) — Loss functions
- [Training Guide](/docs/guides/training) — Full training pipeline
