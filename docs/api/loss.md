---
sidebar_position: 5
title: Loss Functions
---

# Loss Functions API

`#include <aicraft/loss.h>`

## Cross-Entropy Loss

```c
ac_tensor* loss = ac_cross_entropy_loss(logits, labels);
// logits: [batch, num_classes] — raw scores (pre-softmax)
// labels: [batch, num_classes] — one-hot encoded targets
// Returns: scalar tensor (mean loss over batch)
```

**Implementation**: Fused softmax + cross-entropy. The gradient is computed as:

> `∇_logits = softmax(logits) - labels`

This fused formulation is numerically stable and avoids computing `log(softmax)` separately.

## Mean Squared Error

```c
ac_tensor* loss = ac_mse_loss(predictions, targets);
// predictions, targets: same shape
// Returns: scalar tensor = mean((pred - target)^2)
```

**Gradient**:

> `∇_pred = (2/N) * (pred - target)`

SIMD-vectorized using FMA operations.

## Binary Cross-Entropy

```c
ac_tensor* loss = ac_bce_loss(sigmoid_output, labels);
// sigmoid_output: [batch, 1] — values in (0, 1)
// labels: [batch, 1] — binary labels (0 or 1)
// Returns: scalar tensor
```

**Gradient**:

> `∇ = (ŷ - y) / (ŷ * (1 - ŷ))`

Clamped to prevent division by zero at the boundaries.

## Usage in Training

```c
// Classification
ac_tensor* logits = ac_dense_forward(&fc, features);
ac_tensor* loss = ac_cross_entropy_loss(logits, one_hot_labels);
ac_backward(loss);

// Regression
ac_tensor* pred = ac_dense_forward(&fc, features);
ac_tensor* loss = ac_mse_loss(pred, targets);
ac_backward(loss);

// Binary classification
ac_tensor* raw = ac_dense_forward(&fc, features);
ac_tensor* prob = ac_tensor_sigmoid(raw);
ac_tensor* loss = ac_bce_loss(prob, binary_labels);
ac_backward(loss);
```
