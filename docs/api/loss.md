---
sidebar_position: 5
title: Loss API
---

# Loss API

`#include "aicraft/loss.h"`

## Cross-Entropy

```c
AcTensor *ac_cross_entropy(AcTensor *pred, AcTensor *target);
```

Standard cross-entropy loss for classification. Expects `pred` to be softmax probabilities and `target` to be one-hot encoded.

## MSE (Mean Squared Error)

```c
AcTensor *ac_mse(AcTensor *pred, AcTensor *target);
```

Mean squared error for regression tasks.

```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

## Huber Loss

```c
AcTensor *ac_huber(AcTensor *pred, AcTensor *target, float delta);
```

Smooth L1 loss — less sensitive to outliers than MSE.

```
L_δ(a) = ½a²              if |a| ≤ δ
       = δ(|a| - ½δ)       otherwise
```

## Usage Example

```c
AcTensor *pred = ac_forward_seq(net, 2, x);
AcTensor *loss = ac_cross_entropy(pred, target);

float loss_val = ac_scalar(loss);
printf("Loss: %.4f\n", loss_val);

ac_backward(loss);  // Compute gradients
```

All loss functions return a scalar `AcTensor` and are differentiable through the autograd engine.
