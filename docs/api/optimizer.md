---
sidebar_position: 6
title: Optimizer API
---

# Optimizer API

`#include "aicraft/optimizer.h"`

## SGD

```c
AcOptimizer *ac_sgd(AcLayer **layers, int n, float lr);
```

Stochastic Gradient Descent with optional momentum.

```c
AcOptimizer *opt = ac_sgd(net, 2, 0.01f);
opt->momentum = 0.9f;  // Optional
```

## Adam

```c
AcOptimizer *ac_adam(AcLayer **layers, int n, float lr);
```

Adam optimiser with default β₁=0.9, β₂=0.999, ε=1e-8.

```c
AcOptimizer *opt = ac_adam(net, 2, 0.001f);
```

## AdamW

```c
AcOptimizer *ac_adamw(AcLayer **layers, int n, float lr, float wd);
```

Adam with decoupled weight decay.

```c
AcOptimizer *opt = ac_adamw(net, 2, 0.001f, 0.01f);
```

## Optimiser Step

```c
void ac_optimizer_step(AcOptimizer *opt);
```

Update all layer weights using computed gradients.

## Zero Gradients

```c
void ac_optimizer_zero_grad(AcOptimizer *opt);
```

Reset all gradients to zero before the next backward pass.

## Full Example

```c
AcOptimizer *opt = ac_adam(net, 2, 0.001f);

for (int i = 0; i < epochs; i++) {
    AcTensor *pred = ac_forward_seq(net, 2, x);
    AcTensor *loss = ac_cross_entropy(pred, target);
    ac_backward(loss);
    ac_optimizer_step(opt);
    ac_optimizer_zero_grad(opt);
}
```
