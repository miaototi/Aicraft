---
sidebar_position: 6
title: Optimizer
---

# Optimizer API

`#include <aicraft/optimizer.h>`

All optimizer step functions are **SIMD-accelerated** with AVX2/NEON vectorized loops.

## Adam / AdamW

```c
typedef struct {
    ac_param_group* params;
    float lr, beta1, beta2, eps, weight_decay;
    int t;                     // Timestep counter
    float* m;                  // First moment (per-param)
    float* v;                  // Second moment (per-param)
} ac_adam;
```

```c
ac_adam opt;
ac_adam_init(&opt, &params,
    0.001f,   // learning rate
    0.9f,     // beta1 (first moment decay)
    0.999f,   // beta2 (second moment decay)
    1e-8f,    // epsilon
    0.0f,     // weight decay (0 = Adam, >0 = AdamW)
    0         // initial timestep
);

// In training loop:
ac_adam_step(&opt);
```

When `weight_decay > 0`, applies **decoupled weight decay** (AdamW):

> `θ(t+1) = θ(t) - η * (m̂(t) / (√v̂(t) + ε) + λ * θ(t))`

## SGD

```c
typedef struct {
    ac_param_group* params;
    float lr, momentum, weight_decay;
    float* velocity;           // Momentum buffers
} ac_sgd;
```

```c
ac_sgd opt;
ac_sgd_init(&opt, &params,
    0.01f,    // learning rate
    0.9f,     // momentum
    0.0001f   // weight decay
);

// In training loop:
ac_sgd_step(&opt);
```

## Gradient Clipping

### Global L2-norm clipping

```c
ac_clip_grad_norm(&params, 1.0f);
// Computes total L2 norm across all params
// If norm > max_norm, scales all gradients by (max_norm / norm)
```

### Per-element value clipping

```c
ac_clip_grad_value(&params, 0.5f);
// Clamps each gradient element to [-value, +value]
```

## Learning Rate Schedulers

```c
typedef struct {
    ac_lr_type type;
    float base_lr, gamma, min_lr;
    int step_size, total_steps;
} ac_lr_scheduler;
```

### Step Decay

```c
ac_lr_scheduler sched;
ac_lr_scheduler_init(&sched, AC_LR_STEP,
    0.01f,    // base_lr
    30,       // step_size (decay every 30 epochs)
    0.1f,     // gamma (multiply by 0.1)
    0, 0
);

float lr = ac_lr_scheduler_get(&sched, epoch);
// lr = base_lr * gamma^(epoch / step_size)
```

### Cosine Annealing

```c
ac_lr_scheduler_init(&sched, AC_LR_COSINE,
    0.01f,    // base_lr
    0, 0,
    200,      // total_steps
    0.0001f   // min_lr
);
// lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * epoch / total))
```

### Exponential Decay

```c
ac_lr_scheduler_init(&sched, AC_LR_EXPONENTIAL,
    0.01f,    // base_lr
    0,
    0.95f,    // gamma (multiply each epoch)
    0, 0
);
// lr = base_lr * gamma^epoch
```

### Using with Optimizer

```c
for (int epoch = 0; epoch < num_epochs; epoch++) {
    float lr = ac_lr_scheduler_get(&sched, epoch);
    opt.lr = lr;  // Update optimizer's learning rate
    
    // ... training step ...
    ac_adam_step(&opt);
}
```
