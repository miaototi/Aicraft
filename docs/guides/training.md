---
sidebar_position: 1
title: Training Guide
---

# Training Guide

How to train a neural network with Aicraft.

## Overview

Aicraft provides a complete training pipeline: forward pass, loss computation, backward pass (autograd), and optimiser step.

## Basic Training Loop

```c
#include "aicraft/aicraft.h"

int main(void) {
    ac_init();

    // Define network
    AcLayer *net[] = {
        ac_dense(784, 128, AC_RELU),
        ac_dense(128, 10,  AC_SOFTMAX)
    };

    // Create optimiser
    AcOptimizer *opt = ac_adam(net, 2, 0.001f);

    // Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        float total_loss = 0;

        for (int i = 0; i < num_samples; i++) {
            ac_mem_checkpoint();

            AcTensor *x = get_input(i);
            AcTensor *target = get_label(i);

            // Forward
            AcTensor *pred = ac_forward_seq(net, 2, x);

            // Loss
            AcTensor *loss = ac_cross_entropy(pred, target);
            total_loss += ac_scalar(loss);

            // Backward
            ac_backward(loss);

            // Update weights
            ac_optimizer_step(opt);

            ac_mem_restore();
        }

        printf("Epoch %d — loss: %.4f\n", epoch, total_loss / num_samples);
    }

    ac_cleanup();
    return 0;
}
```

## Optimisers

| Optimiser | Function | Key Parameters |
|-----------|----------|----------------|
| SGD | `ac_sgd()` | learning rate, momentum |
| Adam | `ac_adam()` | lr, beta1, beta2, epsilon |
| AdamW | `ac_adamw()` | lr, beta1, beta2, weight decay |

## Loss Functions

| Loss | Function | Use Case |
|------|----------|----------|
| Cross-Entropy | `ac_cross_entropy()` | Classification |
| MSE | `ac_mse()` | Regression |
| Huber | `ac_huber()` | Robust regression |

## Tips

- Use `ac_mem_checkpoint()` / `ac_mem_restore()` inside the inner loop to keep memory constant
- Compile with `-O3` and SIMD flags for maximum throughput
- Add `-lvulkan` to offload heavy operations to the GPU
