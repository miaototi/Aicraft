---
sidebar_position: 1
title: Training Guide
---

# Training Guide

Complete walkthrough for training neural networks with Aicraft, from model definition to checkpoint saving.

## Model Definition

```c
#include <aicraft/aicraft.h>

int main() {
    ac_init();

    // Define layers
    ac_dense fc1, fc2, fc3;
    ac_dense_init(&fc1, 784, 256);   // Input → Hidden 1
    ac_dense_init(&fc2, 256, 128);   // Hidden 1 → Hidden 2
    ac_dense_init(&fc3, 128, 10);    // Hidden 2 → Output
```

All layers use **He initialization** by default, which is optimal for ReLU activations.

## Parameter Registration

Before training, register all learnable parameters in a parameter group:

```c
    ac_param_group params;
    ac_param_group_init(&params);
    
    // Add weights and biases from each layer
    ac_param_group_add(&params, fc1.weight);
    ac_param_group_add(&params, fc1.bias);
    ac_param_group_add(&params, fc2.weight);
    ac_param_group_add(&params, fc2.bias);
    ac_param_group_add(&params, fc3.weight);
    ac_param_group_add(&params, fc3.bias);
```

The parameter group dynamically grows — no static limit on the number of parameters.

## Optimizers

### Adam (recommended)

```c
    ac_adam opt;
    ac_adam_init(&opt, &params,
        0.001f,   // learning rate
        0.9f,     // beta1
        0.999f,   // beta2
        1e-8f,    // epsilon
        0.0f,     // weight decay (0 = no decay)
        0         // timestep (0 = start fresh)
    );
```

### AdamW (with decoupled weight decay)

```c
    ac_adam opt;
    ac_adam_init(&opt, &params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, 0);
    // weight_decay=0.01 enables AdamW behavior
```

### SGD (with momentum)

```c
    ac_sgd opt;
    ac_sgd_init(&opt, &params,
        0.01f,    // learning rate
        0.9f,     // momentum
        0.0001f   // weight decay
    );
```

All optimizer step functions are **SIMD-accelerated** (AVX2/NEON vectorized parameter updates).

## Training Loop

```c
    for (int epoch = 0; epoch < 100; epoch++) {
        // Checkpoint arena — all intermediates will be freed at restore
        ac_arena_checkpoint cp;
        ac_arena_save(&g_tensor_arena, &cp);
        
        // Zero gradients
        ac_zero_grad(&params);
        
        // Forward pass
        ac_tensor* h = ac_dense_forward(&fc1, input);
        h = ac_tensor_relu(h);
        h = ac_dense_forward(&fc2, h);
        h = ac_tensor_relu(h);
        ac_tensor* logits = ac_dense_forward(&fc3, h);
        
        // Compute loss
        ac_tensor* loss = ac_cross_entropy_loss(logits, labels);
        
        // Backward pass
        ac_backward(loss);
        
        // Gradient clipping (prevents exploding gradients)
        ac_clip_grad_norm(&params, 1.0f);
        
        // Optimizer step
        ac_adam_step(&opt);
        
        // Free all intermediates (model params survive)
        ac_arena_restore(&g_tensor_arena, &cp);
        
        if (epoch % 10 == 0) {
            printf("Epoch %d, Loss: %.4f\n", epoch, loss->data[0]);
        }
    }
```

:::tip Arena Checkpointing
The `ac_arena_save` / `ac_arena_restore` pattern is critical: it prevents memory growth during training. All intermediate tensors (activations, gradients) are bulk-freed in O(1) each epoch. Model parameters are allocated *before* the checkpoint so they survive.
:::

## Loss Functions

### Cross-Entropy (classification)

```c
ac_tensor* loss = ac_cross_entropy_loss(logits, labels);
// logits: [batch, classes], labels: [batch, classes] (one-hot)
```

### Mean Squared Error (regression)

```c
ac_tensor* loss = ac_mse_loss(predictions, targets);
```

### Binary Cross-Entropy

```c
ac_tensor* loss = ac_bce_loss(sigmoid_output, binary_labels);
```

## Gradient Clipping

### Global L2-norm clipping

```c
ac_clip_grad_norm(&params, 1.0f);  // clip to max norm of 1.0
```

### Per-element value clipping

```c
ac_clip_grad_value(&params, 0.5f);  // clip each grad element to [-0.5, 0.5]
```

## Learning Rate Schedulers

```c
ac_lr_scheduler sched;

// Step decay: lr *= gamma every step_size epochs
ac_lr_scheduler_init(&sched, AC_LR_STEP, base_lr, step_size, gamma, 0, 0);

// Cosine annealing: lr follows cosine curve
ac_lr_scheduler_init(&sched, AC_LR_COSINE, base_lr, 0, 0, total_epochs, min_lr);

// Exponential decay: lr *= gamma every epoch
ac_lr_scheduler_init(&sched, AC_LR_EXPONENTIAL, base_lr, 0, gamma, 0, 0);

// In training loop:
float lr = ac_lr_scheduler_get(&sched, epoch);
opt.lr = lr;  // update optimizer
```

## Saving and Loading

```c
    // Save trained model
    ac_model_save("model.acml", &params);
    
    // Load model
    ac_error_code err = ac_model_load("model.acml", &params);
    if (err != AC_OK) {
        printf("Load failed: %s\n", ac_get_last_error_message());
    }
```

The `.acml` format uses a versioned binary format with a magic header for corruption detection.

## Convolutional Networks

```c
    ac_conv2d conv1;
    ac_conv2d_init(&conv1, 
        1,    // in_channels
        32,   // out_channels
        3,    // kernel_size
        1,    // stride
        1     // padding
    );
    
    ac_maxpool2d pool1;
    ac_maxpool2d_init(&pool1, 2, 2);  // 2×2 pool, stride 2
    
    ac_batchnorm bn1;
    ac_batchnorm_init(&bn1, 32);  // 32 channels
    
    // Forward
    ac_tensor* x = ac_conv2d_forward(&conv1, input);    // im2col + GEMM
    x = ac_tensor_relu(x);
    x = ac_maxpool2d_forward(&pool1, x);
    x = ac_batchnorm_forward(&bn1, x, 1);               // 1 = training mode
    x = ac_flatten_forward(x);                            // zero-copy view
    x = ac_dense_forward(&fc1, x);
```

## Cleanup

```c
    ac_param_group_destroy(&params);
    ac_cleanup();
    return 0;
}
```
