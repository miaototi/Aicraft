---
sidebar_position: 6
title: Debugging Guide
---

# Debugging Guide

How to diagnose and fix common issues in Aicraft.

## Tensor Inspection

### Print Tensor Contents

```c
AcTensor *t = ac_tensor_rand((int[]){2, 3}, 2);
ac_tensor_print(t);
```

Output:
```
Tensor [2, 3]:
[[0.4521, 0.1234, 0.8901],
 [0.2345, 0.6789, 0.0123]]
```

### Print Shape Only

```c
ac_tensor_print_shape(t);
// Output: [2, 3]
```

### Get Statistics

```c
float min, max, mean, std;
ac_tensor_stats(t, &min, &max, &mean, &std);
printf("min=%.4f max=%.4f mean=%.4f std=%.4f\n", min, max, mean, std);
```

---

## NaN / Inf Detection

### Check a Single Tensor

```c
if (ac_tensor_has_nan(tensor)) {
    printf("WARNING: NaN detected!\n");
    ac_tensor_print(tensor);
}

if (ac_tensor_has_inf(tensor)) {
    printf("WARNING: Inf detected!\n");
}
```

### Enable Global NaN Detection

Add this at the start of your program:

```c
ac_debug_enable_nan_check(true);
```

Aicraft will now throw an error (with stack location) whenever a NaN is produced.

---

## Common Issues

### 1. NaN During Training

**Symptoms**: Loss becomes `nan` after a few iterations.

**Causes**:
- Learning rate too high
- Exploding gradients
- Log of zero (cross-entropy with hard zeros)
- Division by zero in normalisation

**Solutions**:

```c
// Lower learning rate
AcOptimizer *opt = ac_adam(net, n, 0.0001f);  // instead of 0.01

// Enable gradient clipping
ac_optimizer_set_clip_norm(opt, 1.0f);

// Add epsilon to prevent log(0)
// Aicraft's cross_entropy already does this internally
```

### 2. Loss Not Decreasing

**Symptoms**: Loss stays constant or oscillates wildly.

**Causes**:
- Optimizer not updating (forgot `ac_backward` or `ac_optimizer_step`)
- Labels and predictions misaligned
- Wrong loss function for the task

**Debug checklist**:

```c
// 1. Verify backward was called
AcTensor *loss = ac_cross_entropy(pred, target);
printf("Loss: %f\n", ac_scalar(loss));
ac_backward(loss);

// 2. Check gradients exist
AcTensor *weights = net[0]->weights;
if (weights->grad_node && weights->grad_node->grad) {
    ac_tensor_print(weights->grad_node->grad);
} else {
    printf("ERROR: No gradients computed!\n");
}

// 3. Verify optimizer updates
printf("Weight before: %f\n", weights->data[0]);
ac_optimizer_step(opt);
printf("Weight after: %f\n", weights->data[0]);
```

### 3. Out of Memory

**Symptoms**: Program crashes or slows down over time.

**Cause**: Tensors accumulating without being freed.

**Solution**: Use checkpoint/restore:

```c
for (int i = 0; i < num_batches; i++) {
    ac_mem_checkpoint();  // ← Mark position

    // ... forward, backward, step ...

    ac_mem_restore();     // ← Free everything since checkpoint
}
```

### 4. Wrong Predictions (Model Works but Accuracy is Bad)

**Checklist**:

1. **Data normalisation**: Are inputs in [0, 1] or [-1, 1]?
2. **Label encoding**: For classification, are you using one-hot or integer labels correctly?
3. **Shuffle data**: Training on sorted data hurts generalisation
4. **Activation function**: Softmax for multi-class, Sigmoid for binary

```c
// Check prediction distribution
ac_tensor_print(pred);  // Should be a probability distribution

// Verify argmax
int predicted_class = ac_argmax(pred);
int true_class = labels[i];
printf("Predicted: %d, True: %d\n", predicted_class, true_class);
```

---

## Gradient Debugging

### Print Gradient Flow

```c
// After backward pass
void print_grads(AcLayer **net, int n) {
    for (int i = 0; i < n; i++) {
        AcTensor *w = net[i]->weights;
        if (w->grad_node && w->grad_node->grad) {
            printf("Layer %d grad norm: %f\n", i,
                   ac_tensor_norm(w->grad_node->grad));
        }
    }
}
```

### Gradient Check (Numerical vs Analytical)

For debugging custom ops:

```c
bool ac_gradient_check(AcTensor *input, AcTensor *(*forward)(AcTensor*),
                       float epsilon, float tolerance);

// Example
bool ok = ac_gradient_check(x, my_forward_fn, 1e-5, 1e-4);
if (!ok) {
    printf("Gradient check FAILED!\n");
}
```

---

## Logging

### Enable Verbose Mode

```c
ac_set_log_level(AC_LOG_DEBUG);
```

Levels:
- `AC_LOG_NONE` — Silent
- `AC_LOG_ERROR` — Errors only
- `AC_LOG_WARN` — Warnings + errors (default)
- `AC_LOG_INFO` — General info
- `AC_LOG_DEBUG` — Everything

### Custom Log Handler

```c
void my_logger(AcLogLevel level, const char *msg) {
    fprintf(stderr, "[AICRAFT] %s\n", msg);
}

ac_set_log_handler(my_logger);
```

---

## Vulkan Debugging

### Check Vulkan Availability

```c
if (ac_vulkan_available()) {
    printf("Vulkan GPU: %s\n", ac_vulkan_device_name());
} else {
    printf("Vulkan not available, using CPU\n");
}
```

### Enable Vulkan Validation Layers

Set environment variable before running:

```bash
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
./my_program
```

This will print detailed Vulkan errors.

---

## Performance Profiling

See [Performance Tuning](/docs/guides/performance-tuning) for optimisation tips.

```c
// Quick timer
clock_t start = clock();
// ... operation ...
double ms = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
printf("Took %.2f ms\n", ms);
```
