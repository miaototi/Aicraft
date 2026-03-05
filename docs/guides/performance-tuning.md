---
sidebar_position: 7
title: Performance Tuning
---

# Performance Tuning

Get the most out of Aicraft with these optimisation techniques.

## Quick Checklist

| Optimisation | Speedup | Effort |
|--------------|---------|--------|
| `-O3` compiler flag | 3-10× | 🟢 Easy |
| SIMD flags (`-mavx2`) | 2-8× | 🟢 Easy |
| Batch size ≥ 32 | 2-4× | 🟢 Easy |
| Vulkan GPU | 10-100× | 🟡 Medium |
| Memory checkpoints | Constant memory | 🟢 Easy |
| INT8 quantisation | 2-4× inference | 🟡 Medium |

---

## Compiler Flags

### Essential Flags

```bash
# Always use
gcc -O3 -DNDEBUG your_code.c -I./include -o program

# -O3: Maximum optimisation
# -DNDEBUG: Disable assertions
```

### SIMD Flags

```bash
# x86 (Intel/AMD)
gcc -O3 -mavx2 ...             # Most modern CPUs (2013+)
gcc -O3 -mavx512f ...          # Intel Xeon, some i9

# ARM
gcc -O3 -mfpu=neon ...         # Raspberry Pi, older ARM
gcc -O3 -march=armv8-a ...     # Modern ARM (Apple M1 via Rosetta)

# Auto-detect best
gcc -O3 -march=native ...      # Use all features of current CPU
```

### Link-Time Optimisation

```bash
gcc -O3 -flto your_code.c -I./include -o program
```

LTO can provide 5-15% additional speedup.

---

## Batch Size

SIMD and GPU both benefit from larger batches:

| Batch Size | Relative Speed | Memory |
|------------|----------------|--------|
| 1 | 1× | Low |
| 8 | 1.5× | Low |
| 32 | 3× | Medium |
| 64 | 4× | Medium |
| 128 | 4.5× | High |
| 256 | 5× | High |

**Rule of thumb**: Use the largest batch size that fits in memory.

```c
#define BATCH_SIZE 64  // Good default

AcTensor *x = ac_tensor_from_data(batch_data, (int[]){BATCH_SIZE, 784}, 2);
```

---

## Memory Management

### Checkpoint/Restore

Prevents memory growth in training loops:

```c
for (int epoch = 0; epoch < EPOCHS; epoch++) {
    for (int batch = 0; batch < num_batches; batch++) {
        ac_mem_checkpoint();   // ← Mark position

        AcTensor *x = ...;
        AcTensor *pred = ac_forward_seq(net, n, x);
        AcTensor *loss = ac_cross_entropy(pred, target);
        ac_backward(loss);
        ac_optimizer_step(opt);

        ac_mem_restore();      // ← Free all tensors since checkpoint
    }
}
```

### Custom Arena Size

Default arena is 256 MB. For large models:

```c
// 1 GB arena
ac_init_with_arena(1024 * 1024 * 1024);
```

### Preallocate Tensors

For inference, reuse tensors:

```c
// Allocate once
AcTensor *input = ac_tensor_new((int[]){BATCH_SIZE, 784}, 2);
AcTensor *output = ac_tensor_new((int[]){BATCH_SIZE, 10}, 2);

// Reuse in loop
for (int i = 0; i < N; i++) {
    // Copy data into preallocated tensor
    memcpy(input->data, &data[i * 784], BATCH_SIZE * 784 * sizeof(float));
    
    // In-place forward (no allocation)
    ac_forward_seq_inplace(net, n, input, output);
}
```

---

## GPU Acceleration (Vulkan)

### Enable Vulkan

```c
ac_init();
if (ac_vulkan_available()) {
    ac_vulkan_init();
    printf("Using GPU: %s\n", ac_vulkan_device_name());
}
```

### GPU Selection

If you have multiple GPUs:

```c
int num_gpus = ac_vulkan_device_count();
for (int i = 0; i < num_gpus; i++) {
    printf("%d: %s\n", i, ac_vulkan_device_name_at(i));
}
ac_vulkan_select_device(0);  // Use first GPU
```

### When to Use GPU

| Operation | CPU (AVX2) | GPU (Vulkan) | Winner |
|-----------|------------|--------------|--------|
| Small GEMM (&lt;1K) | Fast | Slower (overhead) | CPU |
| Large GEMM (&gt;4K) | Slow | Very fast | GPU |
| Element-wise ops | Fast | Fast | Tie |
| Data transfer | N/A | Slow | CPU for small data |

**Rule**: GPU wins for batch ≥ 64 and layer width ≥ 256.

### Minimise Transfers

```c
// BAD: Transfer every iteration
for (int i = 0; i < N; i++) {
    AcTensor *x = ac_tensor_to_gpu(host_tensor);  // Slow!
    AcTensor *y = ac_forward(net, x);
    ac_tensor_to_cpu(y);  // Slow!
}

// GOOD: Transfer once
AcTensor *x_gpu = ac_tensor_to_gpu(host_tensor);
for (int i = 0; i < N; i++) {
    // Update x_gpu in-place if needed
    AcTensor *y = ac_forward(net, x_gpu);  // Stays on GPU
}
AcTensor *y_cpu = ac_tensor_to_cpu(y);  // Transfer once at end
```

---

## Quantisation

### INT8 Inference

2-4× faster inference, 4× smaller model:

```c
// After training
ac_quantize_model(net, num_layers, AC_QUANT_INT8);
ac_save_weights(net, num_layers, "model_int8.bin");
```

### Calibration

For better INT8 accuracy, calibrate with representative data:

```c
// Run a few batches to collect activation ranges
for (int i = 0; i < 10; i++) {
    ac_forward_seq(net, n, calibration_data[i]);
}
ac_quantize_model_calibrated(net, num_layers, AC_QUANT_INT8);
```

---

## Profiling

### Built-in Timer

```c
ac_timer_start("forward");
AcTensor *y = ac_forward_seq(net, n, x);
ac_timer_stop("forward");

ac_timer_start("backward");
ac_backward(loss);
ac_timer_stop("backward");

ac_timer_print_all();
```

Output:
```
forward:   12.34 ms (avg over 100 calls)
backward:  28.56 ms (avg over 100 calls)
```

### Per-Layer Timing

```c
ac_enable_layer_timing(true);
ac_forward_seq(net, n, x);
ac_print_layer_times();
```

Output:
```
Layer 0 (dense 784->256): 4.21 ms
Layer 1 (dense 256->128): 1.87 ms
Layer 2 (dense 128->10):  0.34 ms
```

---

## Platform-Specific Tips

### Intel/AMD x86

```bash
# Maximum performance
gcc -O3 -march=native -ffast-math -funroll-loops ...
```

### Apple Silicon (M1/M2/M3)

Via Rosetta 2:
```bash
arch -x86_64 gcc -O3 -mavx2 ...
```

Native ARM (experimental):
```bash
clang -O3 -march=armv8-a+simd ...
```

### Raspberry Pi

```bash
# Pi 4 (64-bit OS)
gcc -O3 -march=armv8-a -mtune=cortex-a72 ...

# Pi 3/Zero (32-bit)
gcc -O3 -mfpu=neon-fp-armv8 -mtune=cortex-a53 ...
```

### Windows (MSVC)

```batch
cl /O2 /arch:AVX2 your_code.c /I .\include
```

---

## Benchmarking Tips

1. **Warm up**: Run a few iterations before measuring
2. **Multiple runs**: Average over 10+ runs
3. **Disable Turbo Boost**: For consistent results
4. **Same input data**: Don't measure data loading

```c
// Warmup
for (int i = 0; i < 5; i++) {
    ac_forward_seq(net, n, x);
}

// Benchmark
clock_t start = clock();
for (int i = 0; i < 100; i++) {
    ac_forward_seq(net, n, x);
}
double avg_ms = (double)(clock() - start) / CLOCKS_PER_SEC * 1000 / 100;
printf("Average: %.2f ms per forward pass\n", avg_ms);
```
