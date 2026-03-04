---
sidebar_position: 9
title: Quantize API
---

# Quantize API

`#include "aicraft/quantize.h"`

## Overview

Post-training quantisation to INT8 for reduced model size and faster inference on edge devices.

## Quantise a Model

```c
ac_quantize_model(net, num_layers, AC_QUANT_INT8);
```

This performs **asymmetric per-tensor quantisation**:

```
x_q = round(x / scale) + zero_point
```

## Quantisation Modes

| Mode | Constant | Bits | Compression |
|------|----------|------|-------------|
| INT8 | `AC_QUANT_INT8` | 8 | ~4× |

## Quantised Inference

```c
// Quantise after training
ac_quantize_model(net, 2, AC_QUANT_INT8);

// Inference works the same way
AcTensor *pred = ac_forward_seq(net, 2, x);
```

The forward pass automatically uses quantised arithmetic when the model is quantised.

## Dequantisation

```c
ac_dequantize_model(net, 2);
```

Convert back to float32 (with quantisation error).

## Per-Tensor Parameters

```c
typedef struct AcQuantParams {
    float scale;
    int   zero_point;
    int   bits;
} AcQuantParams;

AcQuantParams params = ac_get_quant_params(layer);
```

## Accuracy Impact

Typical accuracy loss with INT8 quantisation on MNIST:

| Model | Float32 | INT8 | Drop |
|-------|---------|------|------|
| 784→128→10 | 97.8% | 97.5% | 0.3% |
| 784→256→128→10 | 98.3% | 98.1% | 0.2% |
