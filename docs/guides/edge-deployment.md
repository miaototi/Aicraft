---
sidebar_position: 2
title: Edge Deployment
---

# Edge Deployment

Deploy Aicraft models on microcontrollers and embedded devices.

## Overview

Aicraft is designed for edge deployment from the ground up. With zero dependencies, a ~150 KB binary, and INT8 quantisation support, it runs on everything from x86 servers to ARM Cortex-M MCUs.

## Quantisation Workflow

```c
#include "aicraft/aicraft.h"

// After training...
// Quantise the model to INT8
ac_quantize_model(net, 2, AC_QUANT_INT8);

// Save quantised model
ac_serialize(net, 2, "model_int8.bin");
```

## Deployment Targets

| Target | Compiler | Flags |
|--------|----------|-------|
| x86_64 Linux/Win | GCC / Clang / MSVC | `-O3 -mavx2` |
| ARM Cortex-A (RPi) | `arm-linux-gnueabihf-gcc` | `-O3 -mfpu=neon` |
| ARM Cortex-M (STM32) | `arm-none-eabi-gcc` | `-O2 -mcpu=cortex-m4` |
| RISC-V | `riscv64-linux-gnu-gcc` | `-O3` |

## Memory Budget

For constrained devices, configure the arena size at init:

```c
ac_init_with_arena(64 * 1024);  // 64 KB arena
```

## Cross-Compilation Example

```bash
# For Raspberry Pi
arm-linux-gnueabihf-gcc -O3 -mfpu=neon \
    inference.c -I./include -o inference

# For STM32
arm-none-eabi-gcc -O2 -mcpu=cortex-m4 -mthumb \
    inference.c -I./include -o inference.elf
```

## Model Size (MNIST, 784→128→10)

| Format | Size |
|--------|------|
| Float32 (original) | ~400 KB |
| INT8 (quantised) | ~100 KB |
| Binary (code + model) | ~150 KB |
