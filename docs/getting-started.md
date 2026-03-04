---
sidebar_position: 2
title: Getting Started
---

# Getting Started

Get Aicraft running in under 5 minutes.

## Prerequisites

- A C11-compatible compiler (GCC, Clang, or MSVC)
- Optional: Vulkan SDK for GPU acceleration

## Installation

```bash
git clone https://github.com/TobiasTesauri/Aicraft.git
cd Aicraft
```

That's it. Aicraft is header-only — no build system needed.

## Your First Program

Create a file called `demo.c`:

```c
#include "aicraft/aicraft.h"

int main(void) {
    ac_init();

    // Build a feedforward network
    AcLayer *net[] = {
        ac_dense(784, 128, AC_RELU),
        ac_dense(128, 10,  AC_SOFTMAX)
    };

    // Forward pass
    AcTensor *x = ac_tensor_rand((int[]){1, 784}, 2);
    AcTensor *y = ac_forward_seq(net, 2, x);

    // Backward pass
    ac_backward(y);

    ac_cleanup();
    return 0;
}
```

## Compile & Run

```bash
gcc -O3 demo.c -I./include -o demo
./demo
```

Expected output:

```
[ac] init .............. ok
[ac] dense 784->128 .... ok
[ac] dense 128->10 ..... ok
[ac] forward ........... 0.42 ms
[ac] backward .......... 0.87 ms
[ac] loss: 0.0234 ...... ok
[ac] cleanup ........... ok
```

## With Vulkan (optional)

```bash
gcc -O3 demo.c -I./include -lvulkan -o demo
```

## Next Steps

- [Architecture](/docs/architecture) — Understand how Aicraft is structured
- [API Reference](/docs/api/overview) — Full API documentation
- [Training Guide](/docs/guides/training) — Train your first model
