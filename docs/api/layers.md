---
sidebar_position: 4
title: Layers API
---

# Layers API

`#include "aicraft/layers.h"`

## AcLayer

All layers share the `AcLayer` type:

```c
typedef struct AcLayer {
    AcTensor *weights;
    AcTensor *bias;
    AcTensor *(*forward)(struct AcLayer *, AcTensor *);
    int type;
} AcLayer;
```

## Dense (Fully Connected)

```c
AcLayer *ac_dense(int in_features, int out_features, AcActivation act);
```

| Parameter | Description |
|-----------|-------------|
| `in_features` | Input dimension |
| `out_features` | Output dimension |
| `act` | Activation: `AC_RELU`, `AC_SIGMOID`, `AC_SOFTMAX`, `AC_NONE` |

### Example

```c
AcLayer *layer = ac_dense(784, 128, AC_RELU);
AcTensor *out = layer->forward(layer, input);
```

## Sequential Forward

```c
AcTensor *ac_forward_seq(AcLayer **layers, int n, AcTensor *input);
```

Pass input through a sequence of layers:

```c
AcLayer *net[] = {
    ac_dense(784, 128, AC_RELU),
    ac_dense(128, 10,  AC_SOFTMAX)
};
AcTensor *out = ac_forward_seq(net, 2, x);
```

## Activation Functions

| Activation | Constant | Formula |
|------------|----------|---------|
| ReLU | `AC_RELU` | `max(0, x)` |
| Sigmoid | `AC_SIGMOID` | `1 / (1 + e⁻ˣ)` |
| Softmax | `AC_SOFTMAX` | `eˣⁱ / Σeˣʲ` |
| None | `AC_NONE` | Identity |

## Weight Initialisation

Weights are initialised using **Kaiming (He) uniform** initialisation by default. Biases are initialised to zero.
