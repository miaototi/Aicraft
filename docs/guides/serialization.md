---
sidebar_position: 4
title: Serialization
---

# Serialization

Save and load Aicraft models.

## Saving a Model

```c
// Save network weights to a binary file
ac_serialize(net, 2, "model.bin");
```

The file format is a compact binary format containing:
- Layer count and types
- Weight and bias tensors
- Quantisation parameters (if quantised)

## Loading a Model

```c
AcLayer *net[2];
int n = ac_deserialize(net, "model.bin");
// n == 2 (number of layers loaded)
```

## File Format

```
┌──────────────────────────┐
│ Magic: "ACML"  (4 bytes) │
│ Version       (4 bytes)  │
│ Layer count   (4 bytes)  │
├──────────────────────────┤
│ Layer 0 type  (4 bytes)  │
│ Layer 0 shape (N bytes)  │
│ Layer 0 weights (float[])│
│ Layer 0 biases  (float[])│
├──────────────────────────┤
│ Layer 1 ...              │
└──────────────────────────┘
```

## Quantised Models

Quantised models include additional per-tensor scaling parameters:

```c
// Save quantised model
ac_quantize_model(net, 2, AC_QUANT_INT8);
ac_serialize(net, 2, "model_int8.bin");

// Load quantised model
AcLayer *qnet[2];
ac_deserialize(qnet, "model_int8.bin");
// Quantisation params are restored automatically
```

## Portability

The binary format is **platform-independent** (little-endian, IEEE 754 floats). Models saved on x86 can be loaded on ARM and vice versa.
