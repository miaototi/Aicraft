---
sidebar_position: 11
title: Changelog
---

# Changelog

All notable changes to Aicraft are documented here.

Format: [Semantic Versioning](https://semver.org/)

---

## [1.0.0] — 2026-03-01

**Initial public release** 🎉

### Core

- **Tensor engine**: N-dimensional tensors with broadcasting, slicing, and reshape
- **Autograd**: Reverse-mode automatic differentiation with 22 differentiable ops
- **Arena allocator**: Checkpoint/restore memory management with zero per-tensor malloc

### Layers

- `ac_dense` — Fully-connected layer with configurable activation
- `ac_conv2d` — 2D convolution (coming in v1.1)
- `ac_batchnorm` — Batch normalisation
- `ac_dropout` — Dropout regularisation

### Activations

- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU

### Loss Functions

- Mean Squared Error (MSE)
- Cross-Entropy
- Binary Cross-Entropy
- Huber Loss

### Optimisers

- SGD, SGD with Momentum
- Adam, AdamW

### Backends

- **SIMD**: AVX2, AVX-512, ARM NEON hand-tuned kernels
- **Vulkan**: 14 GLSL compute shaders for GPU acceleration

### Quantisation

- Post-training INT8 quantisation with asymmetric per-tensor scaling

### Serialisation

- Binary format for weights (`ac_save_weights` / `ac_load_weights`)

---

## [Unreleased]

### Planned for v1.1

- [ ] Conv2D layer with im2col optimisation
- [ ] MaxPool2D / AvgPool2D layers
- [ ] RNN / LSTM cells
- [ ] ONNX export

### Planned for v1.2

- [ ] Automatic mixed precision (FP16/BF16)
- [ ] Multi-GPU support (Vulkan)
- [ ] Metal backend for Apple Silicon
- [ ] WebAssembly build target

---

## How to Upgrade

Aicraft is header-only, so upgrading is simple:

```bash
cd Aicraft
git pull origin main
```

Then recompile your project. No ABI concerns.

---

## Reporting Issues

Found a bug? Please open an issue at:

https://github.com/TobiasTesauri/Aicraft/issues
