# Changelog

All notable changes to Aicraft will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-13

### Added

#### Vulkan Compute Backend
- Vulkan 1.0 compute-only backend for GPU-accelerated tensor operations
- Automatic GPU device selection (prefers discrete GPU, falls back to integrated)
- GPU GEMM: tiled matrix multiplication with shared-memory blocking (16×16 tiles)
- GPU element-wise ops: add, mul, scale, fused multiply-add
- GPU activations: ReLU, sigmoid, tanh (forward + backward)
- GPU softmax with numerically stable row-wise reduction (shared memory)
- GPU reduction operations: sum, max (workgroup-level tree reduction)
- Smart auto-dispatch: GPU for large tensors (≥4096 elements), CPU SIMD for small
- Staging buffer management for efficient host↔device transfers (16 MB double-buffered)
- Pipeline caching: one-time shader compilation, reused across calls
- Descriptor set management with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
- Push constants for zero-overhead kernel parameterization
- 14 GLSL compute shaders compiled to SPIR-V via CMake custom commands
- Transparent integration: `ac_init()` / `ac_cleanup()` handle Vulkan lifecycle
- `ac_vk_print_info()` for runtime GPU diagnostics
- CMake option `AICRAFT_ENABLE_VULKAN` (OFF by default)
- Full Doxygen documentation for all Vulkan API functions

## [1.0.0] - 2026-02-13

### Added

#### Core
- N-dimensional tensor engine (up to 8D) with SIMD-aligned storage
- Reverse-mode autograd with dynamic topological sorting (22 ops)
- Arena allocator with checkpoint/restore for zero-alloc training loops
- Pool allocator for fixed-size objects
- xoshiro128** PRNG (passes BigCrush)

#### Neural Network Layers
- Dense (fully-connected) with forward + backward
- Conv2D via im2col with forward + backward
- MaxPool2D with forward + backward
- BatchNorm with running mean/variance and forward + backward
- Dropout with training/inference mode switching
- Flatten with forward + backward

#### Activations
- ReLU, Sigmoid, Tanh, Softmax — all with autograd backward

#### Loss Functions
- Mean Squared Error (MSE)
- Cross-Entropy with fused softmax
- Binary Cross-Entropy (BCE)

#### Optimizers
- SGD with momentum and weight decay
- Adam with bias correction
- AdamW (decoupled weight decay)
- Global L2-norm gradient clipping
- Per-element value gradient clipping

#### LR Schedulers
- Step decay
- Cosine annealing
- Exponential decay

#### SIMD Acceleration
- Cascading AVX-512 → AVX2 → SSE → NEON → scalar fallback
- Element-wise ops: add, sub, mul, scale, FMA, dot, sum, max, relu, exp, sigmoid, tanh
- BLIS-style packed GEMM with architecture-specific micro-kernels:
  - AVX-512: 6×32 tile (12 ZMM accumulators)
  - AVX2: 6×16 tile (12 YMM accumulators, FMA3 optional)
  - NEON: 6×8 tile (12 Q-reg accumulators)
  - Scalar fallback for edge tiles
- Panel packing (A row-major MR-contiguous, B column-major NR-contiguous)
- Cache prefetch hints for L1 locality

#### Fast Math Approximations
- Polynomial exp/tanh approximations for AVX2, AVX-512, NEON
- Array wrappers with automatic SIMD vectorization

#### Edge / Embedded
- ARM NEON intrinsics for all element-wise ops + GEMM micro-kernel
- NEON transcendentals: vectorized exp (Cephes polynomial), sigmoid (Newton-Raphson), tanh
- INT8 asymmetric per-tensor affine quantization (float32 → uint8)
- Quantized GEMM with INT32 accumulation
- Quantized dense layer for inference
- Model size estimation (FP32 vs INT8 comparison)

#### Systems
- Thread pool for parallel GEMM
- Production error handling: error codes, user callbacks, debug stderr
- Binary model serialization with versioned format and magic header
- Cross-platform support: Windows, Linux, macOS, ARM

#### Documentation
- Comprehensive Doxygen documentation for all 14 header modules
- Generated HTML docs with dark theme and tree navigation
- Custom tags: `@simd`, `@perf`, `@threadsafe`

#### Testing
- 75 tests across 25 sections covering all modules
- Debug builds with AddressSanitizer + UndefinedBehaviorSanitizer
- Benchmarks for GEMM and key operations

[1.0.0]: https://github.com/AicraftOrg/Aicraft/releases/tag/v1.0.0
[1.1.0]: https://github.com/AicraftOrg/Aicraft/releases/tag/v1.1.0
