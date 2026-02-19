# Contributing to Aicraft

Thank you for your interest in contributing to Aicraft! This document provides guidelines to make the process smooth for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/Aicraft.git`
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Push and open a Pull Request

## How to Contribute

### Reporting Bugs

- Use the [GitHub Issues](https://github.com/AicraftOrg/Aicraft/issues) page
- Include a clear title and description
- Provide a minimal reproducible example when possible
- Mention your OS, compiler, and CPU (especially SIMD support: AVX2/AVX-512/NEON)

### Suggesting Features

- Open an issue with the `enhancement` label
- Describe the use case and expected behavior
- If proposing a new layer/op, outline the forward and backward pass

### Submitting Changes

- Bug fixes, documentation improvements, and new features are all welcome
- For large changes, please open a discussion issue first

## Development Setup

### Prerequisites

- C11-compatible compiler (GCC 7+, Clang 6+, MSVC 2019+)
- C++17 for tests/benchmarks/demo
- CMake 3.16+
- (Optional) Doxygen 1.9+ for documentation generation

### Building

```bash
git clone https://github.com/<your-username>/Aicraft.git
cd Aicraft
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

### Running Tests

```bash
./aicraft_test      # 75 tests across 25 sections
```

### Debug Build (with sanitizers)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
./aicraft_test      # runs with AddressSanitizer + UBSan
```

### Generating Documentation

```bash
doxygen Doxyfile    # output in docs/html/
```

## Coding Standards

### C Code (core library)

- **Standard**: C11
- **Style**: 4-space indentation, no tabs
- **Naming**: `ac_` prefix for all public symbols (functions, types, macros)
  - Functions: `ac_snake_case()`
  - Types: `ac_snake_case` (structs/enums)
  - Macros: `AC_UPPER_CASE`
  - Constants: `AC_UPPER_CASE`
- **Headers**: Include guards using `#ifndef AICRAFT_MODULE_H` / `#define` / `#endif`
- **Inline**: Use `AC_INLINE` macro for hot-path functions
- **Memory**: Use arena/pool allocators — avoid raw `malloc`/`free` in hot paths
- **SIMD**: Always provide a scalar fallback; use the cascading `#if` pattern:
  ```c
  #if defined(AC_SIMD_AVX512)
      // AVX-512 path
  #elif defined(AC_SIMD_AVX2)
      // AVX2 path
  #elif defined(AC_SIMD_NEON)
      // NEON path
  #else
      // scalar fallback
  #endif
  ```

### C++ Code (tests, benchmarks, demo)

- **Standard**: C++17
- **Style**: Same indentation as C code
- Wrapped in `extern "C"` includes when using C headers

### Documentation

- All public API functions must have Doxygen comments
- Use `@brief`, `@param`, `@return`, `@note`, `@see`
- Document SIMD usage with `@simd` custom tag
- Document thread-safety with `@threadsafe` custom tag

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add quantized conv2d layer
fix: correct NEON micro-kernel accumulator reset
docs: add Doxygen comments to loss.h
perf: optimize AVX2 GEMM prefetch pattern
test: add backward pass tests for BatchNorm
chore: update CI to test on ARM64
```

## Pull Request Process

1. Ensure all 75 tests pass: `./aicraft_test`
2. Run with sanitizers (Debug build) to catch memory/UB issues
3. Add tests for any new functionality
4. Update Doxygen comments for new/changed API
5. Update README.md if adding user-facing features
6. Keep PRs focused — one feature/fix per PR
7. Fill out the PR template with a clear description

### Review Criteria

- [ ] All tests pass
- [ ] No sanitizer warnings (ASan + UBSan)
- [ ] Code follows naming conventions (`ac_` prefix)
- [ ] Doxygen comments present on public API
- [ ] SIMD code has scalar fallback
- [ ] No raw `malloc`/`free` in hot paths (use arena)
- [ ] Commit messages follow Conventional Commits

## Architecture Notes

Before contributing a new module, review the existing architecture:

- **Header-only core**: Hot paths are `inline` in headers under `include/aicraft/`
- **`src/core.c`**: Only non-inline global state (arena, error state)
- **Autograd**: Each forward op stores a backward closure in the tensor
- **Memory model**: Arena allocator with checkpoint/restore — intermediates freed per epoch
- **Threading**: Thread pool in `thread_pool.h` — GEMM is the primary parallel workload

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
