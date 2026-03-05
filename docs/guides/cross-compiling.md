---
sidebar_position: 8
title: Cross-Compiling Guide
---

# Cross-Compiling Guide

Deploy Aicraft on embedded systems and different architectures.

## Supported Targets

| Target | Architecture | Toolchain |
|--------|--------------|-----------|
| Linux x86_64 | x86-64 | GCC / Clang |
| Linux ARM64 | AArch64 | aarch64-linux-gnu-gcc |
| Linux ARM32 | ARMv7 | arm-linux-gnueabihf-gcc |
| Raspberry Pi | ARM Cortex-A | arm-linux-gnueabihf-gcc |
| ESP32 | Xtensa | xtensa-esp32-elf-gcc |
| STM32 | ARM Cortex-M | arm-none-eabi-gcc |
| WebAssembly | WASM | Emscripten |
| macOS ARM | Apple Silicon | clang |
| Windows | x86-64 | MSVC / MinGW |

---

## Raspberry Pi

### Pi 4 / Pi 5 (64-bit OS)

```bash
# Install toolchain
sudo apt install gcc-aarch64-linux-gnu

# Cross-compile
aarch64-linux-gnu-gcc -O3 -march=armv8-a -mtune=cortex-a72 \
    your_code.c -I./include -o program_pi4

# Copy to Pi
scp program_pi4 pi@raspberrypi.local:~/
```

### Pi 3 / Pi Zero (32-bit OS)

```bash
# Install toolchain
sudo apt install gcc-arm-linux-gnueabihf

# Cross-compile with NEON
arm-linux-gnueabihf-gcc -O3 -mfpu=neon-fp-armv8 -mtune=cortex-a53 \
    your_code.c -I./include -o program_pi3
```

### Native compilation on Pi

```bash
# On the Pi itself
gcc -O3 -mcpu=native your_code.c -I./include -o program
```

---

## ARM Cortex-M (STM32, etc.)

For bare-metal embedded systems without an OS.

### Requirements

- No heap allocation (use static arena)
- No floating-point by default on Cortex-M0/M3
- Cortex-M4F/M7 have FPU

### Compile Flags

```bash
# Cortex-M4 with FPU
arm-none-eabi-gcc -O3 -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
    -mthumb -ffunction-sections -fdata-sections \
    your_code.c -I./include -o program.elf

# Link with newlib-nano for small footprint
arm-none-eabi-gcc ... -specs=nano.specs -lm
```

### Static Memory Configuration

```c
// Define static arena (no malloc)
#define AC_STATIC_ARENA
#define AC_ARENA_SIZE (64 * 1024)  // 64 KB

#include "aicraft/aicraft.h"

static uint8_t arena_buffer[AC_ARENA_SIZE];

int main(void) {
    ac_init_static(arena_buffer, AC_ARENA_SIZE);
    
    // Your inference code...
    
    return 0;
}
```

---

## ESP32

### Using ESP-IDF

```bash
# Set up ESP-IDF
. ~/esp/esp-idf/export.sh

# In your CMakeLists.txt
idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "." "${AICRAFT_PATH}/include"
)
```

### Configuration

```c
// ESP32 has limited RAM (~320 KB)
#define AC_STATIC_ARENA
#define AC_ARENA_SIZE (32 * 1024)  // 32 KB

// Disable features not needed
#define AC_NO_VULKAN
#define AC_NO_AVX

#include "aicraft/aicraft.h"
```

### Build

```bash
idf.py build
idf.py flash
```

---

## WebAssembly (Emscripten)

### Install Emscripten

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### Compile to WASM

```bash
emcc -O3 -s WASM=1 -s EXPORTED_FUNCTIONS='["_main", "_inference"]' \
    -s EXPORTED_RUNTIME_METHODS='["ccall", "cwrap"]' \
    your_code.c -I./include -o model.js
```

### Use in Browser

```html
<script src="model.js"></script>
<script>
Module.onRuntimeInitialized = async () => {
    const inference = Module.cwrap('inference', 'number', ['number']);
    const result = inference(inputPointer);
    console.log('Prediction:', result);
};
</script>
```

---

## RISC-V

### GCC Toolchain

```bash
# Install toolchain
sudo apt install gcc-riscv64-linux-gnu

# Compile
riscv64-linux-gnu-gcc -O3 -march=rv64gc \
    your_code.c -I./include -o program_riscv
```

### Embedded RISC-V (no OS)

```bash
riscv64-unknown-elf-gcc -O3 -march=rv32imc -mabi=ilp32 \
    -nostartfiles -T linker.ld \
    your_code.c -I./include -o program.elf
```

---

## Apple Silicon (M1/M2/M3)

### Native ARM64

```bash
clang -O3 -arch arm64 your_code.c -I./include -o program
```

### With NEON intrinsics

```bash
clang -O3 -arch arm64 -mcpu=apple-m1 your_code.c -I./include -o program
```

### Universal Binary (Intel + ARM)

```bash
clang -O3 -arch x86_64 -arch arm64 your_code.c -I./include -o program_universal
```

---

## Windows Cross-Compile (from Linux)

### MinGW-w64

```bash
# Install
sudo apt install mingw-w64

# Compile for Windows
x86_64-w64-mingw32-gcc -O3 your_code.c -I./include -o program.exe
```

### With AVX2

```bash
x86_64-w64-mingw32-gcc -O3 -mavx2 your_code.c -I./include -o program.exe
```

---

## Android (NDK)

### Setup

```bash
# Download NDK from https://developer.android.com/ndk/downloads
export NDK_HOME=/path/to/android-ndk
```

### Compile for ARM64

```bash
$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang \
    -O3 your_code.c -I./include -o libmodel.so -shared -fPIC
```

### CMakeLists.txt for Android Studio

```cmake
cmake_minimum_required(VERSION 3.10)
project(aicraft_android)

add_library(model SHARED your_code.c)
target_include_directories(model PRIVATE ${AICRAFT_PATH}/include)
target_compile_options(model PRIVATE -O3)
```

---

## Build System Integration

### CMake

```cmake
cmake_minimum_required(VERSION 3.10)
project(my_model)

# Set Aicraft include path
set(AICRAFT_INCLUDE "${CMAKE_SOURCE_DIR}/Aicraft/include")

add_executable(my_model main.c)
target_include_directories(my_model PRIVATE ${AICRAFT_INCLUDE})
target_compile_options(my_model PRIVATE -O3 -mavx2)
```

### Meson

```meson
project('my_model', 'c')

aicraft_inc = include_directories('Aicraft/include')

executable('my_model',
    'main.c',
    include_directories: aicraft_inc,
    c_args: ['-O3', '-mavx2']
)
```

### Makefile

```makefile
CC = gcc
CFLAGS = -O3 -mavx2 -I./Aicraft/include
TARGET = my_model

$(TARGET): main.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)
```

---

## Troubleshooting

### "Undefined reference to sqrt/sin/cos"

Add `-lm` to link the math library:

```bash
gcc ... -lm
```

### "Illegal instruction" at runtime

Your CPU doesn't support the SIMD instructions. Compile without AVX:

```bash
gcc -O3 -mno-avx2 ...
```

### Binary too large for embedded

1. Use `-Os` instead of `-O3`
2. Enable `-ffunction-sections -fdata-sections` and `-Wl,--gc-sections`
3. Disable unused features with `#define AC_NO_*`
4. Use INT8 quantised models

### Floating-point on Cortex-M0/M3

These cores lack FPU. Use soft-float:

```bash
arm-none-eabi-gcc -mfloat-abi=soft ...
```

Or quantise to INT8 for better performance.
