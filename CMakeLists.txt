cmake_minimum_required(VERSION 3.16)
project(Aicraft VERSION 1.0.0 LANGUAGES C CXX)

# ── Options ──────────────────────────────────────────────────────────────
option(AICRAFT_BUILD_TESTS   "Build test suite"       ON)
option(AICRAFT_BUILD_BENCH   "Build benchmarks"       ON)
option(AICRAFT_BUILD_DEMO    "Build demo"             ON)
option(AICRAFT_ENABLE_AVX512 "Enable AVX-512 support" OFF)
option(AICRAFT_ENABLE_VULKAN "Enable Vulkan compute backend" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Compiler flags ───────────────────────────────────────────────────────
if(MSVC)
    set(AICRAFT_COMMON_C "/W4")
    set(AICRAFT_COMMON_CXX "/W4 /EHsc")
    if(AICRAFT_ENABLE_AVX512)
        set(AICRAFT_ARCH_FLAG "/arch:AVX512")
    else()
        set(AICRAFT_ARCH_FLAG "/arch:AVX2")
    endif()
    set(CMAKE_C_FLAGS_RELEASE   "/O2 /Ob3 /Oi /Ot /GL ${AICRAFT_ARCH_FLAG} /fp:fast /DNDEBUG ${AICRAFT_COMMON_C}")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /Oi /Ot /GL ${AICRAFT_ARCH_FLAG} /fp:fast /DNDEBUG ${AICRAFT_COMMON_CXX}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
    set(CMAKE_C_FLAGS_DEBUG     "/Od /Zi /RTC1 ${AICRAFT_COMMON_C}")
    set(CMAKE_CXX_FLAGS_DEBUG   "/Od /Zi /RTC1 ${AICRAFT_COMMON_CXX}")
else()
    set(AICRAFT_COMMON_C "-Wall -Wextra -Wpedantic")
    set(AICRAFT_COMMON_CXX "-Wall -Wextra -Wpedantic")
    if(AICRAFT_ENABLE_AVX512)
        set(AICRAFT_ARCH_FLAG "-mavx512f -mavx512bw -mavx512vl")
    else()
        set(AICRAFT_ARCH_FLAG "-march=native -mtune=native")
    endif()
    set(CMAKE_C_FLAGS_RELEASE   "-O3 ${AICRAFT_ARCH_FLAG} -ffast-math -funroll-loops -flto -DNDEBUG ${AICRAFT_COMMON_C}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${AICRAFT_ARCH_FLAG} -ffast-math -funroll-loops -flto -DNDEBUG ${AICRAFT_COMMON_CXX}")
    set(CMAKE_C_FLAGS_DEBUG     "-O0 -g -fsanitize=address,undefined ${AICRAFT_COMMON_C}")
    set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined ${AICRAFT_COMMON_CXX}")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ── Core library ─────────────────────────────────────────────────────────
file(GLOB_RECURSE AICRAFT_SOURCES "src/*.c" "src/*.cpp")
file(GLOB_RECURSE AICRAFT_HEADERS "include/*.h" "include/*.hpp")

add_library(aicraft STATIC ${AICRAFT_SOURCES})
target_include_directories(aicraft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Platform-specific threading
if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(aicraft PUBLIC Threads::Threads)
endif()

# Math library
if(UNIX AND NOT APPLE)
    target_link_libraries(aicraft PUBLIC m)
endif()

# ── Vulkan compute backend ───────────────────────────────────────────────
if(AICRAFT_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    target_compile_definitions(aicraft PUBLIC AICRAFT_ENABLE_VULKAN)
    target_link_libraries(aicraft PUBLIC Vulkan::Vulkan)

    # Find shader compiler (glslc from Vulkan SDK, or glslangValidator)
    find_program(GLSLC_EXECUTABLE
        NAMES glslc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    )
    if(NOT GLSLC_EXECUTABLE)
        find_program(GLSLC_EXECUTABLE
            NAMES glslangValidator
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
        )
    endif()

    if(GLSLC_EXECUTABLE)
        message(STATUS "GLSL compiler found: ${GLSLC_EXECUTABLE}")
    else()
        message(WARNING "No GLSL compiler (glslc/glslangValidator) found. "
                        "Pre-compiled .spv files must be provided.")
    endif()

    # Compile .comp shaders to .spv
    set(SHADER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src/shaders")
    set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

    file(GLOB SHADER_SOURCES "${SHADER_SOURCE_DIR}/*.comp")
    set(SHADER_SPIRV_FILES "")

    if(GLSLC_EXECUTABLE)
        foreach(SHADER ${SHADER_SOURCES})
            get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
            set(SPV_OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
            add_custom_command(
                OUTPUT ${SPV_OUTPUT}
                COMMAND ${GLSLC_EXECUTABLE} -O ${SHADER} -o ${SPV_OUTPUT}
                DEPENDS ${SHADER}
                COMMENT "Compiling GLSL → SPIR-V: ${SHADER_NAME}.comp"
            )
            list(APPEND SHADER_SPIRV_FILES ${SPV_OUTPUT})
        endforeach()

        add_custom_target(aicraft_shaders ALL DEPENDS ${SHADER_SPIRV_FILES})
        add_dependencies(aicraft aicraft_shaders)
    endif()

    # Point the runtime to the compiled shaders directory
    target_compile_definitions(aicraft PRIVATE
        AICRAFT_VK_SHADER_DIR="${SHADER_OUTPUT_DIR}"
    )
endif()

# ── Demo ─────────────────────────────────────────────────────────────────
if(AICRAFT_BUILD_DEMO)
    add_executable(aicraft_demo demo/main.cpp)
    target_link_libraries(aicraft_demo PRIVATE aicraft)
endif()

# ── Tests ────────────────────────────────────────────────────────────────
if(AICRAFT_BUILD_TESTS)
    enable_testing()
    add_executable(aicraft_test tests/test_main.cpp)
    target_link_libraries(aicraft_test PRIVATE aicraft)
    add_test(NAME AllTests COMMAND aicraft_test)
endif()

# ── Benchmarks ───────────────────────────────────────────────────────────
if(AICRAFT_BUILD_BENCH)
    add_executable(aicraft_bench benchmarks/bench_main.cpp)
    target_link_libraries(aicraft_bench PRIVATE aicraft)
endif()

# ── Install ──────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS aicraft
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/aicraft DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ── Summary ──────────────────────────────────────────────────────────────
message(STATUS "=== Aicraft ML Framework v${PROJECT_VERSION} ===")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Tests      : ${AICRAFT_BUILD_TESTS}")
message(STATUS "Benchmarks : ${AICRAFT_BUILD_BENCH}")
message(STATUS "Demo       : ${AICRAFT_BUILD_DEMO}")
message(STATUS "AVX-512    : ${AICRAFT_ENABLE_AVX512}")
