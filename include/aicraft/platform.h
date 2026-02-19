/**
 * @file platform.h
 * @brief Platform detection, SIMD feature selection, compiler hints, and type aliases.
 *
 * Detects the target OS and SIMD instruction set at compile time via predefined
 * macros.  Provides portable wrappers for forced-inline, restrict, branch
 * prediction, memory alignment, and cache-line constants.
 *
 * SIMD detection cascades: AVX-512 → AVX2 → SSE → NEON → scalar.
 */

#ifndef AICRAFT_PLATFORM_H
#define AICRAFT_PLATFORM_H

#include <stdint.h>
#include <stddef.h>

/** @defgroup platform Platform Detection */
/** @{ */

/** @name OS Detection */
/** @{ */

#if defined(_WIN32) || defined(_WIN64)
    #define AC_PLATFORM_WINDOWS 1
#elif defined(__linux__)
    #define AC_PLATFORM_LINUX 1
#elif defined(__APPLE__)
    #define AC_PLATFORM_MACOS 1
#endif

/** @} */ /* OS Detection */

/** @name SIMD Detection
 *  Cascading feature selection: AVX-512 ⇒ AVX2 ⇒ SSE ⇒ NEON ⇒ scalar.
 *  @c AC_SIMD_WIDTH contains the number of floats processed per SIMD register.
 *  @{ */

#if defined(__AVX512F__)
    #define AC_SIMD_AVX512 1
    #define AC_SIMD_AVX2   1
    #define AC_SIMD_SSE    1
    #define AC_SIMD_WIDTH 16
#elif defined(__AVX2__) || defined(__AVX__)
    #define AC_SIMD_AVX2 1
    #define AC_SIMD_SSE  1
    #define AC_SIMD_WIDTH 8
#elif defined(__SSE4_1__) || defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
    #define AC_SIMD_SSE 1
    #define AC_SIMD_WIDTH 4
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define AC_SIMD_NEON 1
    #define AC_SIMD_WIDTH 4
#else
    #define AC_SIMD_NONE 1
    #define AC_SIMD_WIDTH 1
#endif
/** @} */ /* SIMD Detection */
/** @} */ /* platform */

/** @defgroup compiler Compiler Hints
 *  Portable wrappers for forced-inline, restrict, branch hints, alignment, and
 *  data prefetch.
 *  @{ */

#if defined(__GNUC__) || defined(__clang__)
    #define AC_INLINE __attribute__((always_inline)) inline
    #define AC_RESTRICT __restrict__
    #define AC_LIKELY(x) __builtin_expect(!!(x), 1)
    #define AC_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define AC_ALIGNED(x) __attribute__((aligned(x)))
    #define AC_PREFETCH(addr) __builtin_prefetch(addr)
#elif defined(_MSC_VER)
    #define AC_INLINE __forceinline
    #define AC_RESTRICT __restrict
    #define AC_LIKELY(x) (x)
    #define AC_UNLIKELY(x) (x)
    #define AC_ALIGNED(x) __declspec(align(x))
    #define AC_PREFETCH(addr)
#else
    #define AC_INLINE inline
    #define AC_RESTRICT
    #define AC_LIKELY(x) (x)
    #define AC_UNLIKELY(x) (x)
    #define AC_ALIGNED(x)
    #define AC_PREFETCH(addr)
#endif
/** @} */ /* compiler */

/** @defgroup alignment Memory Alignment Constants
 *  Cache-line and SIMD alignment values.
 *  @{ */

#define AC_CACHE_LINE 64  /**< Typical L1 data-cache line size in bytes. */
#if defined(AC_SIMD_AVX512)
    #define AC_SIMD_ALIGN 64
#elif defined(AC_SIMD_AVX2)
    #define AC_SIMD_ALIGN 32
#elif defined(AC_SIMD_NEON)
    #define AC_SIMD_ALIGN 16
#else
    #define AC_SIMD_ALIGN 32
#endif
/** @} */ /* alignment */

/** @defgroup types Type Aliases
 *  Fixed-width and convenience type definitions used throughout the framework.
 *  @{ */

typedef float    ac_float32;  /**< 32-bit floating-point. */
typedef double   ac_float64;  /**< 64-bit floating-point. */
typedef int32_t  ac_int32;    /**< Signed 32-bit integer. */
typedef int64_t  ac_int64;    /**< Signed 64-bit integer. */
typedef uint32_t ac_uint32;   /**< Unsigned 32-bit integer. */
typedef uint64_t ac_uint64;   /**< Unsigned 64-bit integer. */
typedef size_t   ac_size;     /**< Size / index type (pointer-width unsigned). */
/** @} */ /* types */

#endif /* AICRAFT_PLATFORM_H */
