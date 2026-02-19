/**
 * @file aicraft.h
 * @brief Main header — include this single file to use the entire Aicraft framework.
 *
 * @code{.c}
 * #include <aicraft/aicraft.h>
 * @endcode
 *
 * Aicraft is a high-performance machine learning framework written in pure C/C++
 * with zero external dependencies. All 14 modules are header-only with inlined
 * hot paths, SIMD-accelerated kernels (AVX-512/AVX2/SSE/NEON), arena-based
 * memory management, and reverse-mode automatic differentiation.
 *
 * @version 1.0.0
 * @author Aicraft Contributors
 * @copyright MIT License
 */

#ifndef AICRAFT_H
#define AICRAFT_H

/** @name Version Macros
 *  Compile-time version identification.
 *  @{ */
#define AICRAFT_VERSION_MAJOR 1   /**< Major version number. */
#define AICRAFT_VERSION_MINOR 0   /**< Minor version number. */
#define AICRAFT_VERSION_PATCH 0   /**< Patch version number. */
#define AICRAFT_VERSION_STRING "1.0.0" /**< Human-readable version string. */
/** @} */

#include "aicraft/platform.h"
#include "aicraft/error.h"
#include "aicraft/memory.h"
#include "aicraft/thread_pool.h"
#include "aicraft/fast_math.h"
#include "aicraft/simd_math.h"
#include "aicraft/tensor.h"
#include "aicraft/tensor_ops.h"
#include "aicraft/autograd.h"
#include "aicraft/layers.h"
#include "aicraft/loss.h"
#include "aicraft/optimizer.h"
#include "aicraft/serialize.h"
#include "aicraft/quantize.h"
#include "aicraft/vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup lifecycle Framework Lifecycle
 *  Initialisation and cleanup of global resources.
 *  @{ */

/**
 * @brief Initialise the Aicraft framework.
 *
 * Allocates the global tensor arena (64 MB default), starts the thread pool
 * for parallel GEMM, and seeds the xoshiro128** PRNG to 42.
 *
 * Must be called once before any other Aicraft API.  It is safe to call
 * multiple times — subsequent calls are no-ops.
 *
 * @see ac_cleanup
 */
static AC_INLINE void ac_init(void) {
    ac_ensure_arena();
    ac_ensure_thread_pool();
    ac_rng_seed(&g_rng, 42);
#ifdef AICRAFT_ENABLE_VULKAN
    ac_vk_init();  /* best-effort: silent no-op if no GPU */
#endif
}

/**
 * @brief Destroy all global resources (arena, thread pool).
 *
 * Call at program exit to release the tensor arena and join worker threads.
 * After cleanup the framework must be re-initialised with ac_init() before
 * further use.
 *
 * @see ac_init
 */
static AC_INLINE void ac_cleanup(void) {
#ifdef AICRAFT_ENABLE_VULKAN
    ac_vk_cleanup();
#endif
    if (g_thread_pool_initialized) {
        ac_thread_pool_destroy(&g_thread_pool);
        g_thread_pool_initialized = 0;
    }
    if (g_arena_initialized) {
        ac_arena_destroy(&g_tensor_arena);
        g_arena_initialized = 0;
    }
}
/** @} */ /* lifecycle */

/** @defgroup timer High-Precision Timer
 *  Portable wall-clock timer for micro-benchmarks.
 *  Uses QueryPerformanceCounter on Windows, clock_gettime on POSIX.
 *  @{ */

#if defined(AC_PLATFORM_WINDOWS)
    #include <windows.h>
    /** @brief High-resolution timer (Windows). */
    typedef struct {
        LARGE_INTEGER start, end, freq;
    } ac_timer;

    /** @brief Record the start timestamp. */
    AC_INLINE void ac_timer_start(ac_timer* t) {
        QueryPerformanceFrequency(&t->freq);
        QueryPerformanceCounter(&t->start);
    }
    /**
     * @brief Record the end timestamp and return elapsed seconds.
     * @param t  Timer previously started with ac_timer_start().
     * @return   Elapsed wall-clock time in seconds.
     */
    AC_INLINE double ac_timer_stop(ac_timer* t) {
        QueryPerformanceCounter(&t->end);
        return (double)(t->end.QuadPart - t->start.QuadPart) / (double)t->freq.QuadPart;
    }
#else
    #include <time.h>
    /** @brief High-resolution timer (POSIX). */
    typedef struct {
        struct timespec start, end;
    } ac_timer;

    /** @brief Record the start timestamp. */
    AC_INLINE void ac_timer_start(ac_timer* t) {
        clock_gettime(CLOCK_MONOTONIC, &t->start);
    }
    /**
     * @brief Record the end timestamp and return elapsed seconds.
     * @param t  Timer previously started with ac_timer_start().
     * @return   Elapsed wall-clock time in seconds.
     */
    AC_INLINE double ac_timer_stop(ac_timer* t) {
        clock_gettime(CLOCK_MONOTONIC, &t->end);
        return (t->end.tv_sec - t->start.tv_sec) + 
               (t->end.tv_nsec - t->start.tv_nsec) / 1e9;
    }
#endif
/** @} */ /* timer */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_H */