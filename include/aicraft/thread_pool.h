/**
 * @file thread_pool.h
 * @brief Lightweight thread pool for parallel execution.
 *
 * Zero-dependency parallel execution using platform-native threads
 * (Win32 or POSIX). Persistent worker threads with event-based
 * synchronization.
 *
 * @warning Global state (g_rng, g_last_error, g_autograd_epoch) is
 *          NOT thread-safe. Autograd, PRNG, and error reporting must
 *          run on the main thread only. Only GEMM kernels run in parallel.
 */

#ifndef AICRAFT_THREAD_POOL_H
#define AICRAFT_THREAD_POOL_H

#include "aicraft/platform.h"

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
    #include <unistd.h>
#endif

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup threadpool Thread Pool */
/** @{ */

/** @name Data Structures */
/** @{ */

/** @brief Maximum number of worker threads supported. */
#define AC_MAX_THREADS 32

/**
 * @brief Function signature for parallel work items.
 *
 * @param ctx       User-supplied context pointer.
 * @param thread_id Index of the executing thread.
 * @param start     Start of the assigned range (inclusive).
 * @param end       End of the assigned range (exclusive).
 */
typedef void (*ac_parallel_fn)(void* ctx, int thread_id, int start, int end);

/**
 * @brief A single unit of work dispatched to a thread.
 */
typedef struct {
    ac_parallel_fn  func;   /**< @brief Function to execute. */
    void*           ctx;    /**< @brief User context pointer. */
    int             start;  /**< @brief Range start (inclusive). */
    int             end;    /**< @brief Range end (exclusive). */
} ac_work_item;

/**
 * @brief Thread pool with persistent worker threads.
 *
 * Platform-specific synchronization (Win32 events / POSIX condvars).
 */
typedef struct {
#ifdef _WIN32
    HANDLE          threads[AC_MAX_THREADS];
    HANDLE          start_events[AC_MAX_THREADS];
    HANDLE          done_events[AC_MAX_THREADS];
#else
    pthread_t       threads[AC_MAX_THREADS];
    pthread_mutex_t mutex[AC_MAX_THREADS];
    pthread_cond_t  start_cond[AC_MAX_THREADS];
    pthread_cond_t  done_cond[AC_MAX_THREADS];
    int             start_flags[AC_MAX_THREADS];
    int             done_flags[AC_MAX_THREADS];
#endif
    ac_work_item    work[AC_MAX_THREADS];
    int             num_threads;
    volatile int    shutdown;
} ac_thread_pool;

/** @brief Global thread pool singleton. */
extern ac_thread_pool g_thread_pool;
/** @brief Non-zero once the global pool has been initialised. */
extern int g_thread_pool_initialized;

/** @} */

/** @name Worker Implementation @internal */
/** @{ */

/**
 * @brief Per-worker context passed as the thread argument.
 */
typedef struct {
    ac_thread_pool* pool;   /**< @brief Owning pool. */
    int             id;     /**< @brief Worker index. */
} ac_worker_ctx;

/**
 * @brief Stable storage for worker contexts (defined in core.c).
 * @internal
 */
extern ac_worker_ctx g_worker_ctxs[AC_MAX_THREADS];

#ifdef _WIN32

/**
 * @brief Worker thread entry point (Win32).
 * @internal
 * @param param Pointer to an ac_worker_ctx.
 * @return Always 0.
 */
static DWORD WINAPI ac_worker_func(LPVOID param) {
    ac_worker_ctx* wctx = (ac_worker_ctx*)param;
    ac_thread_pool* pool = wctx->pool;
    int id = wctx->id;

    while (1) {
        WaitForSingleObject(pool->start_events[id], INFINITE);

        if (pool->shutdown) break;

        ac_work_item* w = &pool->work[id];
        if (w->func && w->start < w->end) {
            w->func(w->ctx, id, w->start, w->end);
        }

        SetEvent(pool->done_events[id]);
    }
    return 0;
}

#else /* POSIX */

/**
 * @brief Worker thread entry point (POSIX).
 * @internal
 * @param param Pointer to an ac_worker_ctx.
 * @return Always NULL.
 */
static void* ac_worker_func(void* param) {
    ac_worker_ctx* wctx = (ac_worker_ctx*)param;
    ac_thread_pool* pool = wctx->pool;
    int id = wctx->id;

    while (1) {
        pthread_mutex_lock(&pool->mutex[id]);
        while (!pool->start_flags[id] && !pool->shutdown)
            pthread_cond_wait(&pool->start_cond[id], &pool->mutex[id]);
        pool->start_flags[id] = 0;
        pthread_mutex_unlock(&pool->mutex[id]);

        if (pool->shutdown) break;

        ac_work_item* w = &pool->work[id];
        if (w->func && w->start < w->end) {
            w->func(w->ctx, id, w->start, w->end);
        }

        pthread_mutex_lock(&pool->mutex[id]);
        pool->done_flags[id] = 1;
        pthread_cond_signal(&pool->done_cond[id]);
        pthread_mutex_unlock(&pool->mutex[id]);
    }
    return NULL;
}

#endif

/** @} */

/** @name Thread Pool API */
/** @{ */

/**
 * @brief Return the number of logical CPUs on the current machine.
 *
 * @return Number of online processors.
 */
static AC_INLINE int ac_get_num_cpus(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

/**
 * @brief Initialise a thread pool with the given number of workers.
 *
 * @param pool        Pool to initialise.
 * @param num_threads Desired worker count (0 = auto-detect from CPU count).
 * @note One thread is reserved for the caller; actual workers = num_threads - 1.
 */
static AC_INLINE void ac_thread_pool_init(ac_thread_pool* pool, int num_threads) {
    if (num_threads <= 0) num_threads = ac_get_num_cpus();
    if (num_threads > AC_MAX_THREADS) num_threads = AC_MAX_THREADS;
    /* Reserve 1 thread for the caller (main thread participates) */
    if (num_threads > 1) num_threads -= 1;

    pool->num_threads = num_threads;
    pool->shutdown = 0;
    memset(pool->work, 0, sizeof(pool->work));

    for (int i = 0; i < num_threads; i++) {
        g_worker_ctxs[i].pool = pool;
        g_worker_ctxs[i].id = i;

#ifdef _WIN32
        pool->start_events[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
        pool->done_events[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
        pool->threads[i] = CreateThread(NULL, 0, ac_worker_func,
                                        &g_worker_ctxs[i], 0, NULL);
#else
        pthread_mutex_init(&pool->mutex[i], NULL);
        pthread_cond_init(&pool->start_cond[i], NULL);
        pthread_cond_init(&pool->done_cond[i], NULL);
        pool->start_flags[i] = 0;
        pool->done_flags[i] = 0;
        pthread_create(&pool->threads[i], NULL, ac_worker_func,
                       &g_worker_ctxs[i]);
#endif
    }
}

/**
 * @brief Shut down all worker threads and release resources.
 *
 * @param pool Pool to destroy.
 */
static AC_INLINE void ac_thread_pool_destroy(ac_thread_pool* pool) {
    pool->shutdown = 1;

    for (int i = 0; i < pool->num_threads; i++) {
#ifdef _WIN32
        SetEvent(pool->start_events[i]);
#else
        pthread_mutex_lock(&pool->mutex[i]);
        pool->start_flags[i] = 1;
        pthread_cond_signal(&pool->start_cond[i]);
        pthread_mutex_unlock(&pool->mutex[i]);
#endif
    }

    for (int i = 0; i < pool->num_threads; i++) {
#ifdef _WIN32
        WaitForSingleObject(pool->threads[i], INFINITE);
        CloseHandle(pool->threads[i]);
        CloseHandle(pool->start_events[i]);
        CloseHandle(pool->done_events[i]);
#else
        pthread_join(pool->threads[i], NULL);
        pthread_mutex_destroy(&pool->mutex[i]);
        pthread_cond_destroy(&pool->start_cond[i]);
        pthread_cond_destroy(&pool->done_cond[i]);
#endif
    }
    pool->num_threads = 0;
}

/**
 * @brief Distribute [start, end) across all worker threads plus the caller.
 *
 * The calling thread participates as the last worker
 * (thread_id = num_threads). Total parallelism is N + 1 where N is the
 * pool's worker count.
 *
 * @param pool  Thread pool (may be NULL for single-threaded fallback).
 * @param start Range start (inclusive).
 * @param end   Range end (exclusive).
 * @param func  Work function to invoke per chunk.
 * @param ctx   User context forwarded to @p func.
 */
static AC_INLINE void ac_parallel_for(ac_thread_pool* pool,
                                       int start, int end,
                                       ac_parallel_fn func, void* ctx)
{
    if (!pool || pool->num_threads <= 0 || (end - start) < 64) {
        /* Fallback: run single-threaded if pool not ready or work is tiny */
        func(ctx, 0, start, end);
        return;
    }

    int total = end - start;
    int n_workers = pool->num_threads;
    int n_all = n_workers + 1; /* workers + main thread */
    int chunk = (total + n_all - 1) / n_all;

    /* Dispatch work to worker threads */
    for (int i = 0; i < n_workers; i++) {
        int s = start + (i + 1) * chunk; /* workers take chunks 1..N */
        int e = s + chunk;
        if (s >= end) s = end;
        if (e > end) e = end;

        pool->work[i].func = func;
        pool->work[i].ctx = ctx;
        pool->work[i].start = s;
        pool->work[i].end = e;

#ifdef _WIN32
        SetEvent(pool->start_events[i]);
#else
        pthread_mutex_lock(&pool->mutex[i]);
        pool->done_flags[i] = 0;
        pool->start_flags[i] = 1;
        pthread_cond_signal(&pool->start_cond[i]);
        pthread_mutex_unlock(&pool->mutex[i]);
#endif
    }

    /* Main thread does chunk 0 */
    int main_end = start + chunk;
    if (main_end > end) main_end = end;
    func(ctx, n_workers, start, main_end);

    /* Wait for all workers to finish */
#ifdef _WIN32
    WaitForMultipleObjects(n_workers, pool->done_events, TRUE, INFINITE);
#else
    for (int i = 0; i < n_workers; i++) {
        pthread_mutex_lock(&pool->mutex[i]);
        while (!pool->done_flags[i])
            pthread_cond_wait(&pool->done_cond[i], &pool->mutex[i]);
        pthread_mutex_unlock(&pool->mutex[i]);
    }
#endif
}

/** @} */

/** @name Global Pool Helpers */
/** @{ */

/**
 * @brief Lazily initialise the global thread pool on first use.
 *
 * Safe to call multiple times; only the first call creates threads.
 */
static AC_INLINE void ac_ensure_thread_pool(void) {
    if (!g_thread_pool_initialized) {
        ac_thread_pool_init(&g_thread_pool, 0);
        g_thread_pool_initialized = 1;
    }
}

/** @} */
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_THREAD_POOL_H */
