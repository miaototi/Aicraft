/**
 * @file memory.h
 * @brief Arena & pool allocators for zero-overhead tensor memory management.
 *
 * Eliminates per-tensor malloc/free overhead that slows PyTorch / TF.
 * - **Arena allocator**: Bump-pointer with 64 MB blocks, checkpoint/restore
 *   for epoch-level lifetime management.
 * - **Pool allocator**: Fixed-size free-list allocator for small, uniform
 *   objects (gradient nodes, etc.).
 * - **Aligned allocation**: Cross-platform SIMD-aligned malloc.
 */

#ifndef AICRAFT_MEMORY_H
#define AICRAFT_MEMORY_H

#include "aicraft/platform.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <malloc.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup memory Memory Management
 *  Arena, pool, and aligned allocators.
 *  @{ */

/** @name Aligned Allocation
 *  Cross-platform SIMD-aligned malloc / free.
 *  @{ */

/**
 * @brief Allocate @p size bytes aligned to @p alignment.
 * @param size       Number of bytes.
 * @param alignment  Required alignment (must be power of two).
 * @return Pointer to the allocated block, or NULL on failure.
 * @see ac_aligned_free()
 */
AC_INLINE void* ac_aligned_alloc(ac_size size, ac_size alignment) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L) && !defined(__MINGW32__)
    return aligned_alloc(alignment, size);
#else
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

/**
 * @brief Free a block previously returned by ac_aligned_alloc().
 * @param ptr  Pointer to free (NULL is safe).
 */
AC_INLINE void ac_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
/** @} */ /* Aligned Allocation */

/** @name Arena Allocator
 *  Bump-pointer allocator with configurable block size.
 *  @{ */

/** Default block size: 64 MB. */
#define AC_ARENA_DEFAULT_SIZE (1024 * 1024 * 64)

/** @brief A single contiguous block in the arena chain. */
typedef struct ac_arena_block {
    uint8_t* data;                   /**< Backing buffer (SIMD-aligned). */
    ac_size  capacity;               /**< Total capacity of @c data. */
    ac_size  used;                   /**< Bytes consumed so far. */
    struct ac_arena_block* next;     /**< Next block in the linked list. */
} ac_arena_block;

/** @brief Arena allocator state. */
typedef struct {
    ac_arena_block* head;            /**< First block. */
    ac_arena_block* current;         /**< Block currently accepting allocations. */
    ac_size         block_size;      /**< Default new-block size. */
    ac_size         total_allocated; /**< Cumulative allocated bytes. */
    ac_size         num_blocks;      /**< Total number of blocks. */
} ac_arena;

/** @brief Create a single arena block of @p capacity bytes. @internal */
AC_INLINE ac_arena_block* ac_arena_block_create(ac_size capacity) {
    ac_arena_block* block = (ac_arena_block*)malloc(sizeof(ac_arena_block));
    if (!block) return NULL;
    block->data = (uint8_t*)ac_aligned_alloc(capacity, AC_SIMD_ALIGN);
    if (!block->data) { free(block); return NULL; }
    block->capacity = capacity;
    block->used = 0;
    block->next = NULL;
    return block;
}

/**
 * @brief Initialise an arena with the given default block size.
 * @param arena       Arena to initialise.
 * @param block_size  Size of each block (0 = AC_ARENA_DEFAULT_SIZE).
 */
AC_INLINE void ac_arena_init(ac_arena* arena, ac_size block_size) {
    arena->block_size = block_size > 0 ? block_size : AC_ARENA_DEFAULT_SIZE;
    arena->head = ac_arena_block_create(arena->block_size);
    arena->current = arena->head;
    arena->total_allocated = 0;
    arena->num_blocks = 1;
}

/**
 * @brief Allocate @p size bytes from the arena (SIMD-aligned).
 * @param arena  Target arena.
 * @param size   Number of bytes to allocate.
 * @return Aligned pointer, or NULL if out of memory.
 */
AC_INLINE void* ac_arena_alloc(ac_arena* arena, ac_size size) {
    /* Align to SIMD boundary */
    ac_size aligned_size = (size + AC_SIMD_ALIGN - 1) & ~(AC_SIMD_ALIGN - 1);
    
    if (arena->current->used + aligned_size > arena->current->capacity) {
        /* Need a new block */
        ac_size new_cap = aligned_size > arena->block_size ? aligned_size : arena->block_size;
        ac_arena_block* new_block = ac_arena_block_create(new_cap);
        if (!new_block) return NULL;
        arena->current->next = new_block;
        arena->current = new_block;
        arena->num_blocks++;
    }
    
    void* ptr = arena->current->data + arena->current->used;
    arena->current->used += aligned_size;
    arena->total_allocated += aligned_size;
    return ptr;
}

/** @brief Reset every block's @c used counter to zero (no deallocation). */
AC_INLINE void ac_arena_reset(ac_arena* arena) {
    ac_arena_block* block = arena->head;
    while (block) {
        block->used = 0;
        block = block->next;
    }
    arena->current = arena->head;
    arena->total_allocated = 0;
}

/**
 * @name Arena Checkpoint / Restore
 * Save arena state before a forward pass, restore after optimizer step.
 * This reclaims all intermediate tensors (activations, gradients, etc.)
 * while preserving model parameters allocated before the checkpoint.
 *
 * @code
 *   ac_arena_checkpoint cp;
 *   ac_arena_save(&g_tensor_arena, &cp);
 *   // ... forward, backward, optimizer step ...
 *   ac_arena_restore(&g_tensor_arena, &cp);  // reclaims intermediates
 * @endcode
 * @{ */

/** @brief Snapshot of arena state for later restore. */
typedef struct {
    ac_arena_block* block;           /**< Which block was current. */
    ac_size         block_used;      /**< How many bytes of that block were used. */
    ac_size         total_allocated; /**< Cumulative allocated at save time. */
} ac_arena_checkpoint;

/**
 * @brief Save current arena state into a checkpoint.
 * @param arena  Arena to snapshot.
 * @param cp     Destination checkpoint.
 */
AC_INLINE void ac_arena_save(ac_arena* arena, ac_arena_checkpoint* cp) {
    cp->block = arena->current;
    cp->block_used = arena->current->used;
    cp->total_allocated = arena->total_allocated;
}

/**
 * @brief Restore arena to a previously saved checkpoint.
 *
 * All blocks allocated after the checkpoint are freed;
 * the checkpoint block's @c used counter is rolled back.
 *
 * @param arena  Arena to restore.
 * @param cp     Checkpoint snapshot from ac_arena_save().
 */
AC_INLINE void ac_arena_restore(ac_arena* arena, ac_arena_checkpoint* cp) {
    /* Free all blocks allocated after the checkpoint block */
    ac_arena_block* block = cp->block->next;
    while (block) {
        ac_arena_block* next = block->next;
        ac_aligned_free(block->data);
        free(block);
        arena->num_blocks--;
        block = next;
    }
    cp->block->next = NULL; /* sever link to freed blocks */
    /* Restore the checkpoint block's used pointer */
    arena->current = cp->block;
    arena->current->used = cp->block_used;
    arena->total_allocated = cp->total_allocated;
}

/**
 * @brief Destroy the arena and free all underlying memory.
 * @param arena  Arena to destroy.
 */
AC_INLINE void ac_arena_destroy(ac_arena* arena) {
    ac_arena_block* block = arena->head;
    while (block) {
        ac_arena_block* next = block->next;
        ac_aligned_free(block->data);
        free(block);
        block = next;
    }
    arena->head = NULL;
    arena->current = NULL;
    arena->total_allocated = 0;
    arena->num_blocks = 0;
}
/** @} */ /* Arena Checkpoint / Restore */
/** @} */ /* Arena Allocator */

/** @name Pool Allocator
 *  Fixed-size block free-list allocator.
 *  @{ */

/** @brief Free-list intrusive node. @internal */
typedef struct ac_pool_free_node {
    struct ac_pool_free_node* next;  /**< Next free node. */
} ac_pool_free_node;

/** @brief Pool allocator state (fixed-size blocks). */
typedef struct {
    uint8_t*          memory;       /**< Contiguous backing buffer. */
    ac_pool_free_node* free_list;   /**< Head of the free list. */
    ac_size           block_size;   /**< Size of each block (SIMD-aligned). */
    ac_size           capacity;     /**< Total number of blocks. */
    ac_size           num_free;     /**< Blocks currently free. */
} ac_pool;

/**
 * @brief Initialise a pool with @p num_blocks blocks of @p block_size bytes.
 * @param pool        Pool to initialise.
 * @param block_size  Size of each block (rounded up to SIMD alignment).
 * @param num_blocks  Number of blocks to pre-allocate.
 */
AC_INLINE void ac_pool_init(ac_pool* pool, ac_size block_size, ac_size num_blocks) {
    if (block_size < sizeof(ac_pool_free_node))
        block_size = sizeof(ac_pool_free_node);
    block_size = (block_size + AC_SIMD_ALIGN - 1) & ~(AC_SIMD_ALIGN - 1);
    
    pool->block_size = block_size;
    pool->capacity = num_blocks;
    pool->num_free = num_blocks;
    pool->memory = (uint8_t*)ac_aligned_alloc(block_size * num_blocks, AC_SIMD_ALIGN);
    
    /* Build free list */
    pool->free_list = NULL;
    for (ac_size i = 0; i < num_blocks; i++) {
        ac_pool_free_node* node = (ac_pool_free_node*)(pool->memory + i * block_size);
        node->next = pool->free_list;
        pool->free_list = node;
    }
}

/**
 * @brief Allocate one block from the pool.
 * @return Pointer to a block, or NULL if the pool is exhausted.
 */
AC_INLINE void* ac_pool_alloc(ac_pool* pool) {
    if (!pool->free_list) return NULL;
    ac_pool_free_node* node = pool->free_list;
    pool->free_list = node->next;
    pool->num_free--;
    return (void*)node;
}

/**
 * @brief Return a block to the pool.
 * @param pool  Owning pool.
 * @param ptr   Block to return (NULL-safe, bounds-checked).
 */
AC_INLINE void ac_pool_free(ac_pool* pool, void* ptr) {
    if (!ptr) return;
    /* Bounds check: ensure ptr is within pool memory range */
    uint8_t* p = (uint8_t*)ptr;
    if (p < pool->memory || p >= pool->memory + pool->block_size * pool->capacity) return;
    ac_pool_free_node* node = (ac_pool_free_node*)ptr;
    node->next = pool->free_list;
    pool->free_list = node;
    pool->num_free++;
}

/** @brief Destroy the pool and free the backing buffer. */
AC_INLINE void ac_pool_destroy(ac_pool* pool) {
    ac_aligned_free(pool->memory);
    pool->memory = NULL;
    pool->free_list = NULL;
}
/** @} */ /* Pool Allocator */
/** @} */ /* memory */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_MEMORY_H */
