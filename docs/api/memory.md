---
sidebar_position: 7
title: Memory
---

# Memory API

`#include <aicraft/memory.h>`

## Aligned Allocation

```c
void* ac_aligned_alloc(size_t size, size_t alignment);
void  ac_aligned_free(void* ptr);
```

Cross-platform SIMD-aligned allocation:
- **Windows**: `_aligned_malloc` / `_aligned_free`
- **C11**: `aligned_alloc`
- **POSIX**: `posix_memalign`

Default alignment: 64 bytes (cache-line and AVX-512 aligned).

## Arena Allocator

The arena allocator provides fast bump-pointer allocation with bulk deallocation.

### Structure

```c
typedef struct ac_arena_block {
    uint8_t* data;
    size_t size, used;
    struct ac_arena_block* next;
} ac_arena_block;

typedef struct {
    ac_arena_block* head;
    ac_arena_block* current;
} ac_arena;
```

### Functions

```c
void  ac_arena_init(ac_arena* a, size_t block_size);    // Default: 64 MB
void* ac_arena_alloc(ac_arena* a, size_t size);          // Bump-pointer alloc
void  ac_arena_reset(ac_arena* a);                       // Reset all blocks
void  ac_arena_destroy(ac_arena* a);                     // Free all memory
```

### Checkpoint / Restore

The key optimization for training loops:

```c
ac_arena_checkpoint cp;

// Save current position
ac_arena_save(&g_tensor_arena, &cp);

// ... allocate intermediates (forward pass, backward, etc.) ...

// Restore — all allocations since save are instantly freed
ac_arena_restore(&g_tensor_arena, &cp);
```

```
Memory layout during training:

┌──────────────┬──────────────────────────┐
│ Model params │ Intermediates (freed)     │
│ (survive)    │◄── checkpoint here        │
└──────────────┴──────────────────────────┘
                ↑ save()          restore() ↑
```

### Properties

- **O(1) allocation**: Bump pointer, no free-list traversal
- **O(1) bulk free**: Reset pointer position
- **SIMD-aligned**: All allocations are 64-byte aligned
- **Auto-growing**: New 64 MB blocks allocated as needed
- **No fragmentation**: Linear allocation pattern

## Pool Allocator

Fixed-size free-list allocator for uniform objects.

```c
typedef struct {
    void* buffer;
    size_t obj_size, capacity;
    ac_pool_free_node* free_list;
} ac_pool;
```

```c
ac_pool pool;
ac_pool_init(&pool, sizeof(my_struct), 1024);  // 1024 objects

void* obj = ac_pool_alloc(&pool);   // O(1) from free list
ac_pool_free(&pool, obj);           // O(1) return to free list

ac_pool_destroy(&pool);
```

Includes bounds checking: `ac_pool_free` validates that the pointer belongs to the pool's buffer range.

## Global Arena

```c
extern ac_arena g_tensor_arena;  // Used by all tensor allocations
```

Initialized by `ac_init()`, destroyed by `ac_cleanup()`.
