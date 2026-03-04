---
sidebar_position: 7
title: Memory API
---

# Memory API

`#include "aicraft/memory.h"`

## Arena Allocator

Aicraft uses a custom arena allocator for all tensor memory. This provides:

- **O(1) allocation** — simple bump pointer
- **O(1) deallocation** — restore to checkpoint
- **Zero fragmentation** — contiguous memory block
- **Constant memory** during training loops

## Initialisation

```c
ac_init();                     // Default arena (16 MB)
ac_init_with_arena(64 * 1024); // Custom arena size (64 KB)
```

## Checkpoint / Restore

```c
AcMemCheckpoint cp = ac_mem_checkpoint();

// Allocate tensors, run forward/backward...
AcTensor *x = ac_tensor_rand((int[]){1, 784}, 2);
AcTensor *y = ac_forward_seq(net, 2, x);

ac_mem_restore(cp);  // Free everything since checkpoint
```

## Memory Query

```c
size_t used  = ac_mem_used();      // Bytes currently allocated
size_t total = ac_mem_capacity();  // Total arena size
size_t free  = ac_mem_free();      // Remaining capacity
```

## Why Not malloc?

| Feature | Arena | malloc/free |
|---------|-------|-------------|
| Allocation speed | O(1) bump | O(n) search |
| Deallocation | O(1) restore | O(1) per free |
| Fragmentation | None | Grows over time |
| Bulk free | Checkpoint/restore | Manual tracking |
| Overhead | 0 bytes/alloc | 16-32 bytes/alloc |

## Training Loop Pattern

```c
for (int epoch = 0; epoch < N; epoch++) {
    for (int batch = 0; batch < batches; batch++) {
        AcMemCheckpoint cp = ac_mem_checkpoint();

        // All intermediate tensors allocated here
        AcTensor *pred = ac_forward_seq(net, 2, x);
        AcTensor *loss = ac_cross_entropy(pred, target);
        ac_backward(loss);
        ac_optimizer_step(opt);

        ac_mem_restore(cp);  // Memory usage stays constant
    }
}
```
