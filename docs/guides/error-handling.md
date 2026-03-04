---
sidebar_position: 5
title: Error Handling
---

# Error Handling

How Aicraft reports and handles errors.

## Error Strategy

Aicraft uses a simple, C-idiomatic error handling approach:

1. **Return codes** for recoverable errors
2. **Assertions** for programming errors (debug builds)
3. **Error callback** for custom handling

## Error Callback

```c
void my_error_handler(AcError err, const char *msg) {
    fprintf(stderr, "[aicraft] error %d: %s\n", err, msg);
}

int main(void) {
    ac_set_error_handler(my_error_handler);
    ac_init();
    // ...
}
```

## Error Codes

| Code | Name | Description |
|------|------|-------------|
| `AC_OK` | Success | No error |
| `AC_ERR_OOM` | Out of memory | Arena is full |
| `AC_ERR_SHAPE` | Shape mismatch | Incompatible tensor shapes |
| `AC_ERR_NULL` | Null pointer | Unexpected null argument |
| `AC_ERR_VULKAN` | Vulkan error | GPU operation failed |
| `AC_ERR_IO` | I/O error | File read/write failed |
| `AC_ERR_QUANT` | Quantisation error | Invalid quantisation params |

## Debug Mode

Compile with `-DAC_DEBUG` to enable verbose logging:

```bash
gcc -O0 -DAC_DEBUG -g demo.c -I./include -o demo_debug
```

This enables:
- Shape checks on every operation
- Memory tracking and leak detection
- Verbose operation logging to stderr

## Common Pitfalls

- **Forgetting `ac_init()`** — all operations will fail
- **Arena overflow** — increase arena size or use checkpoint/restore
- **Shape mismatches** — verify layer dimensions match
- **Missing `ac_cleanup()`** — memory leak (non-arena allocations)
