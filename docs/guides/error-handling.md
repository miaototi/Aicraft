---
sidebar_position: 5
title: Error Handling
---

# Error Handling

Aicraft uses a production-grade error handling system based on error codes and optional user callbacks — no C++ exceptions, no `abort()` in release mode.

## Error Codes

```c
typedef enum {
    AC_OK = 0,
    AC_ERROR_ALLOC,       // Memory allocation failure
    AC_ERROR_SHAPE,       // Shape mismatch
    AC_ERROR_NULL,        // Null pointer
    AC_ERROR_IO,          // File I/O error
    AC_ERROR_FORMAT,      // Invalid file format
    AC_ERROR_OVERFLOW,    // Buffer overflow
    AC_ERROR_INVALID,     // Invalid argument
    AC_ERROR_VULKAN,      // Vulkan backend error
} ac_error_code;
```

## Checking Errors

```c
ac_error_code err = ac_model_load("weights.acml", &params);
if (err != AC_OK) {
    printf("Error: %s\n", ac_error_string(err));  // Human-readable name
    printf("Detail: %s\n", ac_get_last_error_message());  // Full message
    printf("At: %s:%d\n", g_last_error.file, g_last_error.line);
    ac_clear_error();  // Reset error state
}
```

## Custom Error Handler

Register a callback to be notified of all errors:

```c
void my_handler(const ac_error* err, void* user_data) {
    FILE* log = (FILE*)user_data;
    fprintf(log, "[AICRAFT ERROR] %s:%d — %s (code %d)\n",
            err->file, err->line, err->message, err->code);
}

// Register handler
FILE* logfile = fopen("aicraft.log", "w");
ac_set_error_handler(my_handler, logfile);
```

## Error Macros

Internal macros for reporting errors from library code:

```c
// Reports error with file/line info
AC_RETURN_ERROR(AC_ERROR_SHAPE, "matmul: inner dimensions mismatch (%d vs %d)", a, b);

// Check condition and return error
AC_CHECK(ptr != NULL, AC_ERROR_NULL, "input tensor is NULL");
```

## Debug Mode

In debug builds (`NDEBUG` not defined), errors are also printed to `stderr` for immediate visibility during development.
