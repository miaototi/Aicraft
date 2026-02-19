/**
 * @file error.h
 * @brief Production-grade error handling with codes, messages, and user callbacks.
 *
 * Replaces raw @c assert() with structured error reporting that allows
 * graceful recovery.  Every error sets a thread-local (simplified: global)
 * error state that can be queried with ac_get_last_error() and cleared
 * with ac_clear_error().  An optional user callback is invoked on every
 * error for custom logging or recovery.
 */

#ifndef AICRAFT_ERROR_H
#define AICRAFT_ERROR_H

#include "aicraft/platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup error Error Handling
 *  Error codes, state management, callbacks, and convenience macros.
 *  @{ */

/** @brief Error codes returned by Aicraft functions. */
typedef enum {
    AC_OK = 0,                /**< Success — no error. */
    AC_ERR_NULL_PTR,          /**< NULL pointer passed to an API that requires non-NULL. */
    AC_ERR_OUT_OF_MEMORY,     /**< Arena or system allocation failed. */
    AC_ERR_SHAPE_MISMATCH,    /**< Tensor shapes are incompatible for the operation. */
    AC_ERR_INVALID_DIM,       /**< Dimensionality is invalid (e.g. 5-D to matmul). */
    AC_ERR_INVALID_ARGUMENT,  /**< A generic bad argument. */
    AC_ERR_OVERFLOW,          /**< Graph or parameter list exceeded capacity. */
    AC_ERR_FILE_IO,           /**< File read/write failure. */
    AC_ERR_GRAPH_TOO_LARGE,   /**< Autograd graph exceeds the maximum number of nodes. */
    AC_ERR_UNSUPPORTED_OP,    /**< Operation not supported (e.g. missing backward). */
    AC_ERR_INTERNAL,          /**< Unexpected internal error. */
} ac_error_code;

/** @brief Snapshot of the last error — code + source location + message. */
typedef struct {
    ac_error_code code;       /**< Numeric error code. */
    const char*   file;       /**< Source file where the error was raised. */
    int           line;       /**< Source line number. */
    char          message[256]; /**< Human-readable description. */
} ac_error;

/** Global (thread-local simplified) last-error state. */
extern ac_error g_last_error;

/**
 * @brief User-defined error callback signature.
 * @param err        Pointer to the error that just occurred.
 * @param user_data  Opaque pointer passed during registration.
 */
typedef void (*ac_error_handler)(const ac_error* err, void* user_data);
extern ac_error_handler g_error_handler;  /**< Currently registered handler (NULL = none). */
extern void* g_error_user_data;           /**< Opaque data forwarded to the handler. */

/** @name Error Query API
 *  @{ */

/**
 * @brief Return a human-readable string for an error code.
 * @param code  The error code to stringify.
 * @return Static string (never NULL).
 */
AC_INLINE const char* ac_error_string(ac_error_code code) {
    switch (code) {
        case AC_OK:                 return "OK";
        case AC_ERR_NULL_PTR:       return "Null pointer";
        case AC_ERR_OUT_OF_MEMORY:  return "Out of memory";
        case AC_ERR_SHAPE_MISMATCH: return "Shape mismatch";
        case AC_ERR_INVALID_DIM:    return "Invalid dimension";
        case AC_ERR_INVALID_ARGUMENT: return "Invalid argument";
        case AC_ERR_OVERFLOW:       return "Overflow (graph/param limit exceeded)";
        case AC_ERR_FILE_IO:        return "File I/O error";
        case AC_ERR_GRAPH_TOO_LARGE: return "Computation graph exceeds max nodes";
        case AC_ERR_UNSUPPORTED_OP: return "Unsupported operation";
        case AC_ERR_INTERNAL:       return "Internal error";
        default:                    return "Unknown error";
    }
}

/**
 * @brief Register a user-defined error callback.
 * @param handler    Function to call on every error (NULL to unregister).
 * @param user_data  Opaque pointer forwarded to @p handler.
 */
AC_INLINE void ac_set_error_handler(ac_error_handler handler, void* user_data) {
    g_error_handler = handler;
    g_error_user_data = user_data;
}

/**
 * @brief Record an error and optionally invoke the user callback.
 * @param code  Error code (e.g. AC_ERR_NULL_PTR).
 * @param file  Source file name (typically __FILE__).
 * @param line  Source line number (typically __LINE__).
 * @param fmt   printf-style format string (may be NULL).
 * @param ...   Format arguments.
 */
static void ac_set_error(ac_error_code code, const char* file, int line,
                                   const char* fmt, ...) {
    g_last_error.code = code;
    g_last_error.file = file;
    g_last_error.line = line;
    
    /* Format message */
    if (fmt) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(g_last_error.message, sizeof(g_last_error.message), fmt, args);
        va_end(args);
    } else {
        strncpy(g_last_error.message, ac_error_string(code), 
                sizeof(g_last_error.message) - 1);
    }
    
    /* Call user handler if registered */
    if (g_error_handler) {
        g_error_handler(&g_last_error, g_error_user_data);
    }
    
#ifndef NDEBUG
    /* In debug builds, print to stderr */
    fprintf(stderr, "[AICRAFT ERROR] %s:%d: %s (code=%d)\n",
            file, line, g_last_error.message, code);
#endif
}

/** @brief Return the last error code (AC_OK if none). */
AC_INLINE ac_error_code ac_get_last_error(void) {
    return g_last_error.code;
}

/** @brief Return the message from the last error (empty string if none). */
AC_INLINE const char* ac_get_last_error_message(void) {
    return g_last_error.message;
}

/** @brief Clear the last-error state back to AC_OK. */
AC_INLINE void ac_clear_error(void) {
    g_last_error.code = AC_OK;
    g_last_error.message[0] = '\0';
}
/** @} */ /* Error Query API */

/** @name Error Convenience Macros
 *  Report errors and/or return from the calling function.
 *  @{ */

/** @brief Report error and @c return @c NULL. */
#define AC_ERROR_NULL(code, ...) do { \
    ac_set_error((code), __FILE__, __LINE__, __VA_ARGS__); \
    return NULL; \
} while(0)

/** @brief Report error and @c return the error code. */
#define AC_ERROR_RET(code, ...) do { \
    ac_set_error((code), __FILE__, __LINE__, __VA_ARGS__); \
    return (code); \
} while(0)

/** @brief Check condition; report error and @c return @c NULL on failure. */
#define AC_CHECK_NULL(cond, code, ...) do { \
    if (!(cond)) { \
        ac_set_error((code), __FILE__, __LINE__, __VA_ARGS__); \
        return NULL; \
    } \
} while(0)

/** @brief Check condition; report error and @c return code on failure. */
#define AC_CHECK_RET(cond, code, ...) do { \
    if (!(cond)) { \
        ac_set_error((code), __FILE__, __LINE__, __VA_ARGS__); \
        return (code); \
    } \
} while(0)
/** @} */ /* Error Convenience Macros */
/** @} */ /* error */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_ERROR_H */
