/**
 * @file vulkan.h
 * @brief Vulkan compute backend for GPU-accelerated tensor operations.
 *
 * Provides a GPU compute pipeline using Vulkan 1.0+ for offloading
 * heavy operations (GEMM, element-wise, activations, reductions) to
 * the GPU.  Falls back transparently to the CPU SIMD path when Vulkan
 * is unavailable or the tensor is too small to benefit from GPU dispatch.
 *
 * @par Design
 *   - Compute-only (no graphics / no window surface required)
 *   - Dynamic Vulkan function loading (no compile-time Vulkan SDK dependency)
 *   - Shader SPIR-V embedded as C arrays or loaded from .spv files
 *   - Double-buffered staging for async host↔device transfers
 *   - Descriptor-set caching — one pipeline per kernel, reused across calls
 *
 * @par Usage
 * @code{.c}
 *   ac_init();                             // framework init (calls ac_vk_init internally)
 *   ac_tensor* a = ac_tensor_2d(1024, 1024, 0);
 *   ac_tensor* b = ac_tensor_2d(1024, 1024, 0);
 *   ac_tensor* c = ac_tensor_2d(1024, 1024, 0);
 *   // ... fill a, b ...
 *   ac_vk_gemm(a, b, c);                   // runs on GPU if available
 * @endcode
 *
 * @defgroup vulkan Vulkan Compute Backend
 * @{
 */

#ifndef AICRAFT_VULKAN_H
#define AICRAFT_VULKAN_H

#include "aicraft/platform.h"
#include "aicraft/error.h"

/* ── Guard: only compile Vulkan support if requested ─────────────────── */
#ifdef AICRAFT_ENABLE_VULKAN

#ifdef _WIN32
    #define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__linux__)
    #define VK_USE_PLATFORM_XLIB_KHR
#elif defined(__APPLE__)
    #define VK_USE_PLATFORM_MACOS_MVK
#endif

#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ====================================================================== */
/*  Constants                                                              */
/* ====================================================================== */

/** Maximum number of GPU buffers tracked by the context. */
#define AC_VK_MAX_BUFFERS       256

/** Maximum number of cached compute pipelines. */
#define AC_VK_MAX_PIPELINES     32

/** Workgroup size for 1-D compute shaders. */
#define AC_VK_WORKGROUP_SIZE    256

/** Minimum element count to prefer GPU dispatch over CPU SIMD. */
#define AC_VK_MIN_GPU_ELEMENTS  4096

/** Staging buffer size for host ↔ device transfers (16 MB). */
#define AC_VK_STAGING_SIZE      (1024 * 1024 * 16)

/* ====================================================================== */
/*  Shader IDs (index into pipeline cache)                                 */
/* ====================================================================== */

/** @brief Identifiers for each compiled compute shader / pipeline. */
typedef enum {
    AC_VK_SHADER_ADD = 0,       /**< Element-wise addition. */
    AC_VK_SHADER_MUL,           /**< Element-wise multiplication. */
    AC_VK_SHADER_SCALE,         /**< Scalar-vector multiply. */
    AC_VK_SHADER_FMA,           /**< Fused multiply-add. */
    AC_VK_SHADER_RELU,          /**< ReLU activation. */
    AC_VK_SHADER_SIGMOID,       /**< Sigmoid activation. */
    AC_VK_SHADER_TANH,          /**< Tanh activation. */
    AC_VK_SHADER_SOFTMAX,       /**< Softmax (two-pass). */
    AC_VK_SHADER_GEMM,          /**< Tiled matrix multiplication. */
    AC_VK_SHADER_SUM,           /**< Reduction sum. */
    AC_VK_SHADER_MAX,           /**< Reduction max. */
    AC_VK_SHADER_RELU_BACKWARD, /**< ReLU backward pass. */
    AC_VK_SHADER_SIGMOID_BACKWARD,/**< Sigmoid backward pass. */
    AC_VK_SHADER_TANH_BACKWARD, /**< Tanh backward pass. */
    AC_VK_SHADER_COUNT          /**< Sentinel — total number of shaders. */
} ac_vk_shader_id;

/* ====================================================================== */
/*  GPU buffer handle                                                      */
/* ====================================================================== */

/**
 * @brief Wrapper around a Vulkan device-local buffer + its memory.
 *
 * Buffers are created as STORAGE_BUFFER usage so they can be bound
 * directly to compute shader descriptors.
 */
typedef struct {
    VkBuffer        buffer;     /**< Vulkan buffer handle. */
    VkDeviceMemory  memory;     /**< Backing device memory. */
    VkDeviceSize    size;       /**< Size in bytes. */
    int             in_use;     /**< Non-zero if this slot is occupied. */
} ac_vk_buffer;

/* ====================================================================== */
/*  Compute pipeline                                                       */
/* ====================================================================== */

/**
 * @brief Cached compute pipeline (shader module + pipeline + layout).
 */
typedef struct {
    VkShaderModule          shader_module;   /**< Compiled SPIR-V module. */
    VkPipelineLayout        pipeline_layout; /**< Push-constant & descriptor layout. */
    VkDescriptorSetLayout   desc_layout;     /**< Descriptor set layout. */
    VkPipeline              pipeline;        /**< Compute pipeline handle. */
    int                     valid;           /**< Non-zero if initialised. */
} ac_vk_pipeline;

/* ====================================================================== */
/*  Vulkan context (singleton)                                             */
/* ====================================================================== */

/**
 * @brief Top-level Vulkan compute context.
 *
 * Holds instance, device, queue, command pool, descriptor pool, staging
 * buffers, and all cached pipelines.  Created once by @ref ac_vk_init
 * and destroyed by @ref ac_vk_cleanup.
 */
typedef struct {
    /* --- Vulkan core --- */
    VkInstance                  instance;
    VkPhysicalDevice            physical_device;
    VkDevice                    device;
    VkQueue                     compute_queue;
    uint32_t                    compute_queue_family;

    /* --- Physical device info --- */
    VkPhysicalDeviceProperties  device_props;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkDeviceSize                max_alloc_size;

    /* --- Command submission --- */
    VkCommandPool               cmd_pool;
    VkCommandBuffer             cmd_buf;        /**< Reusable primary cmd buffer. */
    VkFence                     fence;          /**< Fence for sync after dispatch. */

    /* --- Descriptors --- */
    VkDescriptorPool            desc_pool;

    /* --- Staging (host-visible) --- */
    VkBuffer                    staging_buf;
    VkDeviceMemory              staging_mem;
    void*                       staging_mapped; /**< Persistently mapped pointer. */

    /* --- Pipeline cache --- */
    ac_vk_pipeline              pipelines[AC_VK_MAX_PIPELINES];

    /* --- State flags --- */
    int                         initialized;    /**< Non-zero after successful init. */
    int                         available;       /**< Non-zero if GPU is usable for compute. */
} ac_vk_context;

/** @brief Global Vulkan context singleton. */
extern ac_vk_context g_vk_ctx;

/* ====================================================================== */
/*  Push-constant structures (mirror the GLSL layout)                      */
/* ====================================================================== */

/** @brief Push constants for element-wise kernels (add, mul, relu …). */
typedef struct {
    uint32_t n;         /**< Number of elements. */
    float    scalar;    /**< Scalar parameter (used by scale, etc.). */
} ac_vk_push_elementwise;

/** @brief Push constants for the GEMM kernel. */
typedef struct {
    uint32_t M;     /**< Rows of A / C. */
    uint32_t N;     /**< Columns of B / C. */
    uint32_t K;     /**< Columns of A / rows of B. */
    float    alpha;  /**< Scalar multiplier. */
    float    beta;   /**< Scalar for C accumulation. */
} ac_vk_push_gemm;

/** @brief Push constants for the softmax kernel. */
typedef struct {
    uint32_t rows;      /**< Number of rows. */
    uint32_t cols;      /**< Number of columns (softmax dimension). */
} ac_vk_push_softmax;

/* ====================================================================== */
/*  Lifecycle                                                              */
/* ====================================================================== */

/**
 * @brief Initialise the Vulkan compute backend.
 *
 * Creates a Vulkan instance (compute-only, no extensions), selects the
 * best discrete GPU (or falls back to integrated), creates a logical
 * device with a compute queue, and prepares the command pool, descriptor
 * pool, and staging buffer.
 *
 * Safe to call multiple times — subsequent calls are no-ops.
 *
 * @return AC_OK on success, or an error code if Vulkan is unavailable.
 * @see ac_vk_cleanup
 */
ac_error_code ac_vk_init(void);

/**
 * @brief Destroy all Vulkan resources.
 *
 * Releases pipelines, descriptor pool, command pool, staging buffer,
 * device, and instance.  After cleanup the backend must be re-initialised
 * with ac_vk_init() before further use.
 *
 * @see ac_vk_init
 */
void ac_vk_cleanup(void);

/**
 * @brief Check whether the Vulkan backend is available and ready.
 * @return Non-zero if GPU compute is available.
 */
AC_INLINE int ac_vk_is_available(void) {
    return g_vk_ctx.initialized && g_vk_ctx.available;
}

/**
 * @brief Return the name of the selected GPU.
 * @return Device name string, or "N/A" if Vulkan is not initialised.
 */
AC_INLINE const char* ac_vk_device_name(void) {
    return g_vk_ctx.initialized
         ? g_vk_ctx.device_props.deviceName
         : "N/A";
}

/* ====================================================================== */
/*  Buffer management                                                      */
/* ====================================================================== */

/**
 * @brief Create a device-local GPU buffer.
 * @param size  Size in bytes.
 * @param[out] buf  Pointer to the buffer handle to fill.
 * @return AC_OK on success.
 */
ac_error_code ac_vk_buffer_create(VkDeviceSize size, ac_vk_buffer* buf);

/**
 * @brief Destroy a GPU buffer and free its memory.
 * @param buf  Buffer to destroy.
 */
void ac_vk_buffer_destroy(ac_vk_buffer* buf);

/**
 * @brief Upload data from host memory to a device-local buffer.
 * @param buf   Destination GPU buffer.
 * @param data  Source host pointer.
 * @param size  Number of bytes to upload.
 * @return AC_OK on success.
 */
ac_error_code ac_vk_buffer_upload(const ac_vk_buffer* buf,
                                  const void* data, VkDeviceSize size);

/**
 * @brief Download data from a device-local buffer to host memory.
 * @param buf   Source GPU buffer.
 * @param data  Destination host pointer.
 * @param size  Number of bytes to download.
 * @return AC_OK on success.
 */
ac_error_code ac_vk_buffer_download(const ac_vk_buffer* buf,
                                    void* data, VkDeviceSize size);

/* ====================================================================== */
/*  Pipeline / shader management                                           */
/* ====================================================================== */

/**
 * @brief Load or retrieve a cached compute pipeline.
 *
 * On first call for a given shader ID, compiles the embedded SPIR-V,
 * creates the pipeline layout and descriptor set layout, and caches
 * the result.  Subsequent calls return the cached pipeline.
 *
 * @param id  Shader identifier.
 * @return Pointer to the pipeline, or NULL on failure.
 */
ac_vk_pipeline* ac_vk_get_pipeline(ac_vk_shader_id id);

/* ====================================================================== */
/*  High-level compute operations                                          */
/* ====================================================================== */

/**
 * @brief GPU matrix multiplication: C = alpha * A @ B + beta * C.
 *
 * Uses a tiled GEMM shader with shared-memory blocking.
 *
 * @param A     Input tensor (M × K).
 * @param B     Input tensor (K × N).
 * @param C     Output tensor (M × N).
 * @param alpha Scalar multiplier for A@B.
 * @param beta  Scalar multiplier for existing C values.
 * @return AC_OK on success.
 */
ac_error_code ac_vk_gemm(const float* A, const float* B, float* C,
                         uint32_t M, uint32_t N, uint32_t K,
                         float alpha, float beta);

/**
 * @brief GPU element-wise addition: out[i] = a[i] + b[i].
 */
ac_error_code ac_vk_add(const float* a, const float* b, float* out,
                        uint32_t n);

/**
 * @brief GPU element-wise multiplication: out[i] = a[i] * b[i].
 */
ac_error_code ac_vk_mul(const float* a, const float* b, float* out,
                        uint32_t n);

/**
 * @brief GPU scalar-vector multiply: out[i] = a[i] * scalar.
 */
ac_error_code ac_vk_scale(const float* a, float scalar, float* out,
                          uint32_t n);

/**
 * @brief GPU fused multiply-add: out[i] = a[i] * b[i] + c[i].
 */
ac_error_code ac_vk_fma(const float* a, const float* b, const float* c,
                        float* out, uint32_t n);

/**
 * @brief GPU ReLU activation: out[i] = max(0, x[i]).
 */
ac_error_code ac_vk_relu(const float* x, float* out, uint32_t n);

/**
 * @brief GPU sigmoid activation: out[i] = 1 / (1 + exp(-x[i])).
 */
ac_error_code ac_vk_sigmoid(const float* x, float* out, uint32_t n);

/**
 * @brief GPU tanh activation: out[i] = tanh(x[i]).
 */
ac_error_code ac_vk_tanh_act(const float* x, float* out, uint32_t n);

/**
 * @brief GPU softmax: out[r][c] = exp(x[r][c]) / sum(exp(x[r][:])).
 * @param x     Input (rows × cols).
 * @param out   Output (rows × cols).
 * @param rows  Number of rows.
 * @param cols  Number of columns (softmax axis).
 */
ac_error_code ac_vk_softmax(const float* x, float* out,
                            uint32_t rows, uint32_t cols);

/**
 * @brief GPU reduction sum.
 * @param x   Input array.
 * @param n   Number of elements.
 * @param[out] result  Scalar sum result.
 */
ac_error_code ac_vk_sum(const float* x, uint32_t n, float* result);

/* ====================================================================== */
/*  Auto-dispatch helpers                                                  */
/* ====================================================================== */

/**
 * @brief Decide whether GPU dispatch is worthwhile for @p n elements.
 *
 * Returns non-zero if Vulkan is available AND the workload is large
 * enough to justify the overhead of buffer upload + dispatch + download.
 *
 * @param n  Number of elements in the operation.
 * @return Non-zero if GPU should be used.
 */
AC_INLINE int ac_vk_should_use_gpu(ac_size n) {
    return ac_vk_is_available() && (n >= AC_VK_MIN_GPU_ELEMENTS);
}

/**
 * @brief Print a summary of the Vulkan backend to @p stream.
 * @param stream  Output file (e.g. stdout).
 */
void ac_vk_print_info(FILE* stream);

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_ENABLE_VULKAN */

/** @} */ /* vulkan */

#endif /* AICRAFT_VULKAN_H */
