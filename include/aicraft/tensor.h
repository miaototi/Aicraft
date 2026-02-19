/**
 * @file tensor.h
 * @brief Lightweight tensor with autograd support and SIMD-aligned storage.
 *
 * Provides the core tensor data structure, shape utilities, element-wise fill
 * operations, pseudorandom number generation (xoshiro128**), weight
 * initialisation helpers (Xavier / He / Uniform), gradient management, and
 * pretty-printing.
 *
 * No dynamic dispatch overhead — pure compile-time polymorphism.
 *
 * @defgroup tensor Tensor Core
 * @{
 */

#ifndef AICRAFT_TENSOR_H
#define AICRAFT_TENSOR_H

#include "aicraft/platform.h"
#include "aicraft/memory.h"
#include "aicraft/simd_math.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @name Tensor Shape
 *  Utilities for describing and constructing N-dimensional shapes.
 *  @{
 */

/** Maximum number of dimensions a tensor shape can hold. */
#define AC_MAX_DIMS 8

/**
 * @brief Compact N-dimensional shape descriptor.
 *
 * Stores up to @ref AC_MAX_DIMS dimension sizes together with the
 * pre-computed total element count for fast allocation and iteration.
 */
typedef struct {
    ac_size dims[AC_MAX_DIMS]; /**< Size of each dimension. */
    ac_size ndim;              /**< Number of active dimensions. */
    ac_size total_size;        /**< Cached total element count (product of dims). */
} ac_shape;

/**
 * @brief Build an ac_shape from an explicit dimension array.
 * @param ndim  Number of dimensions (clamped to @ref AC_MAX_DIMS).
 * @param dims  Pointer to @p ndim dimension sizes.
 * @return A fully initialised ac_shape with total_size computed.
 */
AC_INLINE ac_shape ac_shape_make(ac_size ndim, const ac_size* dims) {
    ac_shape s;
    memset(&s, 0, sizeof(s));
    if (ndim > AC_MAX_DIMS) ndim = AC_MAX_DIMS; /* prevent buffer overflow */
    s.ndim = ndim;
    s.total_size = 1;
    for (ac_size i = 0; i < ndim; i++) {
        s.dims[i] = dims[i];
        s.total_size *= dims[i];
    }
    return s;
}

/**
 * @brief Convenience: 1-D shape.
 * @param d0 Size of the single dimension.
 * @return A 1-D ac_shape.
 */
AC_INLINE ac_shape ac_shape_1d(ac_size d0) {
    ac_size dims[] = {d0};
    return ac_shape_make(1, dims);
}

/**
 * @brief Convenience: 2-D shape (rows × cols).
 * @param d0 Number of rows.
 * @param d1 Number of columns.
 * @return A 2-D ac_shape.
 */
AC_INLINE ac_shape ac_shape_2d(ac_size d0, ac_size d1) {
    ac_size dims[] = {d0, d1};
    return ac_shape_make(2, dims);
}

/**
 * @brief Convenience: 3-D shape.
 * @param d0 First dimension size.
 * @param d1 Second dimension size.
 * @param d2 Third dimension size.
 * @return A 3-D ac_shape.
 */
AC_INLINE ac_shape ac_shape_3d(ac_size d0, ac_size d1, ac_size d2) {
    ac_size dims[] = {d0, d1, d2};
    return ac_shape_make(3, dims);
}

/**
 * @brief Convenience: 4-D shape (e.g. batch × channels × height × width).
 * @param d0 First dimension size.
 * @param d1 Second dimension size.
 * @param d2 Third dimension size.
 * @param d3 Fourth dimension size.
 * @return A 4-D ac_shape.
 */
AC_INLINE ac_shape ac_shape_4d(ac_size d0, ac_size d1, ac_size d2, ac_size d3) {
    ac_size dims[] = {d0, d1, d2, d3};
    return ac_shape_make(4, dims);
}

/** @} */ /* end Tensor Shape */

/** @name Autograd Operation Tags
 *  Enum values that record which forward operation produced a tensor so the
 *  backward pass can dispatch the correct gradient rule.
 *  @{
 */

/**
 * @brief Tags for every differentiable operation tracked by the autograd graph.
 * @see ac_tensor::op
 */
typedef enum {
    AC_OP_NONE = 0,     /**< Leaf tensor / no operation. */
    AC_OP_ADD,          /**< Element-wise addition. */
    AC_OP_SUB,          /**< Element-wise subtraction. */
    AC_OP_MUL,          /**< Element-wise multiplication. */
    AC_OP_DIV,          /**< Element-wise division. */
    AC_OP_MATMUL,       /**< Matrix multiplication. */
    AC_OP_RELU,         /**< Rectified Linear Unit activation. */
    AC_OP_SIGMOID,      /**< Sigmoid activation. */
    AC_OP_TANH,         /**< Hyperbolic tangent activation. */
    AC_OP_SOFTMAX,      /**< Softmax normalisation. */
    AC_OP_MSE_LOSS,     /**< Mean Squared Error loss. */
    AC_OP_CE_LOSS,      /**< Cross-Entropy loss. */
    AC_OP_RESHAPE,      /**< Reshape (view) operation. */
    AC_OP_SCALE,        /**< Scalar multiplication. */
    AC_OP_BIAS_ADD,     /**< Bias addition. */
    AC_OP_CONV2D,       /**< 2-D convolution. */
    AC_OP_MAXPOOL,      /**< 2-D max-pooling. */
    AC_OP_BATCHNORM,    /**< Batch normalisation. */
    AC_OP_DROPOUT,      /**< Dropout regularisation. */
    AC_OP_FLATTEN,      /**< Flatten to 1-D. */
    AC_OP_SUM,          /**< Reduction sum. */
    AC_OP_MEAN,         /**< Reduction mean. */
    AC_OP_BCE_LOSS,     /**< Binary Cross-Entropy loss. */
} ac_op_type;

/** @} */ /* end Autograd Operation Tags */

/** @name Tensor Core Structure
 *  The primary tensor type that holds data, gradient, shape, and autograd
 *  book-keeping.
 *  @{
 */

/**
 * @brief Core tensor with optional autograd metadata.
 *
 * All data pointers are SIMD-aligned and allocated from the global arena.
 * The autograd graph links each result tensor back to at most two parent
 * tensors so that gradients can be propagated during back-propagation.
 *
 * @see ac_tensor_create, ac_tensor_alloc
 */
typedef struct ac_tensor {
    float*    data;       /**< SIMD-aligned float storage. */
    float*    grad;       /**< Gradient buffer (NULL if @c requires_grad is 0). */
    ac_shape  shape;      /**< Shape descriptor. */

    /* Autograd graph */
    ac_op_type          op;             /**< Operation that created this tensor. */
    struct ac_tensor*   parents[2];     /**< Parent tensors (max 2 for binary ops). */
    int                 requires_grad;  /**< Non-zero if gradients are tracked. */
    int                 grad_computed;  /**< Non-zero after backward has run. */
    int                 visited_epoch;  /**< Epoch counter for O(1) visited check. */

    /* Metadata */
    float     scalar;         /**< Cached scalar for AC_OP_SCALE backward. */
    float*    aux;            /**< Auxiliary buffer (dropout mask, im2col, …). */
    ac_size   aux_size;       /**< Size of @c aux buffer in bytes. */
    int       owns_data;      /**< Whether the arena owns / should free data. */
    int       is_contiguous;  /**< Non-zero if memory layout is contiguous. */
} ac_tensor;

/** @} */ /* end Tensor Core Structure */

/** @name Global Arena
 *  Process-wide arena used for all tensor data and gradient allocations.
 *  @{
 */

/** @brief Global arena instance for tensor storage. */
extern ac_arena g_tensor_arena;
/** @brief Flag indicating whether @ref g_tensor_arena has been initialised. */
extern int g_arena_initialized;

/**
 * @brief Lazily initialise the global tensor arena.
 *
 * Called automatically by tensor creation helpers.  Safe to call multiple
 * times; initialisation happens only once.
 *
 * @note Uses @ref AC_ARENA_DEFAULT_SIZE as the initial capacity.
 * @see g_tensor_arena
 */
static AC_INLINE void ac_ensure_arena(void) {
    if (!g_arena_initialized) {
        ac_arena_init(&g_tensor_arena, AC_ARENA_DEFAULT_SIZE);
        g_arena_initialized = 1;
    }
}

/** @} */ /* end Global Arena */

/** @name Tensor Creation
 *  Functions for allocating and constructing tensors.
 *  @{
 */

/**
 * @brief Allocate a zero-initialised ac_tensor from the global arena.
 *
 * Does **not** allocate the data buffer; use @ref ac_tensor_create for full
 * construction.
 *
 * @return Pointer to the newly allocated tensor.
 * @see ac_tensor_create
 */
static AC_INLINE ac_tensor* ac_tensor_alloc(void) {
    ac_ensure_arena();
    ac_tensor* t = (ac_tensor*)ac_arena_alloc(&g_tensor_arena, sizeof(ac_tensor));
    memset(t, 0, sizeof(ac_tensor));
    t->is_contiguous = 1;
    return t;
}

/**
 * @brief Create a tensor with the given shape and optional gradient storage.
 *
 * Allocates SIMD-aligned data (and gradient) buffers from the global arena,
 * zero-filled.
 *
 * @param shape         Desired tensor shape.
 * @param requires_grad Non-zero to allocate a gradient buffer.
 * @return Pointer to the newly created tensor.
 * @see ac_tensor_alloc, ac_tensor_1d, ac_tensor_2d
 */
static AC_INLINE ac_tensor* ac_tensor_create(ac_shape shape, int requires_grad) {
    ac_ensure_arena();
    ac_tensor* t = ac_tensor_alloc();
    t->shape = shape;
    
    /* Align size to SIMD boundary */
    ac_size aligned_size = (shape.total_size + AC_SIMD_WIDTH - 1) & ~(AC_SIMD_WIDTH - 1);
    t->data = (float*)ac_arena_alloc(&g_tensor_arena, aligned_size * sizeof(float));
    memset(t->data, 0, aligned_size * sizeof(float));
    
    t->requires_grad = requires_grad;
    if (requires_grad) {
        t->grad = (float*)ac_arena_alloc(&g_tensor_arena, aligned_size * sizeof(float));
        memset(t->grad, 0, aligned_size * sizeof(float));
    }
    
    t->owns_data = 1;
    t->op = AC_OP_NONE;
    return t;
}

/**
 * @brief Convenience: create a 1-D tensor.
 * @param n             Number of elements.
 * @param requires_grad Non-zero to allocate gradient storage.
 * @return Pointer to the new 1-D tensor.
 */
static AC_INLINE ac_tensor* ac_tensor_1d(ac_size n, int requires_grad) {
    return ac_tensor_create(ac_shape_1d(n), requires_grad);
}

/**
 * @brief Convenience: create a 2-D tensor.
 * @param rows          Number of rows.
 * @param cols          Number of columns.
 * @param requires_grad Non-zero to allocate gradient storage.
 * @return Pointer to the new 2-D tensor.
 */
static AC_INLINE ac_tensor* ac_tensor_2d(ac_size rows, ac_size cols, int requires_grad) {
    return ac_tensor_create(ac_shape_2d(rows, cols), requires_grad);
}

/** @} */ /* end Tensor Creation */

/** @name Fill Operations
 *  Set every element of a tensor to a constant value.
 *  @{
 */

/**
 * @brief Fill every element with @p value.
 * @param t     Target tensor.
 * @param value Scalar fill value.
 */
AC_INLINE void ac_tensor_fill(ac_tensor* t, float value) {
    for (ac_size i = 0; i < t->shape.total_size; i++) {
        t->data[i] = value;
    }
}

/**
 * @brief Zero-fill the tensor data buffer.
 * @param t Target tensor.
 */
AC_INLINE void ac_tensor_zeros(ac_tensor* t) {
    memset(t->data, 0, t->shape.total_size * sizeof(float));
}

/**
 * @brief Fill every element with 1.0.
 * @param t Target tensor.
 */
AC_INLINE void ac_tensor_ones(ac_tensor* t) {
    ac_tensor_fill(t, 1.0f);
}

/** @} */ /* end Fill Operations */

/** @name PRNG & Weight Initialisation
 *  High-quality pseudorandom number generator (xoshiro128**) and weight
 *  initialisation strategies (Xavier / He / Uniform).
 *
 *  The PRNG has a period of $2^{128}-1$ and passes all BigCrush tests,
 *  making it suitable for dropout masks, weight init, and data augmentation.
 *  @{
 */

/**
 * @brief State for the xoshiro128** PRNG.
 *
 * Holds four 32-bit state words.  Initialise with @ref ac_rng_seed before
 * use.
 */
typedef struct {
    uint32_t s[4]; /**< Internal 128-bit state. */
} ac_rng;

/** @brief Global default RNG instance used by @ref ac_randf / @ref ac_randn. */
extern ac_rng g_rng;

/**
 * @brief Rotate-left helper (internal).
 * @param x Value to rotate.
 * @param k Number of bits to rotate.
 * @return Rotated value.
 */
static AC_INLINE uint32_t ac_rotl(const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

/**
 * @brief Advance the RNG and return the next 32-bit pseudorandom value.
 * @param rng Pointer to the RNG state.
 * @return Next pseudorandom @c uint32_t.
 */
static AC_INLINE uint32_t ac_rng_next(ac_rng* rng) {
    const uint32_t result = ac_rotl(rng->s[1] * 5, 7) * 9;
    const uint32_t t = rng->s[1] << 9;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = ac_rotl(rng->s[3], 11);

    return result;
}

/**
 * @brief Seed an RNG state from a single 64-bit value using SplitMix64.
 * @param rng  Pointer to the RNG state to initialise.
 * @param seed 64-bit seed value.
 */
static AC_INLINE void ac_rng_seed(ac_rng* rng, uint64_t seed) {
    /* SplitMix64 to initialize state from a single seed */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        rng->s[i] = (uint32_t)z;
    }
}

/**
 * @brief Return a uniform random float in [0, 1) from the global RNG.
 * @return Pseudorandom float.
 */
static AC_INLINE float ac_randf(void) {
    return (float)(ac_rng_next(&g_rng) >> 8) / 16777216.0f; /* [0, 1) */
}

/**
 * @brief Return a standard-normal random float (Box–Muller transform).
 * @return Pseudorandom float ~ N(0, 1).
 */
static AC_INLINE float ac_randn(void) {
    /* Box-Muller transform */
    float u1 = ac_randf() + 1e-10f;
    float u2 = ac_randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
}

/**
 * @brief Xavier / Glorot normal initialisation.
 *
 * Fills the tensor with values drawn from N(0, sqrt(2 / (fan_in + fan_out))).
 *
 * @param t        Target tensor.
 * @param fan_in   Number of input units.
 * @param fan_out  Number of output units.
 * @see ac_tensor_he_init
 */
static AC_INLINE void ac_tensor_xavier(ac_tensor* t, ac_size fan_in, ac_size fan_out) {
    float std = sqrtf(2.0f / (float)(fan_in + fan_out));
    for (ac_size i = 0; i < t->shape.total_size; i++) {
        t->data[i] = ac_randn() * std;
    }
}

/**
 * @brief He (Kaiming) normal initialisation for ReLU networks.
 *
 * Fills the tensor with values drawn from N(0, sqrt(2 / fan_in)).
 *
 * @param t       Target tensor.
 * @param fan_in  Number of input units.
 * @see ac_tensor_xavier
 */
static AC_INLINE void ac_tensor_he_init(ac_tensor* t, ac_size fan_in) {
    float std = sqrtf(2.0f / (float)fan_in);
    for (ac_size i = 0; i < t->shape.total_size; i++) {
        t->data[i] = ac_randn() * std;
    }
}

/**
 * @brief Fill tensor with uniform random values in [low, high).
 * @param t    Target tensor.
 * @param low  Lower bound (inclusive).
 * @param high Upper bound (exclusive).
 */
static AC_INLINE void ac_tensor_uniform(ac_tensor* t, float low, float high) {
    float range = high - low;
    for (ac_size i = 0; i < t->shape.total_size; i++) {
        t->data[i] = low + ac_randf() * range;
    }
}

/** @} */ /* end PRNG & Weight Initialisation */

/** @name Gradient Management
 *  Helpers for resetting gradient buffers between training steps.
 *  @{
 */

/**
 * @brief Zero the gradient buffer and clear the @c grad_computed flag.
 * @param t Target tensor (must have a gradient buffer, or the call is a no-op).
 */
AC_INLINE void ac_tensor_zero_grad(ac_tensor* t) {
    if (t->grad) {
        memset(t->grad, 0, t->shape.total_size * sizeof(float));
    }
    t->grad_computed = 0;
}

/** @} */ /* end Gradient Management */

/** @name Print
 *  Diagnostic pretty-printing.
 *  @{
 */

/**
 * @brief Print a human-readable summary of a tensor to @c stdout.
 *
 * Displays the name, shape, and up to the first 10 data values.
 *
 * @param t    Tensor to print.
 * @param name Label shown before the shape and data.
 */
AC_INLINE void ac_tensor_print(const ac_tensor* t, const char* name) {
    printf("Tensor '%s' shape=(", name);
    for (ac_size i = 0; i < t->shape.ndim; i++) {
        printf("%zu%s", t->shape.dims[i], i < t->shape.ndim - 1 ? ", " : "");
    }
    printf(") [");
    ac_size show = t->shape.total_size < 10 ? t->shape.total_size : 10;
    for (ac_size i = 0; i < show; i++) {
        printf("%.4f%s", t->data[i], i < show - 1 ? ", " : "");
    }
    if (t->shape.total_size > 10) printf(", ...");
    printf("]\n");
}

/** @} */ /* end Print */

/** @} */ /* end defgroup tensor */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_TENSOR_H */
