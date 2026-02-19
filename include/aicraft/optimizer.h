/**
 * @file optimizer.h
 * @brief SGD (momentum), Adam, and AdamW optimisers — all SIMD-accelerated.
 *
 * Direct parameter update loop with AVX2 intrinsics where available.
 * Includes gradient clipping (norm + value) and learning-rate schedulers.
 */

#ifndef AICRAFT_OPTIMIZER_H
#define AICRAFT_OPTIMIZER_H

#include "aicraft/tensor.h"
#include "aicraft/simd_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup optimizer Optimisers */
/** @{ */

/** @name Parameter Group */
/** @{ */

/** @brief Default initial capacity for a parameter group. */
#define AC_PARAMS_INITIAL_CAPACITY 64

/** @brief Dynamically growable collection of tensor parameters. */
typedef struct {
    ac_tensor** params;     /**< Array of pointers to parameter tensors. */
    ac_size     num_params; /**< Current number of parameters. */
    ac_size     capacity;   /**< Allocated capacity of the array. */
} ac_param_group;

/**
 * @brief Initialise a parameter group with #AC_PARAMS_INITIAL_CAPACITY slots.
 * @param group  Pointer to the group to initialise.
 */
AC_INLINE void ac_param_group_init(ac_param_group* group) {
    group->num_params = 0;
    group->capacity = AC_PARAMS_INITIAL_CAPACITY;
    group->params = (ac_tensor**)malloc(group->capacity * sizeof(ac_tensor*));
}

/**
 * @brief Append a tensor to the parameter group, growing if needed.
 * @param group  Target parameter group.
 * @param param  Tensor to add.
 */
AC_INLINE void ac_param_group_add(ac_param_group* group, ac_tensor* param) {
    if (group->num_params >= group->capacity) {
        group->capacity *= 2;
        group->params = (ac_tensor**)realloc(group->params, 
                                             group->capacity * sizeof(ac_tensor*));
    }
    group->params[group->num_params++] = param;
}

/**
 * @brief Free the parameter group's internal storage.
 * @param group  Group to destroy.
 * @note  Does not free the tensors themselves.
 */
AC_INLINE void ac_param_group_destroy(ac_param_group* group) {
    free(group->params);
    group->params = NULL;
    group->num_params = 0;
    group->capacity = 0;
}

/** @} */

/** @name SGD Optimiser */
/** @{ */

/** @brief Stochastic Gradient Descent optimiser with optional momentum.
 *  @simd  AVX2 path for momentum and vanilla updates.
 */
typedef struct {
    ac_param_group* params;       /**< Parameter group to optimise. */
    float           lr;           /**< Learning rate. */
    float           momentum;     /**< Momentum coefficient (0 = disabled). */
    float           weight_decay; /**< L2 weight-decay factor. */
    float**         velocity;     /**< Momentum buffers (one per parameter). */
    int             initialized;  /**< Non-zero after velocity buffers are allocated. */
} ac_sgd;

/**
 * @brief Initialise an SGD optimiser.
 * @param opt          Pointer to the SGD state.
 * @param params       Parameter group to optimise.
 * @param lr           Learning rate.
 * @param momentum     Momentum coefficient (0 to disable).
 * @param weight_decay L2 weight-decay factor.
 * @see ac_sgd_step
 */
static AC_INLINE void ac_sgd_init(ac_sgd* opt, ac_param_group* params, 
                           float lr, float momentum, float weight_decay) {
    opt->params = params;
    opt->lr = lr;
    opt->momentum = momentum;
    opt->weight_decay = weight_decay;
    opt->initialized = 0;
    opt->velocity = NULL;
    
    if (momentum > 0.0f) {
        ac_ensure_arena();
        opt->velocity = (float**)ac_arena_alloc(&g_tensor_arena, 
                                                 params->num_params * sizeof(float*));
        for (ac_size i = 0; i < params->num_params; i++) {
            ac_size n = params->params[i]->shape.total_size;
            ac_size aligned = (n + AC_SIMD_WIDTH - 1) & ~(AC_SIMD_WIDTH - 1);
            opt->velocity[i] = (float*)ac_arena_alloc(&g_tensor_arena, aligned * sizeof(float));
            memset(opt->velocity[i], 0, aligned * sizeof(float));
        }
        opt->initialized = 1;
    }
}

/**
 * @brief Perform one SGD update step over all parameters.
 * @param opt  SGD optimiser state.
 * @simd  Uses AVX2 / FMA when available.
 * @see ac_sgd_init, ac_zero_grad
 */
AC_INLINE void ac_sgd_step(ac_sgd* opt) {
    for (ac_size i = 0; i < opt->params->num_params; i++) {
        ac_tensor* p = opt->params->params[i];
        if (!p->grad) continue;
        
        ac_size n = p->shape.total_size;
        
        if (opt->weight_decay > 0.0f) {
            /* L2 regularization: grad += wd * param */
            for (ac_size j = 0; j < n; j++) {
                p->grad[j] += opt->weight_decay * p->data[j];
            }
        }
        
        if (opt->momentum > 0.0f && opt->velocity) {
            /* v = momentum * v + grad */
            /* param -= lr * v */
            float* v = opt->velocity[i];
            ac_size j = 0;
#if defined(AC_SIMD_AVX2)
            __m256 vm = _mm256_set1_ps(opt->momentum);
            __m256 vlr = _mm256_set1_ps(-opt->lr);
            for (; j + 8 <= n; j += 8) {
                __m256 vv = _mm256_load_ps(v + j);
                __m256 vg = _mm256_load_ps(p->grad + j);
                vv = _mm256_add_ps(_mm256_mul_ps(vm, vv), vg);
                _mm256_store_ps(v + j, vv);
                __m256 vp = _mm256_load_ps(p->data + j);
                #ifdef __FMA__
                _mm256_store_ps(p->data + j, _mm256_fmadd_ps(vlr, vv, vp));
                #else
                _mm256_store_ps(p->data + j, _mm256_add_ps(vp, _mm256_mul_ps(vlr, vv)));
                #endif
            }
#endif
            for (; j < n; j++) {
                v[j] = opt->momentum * v[j] + p->grad[j];
                p->data[j] -= opt->lr * v[j];
            }
        } else {
            /* Simple SGD: param -= lr * grad */
            ac_size j = 0;
#if defined(AC_SIMD_AVX2)
            __m256 vlr = _mm256_set1_ps(-opt->lr);
            for (; j + 8 <= n; j += 8) {
                __m256 vp = _mm256_load_ps(p->data + j);
                __m256 vg = _mm256_load_ps(p->grad + j);
                #ifdef __FMA__
                _mm256_store_ps(p->data + j, _mm256_fmadd_ps(vlr, vg, vp));
                #else
                _mm256_store_ps(p->data + j, _mm256_add_ps(vp, _mm256_mul_ps(vlr, vg)));
                #endif
            }
#endif
            for (; j < n; j++) {
                p->data[j] -= opt->lr * p->grad[j];
            }
        }
    }
}

/** @} */

/** @name Adam / AdamW Optimiser */
/** @{ */

/** @brief Adam / AdamW adaptive optimiser.
 *  @simd  AVX2 path for moment updates and parameter step.
 */
typedef struct {
    ac_param_group* params;       /**< Parameter group to optimise. */
    float           lr;           /**< Learning rate. */
    float           beta1;        /**< Exponential decay rate for 1st moment. */
    float           beta2;        /**< Exponential decay rate for 2nd moment. */
    float           eps;          /**< Numerical stability constant. */
    float           weight_decay; /**< Weight-decay factor (L2 or decoupled). */
    float**         m;            /**< First-moment buffers. */
    float**         v;            /**< Second-moment buffers. */
    int             t;            /**< Current time step. */
    int             adamw;        /**< Non-zero to use decoupled weight decay (AdamW). */
} ac_adam;

/**
 * @brief Initialise an Adam / AdamW optimiser.
 * @param opt          Pointer to the Adam state.
 * @param params       Parameter group to optimise.
 * @param lr           Learning rate.
 * @param beta1        Exponential decay rate for 1st moment (e.g. 0.9).
 * @param beta2        Exponential decay rate for 2nd moment (e.g. 0.999).
 * @param eps          Small constant for numerical stability (e.g. 1e-8).
 * @param weight_decay Weight-decay factor.
 * @param adamw        Non-zero to enable decoupled weight decay (AdamW).
 * @see ac_adam_step
 */
static AC_INLINE void ac_adam_init(ac_adam* opt, ac_param_group* params,
                            float lr, float beta1, float beta2, float eps,
                            float weight_decay, int adamw) {
    opt->params = params;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->t = 0;
    opt->adamw = adamw;
    
    ac_ensure_arena();
    opt->m = (float**)ac_arena_alloc(&g_tensor_arena, params->num_params * sizeof(float*));
    opt->v = (float**)ac_arena_alloc(&g_tensor_arena, params->num_params * sizeof(float*));
    
    for (ac_size i = 0; i < params->num_params; i++) {
        ac_size n = params->params[i]->shape.total_size;
        ac_size aligned = (n + AC_SIMD_WIDTH - 1) & ~(AC_SIMD_WIDTH - 1);
        opt->m[i] = (float*)ac_arena_alloc(&g_tensor_arena, aligned * sizeof(float));
        opt->v[i] = (float*)ac_arena_alloc(&g_tensor_arena, aligned * sizeof(float));
        memset(opt->m[i], 0, aligned * sizeof(float));
        memset(opt->v[i], 0, aligned * sizeof(float));
    }
}

/**
 * @brief Perform one Adam / AdamW update step over all parameters.
 * @param opt  Adam optimiser state.
 * @simd  Uses AVX2 / FMA when available.
 * @see ac_adam_init, ac_zero_grad
 */
AC_INLINE void ac_adam_step(ac_adam* opt) {
    opt->t++;
    float bc1 = 1.0f - powf(opt->beta1, (float)opt->t);
    float bc2 = 1.0f - powf(opt->beta2, (float)opt->t);
    float lr_t = opt->lr * sqrtf(bc2) / bc1;
    
    for (ac_size i = 0; i < opt->params->num_params; i++) {
        ac_tensor* p = opt->params->params[i];
        if (!p->grad) continue;
        
        ac_size n = p->shape.total_size;
        float* m = opt->m[i];
        float* v = opt->v[i];
        
        if (opt->adamw && opt->weight_decay > 0.0f) {
            /* Decoupled weight decay (AdamW) */
            for (ac_size j = 0; j < n; j++) {
                p->data[j] *= (1.0f - opt->lr * opt->weight_decay);
            }
        }
        
        ac_size j = 0;
#if defined(AC_SIMD_AVX2)
        __m256 vb1 = _mm256_set1_ps(opt->beta1);
        __m256 vb1c = _mm256_set1_ps(1.0f - opt->beta1);
        __m256 vb2 = _mm256_set1_ps(opt->beta2);
        __m256 vb2c = _mm256_set1_ps(1.0f - opt->beta2);
        __m256 vlr = _mm256_set1_ps(lr_t);
        __m256 veps = _mm256_set1_ps(opt->eps);
        __m256 vwd = _mm256_set1_ps(opt->weight_decay);
        
        for (; j + 8 <= n; j += 8) {
            __m256 vg = _mm256_load_ps(p->grad + j);
            
            /* Apply L2 weight decay in SIMD path (when not AdamW) */
            if (!opt->adamw && opt->weight_decay > 0.0f) {
                __m256 vp_wd = _mm256_load_ps(p->data + j);
#ifdef __FMA__
                vg = _mm256_fmadd_ps(vwd, vp_wd, vg);
#else
                vg = _mm256_add_ps(vg, _mm256_mul_ps(vwd, vp_wd));
#endif
            }
            
            /* m = beta1*m + (1-beta1)*g */
            __m256 vm = _mm256_load_ps(m + j);
            vm = _mm256_add_ps(_mm256_mul_ps(vb1, vm), _mm256_mul_ps(vb1c, vg));
            _mm256_store_ps(m + j, vm);
            
            /* v = beta2*v + (1-beta2)*g^2 */
            __m256 vv = _mm256_load_ps(v + j);
            vv = _mm256_add_ps(_mm256_mul_ps(vb2, vv), _mm256_mul_ps(vb2c, _mm256_mul_ps(vg, vg)));
            _mm256_store_ps(v + j, vv);
            
            /* param -= lr_t * m / (sqrt(v) + eps) */
            __m256 vp = _mm256_load_ps(p->data + j);
            __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(vv), veps);
            __m256 update = _mm256_mul_ps(vlr, _mm256_div_ps(vm, denom));
            _mm256_store_ps(p->data + j, _mm256_sub_ps(vp, update));
        }
#endif
        for (; j < n; j++) {
            float g = p->grad[j];
            
            if (!opt->adamw && opt->weight_decay > 0.0f) {
                g += opt->weight_decay * p->data[j]; /* L2 regularization */
            }
            
            m[j] = opt->beta1 * m[j] + (1.0f - opt->beta1) * g;
            v[j] = opt->beta2 * v[j] + (1.0f - opt->beta2) * g * g;
            
            p->data[j] -= lr_t * m[j] / (sqrtf(v[j]) + opt->eps);
        }
    }
}

/** @} */

/** @name Gradient Utilities */
/** @{ */

/**
 * @brief Zero all gradient buffers in a parameter group.
 * @param params  Parameter group whose gradients are cleared.
 * @see ac_tensor_zero_grad
 */
AC_INLINE void ac_zero_grad(ac_param_group* params) {
    for (ac_size i = 0; i < params->num_params; i++) {
        ac_tensor_zero_grad(params->params[i]);
    }
}

/**
 * @brief Clip gradients by global L2 norm.
 * @param params    Parameter group.
 * @param max_norm  Maximum allowable gradient norm.
 * @return The original (unclipped) gradient norm.
 * @simd  AVX2 path for the scaling pass.
 * @see ac_clip_grad_value
 */
AC_INLINE float ac_clip_grad_norm(ac_param_group* params, float max_norm) {
    /* Compute global gradient norm */
    float total_sq = 0.0f;
    for (ac_size i = 0; i < params->num_params; i++) {
        ac_tensor* p = params->params[i];
        if (!p->grad) continue;
        ac_size n = p->shape.total_size;
        total_sq += ac_simd_dot(p->grad, p->grad, n);
    }
    float grad_norm = sqrtf(total_sq);

    if (grad_norm > max_norm && grad_norm > 0.0f) {
        float scale = max_norm / grad_norm;
        for (ac_size i = 0; i < params->num_params; i++) {
            ac_tensor* p = params->params[i];
            if (!p->grad) continue;
            ac_size n = p->shape.total_size;
            ac_size j = 0;
#if defined(AC_SIMD_AVX2)
            __m256 vs = _mm256_set1_ps(scale);
            for (; j + 8 <= n; j += 8) {
                __m256 vg = _mm256_load_ps(p->grad + j);
                _mm256_store_ps(p->grad + j, _mm256_mul_ps(vg, vs));
            }
#endif
            for (; j < n; j++) {
                p->grad[j] *= scale;
            }
        }
    }
    return grad_norm;
}

/**
 * @brief Clip every gradient element to [-clip_value, +clip_value].
 * @param params      Parameter group.
 * @param clip_value  Symmetric clamp bound.
 * @simd  AVX2 path for element-wise clamping.
 * @see ac_clip_grad_norm
 */
AC_INLINE void ac_clip_grad_value(ac_param_group* params, float clip_value) {
    for (ac_size i = 0; i < params->num_params; i++) {
        ac_tensor* p = params->params[i];
        if (!p->grad) continue;
        ac_size n = p->shape.total_size;
        ac_size j = 0;
#if defined(AC_SIMD_AVX2)
        __m256 vhi = _mm256_set1_ps(clip_value);
        __m256 vlo = _mm256_set1_ps(-clip_value);
        for (; j + 8 <= n; j += 8) {
            __m256 vg = _mm256_load_ps(p->grad + j);
            vg = _mm256_min_ps(_mm256_max_ps(vg, vlo), vhi);
            _mm256_store_ps(p->grad + j, vg);
        }
#endif
        for (; j < n; j++) {
            if (p->grad[j] > clip_value) p->grad[j] = clip_value;
            else if (p->grad[j] < -clip_value) p->grad[j] = -clip_value;
        }
    }
}

/** @} */

/** @name Learning Rate Schedulers */
/** @{ */

/** @brief Supported learning-rate schedule types. */
typedef enum {
    AC_LR_STEP,   /**< Step decay: lr *= gamma every @p step_size epochs. */
    AC_LR_COSINE, /**< Cosine annealing to @p min_lr over @p total_epochs. */
    AC_LR_EXP     /**< Exponential decay: lr = base_lr * gamma^epoch. */
} ac_lr_schedule_type;

/** @brief Learning-rate scheduler state. */
typedef struct {
    ac_lr_schedule_type type; /**< Schedule type. */
    float  base_lr;           /**< Initial learning rate. */
    float  min_lr;            /**< Minimum learning rate (cosine). */
    float  gamma;             /**< Decay factor. */
    int    step_size;         /**< Epoch interval for step decay. */
    int    total_epochs;      /**< Total epochs (cosine annealing). */
} ac_lr_scheduler;

/**
 * @brief Initialise a learning-rate scheduler.
 * @param sched        Scheduler to initialise.
 * @param type         Schedule type.
 * @param base_lr      Initial learning rate.
 * @param gamma        Decay factor.
 * @param step_size    Epoch interval (step decay).
 * @param total_epochs Total epochs (cosine annealing).
 * @param min_lr       Minimum learning rate.
 */
AC_INLINE void ac_lr_scheduler_init(ac_lr_scheduler* sched,
                                    ac_lr_schedule_type type,
                                    float base_lr, float gamma,
                                    int step_size, int total_epochs,
                                    float min_lr) {
    sched->type = type;
    sched->base_lr = base_lr;
    sched->min_lr = min_lr;
    sched->gamma = gamma;
    sched->step_size = step_size > 0 ? step_size : 1;
    sched->total_epochs = total_epochs > 0 ? total_epochs : 1;
}

/**
 * @brief Compute the learning rate for a given epoch.
 * @param sched  Scheduler.
 * @param epoch  Current epoch index (0-based).
 * @return Scheduled learning rate (≥ @p min_lr).
 */
AC_INLINE float ac_lr_scheduler_get(const ac_lr_scheduler* sched, int epoch) {
    float lr;
    switch (sched->type) {
    case AC_LR_STEP:
        lr = sched->base_lr * powf(sched->gamma, (float)(epoch / sched->step_size));
        break;
    case AC_LR_COSINE: {
        float pi = 3.14159265358979323846f;
        float t = (float)epoch / (float)sched->total_epochs;
        if (t > 1.0f) t = 1.0f;
        lr = sched->min_lr + 0.5f * (sched->base_lr - sched->min_lr) * (1.0f + cosf(pi * t));
        break;
    }
    case AC_LR_EXP:
        lr = sched->base_lr * powf(sched->gamma, (float)epoch);
        break;
    default:
        lr = sched->base_lr;
        break;
    }
    return lr > sched->min_lr ? lr : sched->min_lr;
}

/**
 * @brief Convenience: update an SGD optimiser's lr for the given epoch.
 * @param sched  Scheduler.
 * @param opt    SGD optimiser to update.
 * @param epoch  Current epoch index.
 * @see ac_lr_scheduler_get
 */
AC_INLINE void ac_lr_scheduler_step_sgd(const ac_lr_scheduler* sched,
                                        ac_sgd* opt, int epoch) {
    opt->lr = ac_lr_scheduler_get(sched, epoch);
}

/**
 * @brief Convenience: update an Adam/AdamW optimiser's lr for the given epoch.
 * @param sched  Scheduler.
 * @param opt    Adam/AdamW optimiser to update.
 * @param epoch  Current epoch index.
 * @see ac_lr_scheduler_get
 */
AC_INLINE void ac_lr_scheduler_step_adam(const ac_lr_scheduler* sched,
                                        ac_adam* opt, int epoch) {
    opt->lr = ac_lr_scheduler_get(sched, epoch);
}

/** @} */
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* AICRAFT_OPTIMIZER_H */
