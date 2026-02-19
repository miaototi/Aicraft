/*
 * ============================================================================
 *  AICRAFT - Benchmark Suite
 *  Raw performance measurement of Aicraft's SIMD kernels, GEMM, MLP,
 *  training pipeline, memory allocation, and element-wise operations.
 *
 *  Benchmarks:
 *  1. Dense matrix multiply (GEMM) — the core of all ML
 *  2. Full forward pass (MLP)
 *  3. Forward + backward pass (training step)
 *  4. Element-wise throughput (operations/second)
 *  5. Memory allocation overhead (arena vs malloc)
 *  6. Dot product throughput (GFLOPS)
 * ============================================================================
 */

#include "aicraft/aicraft.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ── Benchmark Configuration ────────────────────────────────────────────── */

#define BENCH_WARMUP     5
#define BENCH_ITERATIONS 100

/* ── Utility ────────────────────────────────────────────────────────────── */

static void print_header(const char* name) {
    printf("\n");
    printf("  ========================================================\n");
    printf("  BENCHMARK: %s\n", name);
    printf("  ========================================================\n");
}

static void print_result(const char* name, double aicraft_ms) {
    printf("  %-30s %8.3f ms\n", name, aicraft_ms);
    printf("  --------------------------------------------------------\n");
}

static double run_timed(void (*fn)(void), int warmup, int iters) {
    ac_timer timer;
    
    /* Warmup */
    for (int i = 0; i < warmup; i++) fn();
    
    /* Timed run */
    ac_timer_start(&timer);
    for (int i = 0; i < iters; i++) fn();
    double total = ac_timer_stop(&timer);
    
    return (total / (double)iters) * 1000.0; /* Convert to ms */
}

/* ── Benchmark 1: GEMM 512x512 ─────────────────────────────────────────── */

static float* bench_A = NULL;
static float* bench_B = NULL;
static float* bench_C = NULL;

static void bench_gemm_512(void) {
    ac_gemm(bench_A, bench_B, bench_C, 512, 512, 512);
}

static void run_gemm_benchmark(void) {
    print_header("GEMM 512x512 (Core Matrix Multiply)");
    
    bench_A = (float*)ac_aligned_alloc(512 * 512 * sizeof(float), AC_SIMD_ALIGN);
    bench_B = (float*)ac_aligned_alloc(512 * 512 * sizeof(float), AC_SIMD_ALIGN);
    bench_C = (float*)ac_aligned_alloc(512 * 512 * sizeof(float), AC_SIMD_ALIGN);
    
    for (int i = 0; i < 512 * 512; i++) {
        bench_A[i] = (float)(i % 100) / 100.0f;
        bench_B[i] = (float)((i + 37) % 100) / 100.0f;
    }
    
    double ms = run_timed(bench_gemm_512, BENCH_WARMUP, BENCH_ITERATIONS);
    double gflops = (2.0 * 512 * 512 * 512 / (ms / 1000.0)) / 1e9;
    printf("  %-30s %8.3f ms\n", "GEMM 512x512", ms);
    printf("  Throughput: %.2f GFLOPS\n", gflops);
    printf("  --------------------------------------------------------\n");
    
    ac_aligned_free(bench_A);
    ac_aligned_free(bench_B);
    ac_aligned_free(bench_C);
}

/* ── Benchmark 2: MLP Forward Pass ──────────────────────────────────────── */

static ac_dense mlp_layers[4];
static ac_tensor* mlp_input = NULL;

static void bench_mlp_forward(void) {
    ac_tensor* x = mlp_input;
    for (int i = 0; i < 4; i++) {
        x = ac_dense_forward(&mlp_layers[i], x);
        if (i < 3) x = ac_tensor_relu(x);
    }
}

static void run_mlp_forward_benchmark(void) {
    print_header("MLP Forward Pass (4-layer, 256 hidden)");
    
    ac_dense_init(&mlp_layers[0], 784, 256);
    ac_dense_init(&mlp_layers[1], 256, 256);
    ac_dense_init(&mlp_layers[2], 256, 128);
    ac_dense_init(&mlp_layers[3], 128, 10);
    
    mlp_input = ac_tensor_2d(32, 784, 0); /* Batch=32, MNIST-like */
    ac_tensor_uniform(mlp_input, -1.0f, 1.0f);
    
    double ms = run_timed(bench_mlp_forward, BENCH_WARMUP, BENCH_ITERATIONS);
    print_result("MLP Forward (batch=32)", ms);
}

/* ── Benchmark 3: Training Step (Forward + Backward + Update) ───────────── */

static ac_param_group train_params;
static ac_adam train_optimizer;
static ac_tensor* train_input = NULL;
static ac_tensor* train_labels = NULL;

static void bench_train_step(void) {
    ac_zero_grad(&train_params);
    
    ac_tensor* x = train_input;
    x = ac_dense_forward(&mlp_layers[0], x);
    x = ac_tensor_relu(x);
    x = ac_dense_forward(&mlp_layers[1], x);
    x = ac_tensor_relu(x);
    x = ac_dense_forward(&mlp_layers[2], x);
    x = ac_tensor_relu(x);
    x = ac_dense_forward(&mlp_layers[3], x);
    
    ac_tensor* loss = ac_cross_entropy_loss(x, train_labels);
    ac_backward(loss);
    ac_adam_step(&train_optimizer);
}

static void run_train_step_benchmark(void) {
    print_header("Full Training Step (Forward + Backward + Adam)");
    
    ac_param_group_init(&train_params);
    for (int i = 0; i < 4; i++) {
        ac_param_group_add(&train_params, mlp_layers[i].weight);
        ac_param_group_add(&train_params, mlp_layers[i].bias);
    }
    
    ac_adam_init(&train_optimizer, &train_params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, 0);
    
    train_input = ac_tensor_2d(32, 784, 1);
    ac_tensor_uniform(train_input, -1.0f, 1.0f);
    
    train_labels = ac_tensor_1d(32, 0);
    for (int i = 0; i < 32; i++) {
        train_labels->data[i] = (float)(i % 10);
    }
    
    double ms = run_timed(bench_train_step, BENCH_WARMUP, BENCH_ITERATIONS / 2);
    print_result("Train Step (batch=32)", ms);
}

/* ── Benchmark 4: Element-wise Operations ───────────────────────────────── */

static float* elem_a = NULL;
static float* elem_b = NULL;
static float* elem_c = NULL;
#define ELEM_SIZE (1024 * 1024)

static void bench_elementwise(void) {
    ac_simd_add(elem_a, elem_b, elem_c, ELEM_SIZE);
    ac_simd_mul(elem_a, elem_b, elem_c, ELEM_SIZE);
    ac_simd_relu(elem_a, elem_c, ELEM_SIZE);
    ac_simd_scale(elem_a, 0.5f, elem_c, ELEM_SIZE);
}

static void run_elementwise_benchmark(void) {
    print_header("Element-wise Ops (1M elements, add+mul+relu+scale)");
    
    elem_a = (float*)ac_aligned_alloc(ELEM_SIZE * sizeof(float), AC_SIMD_ALIGN);
    elem_b = (float*)ac_aligned_alloc(ELEM_SIZE * sizeof(float), AC_SIMD_ALIGN);
    elem_c = (float*)ac_aligned_alloc(ELEM_SIZE * sizeof(float), AC_SIMD_ALIGN);
    
    for (int i = 0; i < ELEM_SIZE; i++) {
        elem_a[i] = (float)i / ELEM_SIZE;
        elem_b[i] = (float)(ELEM_SIZE - i) / ELEM_SIZE;
    }
    
    double ms = run_timed(bench_elementwise, BENCH_WARMUP, BENCH_ITERATIONS);
    printf("  %-30s %8.3f ms\n", "4x elem-wise ops (1M)", ms);
    printf("  Throughput: %.1f M elements/sec\n", (4.0 * ELEM_SIZE / (ms / 1000.0)) / 1e6);
    printf("  --------------------------------------------------------\n");
    
    ac_aligned_free(elem_a);
    ac_aligned_free(elem_b);
    ac_aligned_free(elem_c);
}

/* ── Benchmark 5: Memory Allocation ─────────────────────────────────────── */

static void bench_arena_alloc(void) {
    ac_arena arena;
    ac_arena_init(&arena, 1024 * 1024);
    
    for (int i = 0; i < 1000; i++) {
        ac_arena_alloc(&arena, 4096);
    }
    
    ac_arena_destroy(&arena);
}

static void bench_malloc_alloc(void) {
    void* ptrs[1000];
    for (int i = 0; i < 1000; i++) {
        ptrs[i] = malloc(4096);
    }
    for (int i = 0; i < 1000; i++) {
        free(ptrs[i]);
    }
}

static void run_memory_benchmark(void) {
    print_header("Memory Allocation (1000x 4KB blocks)");
    
    double arena_ms = run_timed(bench_arena_alloc, BENCH_WARMUP, BENCH_ITERATIONS * 10);
    double malloc_ms = run_timed(bench_malloc_alloc, BENCH_WARMUP, BENCH_ITERATIONS * 10);
    
    double speedup = ((malloc_ms - arena_ms) / malloc_ms) * 100.0;
    
    printf("  %-30s %8.4f ms\n", "Aicraft Arena Allocator:", arena_ms);
    printf("  %-30s %8.4f ms\n", "System malloc/free:", malloc_ms);
    printf("  Arena speedup: %+.1f%%\n", speedup);
    printf("  --------------------------------------------------------\n");
}

/* ── Benchmark 6: Dot Product ───────────────────────────────────────────── */

static float* dot_a = NULL;
static float* dot_b = NULL;
#define DOT_SIZE (1024 * 1024)

static void bench_dot_product(void) {
    volatile float result = ac_simd_dot(dot_a, dot_b, DOT_SIZE);
    (void)result;
}

static void run_dot_benchmark(void) {
    print_header("Dot Product (1M elements)");
    
    dot_a = (float*)ac_aligned_alloc(DOT_SIZE * sizeof(float), AC_SIMD_ALIGN);
    dot_b = (float*)ac_aligned_alloc(DOT_SIZE * sizeof(float), AC_SIMD_ALIGN);
    
    for (int i = 0; i < DOT_SIZE; i++) {
        dot_a[i] = (float)i / DOT_SIZE;
        dot_b[i] = (float)(DOT_SIZE - i) / DOT_SIZE;
    }
    
    double ms = run_timed(bench_dot_product, BENCH_WARMUP, BENCH_ITERATIONS);
    double gflops = (2.0 * DOT_SIZE / (ms / 1000.0)) / 1e9;
    printf("  %-30s %8.3f ms\n", "Dot Product (1M)", ms);
    printf("  Throughput: %.2f GFLOPS\n", gflops);
    printf("  --------------------------------------------------------\n");
    
    ac_aligned_free(dot_a);
    ac_aligned_free(dot_b);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(void) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║     AICRAFT v%s - Performance Benchmark Suite       ║\n", AICRAFT_VERSION_STRING);
    printf("  ║     Zero Dependencies. Pure C/C++. SIMD-Optimized.     ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n");
    
#if defined(AC_SIMD_AVX512)
    printf("  SIMD: AVX-512 (16-wide)\n");
#elif defined(AC_SIMD_AVX2)
    printf("  SIMD: AVX2 (8-wide)\n");
#elif defined(AC_SIMD_SSE)
    printf("  SIMD: SSE (4-wide)\n");
#elif defined(AC_SIMD_NEON)
    printf("  SIMD: NEON (4-wide)\n");
#else
    printf("  SIMD: Scalar fallback\n");
#endif
    
    ac_init();
    
    run_gemm_benchmark();
    run_mlp_forward_benchmark();
    run_train_step_benchmark();
    run_elementwise_benchmark();
    run_memory_benchmark();
    run_dot_benchmark();
    
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║                    BENCHMARK SUMMARY                    ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║  DESIGN ADVANTAGES:                                    ║\n");
    printf("  ║  - No interpreter overhead (pure C/C++)                ║\n");
    printf("  ║  - No dynamic dispatch / virtual function calls        ║\n");
    printf("  ║  - Arena allocator eliminates per-op malloc overhead   ║\n");
    printf("  ║  - Hand-tuned SIMD kernels (AVX-512/AVX2/SSE/NEON)    ║\n");
    printf("  ║  - BLIS-style tiled GEMM with panel packing            ║\n");
    printf("  ║  - Fused operations (FMA, softmax+CE)                  ║\n");
    printf("  ║  - INT8 quantization for ~4x model compression         ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");
    
    ac_cleanup();
    return 0;
}
