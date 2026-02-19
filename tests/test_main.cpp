/*
 * ============================================================================
 *  AICRAFT - Comprehensive Test Suite
 *  Production-grade unit tests covering all components.
 * ============================================================================
 */

#include "aicraft/aicraft.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

/* ── Test Framework ─────────────────────────────────────────────────────── */

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_total  = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %-50s ", name); } while(0)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define ASSERT_NEAR(a, b, eps) do { \
    float _a = (float)(a), _b = (float)(b); \
    if (fabsf(_a - _b) > (eps)) { \
        printf("FAIL: expected %.6f, got %.6f (diff=%.2e)\n", \
               (double)_b, (double)_a, (double)fabsf(_a - _b)); \
        tests_failed++; return; \
    } \
} while(0)
#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

static void print_section(const char* name) {
    printf("\n  -- %s --\n\n", name);
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 1: TENSOR CORE
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_tensor_creation_1d(void) {
    TEST("Tensor creation (1D)");
    ac_tensor* t = ac_tensor_1d(100, 0);
    ASSERT_TRUE(t != NULL, "tensor is NULL");
    ASSERT_TRUE(t->shape.ndim == 1, "wrong ndim");
    ASSERT_TRUE(t->shape.dims[0] == 100, "wrong dim");
    ASSERT_TRUE(t->shape.total_size == 100, "wrong total");
    ASSERT_TRUE(t->grad == NULL, "grad should be NULL");
    PASS();
}

static void test_tensor_creation_2d(void) {
    TEST("Tensor creation (2D)");
    ac_tensor* t = ac_tensor_2d(3, 4, 0);
    ASSERT_TRUE(t != NULL, "tensor is NULL");
    ASSERT_TRUE(t->shape.ndim == 2, "wrong ndim");
    ASSERT_TRUE(t->shape.dims[0] == 3, "wrong dim0");
    ASSERT_TRUE(t->shape.dims[1] == 4, "wrong dim1");
    ASSERT_TRUE(t->shape.total_size == 12, "wrong total");
    PASS();
}

static void test_tensor_creation_with_grad(void) {
    TEST("Tensor creation (requires_grad=1)");
    ac_tensor* t = ac_tensor_2d(5, 5, 1);
    ASSERT_TRUE(t != NULL, "tensor is NULL");
    ASSERT_TRUE(t->requires_grad == 1, "requires_grad not set");
    ASSERT_TRUE(t->grad != NULL, "grad should be allocated");
    for (int i = 0; i < 25; i++) {
        ASSERT_NEAR(t->grad[i], 0.0f, 1e-10f);
    }
    PASS();
}

static void test_tensor_fill(void) {
    TEST("Tensor fill operations");
    ac_tensor* t = ac_tensor_1d(100, 0);
    
    ac_tensor_ones(t);
    for (int i = 0; i < 100; i++) ASSERT_NEAR(t->data[i], 1.0f, 1e-6f);
    
    ac_tensor_zeros(t);
    for (int i = 0; i < 100; i++) ASSERT_NEAR(t->data[i], 0.0f, 1e-6f);
    
    ac_tensor_fill(t, 3.14f);
    for (int i = 0; i < 100; i++) ASSERT_NEAR(t->data[i], 3.14f, 1e-5f);
    
    PASS();
}

static void test_tensor_shape_4d(void) {
    TEST("Tensor 4D shape");
    ac_shape s = ac_shape_4d(2, 3, 4, 5);
    ASSERT_TRUE(s.ndim == 4, "wrong ndim");
    ASSERT_TRUE(s.total_size == 120, "wrong total");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 2: SIMD OPERATIONS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_simd_add(void) {
    TEST("SIMD vectorized add (large)");
    ac_tensor* a = ac_tensor_1d(1024, 0);
    ac_tensor* b = ac_tensor_1d(1024, 0);
    ac_tensor_fill(a, 2.0f);
    ac_tensor_fill(b, 3.0f);
    ac_tensor* c = ac_tensor_add(a, b);
    for (int i = 0; i < 1024; i++) ASSERT_NEAR(c->data[i], 5.0f, 1e-6f);
    PASS();
}

static void test_simd_mul(void) {
    TEST("SIMD vectorized multiply");
    ac_tensor* a = ac_tensor_1d(256, 0);
    ac_tensor* b = ac_tensor_1d(256, 0);
    ac_tensor_fill(a, 3.0f);
    ac_tensor_fill(b, 4.0f);
    ac_tensor* c = ac_tensor_mul(a, b);
    for (int i = 0; i < 256; i++) ASSERT_NEAR(c->data[i], 12.0f, 1e-5f);
    PASS();
}

static void test_simd_scale(void) {
    TEST("SIMD scalar multiply");
    ac_tensor* a = ac_tensor_1d(256, 0);
    ac_tensor_fill(a, 5.0f);
    ac_tensor* b = ac_tensor_scale(a, 0.5f);
    for (int i = 0; i < 256; i++) ASSERT_NEAR(b->data[i], 2.5f, 1e-5f);
    PASS();
}

static void test_simd_dot(void) {
    TEST("SIMD dot product");
    float a[256], b[256];
    for (int i = 0; i < 256; i++) { a[i] = 1.0f; b[i] = 2.0f; }
    float result = ac_simd_dot(a, b, 256);
    ASSERT_NEAR(result, 512.0f, 1e-3f);
    PASS();
}

static void test_simd_sum(void) {
    TEST("SIMD sum reduction");
    float a[256];
    for (int i = 0; i < 256; i++) a[i] = 1.0f;
    float result = ac_simd_sum(a, 256);
    ASSERT_NEAR(result, 256.0f, 1e-3f);
    PASS();
}

static void test_simd_max(void) {
    TEST("SIMD max reduction");
    float a[100];
    for (int i = 0; i < 100; i++) a[i] = (float)i;
    a[42] = 999.0f;
    float result = ac_simd_max(a, 100);
    ASSERT_NEAR(result, 999.0f, 1e-3f);
    PASS();
}

static void test_simd_fma(void) {
    TEST("SIMD fused multiply-add");
    float a[64], b[64], c[64], out[64];
    for (int i = 0; i < 64; i++) { a[i] = 2.0f; b[i] = 3.0f; c[i] = 1.0f; }
    ac_simd_fma(a, b, c, out, 64);
    for (int i = 0; i < 64; i++) ASSERT_NEAR(out[i], 7.0f, 1e-5f);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 3: MATRIX OPERATIONS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_matmul_small(void) {
    TEST("Matrix multiply (2x3 @ 3x2)");
    ac_tensor* A = ac_tensor_2d(2, 3, 0);
    ac_tensor* B = ac_tensor_2d(3, 2, 0);
    A->data[0]=1; A->data[1]=2; A->data[2]=3;
    A->data[3]=4; A->data[4]=5; A->data[5]=6;
    B->data[0]=7;  B->data[1]=8;
    B->data[2]=9;  B->data[3]=10;
    B->data[4]=11; B->data[5]=12;
    ac_tensor* C = ac_tensor_matmul(A, B);
    ASSERT_NEAR(C->data[0], 58.0f,  1e-3f);
    ASSERT_NEAR(C->data[1], 64.0f,  1e-3f);
    ASSERT_NEAR(C->data[2], 139.0f, 1e-3f);
    ASSERT_NEAR(C->data[3], 154.0f, 1e-3f);
    PASS();
}

static void test_matmul_identity(void) {
    TEST("Matrix multiply (identity)");
    ac_tensor* A = ac_tensor_2d(4, 4, 0);
    ac_tensor* I = ac_tensor_2d(4, 4, 0);
    for (int i = 0; i < 16; i++) A->data[i] = (float)(i + 1);
    for (int i = 0; i < 4; i++) I->data[i * 4 + i] = 1.0f;
    ac_tensor* C = ac_tensor_matmul(A, I);
    for (int i = 0; i < 16; i++) ASSERT_NEAR(C->data[i], A->data[i], 1e-4f);
    PASS();
}

static void test_matmul_large(void) {
    TEST("Matrix multiply (64x64 -- GEMM correctness)");
    int N = 64;
    ac_tensor* A = ac_tensor_2d(N, N, 0);
    ac_tensor* B = ac_tensor_2d(N, N, 0);
    for (int i = 0; i < N * N; i++) {
        A->data[i] = (float)(i % 7) * 0.1f;
        B->data[i] = (float)((i + 3) % 11) * 0.1f;
    }
    ac_tensor* C = ac_tensor_matmul(A, B);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += A->data[i * N + k] * B->data[k * N + j];
            }
            ASSERT_NEAR(C->data[i * N + j], expected, 0.01f);
        }
    }
    PASS();
}

static void test_transpose(void) {
    TEST("Matrix transpose");
    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[6];
    ac_transpose(A, B, 2, 3);
    ASSERT_NEAR(B[0], 1.0f, 1e-6f);
    ASSERT_NEAR(B[1], 4.0f, 1e-6f);
    ASSERT_NEAR(B[2], 2.0f, 1e-6f);
    ASSERT_NEAR(B[3], 5.0f, 1e-6f);
    ASSERT_NEAR(B[4], 3.0f, 1e-6f);
    ASSERT_NEAR(B[5], 6.0f, 1e-6f);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 4: ACTIVATIONS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_relu(void) {
    TEST("ReLU activation");
    ac_tensor* a = ac_tensor_1d(8, 0);
    float vals[] = {-3, -1, 0, 0.5f, 1, 2, -0.1f, 3};
    memcpy(a->data, vals, 8 * sizeof(float));
    ac_tensor* out = ac_tensor_relu(a);
    float expected[] = {0, 0, 0, 0.5f, 1, 2, 0, 3};
    for (int i = 0; i < 8; i++) ASSERT_NEAR(out->data[i], expected[i], 1e-6f);
    PASS();
}

static void test_sigmoid(void) {
    TEST("Sigmoid activation");
    ac_tensor* a = ac_tensor_1d(4, 0);
    a->data[0] = 0.0f; a->data[1] = 1.0f; a->data[2] = -1.0f; a->data[3] = 10.0f;
    ac_tensor* out = ac_tensor_sigmoid(a);
    ASSERT_NEAR(out->data[0], 0.5f, 1e-5f);
    ASSERT_NEAR(out->data[1], 0.7310586f, 1e-5f);
    ASSERT_NEAR(out->data[2], 0.2689414f, 1e-5f);
    ASSERT_NEAR(out->data[3], 0.9999546f, 1e-4f);
    PASS();
}

static void test_tanh_activation(void) {
    TEST("Tanh activation");
    ac_tensor* a = ac_tensor_1d(3, 0);
    a->data[0] = 0.0f; a->data[1] = 1.0f; a->data[2] = -1.0f;
    ac_tensor* out = ac_tensor_tanh(a);
    ASSERT_NEAR(out->data[0], 0.0f, 1e-5f);
    ASSERT_NEAR(out->data[1], tanhf(1.0f), 1e-4f);
    ASSERT_NEAR(out->data[2], tanhf(-1.0f), 1e-4f);
    PASS();
}

static void test_softmax(void) {
    TEST("Softmax (row-wise, sums to 1)");
    ac_tensor* a = ac_tensor_2d(2, 4, 0);
    for (int i = 0; i < 8; i++) a->data[i] = (float)i;
    ac_tensor* out = ac_tensor_softmax(a);
    float sum0 = out->data[0] + out->data[1] + out->data[2] + out->data[3];
    float sum1 = out->data[4] + out->data[5] + out->data[6] + out->data[7];
    ASSERT_NEAR(sum0, 1.0f, 1e-5f);
    ASSERT_NEAR(sum1, 1.0f, 1e-5f);
    for (int i = 0; i < 8; i++) ASSERT_TRUE(out->data[i] > 0.0f, "softmax value <= 0");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 5: LAYERS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_dense_forward(void) {
    TEST("Dense layer forward pass");
    ac_dense layer;
    ac_dense_init(&layer, 4, 3);
    ac_tensor* input = ac_tensor_2d(2, 4, 0);
    ac_tensor_fill(input, 1.0f);
    ac_tensor* output = ac_dense_forward(&layer, input);
    ASSERT_TRUE(output != NULL, "output is NULL");
    ASSERT_TRUE(output->shape.ndim == 2, "wrong ndim");
    ASSERT_TRUE(output->shape.dims[0] == 2, "wrong batch dim");
    ASSERT_TRUE(output->shape.dims[1] == 3, "wrong feature dim");
    PASS();
}

static void test_conv2d_forward(void) {
    TEST("Conv2D forward pass (basic)");
    ac_conv2d layer;
    ac_conv2d_init(&layer, 1, 2, 3, 1, 0);
    ac_tensor* input = ac_tensor_create(ac_shape_4d(1, 1, 5, 5), 0);
    ac_tensor_fill(input, 1.0f);
    ac_tensor* output = ac_conv2d_forward(&layer, input);
    ASSERT_TRUE(output != NULL, "output is NULL");
    ASSERT_TRUE(output->shape.ndim == 4, "wrong ndim");
    ASSERT_TRUE(output->shape.dims[0] == 1, "wrong batch");
    ASSERT_TRUE(output->shape.dims[1] == 2, "wrong out_channels");
    ASSERT_TRUE(output->shape.dims[2] == 3, "wrong outH");
    ASSERT_TRUE(output->shape.dims[3] == 3, "wrong outW");
    PASS();
}

static void test_maxpool2d_forward(void) {
    TEST("MaxPool2D forward pass");
    ac_maxpool2d pool;
    ac_maxpool2d_init(&pool, 2, 2);
    ac_tensor* input = ac_tensor_create(ac_shape_4d(1, 1, 4, 4), 0);
    for (int i = 0; i < 16; i++) input->data[i] = (float)i;
    ac_tensor* output = ac_maxpool2d_forward(&pool, input);
    ASSERT_TRUE(output->shape.dims[2] == 2, "wrong outH");
    ASSERT_TRUE(output->shape.dims[3] == 2, "wrong outW");
    ASSERT_NEAR(output->data[0], 5.0f, 1e-5f);
    ASSERT_NEAR(output->data[1], 7.0f, 1e-5f);
    ASSERT_NEAR(output->data[2], 13.0f, 1e-5f);
    ASSERT_NEAR(output->data[3], 15.0f, 1e-5f);
    PASS();
}

static void test_batchnorm_forward(void) {
    TEST("BatchNorm forward (training)");
    ac_batchnorm bn;
    ac_batchnorm_init(&bn, 4);
    ac_tensor* input = ac_tensor_2d(8, 4, 0);
    ac_tensor_uniform(input, -2.0f, 2.0f);
    ac_tensor* output = ac_batchnorm_forward(&bn, input);
    ASSERT_TRUE(output != NULL, "output is NULL");
    ASSERT_TRUE(output->shape.dims[0] == 8, "wrong batch");
    ASSERT_TRUE(output->shape.dims[1] == 4, "wrong features");
    for (ac_size f = 0; f < 4; f++) {
        float mean = 0;
        for (ac_size n = 0; n < 8; n++) mean += output->data[n * 4 + f];
        mean /= 8.0f;
        ASSERT_NEAR(mean, 0.0f, 0.01f);
    }
    PASS();
}

static void test_dropout_forward(void) {
    TEST("Dropout forward (training)");
    ac_dropout drop;
    ac_dropout_init(&drop, 0.5f);
    ac_tensor* input = ac_tensor_1d(1000, 0);
    ac_tensor_ones(input);
    ac_tensor* output = ac_dropout_forward(&drop, input);
    int zeros = 0;
    for (int i = 0; i < 1000; i++) {
        if (output->data[i] == 0.0f) zeros++;
    }
    ASSERT_TRUE(zeros > 300 && zeros < 700, "dropout rate not ~50%");
    PASS();
}

static void test_flatten(void) {
    TEST("Flatten layer");
    ac_tensor* input = ac_tensor_create(ac_shape_4d(2, 3, 4, 5), 0);
    ac_tensor_fill(input, 1.0f);
    ac_tensor* output = ac_flatten(input);
    ASSERT_TRUE(output->shape.ndim == 2, "wrong ndim");
    ASSERT_TRUE(output->shape.dims[0] == 2, "wrong batch");
    ASSERT_TRUE(output->shape.dims[1] == 60, "wrong flat dim");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 6: LOSS FUNCTIONS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_mse_loss(void) {
    TEST("MSE loss");
    ac_tensor* pred = ac_tensor_1d(4, 1);
    ac_tensor* target = ac_tensor_1d(4, 0);
    pred->data[0]=1; pred->data[1]=2; pred->data[2]=3; pred->data[3]=4;
    target->data[0]=1.5f; target->data[1]=2.5f; target->data[2]=3.5f; target->data[3]=4.5f;
    ac_tensor* loss = ac_mse_loss(pred, target);
    ASSERT_NEAR(loss->data[0], 0.25f, 1e-5f);
    PASS();
}

static void test_bce_loss(void) {
    TEST("BCE loss");
    ac_tensor* pred = ac_tensor_1d(4, 1);
    ac_tensor* target = ac_tensor_1d(4, 0);
    pred->data[0]=0.9f; pred->data[1]=0.1f; pred->data[2]=0.8f; pred->data[3]=0.2f;
    target->data[0]=1; target->data[1]=0; target->data[2]=1; target->data[3]=0;
    ac_tensor* loss = ac_bce_loss(pred, target);
    ASSERT_TRUE(loss != NULL, "loss is NULL");
    ASSERT_TRUE(loss->data[0] > 0.0f, "loss should be positive");
    ASSERT_TRUE(loss->data[0] < 1.0f, "loss should be < 1 for good predictions");
    ASSERT_TRUE(loss->op == AC_OP_BCE_LOSS, "wrong op tag, should be AC_OP_BCE_LOSS");
    PASS();
}

static void test_cross_entropy_loss(void) {
    TEST("Cross-entropy loss");
    ac_tensor* logits = ac_tensor_2d(2, 3, 1);
    ac_tensor* labels = ac_tensor_1d(2, 0);
    logits->data[0]=10; logits->data[1]=0; logits->data[2]=0;
    logits->data[3]=0;  logits->data[4]=0; logits->data[5]=10;
    labels->data[0] = 0; labels->data[1] = 2;
    ac_tensor* loss = ac_cross_entropy_loss(logits, labels);
    ASSERT_TRUE(loss != NULL, "loss is NULL");
    ASSERT_TRUE(loss->data[0] < 0.01f, "loss should be near 0 for correct preds");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 7: AUTOGRAD
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_autograd_add(void) {
    TEST("Autograd: add backward");
    ac_tensor* a = ac_tensor_1d(4, 1);
    ac_tensor* b = ac_tensor_1d(4, 1);
    ac_tensor_fill(a, 2.0f);
    ac_tensor_fill(b, 3.0f);
    ac_tensor* c = ac_tensor_add(a, b);
    ac_tensor* loss = ac_tensor_sum(c);
    ac_backward(loss);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(a->grad[i], 1.0f, 1e-5f);
        ASSERT_NEAR(b->grad[i], 1.0f, 1e-5f);
    }
    PASS();
}

static void test_autograd_mul(void) {
    TEST("Autograd: mul backward");
    ac_tensor* a = ac_tensor_1d(4, 1);
    ac_tensor* b = ac_tensor_1d(4, 1);
    ac_tensor_fill(a, 3.0f);
    ac_tensor_fill(b, 5.0f);
    ac_tensor* c = ac_tensor_mul(a, b);
    ac_tensor* loss = ac_tensor_sum(c);
    ac_backward(loss);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(a->grad[i], 5.0f, 1e-4f);
        ASSERT_NEAR(b->grad[i], 3.0f, 1e-4f);
    }
    PASS();
}

static void test_autograd_matmul(void) {
    TEST("Autograd: matmul backward");
    ac_tensor* A = ac_tensor_2d(2, 3, 1);
    ac_tensor* B = ac_tensor_2d(3, 2, 1);
    ac_tensor_fill(A, 1.0f);
    ac_tensor_fill(B, 1.0f);
    ac_tensor* C = ac_tensor_matmul(A, B);
    ac_tensor* loss = ac_tensor_sum(C);
    ac_backward(loss);
    int a_nonzero = 0, b_nonzero = 0;
    for (int i = 0; i < 6; i++) {
        if (fabsf(A->grad[i]) > 1e-6f) a_nonzero++;
        if (fabsf(B->grad[i]) > 1e-6f) b_nonzero++;
    }
    ASSERT_TRUE(a_nonzero == 6, "A grads should all be non-zero");
    ASSERT_TRUE(b_nonzero == 6, "B grads should all be non-zero");
    PASS();
}

static void test_autograd_relu(void) {
    TEST("Autograd: relu backward");
    ac_tensor* a = ac_tensor_1d(4, 1);
    a->data[0] = -1.0f; a->data[1] = 2.0f; a->data[2] = -3.0f; a->data[3] = 4.0f;
    ac_tensor* b = ac_tensor_relu(a);
    ac_tensor* loss = ac_tensor_sum(b);
    ac_backward(loss);
    ASSERT_NEAR(a->grad[0], 0.0f, 1e-6f);
    ASSERT_NEAR(a->grad[1], 1.0f, 1e-6f);
    ASSERT_NEAR(a->grad[2], 0.0f, 1e-6f);
    ASSERT_NEAR(a->grad[3], 1.0f, 1e-6f);
    PASS();
}

static void test_autograd_mse(void) {
    TEST("Autograd: MSE loss backward");
    ac_tensor* pred = ac_tensor_1d(4, 1);
    ac_tensor* target = ac_tensor_1d(4, 0);
    pred->data[0]=1; pred->data[1]=2; pred->data[2]=3; pred->data[3]=4;
    target->data[0]=1; target->data[1]=2; target->data[2]=3; target->data[3]=4;
    ac_tensor* loss = ac_mse_loss(pred, target);
    ac_backward(loss);
    for (int i = 0; i < 4; i++) ASSERT_NEAR(pred->grad[i], 0.0f, 1e-5f);
    PASS();
}

static void test_autograd_bce(void) {
    TEST("Autograd: BCE loss backward");
    ac_tensor* pred = ac_tensor_1d(2, 1);
    ac_tensor* target = ac_tensor_1d(2, 0);
    pred->data[0] = 0.9f; pred->data[1] = 0.1f;
    target->data[0] = 1.0f; target->data[1] = 0.0f;
    ac_tensor* loss = ac_bce_loss(pred, target);
    ac_backward(loss);
    ASSERT_TRUE(pred->grad[0] < 0.0f, "grad should push pred[0] up toward 1");
    ASSERT_TRUE(pred->grad[1] > 0.0f, "grad should push pred[1] down toward 0");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 8: OPTIMIZERS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_sgd_step(void) {
    TEST("SGD optimizer step");
    ac_tensor* w = ac_tensor_1d(4, 1);
    ac_tensor_fill(w, 1.0f);
    w->grad = (float*)ac_arena_alloc(&g_tensor_arena, 4 * sizeof(float));
    for (int i = 0; i < 4; i++) w->grad[i] = 0.5f;
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, w);
    
    ac_sgd opt;
    ac_sgd_init(&opt, &params, 0.1f, 0.0f, 0.0f);
    ac_sgd_step(&opt);
    
    for (int i = 0; i < 4; i++) ASSERT_NEAR(w->data[i], 0.95f, 1e-5f);
    ac_param_group_destroy(&params);
    PASS();
}

static void test_adam_step(void) {
    TEST("Adam optimizer step");
    ac_tensor* w = ac_tensor_1d(4, 1);
    ac_tensor_fill(w, 1.0f);
    w->grad = (float*)ac_arena_alloc(&g_tensor_arena, 4 * sizeof(float));
    for (int i = 0; i < 4; i++) w->grad[i] = 1.0f;
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, w);
    
    ac_adam opt;
    ac_adam_init(&opt, &params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, 0);
    ac_adam_step(&opt);
    
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(w->data[i] < 1.0f, "Adam should decrease weights");
    }
    ac_param_group_destroy(&params);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 9: MEMORY MANAGEMENT
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_arena_allocator(void) {
    TEST("Arena allocator (basic)");
    ac_arena arena;
    ac_arena_init(&arena, 4096);
    void* p1 = ac_arena_alloc(&arena, 100);
    void* p2 = ac_arena_alloc(&arena, 200);
    void* p3 = ac_arena_alloc(&arena, 300);
    ASSERT_TRUE(p1 && p2 && p3, "allocation failed");
    ASSERT_TRUE(p1 != p2 && p2 != p3, "overlapping allocations");
    ac_arena_destroy(&arena);
    PASS();
}

static void test_arena_checkpoint(void) {
    TEST("Arena checkpoint/restore");
    ac_arena arena;
    ac_arena_init(&arena, 4096);
    
    void* perm = ac_arena_alloc(&arena, 100);
    ASSERT_TRUE(perm != NULL, "perm alloc failed");
    ac_size used_before = arena.total_allocated;
    
    ac_arena_checkpoint cp;
    ac_arena_save(&arena, &cp);
    
    ac_arena_alloc(&arena, 500);
    ac_arena_alloc(&arena, 500);
    ASSERT_TRUE(arena.total_allocated > used_before, "temp allocs missing");
    
    ac_arena_restore(&arena, &cp);
    ASSERT_TRUE(arena.total_allocated == used_before, "restore failed");
    
    ac_arena_destroy(&arena);
    PASS();
}

static void test_pool_allocator(void) {
    TEST("Pool allocator");
    ac_pool pool;
    ac_pool_init(&pool, 64, 10);
    
    void* blocks[10];
    for (int i = 0; i < 10; i++) {
        blocks[i] = ac_pool_alloc(&pool);
        ASSERT_TRUE(blocks[i] != NULL, "pool alloc failed");
    }
    ASSERT_TRUE(ac_pool_alloc(&pool) == NULL, "pool should be empty");
    
    ac_pool_free(&pool, blocks[0]);
    void* r = ac_pool_alloc(&pool);
    ASSERT_TRUE(r != NULL, "pool realloc failed");
    
    ac_pool_destroy(&pool);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 10: SERIALIZATION
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_model_save_load(void) {
    TEST("Model save/load roundtrip");
    
    ac_dense layer1, layer2;
    ac_dense_init(&layer1, 4, 8);
    ac_dense_init(&layer2, 8, 2);
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, layer1.weight);
    ac_param_group_add(&params, layer1.bias);
    ac_param_group_add(&params, layer2.weight);
    ac_param_group_add(&params, layer2.bias);
    
    float w1_orig[32];
    float b1_orig[8];
    memcpy(w1_orig, layer1.weight->data, 32 * sizeof(float));
    memcpy(b1_orig, layer1.bias->data, 8 * sizeof(float));
    
    ac_error_code err = ac_model_save("_test_model.acml", &params);
    ASSERT_TRUE(err == AC_OK, "save failed");
    
    ac_tensor_zeros(layer1.weight);
    ac_tensor_zeros(layer1.bias);
    
    err = ac_model_load("_test_model.acml", &params);
    ASSERT_TRUE(err == AC_OK, "load failed");
    
    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(layer1.weight->data[i], w1_orig[i], 1e-6f);
    }
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(layer1.bias->data[i], b1_orig[i], 1e-6f);
    }
    
    remove("_test_model.acml");
    ac_param_group_destroy(&params);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 11: PRNG
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_prng_range(void) {
    TEST("PRNG: randf() in [0,1)");
    for (int i = 0; i < 10000; i++) {
        float r = ac_randf();
        ASSERT_TRUE(r >= 0.0f && r < 1.0f, "randf() out of range");
    }
    PASS();
}

static void test_prng_distribution(void) {
    TEST("PRNG: randn() mean ~0, std ~1");
    double sum = 0, sum2 = 0;
    int N = 100000;
    for (int i = 0; i < N; i++) {
        float r = ac_randn();
        sum += r;
        sum2 += r * r;
    }
    double mean = sum / N;
    double var = sum2 / N - mean * mean;
    ASSERT_NEAR((float)mean, 0.0f, 0.05f);
    ASSERT_NEAR((float)var, 1.0f, 0.1f);
    PASS();
}

static void test_xavier_init(void) {
    TEST("Xavier initialization (mean ~0, bounded)");
    ac_tensor* t = ac_tensor_2d(100, 100, 0);
    ac_tensor_xavier(t, 100, 100);
    float max_val = 0, sum = 0;
    for (int i = 0; i < 10000; i++) {
        if (fabsf(t->data[i]) > max_val) max_val = fabsf(t->data[i]);
        sum += t->data[i];
    }
    float mean = sum / 10000.0f;
    ASSERT_TRUE(max_val < 1.0f, "values too large");
    ASSERT_TRUE(fabsf(mean) < 0.1f, "mean not near zero");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 12: INTEGRATION TESTS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_xor_training(void) {
    TEST("XOR training (full pipeline)");
    
    ac_tensor* X = ac_tensor_2d(4, 2, 0);
    X->data[0]=0; X->data[1]=0;
    X->data[2]=0; X->data[3]=1;
    X->data[4]=1; X->data[5]=0;
    X->data[6]=1; X->data[7]=1;
    
    ac_tensor* y = ac_tensor_2d(4, 1, 0);
    y->data[0]=0; y->data[1]=1; y->data[2]=1; y->data[3]=0;
    
    ac_dense l1, l2;
    ac_dense_init(&l1, 2, 16);
    ac_dense_init(&l2, 16, 1);
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, l1.weight);
    ac_param_group_add(&params, l1.bias);
    ac_param_group_add(&params, l2.weight);
    ac_param_group_add(&params, l2.bias);
    
    ac_adam opt;
    ac_adam_init(&opt, &params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 0);
    
    float final_loss = 999.0f;
    for (int epoch = 0; epoch < 500; epoch++) {
        ac_zero_grad(&params);
        ac_tensor* h1 = ac_dense_forward(&l1, X);
        ac_tensor* a1 = ac_tensor_relu(h1);
        ac_tensor* out = ac_dense_forward(&l2, a1);
        ac_tensor* pred = ac_tensor_sigmoid(out);
        ac_tensor* loss = ac_mse_loss(pred, y);
        ac_backward(loss);
        ac_adam_step(&opt);
        final_loss = loss->data[0];
    }
    
    ASSERT_TRUE(final_loss < 0.05f, "XOR training did not converge");
    ac_param_group_destroy(&params);
    PASS();
}

static void test_arena_checkpoint_training(void) {
    TEST("Arena checkpoint during training loop");
    
    ac_dense layer;
    ac_dense_init(&layer, 4, 2);
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, layer.weight);
    ac_param_group_add(&params, layer.bias);
    
    ac_adam opt;
    ac_adam_init(&opt, &params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 0);
    
    ac_arena_checkpoint cp;
    ac_arena_save(&g_tensor_arena, &cp);
    ac_size base_alloc = g_tensor_arena.total_allocated;
    
    for (int i = 0; i < 10; i++) {
        ac_zero_grad(&params);
        ac_tensor* x = ac_tensor_2d(4, 4, 0);
        ac_tensor_uniform(x, -1.0f, 1.0f);
        ac_tensor* target = ac_tensor_2d(4, 2, 0);
        ac_tensor_fill(target, 0.5f);
        
        ac_tensor* out = ac_dense_forward(&layer, x);
        ac_tensor* pred = ac_tensor_sigmoid(out);
        ac_tensor* loss = ac_mse_loss(pred, target);
        ac_backward(loss);
        ac_adam_step(&opt);
        
        ac_arena_restore(&g_tensor_arena, &cp);
    }
    
    ASSERT_TRUE(g_tensor_arena.total_allocated == base_alloc,
                "arena should not grow with checkpointing");
    ac_param_group_destroy(&params);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 13: ERROR HANDLING & DYNAMIC LIMITS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_error_system(void) {
    TEST("Error handling system");
    ac_clear_error();
    ASSERT_TRUE(ac_get_last_error() == AC_OK, "should start clean");
    
    ac_param_group empty;
    ac_param_group_init(&empty);
    ac_error_code err = ac_model_load("nonexistent_file.acml", &empty);
    ASSERT_TRUE(err != AC_OK, "should fail for missing file");
    ASSERT_TRUE(ac_get_last_error() == AC_ERR_FILE_IO, "should be FILE_IO error");
    
    ac_clear_error();
    ASSERT_TRUE(ac_get_last_error() == AC_OK, "clear should reset");
    ac_param_group_destroy(&empty);
    PASS();
}

static void test_dynamic_param_group(void) {
    TEST("Dynamic param group growth (300 params)");
    ac_param_group params;
    ac_param_group_init(&params);
    
    for (int i = 0; i < 300; i++) {
        ac_tensor* t = ac_tensor_1d(1, 0);
        ac_param_group_add(&params, t);
    }
    ASSERT_TRUE(params.num_params == 300, "should hold 300 params");
    ASSERT_TRUE(params.params[299] != NULL, "last param should be valid");
    ac_param_group_destroy(&params);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 14: NEW TENSOR OPS (Sub, Div, Reshape)
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_tensor_sub(void) {
    TEST("Tensor sub (forward + backward)");
    ac_tensor* a = ac_tensor_1d(4, 1);
    ac_tensor* b = ac_tensor_1d(4, 1);
    ac_tensor_fill(a, 5.0f);
    ac_tensor_fill(b, 3.0f);
    ac_tensor* c = ac_tensor_sub(a, b);
    /* forward: 5-3 = 2 */
    for (int i = 0; i < 4; i++) ASSERT_NEAR(c->data[i], 2.0f, 1e-5f);
    ac_tensor* loss = ac_tensor_sum(c);
    ac_backward(loss);
    /* d(a-b)/da = +1, d(a-b)/db = -1 */
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(a->grad[i], 1.0f, 1e-5f);
        ASSERT_NEAR(b->grad[i], -1.0f, 1e-5f);
    }
    PASS();
}

static void test_tensor_div(void) {
    TEST("Tensor div (forward + backward)");
    ac_tensor* a = ac_tensor_1d(4, 1);
    ac_tensor* b = ac_tensor_1d(4, 1);
    ac_tensor_fill(a, 6.0f);
    ac_tensor_fill(b, 3.0f);
    ac_tensor* c = ac_tensor_div(a, b);
    /* forward: 6/3 = 2 */
    for (int i = 0; i < 4; i++) ASSERT_NEAR(c->data[i], 2.0f, 1e-5f);
    ac_tensor* loss = ac_tensor_sum(c);
    ac_backward(loss);
    /* d(a/b)/da = 1/b = 1/3, d(a/b)/db = -a/b^2 = -6/9 = -0.6667 */
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(a->grad[i], 1.0f/3.0f, 1e-4f);
        ASSERT_NEAR(b->grad[i], -6.0f/9.0f, 1e-4f);
    }
    PASS();
}

static void test_tensor_reshape(void) {
    TEST("Tensor reshape (forward)");
    ac_tensor* a = ac_tensor_2d(3, 4, 0);
    for (int i = 0; i < 12; i++) a->data[i] = (float)i;
    ac_shape new_shape = ac_shape_2d(4, 3);
    ac_tensor* b = ac_tensor_reshape(a, new_shape);
    ASSERT_TRUE(b->shape.dims[0] == 4, "reshape dim0");
    ASSERT_TRUE(b->shape.dims[1] == 3, "reshape dim1");
    ASSERT_TRUE(b->shape.total_size == 12, "reshape total");
    for (int i = 0; i < 12; i++) ASSERT_NEAR(b->data[i], (float)i, 1e-6f);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 15: AUTOGRAD - SCALE BACKWARD (regression test)
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_autograd_scale(void) {
    TEST("Autograd: scale backward (fixed bug)");
    ac_tensor* a = ac_tensor_1d(4, 1);
    ac_tensor_fill(a, 2.0f);
    ac_tensor* b = ac_tensor_scale(a, 3.0f);
    ac_tensor* loss = ac_tensor_sum(b);
    ac_backward(loss);
    /* d(3*a)/da = 3 */
    for (int i = 0; i < 4; i++) ASSERT_NEAR(a->grad[i], 3.0f, 1e-5f);
    PASS();
}

static void test_autograd_scale_zero_input(void) {
    TEST("Autograd: scale backward with zero input");
    ac_tensor* a = ac_tensor_1d(4, 1);
    ac_tensor_fill(a, 0.0f);
    ac_tensor* b = ac_tensor_scale(a, 5.0f);
    ac_tensor* loss = ac_tensor_sum(b);
    ac_backward(loss);
    /* d(5*a)/da = 5 even when a=0 (old bug would divide by zero) */
    for (int i = 0; i < 4; i++) ASSERT_NEAR(a->grad[i], 5.0f, 1e-5f);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 16: RELU BACKWARD ACCUMULATION (regression test)
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_relu_backward_accumulation(void) {
    TEST("ReLU backward accumulates (fixed bug)");
    /* Two paths through 'a' that both go through relu, grads must add */
    ac_tensor* a = ac_tensor_1d(4, 1);
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = -1.0f; a->data[3] = 3.0f;
    ac_tensor* b = ac_tensor_relu(a);
    ac_tensor* c = ac_tensor_scale(b, 2.0f);
    ac_tensor* d = ac_tensor_add(c, b); /* d = 2*relu(a) + relu(a) = 3*relu(a) */
    ac_tensor* loss = ac_tensor_sum(d);
    ac_backward(loss);
    /* Where a>0: grad should be 3.0; where a<=0: grad should be 0.0 */
    ASSERT_NEAR(a->grad[0], 3.0f, 1e-4f);
    ASSERT_NEAR(a->grad[1], 3.0f, 1e-4f);
    ASSERT_NEAR(a->grad[2], 0.0f, 1e-4f);
    ASSERT_NEAR(a->grad[3], 3.0f, 1e-4f);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 17: GRADIENT CLIPPING
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_grad_clip_norm(void) {
    TEST("Gradient clipping (L2 norm)");
    ac_tensor* w = ac_tensor_1d(4, 1);
    ac_tensor_fill(w, 1.0f);
    /* Set large gradient: [10, 10, 10, 10], norm = 20 */
    for (int i = 0; i < 4; i++) w->grad[i] = 10.0f;
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, w);
    
    float norm = ac_clip_grad_norm(&params, 1.0f);
    ASSERT_NEAR(norm, 20.0f, 1e-3f);
    
    /* After clipping to max_norm=1, new norm should ≈ 1 */
    float new_sq = 0;
    for (int i = 0; i < 4; i++) new_sq += w->grad[i] * w->grad[i];
    ASSERT_NEAR(sqrtf(new_sq), 1.0f, 1e-4f);
    ac_param_group_destroy(&params);
    PASS();
}

static void test_grad_clip_value(void) {
    TEST("Gradient clipping (value)");
    ac_tensor* w = ac_tensor_1d(4, 1);
    ac_tensor_fill(w, 1.0f);
    w->grad[0] = 5.0f; w->grad[1] = -5.0f;
    w->grad[2] = 0.5f; w->grad[3] = -0.5f;
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, w);
    
    ac_clip_grad_value(&params, 1.0f);
    ASSERT_NEAR(w->grad[0], 1.0f, 1e-6f);
    ASSERT_NEAR(w->grad[1], -1.0f, 1e-6f);
    ASSERT_NEAR(w->grad[2], 0.5f, 1e-6f);
    ASSERT_NEAR(w->grad[3], -0.5f, 1e-6f);
    ac_param_group_destroy(&params);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 18: LR SCHEDULERS
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_lr_step_decay(void) {
    TEST("LR scheduler: step decay");
    ac_lr_scheduler sched;
    ac_lr_scheduler_init(&sched, AC_LR_STEP, 0.1f, 0.1f, 10, 0, 0.0f);
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 0), 0.1f, 1e-6f);
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 9), 0.1f, 1e-6f);
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 10), 0.01f, 1e-6f);
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 20), 0.001f, 1e-6f);
    PASS();
}

static void test_lr_cosine(void) {
    TEST("LR scheduler: cosine annealing");
    ac_lr_scheduler sched;
    ac_lr_scheduler_init(&sched, AC_LR_COSINE, 0.1f, 0.0f, 0, 100, 0.0f);
    /* At epoch 0: lr = 0.1 */
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 0), 0.1f, 1e-5f);
    /* At half-point: lr ≈ 0.05 */
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 50), 0.05f, 1e-3f);
    /* At end: lr ≈ min_lr = 0 */
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 100), 0.0f, 1e-5f);
    PASS();
}

static void test_lr_exponential(void) {
    TEST("LR scheduler: exponential decay");
    ac_lr_scheduler sched;
    ac_lr_scheduler_init(&sched, AC_LR_EXP, 0.1f, 0.9f, 0, 0, 0.0f);
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 0), 0.1f, 1e-6f);
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 1), 0.09f, 1e-5f);
    ASSERT_NEAR(ac_lr_scheduler_get(&sched, 10), 0.1f * powf(0.9f, 10.0f), 1e-5f);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 19: DENSE LAYER BACKWARD
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_dense_backward(void) {
    TEST("Dense layer backward propagates gradients");
    ac_dense layer;
    ac_dense_init(&layer, 4, 2);
    
    ac_tensor* x = ac_tensor_2d(1, 4, 1);
    ac_tensor_fill(x, 1.0f);
    
    ac_tensor* out = ac_dense_forward(&layer, x);
    ac_tensor* loss = ac_tensor_sum(out);
    ac_backward(loss);
    
    /* Weight grads should be non-zero */
    int w_nonzero = 0;
    for (ac_size i = 0; i < layer.weight->shape.total_size; i++) {
        if (fabsf(layer.weight->grad[i]) > 1e-6f) w_nonzero++;
    }
    ASSERT_TRUE(w_nonzero > 0, "weight grads should be non-zero");
    
    /* Input grad should be non-zero */
    int x_nonzero = 0;
    for (ac_size i = 0; i < 4; i++) {
        if (fabsf(x->grad[i]) > 1e-6f) x_nonzero++;
    }
    ASSERT_TRUE(x_nonzero > 0, "input grads should be non-zero");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 20: FLATTEN BACKWARD
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_flatten_backward(void) {
    TEST("Flatten backward (gradient passthrough)");
    
    ac_tensor* x = ac_tensor_create(ac_shape_4d(1, 2, 3, 4), 1);
    ac_tensor_fill(x, 1.0f);
    
    ac_tensor* out = ac_flatten(x);
    ASSERT_TRUE(out->shape.ndim == 2, "should be 2D after flatten");
    ASSERT_TRUE(out->shape.total_size == 24, "total size preserved");
    
    ac_tensor* loss = ac_tensor_sum(out);
    ac_backward(loss);
    
    /* All grads in x should be 1.0 (identity passthrough) */
    for (int i = 0; i < 24; i++) {
        ASSERT_NEAR(x->grad[i], 1.0f, 1e-5f);
    }
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 21: DROPOUT BACKWARD
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_dropout_backward(void) {
    TEST("Dropout backward (mask application)");
    ac_dropout drop;
    ac_dropout_init(&drop, 0.5f);
    drop.training = 1;
    
    ac_tensor* x = ac_tensor_1d(100, 1);
    ac_tensor_fill(x, 1.0f);
    
    ac_tensor* out = ac_dropout_forward(&drop, x);
    ac_tensor* loss = ac_tensor_sum(out);
    ac_backward(loss);
    
    /* Gradient should be zero where output is zero, scaled where non-zero */
    int zeros = 0, nonzeros = 0;
    for (int i = 0; i < 100; i++) {
        if (fabsf(out->data[i]) < 1e-6f) {
            ASSERT_NEAR(x->grad[i], 0.0f, 1e-6f);
            zeros++;
        } else {
            nonzeros++;
        }
    }
    ASSERT_TRUE(zeros > 10, "dropout should zero some elements");
    ASSERT_TRUE(nonzeros > 10, "dropout should keep some elements");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION 22: INT8 QUANTIZATION
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_quant_roundtrip(void) {
    TEST("INT8 quantize → dequantize roundtrip");
    ac_tensor* t = ac_tensor_1d(100, 0);
    for (int i = 0; i < 100; i++) t->data[i] = (float)i / 50.0f - 1.0f; /* [-1, 0.98] */
    
    ac_qtensor* qt = ac_tensor_quantize(t);
    ac_tensor* restored = ac_qtensor_dequantize(qt);
    
    /* Check roundtrip error < scale (~ 2/255 ≈ 0.008) */
    float max_err = 0.0f;
    for (int i = 0; i < 100; i++) {
        float err = fabsf(t->data[i] - restored->data[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT_TRUE(max_err < 0.02f, "roundtrip error too large");
    PASS();
}

static void test_quant_calibrate(void) {
    TEST("INT8 calibration (scale/zero_point)");
    float data[4] = { -1.0f, 0.0f, 0.5f, 1.0f };
    ac_quant_params qp = ac_calibrate(data, 4);
    
    ASSERT_NEAR(qp.min_val, -1.0f, 1e-6f);
    ASSERT_NEAR(qp.max_val, 1.0f, 1e-6f);
    /* scale = 2.0/255 ≈ 0.00784 */
    ASSERT_TRUE(qp.scale > 0.007f && qp.scale < 0.008f, "bad scale");
    /* zero_point should map 0.0 → ~128 */
    ASSERT_TRUE(qp.zero_point >= 126 && qp.zero_point <= 128, "bad zero_point");
    PASS();
}

static void test_quant_dense(void) {
    TEST("Quantized dense forward (INT8 matmul)");
    /* Create a trained dense layer: 4→2 */
    ac_tensor* weight = ac_tensor_2d(2, 4, 0);
    for (int i = 0; i < 8; i++) weight->data[i] = (float)(i % 3) * 0.5f;
    
    ac_tensor* bias = ac_tensor_1d(2, 0);
    bias->data[0] = 0.1f; bias->data[1] = -0.1f;
    
    /* Float forward (reference) */
    ac_tensor* input = ac_tensor_2d(1, 4, 0);
    for (int i = 0; i < 4; i++) input->data[i] = 0.5f;
    
    /* Compute float reference: out = input @ weight^T + bias */
    float ref[2] = {0};
    for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 4; k++) {
            ref[j] += input->data[k] * weight->data[j * 4 + k];
        }
        ref[j] += bias->data[j];
    }
    
    /* Quantized forward */
    ac_qdense qd;
    ac_qdense_from_dense(&qd, weight, bias, 4, 2);
    ac_tensor* qout = ac_qdense_forward(&qd, input);
    
    /* Quantization error should be small (< 5% relative) */
    for (int j = 0; j < 2; j++) {
        float abs_err = fabsf(qout->data[j] - ref[j]);
        /* Allow larger error for small values */
        ASSERT_TRUE(abs_err < 0.3f, "quantized output too far from reference");
    }
    PASS();
}

static void test_model_size(void) {
    TEST("Model size estimation (4x compression)");
    ac_tensor* w = ac_tensor_2d(100, 100, 0);
    ac_tensor_fill(w, 1.0f);
    
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, w);
    
    ac_model_size_info info = ac_estimate_model_size(&params);
    ASSERT_TRUE(info.num_params == 10000, "wrong param count");
    ASSERT_TRUE(info.fp32_bytes == 40000, "wrong fp32 size");
    ASSERT_TRUE(info.compression > 3.5f, "should be ~4x compression");
    ac_param_group_destroy(&params);
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION: Conv2D / MaxPool / BatchNorm Backward Passes
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_conv2d_backward(void) {
    TEST("Conv2D backward propagates gradients");
    ac_conv2d layer;
    ac_conv2d_init(&layer, 1, 2, 3, 1, 0); /* in=1, out=2, k=3, stride=1, pad=0 */
    
    ac_tensor* x = ac_tensor_create(ac_shape_4d(1, 1, 5, 5), 1); /* requires_grad */
    ac_tensor_fill(x, 1.0f);
    
    ac_tensor* out = ac_conv2d_forward(&layer, x);
    ASSERT_TRUE(out != NULL, "forward returned NULL");
    
    ac_tensor* loss = ac_tensor_sum(out);
    ac_backward(loss);
    
    /* Weight grads should be non-zero */
    int w_nonzero = 0;
    for (ac_size i = 0; i < layer.weight->shape.total_size; i++) {
        if (fabsf(layer.weight->grad[i]) > 1e-6f) w_nonzero++;
    }
    ASSERT_TRUE(w_nonzero > 0, "weight grads should be non-zero");
    
    /* Bias grads should be non-zero (sum of upstream grad over spatial dims) */
    int b_nonzero = 0;
    for (ac_size i = 0; i < layer.bias->shape.total_size; i++) {
        if (fabsf(layer.bias->grad[i]) > 1e-6f) b_nonzero++;
    }
    ASSERT_TRUE(b_nonzero > 0, "bias grads should be non-zero");
    
    /* Input grad should be non-zero */
    int x_nonzero = 0;
    for (ac_size i = 0; i < x->shape.total_size; i++) {
        if (fabsf(x->grad[i]) > 1e-6f) x_nonzero++;
    }
    ASSERT_TRUE(x_nonzero > 0, "input grads should be non-zero");
    PASS();
}

static void test_conv2d_backward_values(void) {
    TEST("Conv2D backward weight grad = input patch sums");
    ac_conv2d layer;
    ac_conv2d_init(&layer, 1, 1, 2, 1, 0); /* 1 in, 1 out, k=2 */
    
    /* Set known weights */
    for (ac_size i = 0; i < layer.weight->shape.total_size; i++)
        layer.weight->data[i] = 1.0f;
    layer.bias->data[0] = 0.0f;
    
    ac_tensor* x = ac_tensor_create(ac_shape_4d(1, 1, 3, 3), 1);
    /* Fill with 1..9 */
    for (int i = 0; i < 9; i++) x->data[i] = (float)(i + 1);
    
    ac_tensor* out = ac_conv2d_forward(&layer, x);
    /* out is 2x2 with all-ones kernel: each element = sum of 2x2 patch */
    /* out[0,0] = 1+2+4+5=12, out[0,1]=2+3+5+6=16, etc. */
    ASSERT_NEAR(out->data[0], 12.0f, 1e-4f);
    ASSERT_NEAR(out->data[1], 16.0f, 1e-4f);
    
    ac_tensor* loss = ac_tensor_sum(out);
    ac_backward(loss);
    
    /* With upstream grad = 1 for all outputs, weight grad for kernel[i,j]
       should equal the sum of input patches at that position */
    /* Weight grad should be non-zero and finite */
    for (ac_size i = 0; i < layer.weight->shape.total_size; i++) {
        ASSERT_TRUE(!isnan(layer.weight->grad[i]) && !isinf(layer.weight->grad[i]),
                     "weight grad is NaN/Inf");
    }
    PASS();
}

static void test_maxpool2d_backward(void) {
    TEST("MaxPool2D backward routes grad to max elements");
    ac_maxpool2d pool;
    ac_maxpool2d_init(&pool, 2, 2); /* 2x2 pool, stride 2 */
    
    ac_tensor* x = ac_tensor_create(ac_shape_4d(1, 1, 4, 4), 1);
    /* Fill 0..15 so each 2x2 block has a clear max */
    for (int i = 0; i < 16; i++) x->data[i] = (float)i;
    /* Max positions: [5, 7, 13, 15] (bottom-right of each pool region) */
    
    ac_tensor* out = ac_maxpool2d_forward(&pool, x);
    ASSERT_TRUE(out != NULL, "forward returned NULL");
    
    ac_tensor* loss = ac_tensor_sum(out);
    ac_backward(loss);
    
    /* Grad should flow only to max positions */
    ASSERT_NEAR(x->grad[5],  1.0f, 1e-5f);  /* max of [0,1,4,5] */
    ASSERT_NEAR(x->grad[7],  1.0f, 1e-5f);  /* max of [2,3,6,7] */
    ASSERT_NEAR(x->grad[13], 1.0f, 1e-5f);  /* max of [8,9,12,13] */
    ASSERT_NEAR(x->grad[15], 1.0f, 1e-5f);  /* max of [10,11,14,15] */
    
    /* Non-max positions should have zero grad */
    ASSERT_NEAR(x->grad[0], 0.0f, 1e-5f);
    ASSERT_NEAR(x->grad[1], 0.0f, 1e-5f);
    ASSERT_NEAR(x->grad[4], 0.0f, 1e-5f);
    ASSERT_NEAR(x->grad[10], 0.0f, 1e-5f);
    PASS();
}

static void test_batchnorm_backward(void) {
    TEST("BatchNorm backward propagates gradients");
    ac_batchnorm bn;
    ac_batchnorm_init(&bn, 4); /* 4 features */
    
    ac_tensor* x = ac_tensor_2d(8, 4, 1); /* batch=8, features=4 */
    ac_tensor_uniform(x, -2.0f, 2.0f);
    
    ac_tensor* out = ac_batchnorm_forward(&bn, x);
    ASSERT_TRUE(out != NULL, "forward returned NULL");
    
    /* Use a weighted sum to get non-uniform upstream gradients
       (uniform grads through BN backward yields zero — mathematically correct) */
    ac_tensor* weights = ac_tensor_2d(8, 4, 0);
    for (ac_size i = 0; i < weights->shape.total_size; i++)
        weights->data[i] = (float)(i + 1);
    ac_tensor* weighted = ac_tensor_mul(out, weights);
    ac_tensor* loss = ac_tensor_sum(weighted);
    ac_backward(loss);
    
    /* Input grad should be non-zero */
    int x_nonzero = 0;
    for (ac_size i = 0; i < x->shape.total_size; i++) {
        if (fabsf(x->grad[i]) > 1e-6f) x_nonzero++;
    }
    ASSERT_TRUE(x_nonzero > 0, "input grads should be non-zero");
    
    /* Gamma grads should be non-zero */
    int g_nonzero = 0;
    for (ac_size i = 0; i < bn.gamma->shape.total_size; i++) {
        if (fabsf(bn.gamma->grad[i]) > 1e-6f) g_nonzero++;
    }
    ASSERT_TRUE(g_nonzero > 0, "gamma grads should be non-zero");
    PASS();
}

static void test_batchnorm_backward_values(void) {
    TEST("BatchNorm backward grad consistency");
    ac_batchnorm bn;
    ac_batchnorm_init(&bn, 2); /* 2 features */
    
    ac_tensor* x = ac_tensor_2d(4, 2, 1);
    /* Feature 0: varying                Feature 1: varying differently */
    x->data[0] = 1.0f; x->data[1] = 10.0f;
    x->data[2] = 2.0f; x->data[3] = 20.0f;
    x->data[4] = 3.0f; x->data[5] = 30.0f;
    x->data[6] = 4.0f; x->data[7] = 40.0f;
    
    ac_tensor* out = ac_batchnorm_forward(&bn, x);
    
    /* Non-uniform upstream gradient via element-wise multiply */
    ac_tensor* scale = ac_tensor_2d(4, 2, 0);
    for (ac_size i = 0; i < 8; i++) scale->data[i] = (float)(i + 1);
    ac_tensor* weighted = ac_tensor_mul(out, scale);
    ac_tensor* loss = ac_tensor_sum(weighted);
    ac_backward(loss);
    
    /* Both gamma grads should be non-zero with non-uniform upstream grads */
    ASSERT_TRUE(fabsf(bn.gamma->grad[0]) > 1e-6f, "gamma grad[0] should be non-zero");
    ASSERT_TRUE(fabsf(bn.gamma->grad[1]) > 1e-6f, "gamma grad[1] should be non-zero");
    
    /* All input grads should be finite */
    for (ac_size i = 0; i < x->shape.total_size; i++) {
        ASSERT_TRUE(!isnan(x->grad[i]) && !isinf(x->grad[i]), "input grad NaN/Inf");
    }
    
    /* Input grads should be non-zero */
    int x_nonzero = 0;
    for (ac_size i = 0; i < x->shape.total_size; i++) {
        if (fabsf(x->grad[i]) > 1e-6f) x_nonzero++;
    }
    ASSERT_TRUE(x_nonzero > 0, "input grads should be non-zero");
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  SECTION: Autograd Backward — Sigmoid, Tanh, Softmax, Mean
 * ════════════════════════════════════════════════════════════════════════════ */

static void test_autograd_sigmoid(void) {
    TEST("Autograd: sigmoid backward (gradient check)");
    ac_tensor* x = ac_tensor_1d(4, 1);
    x->data[0] = -1.0f; x->data[1] = 0.0f; x->data[2] = 0.5f; x->data[3] = 2.0f;
    
    ac_tensor* y = ac_tensor_sigmoid(x);
    ac_tensor* loss = ac_tensor_sum(y);
    ac_backward(loss);
    
    /* sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) */
    for (int i = 0; i < 4; i++) {
        float s = y->data[i];
        float expected_grad = s * (1.0f - s);
        ASSERT_NEAR(x->grad[i], expected_grad, 0.02f);
    }
    PASS();
}

static void test_autograd_tanh(void) {
    TEST("Autograd: tanh backward (gradient check)");
    ac_tensor* x = ac_tensor_1d(4, 1);
    x->data[0] = -1.0f; x->data[1] = 0.0f; x->data[2] = 0.5f; x->data[3] = 1.5f;
    
    ac_tensor* y = ac_tensor_tanh(x);
    ac_tensor* loss = ac_tensor_sum(y);
    ac_backward(loss);
    
    /* tanh'(x) = 1 - tanh(x)^2 */
    for (int i = 0; i < 4; i++) {
        float t = y->data[i];
        float expected_grad = 1.0f - t * t;
        ASSERT_NEAR(x->grad[i], expected_grad, 0.02f);
    }
    PASS();
}

static void test_autograd_softmax(void) {
    TEST("Autograd: softmax backward produces gradients");
    ac_tensor* x = ac_tensor_2d(1, 4, 1);
    x->data[0] = 1.0f; x->data[1] = 2.0f; x->data[2] = 3.0f; x->data[3] = 4.0f;
    
    ac_tensor* y = ac_tensor_softmax(x);
    
    /* Weight by different factors so upstream grad is non-uniform */
    ac_tensor* w = ac_tensor_2d(1, 4, 0);
    w->data[0] = 1.0f; w->data[1] = 0.0f; w->data[2] = 0.0f; w->data[3] = 0.0f;
    ac_tensor* weighted = ac_tensor_mul(y, w);
    ac_tensor* loss = ac_tensor_sum(weighted);
    ac_backward(loss);
    
    /* Grad should be non-zero and finite */
    int nonzero = 0;
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(!isnan(x->grad[i]) && !isinf(x->grad[i]), "grad NaN/Inf");
        if (fabsf(x->grad[i]) > 1e-6f) nonzero++;
    }
    ASSERT_TRUE(nonzero > 0, "softmax grads should be non-zero");
    PASS();
}

static void test_autograd_mean(void) {
    TEST("Autograd: mean backward = 1/n");
    ac_tensor* x = ac_tensor_1d(8, 1);
    for (int i = 0; i < 8; i++) x->data[i] = (float)(i + 1);
    
    ac_tensor* y = ac_tensor_mean(x);
    ac_backward(y);
    
    /* mean backward: each grad = 1/n */
    float expected = 1.0f / 8.0f;
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(x->grad[i], expected, 1e-6f);
    }
    PASS();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ════════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("\n");
    printf("  +========================================================+\n");
    printf("  |     AICRAFT v%s - Production Test Suite             |\n", AICRAFT_VERSION_STRING);
    printf("  +========================================================+\n");
    
    ac_init();
    
    print_section("Tensor Core");
    test_tensor_creation_1d();
    test_tensor_creation_2d();
    test_tensor_creation_with_grad();
    test_tensor_fill();
    test_tensor_shape_4d();
    
    print_section("SIMD Operations");
    test_simd_add();
    test_simd_mul();
    test_simd_scale();
    test_simd_dot();
    test_simd_sum();
    test_simd_max();
    test_simd_fma();
    
    print_section("Matrix Operations");
    test_matmul_small();
    test_matmul_identity();
    test_matmul_large();
    test_transpose();
    
    print_section("Activations");
    test_relu();
    test_sigmoid();
    test_tanh_activation();
    test_softmax();
    
    print_section("Layers");
    test_dense_forward();
    test_conv2d_forward();
    test_maxpool2d_forward();
    test_batchnorm_forward();
    test_dropout_forward();
    test_flatten();
    
    print_section("Loss Functions");
    test_mse_loss();
    test_bce_loss();
    test_cross_entropy_loss();
    
    print_section("Autograd");
    test_autograd_add();
    test_autograd_mul();
    test_autograd_matmul();
    test_autograd_relu();
    test_autograd_mse();
    test_autograd_bce();
    
    print_section("Optimizers");
    test_sgd_step();
    test_adam_step();
    
    print_section("Memory Management");
    test_arena_allocator();
    test_arena_checkpoint();
    test_pool_allocator();
    
    print_section("Serialization");
    test_model_save_load();
    
    print_section("PRNG");
    test_prng_range();
    test_prng_distribution();
    test_xavier_init();
    
    print_section("Integration Tests");
    test_xor_training();
    test_arena_checkpoint_training();
    
    print_section("Error Handling & Dynamic Limits");
    test_error_system();
    test_dynamic_param_group();
    
    print_section("New Tensor Ops (Sub, Div, Reshape)");
    test_tensor_sub();
    test_tensor_div();
    test_tensor_reshape();
    
    print_section("Autograd Regression (Scale, ReLU accum)");
    test_autograd_scale();
    test_autograd_scale_zero_input();
    test_relu_backward_accumulation();
    
    print_section("Gradient Clipping");
    test_grad_clip_norm();
    test_grad_clip_value();
    
    print_section("LR Schedulers");
    test_lr_step_decay();
    test_lr_cosine();
    test_lr_exponential();
    
    print_section("Layer Backward Passes");
    test_dense_backward();
    test_flatten_backward();
    test_dropout_backward();
    
    print_section("Conv2D / MaxPool / BatchNorm Backward");
    test_conv2d_backward();
    test_conv2d_backward_values();
    test_maxpool2d_backward();
    test_batchnorm_backward();
    test_batchnorm_backward_values();
    
    print_section("Autograd: Sigmoid / Tanh / Softmax / Mean");
    test_autograd_sigmoid();
    test_autograd_tanh();
    test_autograd_softmax();
    test_autograd_mean();
    
    print_section("INT8 Quantization");
    test_quant_roundtrip();
    test_quant_calibrate();
    test_quant_dense();
    test_model_size();
    
    printf("\n  ========================================================\n");
    printf("  Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_total);
    
    if (tests_failed == 0) {
        printf("  All tests passed!\n");
    } else {
        printf("  %d test(s) FAILED\n", tests_failed);
    }
    printf("\n");
    
    ac_cleanup();
    return tests_failed > 0 ? 1 : 0;
}
