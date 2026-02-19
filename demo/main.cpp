/*
 * ============================================================================
 *  AICRAFT - Demo: Train an MLP on XOR Problem
 *  Shows the full pipeline: create model, train, evaluate.
 * ============================================================================
 */

#include "aicraft/aicraft.h"
#include <stdio.h>

int main(void) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║          AICRAFT v%s - XOR Training Demo            ║\n", AICRAFT_VERSION_STRING);
    printf("  ║     Zero Dependencies. Pure C/C++. SIMD-Optimized.     ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");
    
    ac_init();
    
    /* ── Define XOR dataset ─────────────────────────────────────────────── */
    
    /* Input: 4 samples of 2 features */
    ac_tensor* X = ac_tensor_2d(4, 2, 0);
    X->data[0] = 0; X->data[1] = 0;  /* 0 XOR 0 = 0 */
    X->data[2] = 0; X->data[3] = 1;  /* 0 XOR 1 = 1 */
    X->data[4] = 1; X->data[5] = 0;  /* 1 XOR 0 = 1 */
    X->data[6] = 1; X->data[7] = 1;  /* 1 XOR 1 = 0 */
    
    /* Target: [0, 1, 1, 0] */
    ac_tensor* y = ac_tensor_2d(4, 1, 0);
    y->data[0] = 0;
    y->data[1] = 1;
    y->data[2] = 1;
    y->data[3] = 0;
    
    /* ── Build Model: 2 -> 16 -> 16 -> 1 ───────────────────────────────── */
    
    ac_dense layer1, layer2, layer3;
    ac_dense_init(&layer1, 2, 16);
    ac_dense_init(&layer2, 16, 16);
    ac_dense_init(&layer3, 16, 1);
    
    /* Register parameters */
    ac_param_group params;
    ac_param_group_init(&params);
    ac_param_group_add(&params, layer1.weight);
    ac_param_group_add(&params, layer1.bias);
    ac_param_group_add(&params, layer2.weight);
    ac_param_group_add(&params, layer2.bias);
    ac_param_group_add(&params, layer3.weight);
    ac_param_group_add(&params, layer3.bias);
    
    /* Adam optimizer */
    ac_adam optimizer;
    ac_adam_init(&optimizer, &params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 0);
    
    /* ── Training Loop ──────────────────────────────────────────────────── */
    
    printf("  Training XOR network (2->16->16->1)...\n\n");
    printf("  %6s  %12s\n", "Epoch", "Loss");
    printf("  ------  ------------\n");
    
    ac_timer timer;
    ac_timer_start(&timer);
    
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        /* Save arena state so intermediates are freed each iteration */
        ac_arena_checkpoint cp;
        ac_arena_save(&g_tensor_arena, &cp);
        
        /* Zero gradients */
        ac_zero_grad(&params);
        
        /* Forward pass */
        ac_tensor* h1 = ac_dense_forward(&layer1, X);
        ac_tensor* a1 = ac_tensor_relu(h1);
        ac_tensor* h2 = ac_dense_forward(&layer2, a1);
        ac_tensor* a2 = ac_tensor_relu(h2);
        ac_tensor* out = ac_dense_forward(&layer3, a2);
        ac_tensor* pred = ac_tensor_sigmoid(out);
        
        /* MSE Loss */
        ac_tensor* loss = ac_mse_loss(pred, y);
        
        /* Backward */
        ac_backward(loss);
        
        /* Update */
        ac_adam_step(&optimizer);
        
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            printf("  %6d  %12.6f\n", epoch, loss->data[0]);
        }
        
        /* Restore arena — frees all intermediates from this iteration */
        ac_arena_restore(&g_tensor_arena, &cp);
    }
    
    double elapsed = ac_timer_stop(&timer);
    
    /* ── Evaluate ───────────────────────────────────────────────────────── */
    
    printf("\n  Training completed in %.3f ms (%.0f epochs/sec)\n\n", 
           elapsed * 1000.0, epochs / elapsed);
    
    /* Final forward pass */
    ac_tensor* h1 = ac_dense_forward(&layer1, X);
    ac_tensor* a1 = ac_tensor_relu(h1);
    ac_tensor* h2 = ac_dense_forward(&layer2, a1);
    ac_tensor* a2 = ac_tensor_relu(h2);
    ac_tensor* out = ac_dense_forward(&layer3, a2);
    ac_tensor* pred = ac_tensor_sigmoid(out);
    
    printf("  Results:\n");
    printf("  %5s %5s  -> %12s  %8s\n", "X1", "X2", "Prediction", "Target");
    printf("  -----  -----  ------------  --------\n");
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        int ok = (pred->data[i] > 0.5f) == (y->data[i] > 0.5f);
        correct += ok;
        printf("  %5.0f  %5.0f  -> %12.6f  %8.0f  %s\n", 
               X->data[i*2], X->data[i*2+1],
               pred->data[i], y->data[i],
               ok ? "OK" : "MISS");
    }
    
    if (correct == 4)
        printf("\n  All predictions correct!\n");
    else
        printf("\n  %d/4 predictions correct.\n", correct);
    
    /* ── Cleanup ────────────────────────────────────────────────────────── */
    
    ac_cleanup();
    printf("\n  Done.\n\n");
    return 0;
}
