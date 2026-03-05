---
sidebar_position: 9
title: Custom Layers Guide
---

# Writing Custom Layers

Extend Aicraft with your own layers.

## Layer Structure

Every layer in Aicraft implements this interface:

```c
typedef struct AcLayer {
    // Forward pass: compute output from input
    AcTensor *(*forward)(struct AcLayer *self, AcTensor *input);
    
    // Parameters (optional)
    AcTensor *weights;
    AcTensor *bias;
    
    // Layer-specific data
    void *data;
    
    // Cleanup function
    void (*free)(struct AcLayer *self);
} AcLayer;
```

---

## Example: Dropout Layer

Let's implement a dropout layer that randomly zeroes activations during training.

### Header

```c
// dropout.h
#ifndef AC_DROPOUT_H
#define AC_DROPOUT_H

#include "aicraft/aicraft.h"

AcLayer *ac_dropout(float p);

#endif
```

### Implementation

```c
// dropout.c
#include "dropout.h"
#include <stdlib.h>

typedef struct {
    float p;        // Dropout probability
    bool training;  // Training mode flag
    AcTensor *mask; // Random mask
} DropoutData;

static AcTensor *dropout_forward(AcLayer *self, AcTensor *input) {
    DropoutData *d = (DropoutData *)self->data;
    
    // During inference, just scale by (1 - p)
    if (!d->training) {
        AcTensor *output = ac_tensor_new(input->shape, input->ndim);
        float scale = 1.0f - d->p;
        for (int i = 0; i < input->size; i++) {
            output->data[i] = input->data[i] * scale;
        }
        return output;
    }
    
    // During training, apply random mask
    AcTensor *output = ac_tensor_new(input->shape, input->ndim);
    d->mask = ac_tensor_rand(input->shape, input->ndim);
    
    for (int i = 0; i < input->size; i++) {
        if (d->mask->data[i] < d->p) {
            output->data[i] = 0;
            d->mask->data[i] = 0;  // Store mask for backward
        } else {
            output->data[i] = input->data[i] / (1.0f - d->p);
            d->mask->data[i] = 1.0f / (1.0f - d->p);
        }
    }
    
    // Connect autograd
    if (input->requires_grad) {
        output->requires_grad = true;
        output->grad_node = ac_grad_node_new(output);
        output->grad_node->inputs[0] = input;
        output->grad_node->backward = dropout_backward;
        output->grad_node->layer = self;
    }
    
    return output;
}

static void dropout_backward(AcGradNode *node, AcTensor *grad_output) {
    AcLayer *layer = (AcLayer *)node->layer;
    DropoutData *d = (DropoutData *)layer->data;
    AcTensor *input = node->inputs[0];
    
    if (!input->grad_node) return;
    
    // Gradient flows through non-dropped neurons
    AcTensor *grad_input = ac_tensor_new(grad_output->shape, grad_output->ndim);
    for (int i = 0; i < grad_output->size; i++) {
        grad_input->data[i] = grad_output->data[i] * d->mask->data[i];
    }
    
    ac_grad_accumulate(input->grad_node, grad_input);
}

static void dropout_free(AcLayer *self) {
    free(self->data);
    free(self);
}

AcLayer *ac_dropout(float p) {
    AcLayer *layer = calloc(1, sizeof(AcLayer));
    DropoutData *data = calloc(1, sizeof(DropoutData));
    
    data->p = p;
    data->training = true;
    data->mask = NULL;
    
    layer->data = data;
    layer->forward = dropout_forward;
    layer->free = dropout_free;
    layer->weights = NULL;
    layer->bias = NULL;
    
    return layer;
}

// Set training mode
void ac_dropout_set_training(AcLayer *layer, bool training) {
    DropoutData *d = (DropoutData *)layer->data;
    d->training = training;
}
```

### Usage

```c
AcLayer *net[] = {
    ac_dense(784, 256, AC_RELU),
    ac_dropout(0.5),              // 50% dropout
    ac_dense(256, 128, AC_RELU),
    ac_dropout(0.3),              // 30% dropout
    ac_dense(128, 10, AC_SOFTMAX)
};

// Training
for (int i = 0; i < num_samples; i++) {
    AcTensor *y = ac_forward_seq(net, 5, x);
    // ... train ...
}

// Inference: disable dropout
ac_dropout_set_training(net[1], false);
ac_dropout_set_training(net[3], false);
AcTensor *y = ac_forward_seq(net, 5, x);
```

---

## Example: Batch Normalisation

```c
typedef struct {
    float epsilon;
    float momentum;
    int features;
    AcTensor *running_mean;
    AcTensor *running_var;
    bool training;
} BatchNormData;

static AcTensor *batchnorm_forward(AcLayer *self, AcTensor *input) {
    BatchNormData *d = (BatchNormData *)self->data;
    int batch = input->shape[0];
    int features = input->shape[1];
    
    AcTensor *output = ac_tensor_new(input->shape, input->ndim);
    
    if (d->training) {
        // Compute batch statistics
        for (int f = 0; f < features; f++) {
            float mean = 0, var = 0;
            
            // Mean
            for (int b = 0; b < batch; b++) {
                mean += input->data[b * features + f];
            }
            mean /= batch;
            
            // Variance
            for (int b = 0; b < batch; b++) {
                float diff = input->data[b * features + f] - mean;
                var += diff * diff;
            }
            var /= batch;
            
            // Normalise
            float std = sqrtf(var + d->epsilon);
            for (int b = 0; b < batch; b++) {
                int idx = b * features + f;
                output->data[idx] = (input->data[idx] - mean) / std;
                // Apply scale and shift (gamma, beta)
                output->data[idx] = output->data[idx] * self->weights->data[f]
                                  + self->bias->data[f];
            }
            
            // Update running statistics
            d->running_mean->data[f] = d->momentum * d->running_mean->data[f]
                                     + (1 - d->momentum) * mean;
            d->running_var->data[f] = d->momentum * d->running_var->data[f]
                                    + (1 - d->momentum) * var;
        }
    } else {
        // Use running statistics for inference
        for (int f = 0; f < features; f++) {
            float mean = d->running_mean->data[f];
            float std = sqrtf(d->running_var->data[f] + d->epsilon);
            
            for (int b = 0; b < batch; b++) {
                int idx = b * features + f;
                output->data[idx] = (input->data[idx] - mean) / std;
                output->data[idx] = output->data[idx] * self->weights->data[f]
                                  + self->bias->data[f];
            }
        }
    }
    
    return output;
}

AcLayer *ac_batchnorm(int features) {
    AcLayer *layer = calloc(1, sizeof(AcLayer));
    BatchNormData *data = calloc(1, sizeof(BatchNormData));
    
    data->epsilon = 1e-5f;
    data->momentum = 0.1f;
    data->features = features;
    data->training = true;
    data->running_mean = ac_tensor_zeros((int[]){features}, 1);
    data->running_var = ac_tensor_ones((int[]){features}, 1);
    
    // Learnable parameters: gamma (scale) and beta (shift)
    layer->weights = ac_tensor_ones((int[]){features}, 1);  // gamma
    layer->bias = ac_tensor_zeros((int[]){features}, 1);    // beta
    layer->weights->requires_grad = true;
    layer->bias->requires_grad = true;
    
    layer->data = data;
    layer->forward = batchnorm_forward;
    
    return layer;
}
```

---

## Layer Checklist

When implementing a custom layer:

1. **Forward pass**: Compute output from input
2. **Backward pass**: Compute gradients w.r.t. inputs and parameters
3. **Parameters**: Mark `requires_grad = true` for learnable weights
4. **Autograd connection**: Create grad node and set backward function
5. **Memory**: Use arena allocator for temporary tensors
6. **Cleanup**: Implement free function

---

## Testing Your Layer

### Gradient Check

```c
bool test_my_layer_gradients() {
    AcLayer *layer = my_custom_layer(...);
    AcTensor *input = ac_tensor_rand((int[]){4, 10}, 2);
    input->requires_grad = true;
    
    // Numerical gradient check
    return ac_gradient_check_layer(layer, input, 1e-5, 1e-4);
}
```

### Shape Check

```c
void test_my_layer_shapes() {
    AcLayer *layer = my_custom_layer(...);
    AcTensor *input = ac_tensor_rand((int[]){32, 100}, 2);
    AcTensor *output = layer->forward(layer, input);
    
    assert(output->ndim == 2);
    assert(output->shape[0] == 32);  // Batch preserved
    assert(output->shape[1] == expected_features);
}
```

---

## Next Steps

- [Autograd Internals](/docs/internals/autograd) — How the computation graph works
- [API Reference: Layers](/docs/api/layers) — Built-in layer implementations
- [Performance Tuning](/docs/guides/performance-tuning) — Optimise your layers
