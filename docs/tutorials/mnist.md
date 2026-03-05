---
sidebar_position: 1
title: "Tutorial: MNIST from Scratch"
---

# Tutorial: MNIST Digit Recognition

Build a complete handwritten digit classifier from scratch using Aicraft.

## What You'll Learn

- Loading and preprocessing image data
- Building a feedforward neural network
- Training with cross-entropy loss
- Evaluating accuracy on test data
- Saving and loading trained models

## Prerequisites

- Aicraft installed (`git clone https://github.com/TobiasTesauri/Aicraft.git`)
- MNIST dataset (we'll show how to download it)
- Basic C knowledge

---

## Step 1: Download MNIST

MNIST comes as 4 gzipped files. Download them:

```bash
mkdir data && cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

---

## Step 2: Data Loading Helper

Create `mnist_loader.h`:

```c
#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Swap big-endian to little-endian
static inline uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) | ((val >> 8) & 0xff00) |
           ((val << 8) & 0xff0000) | ((val << 24) & 0xff000000);
}

typedef struct {
    float *images;    // [N, 784] flattened, normalized to [0,1]
    int   *labels;    // [N] integer labels 0-9
    int    count;
} MnistData;

MnistData load_mnist_images(const char *img_path, const char *label_path) {
    MnistData data = {0};

    // Load images
    FILE *fimg = fopen(img_path, "rb");
    if (!fimg) { fprintf(stderr, "Cannot open %s\n", img_path); exit(1); }

    uint32_t magic, num_images, rows, cols;
    fread(&magic, 4, 1, fimg);
    fread(&num_images, 4, 1, fimg);
    fread(&rows, 4, 1, fimg);
    fread(&cols, 4, 1, fimg);

    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    int img_size = rows * cols;  // 784
    data.count = num_images;
    data.images = malloc(num_images * img_size * sizeof(float));

    uint8_t *buffer = malloc(img_size);
    for (uint32_t i = 0; i < num_images; i++) {
        fread(buffer, 1, img_size, fimg);
        for (int j = 0; j < img_size; j++) {
            data.images[i * img_size + j] = buffer[j] / 255.0f;
        }
    }
    free(buffer);
    fclose(fimg);

    // Load labels
    FILE *flbl = fopen(label_path, "rb");
    if (!flbl) { fprintf(stderr, "Cannot open %s\n", label_path); exit(1); }

    uint32_t lbl_magic, num_labels;
    fread(&lbl_magic, 4, 1, flbl);
    fread(&num_labels, 4, 1, flbl);
    num_labels = swap_endian(num_labels);

    data.labels = malloc(num_labels * sizeof(int));
    for (uint32_t i = 0; i < num_labels; i++) {
        uint8_t lbl;
        fread(&lbl, 1, 1, flbl);
        data.labels[i] = lbl;
    }
    fclose(flbl);

    return data;
}

void free_mnist(MnistData *data) {
    free(data->images);
    free(data->labels);
}

#endif
```

---

## Step 3: Build the Network

Create `mnist_train.c`:

```c
#include "aicraft/aicraft.h"
#include "mnist_loader.h"
#include <stdio.h>
#include <time.h>

#define BATCH_SIZE  64
#define EPOCHS      10
#define LR          0.001f

int main(void) {
    ac_init();

    // ─── Load Data ────────────────────────────────────────
    printf("Loading MNIST...\n");
    MnistData train = load_mnist_images(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte"
    );
    MnistData test = load_mnist_images(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );
    printf("Train: %d samples, Test: %d samples\n", train.count, test.count);

    // ─── Define Network ───────────────────────────────────
    AcLayer *net[] = {
        ac_dense(784, 256, AC_RELU),
        ac_dense(256, 128, AC_RELU),
        ac_dense(128, 10,  AC_SOFTMAX)
    };
    int num_layers = 3;

    // ─── Optimiser ────────────────────────────────────────
    AcOptimizer *opt = ac_adam(net, num_layers, LR);

    // ─── Training Loop ────────────────────────────────────
    int num_batches = train.count / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0;
        clock_t start = clock();

        for (int b = 0; b < num_batches; b++) {
            ac_mem_checkpoint();

            // Get batch
            int offset = b * BATCH_SIZE;
            AcTensor *x = ac_tensor_from_data(
                &train.images[offset * 784],
                (int[]){BATCH_SIZE, 784}, 2
            );

            // One-hot encode labels
            AcTensor *y_true = ac_tensor_zeros((int[]){BATCH_SIZE, 10}, 2);
            for (int i = 0; i < BATCH_SIZE; i++) {
                int label = train.labels[offset + i];
                y_true->data[i * 10 + label] = 1.0f;
            }

            // Forward
            AcTensor *y_pred = ac_forward_seq(net, num_layers, x);

            // Loss
            AcTensor *loss = ac_cross_entropy(y_pred, y_true);
            epoch_loss += ac_scalar(loss);

            // Backward
            ac_backward(loss);

            // Update
            ac_optimizer_step(opt);

            ac_mem_restore();
        }

        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("Epoch %2d | Loss: %.4f | Time: %.2fs\n",
               epoch + 1, epoch_loss / num_batches, elapsed);
    }

    // ─── Evaluation ───────────────────────────────────────
    printf("\nEvaluating on test set...\n");
    int correct = 0;

    for (int i = 0; i < test.count; i++) {
        ac_mem_checkpoint();

        AcTensor *x = ac_tensor_from_data(
            &test.images[i * 784],
            (int[]){1, 784}, 2
        );

        AcTensor *y = ac_forward_seq(net, num_layers, x);

        // Find argmax
        int pred = 0;
        float max_val = y->data[0];
        for (int j = 1; j < 10; j++) {
            if (y->data[j] > max_val) {
                max_val = y->data[j];
                pred = j;
            }
        }

        if (pred == test.labels[i]) correct++;

        ac_mem_restore();
    }

    printf("Accuracy: %.2f%% (%d/%d)\n",
           100.0f * correct / test.count, correct, test.count);

    // ─── Save Model ───────────────────────────────────────
    ac_save_weights(net, num_layers, "mnist_model.bin");
    printf("Model saved to mnist_model.bin\n");

    // ─── Cleanup ──────────────────────────────────────────
    free_mnist(&train);
    free_mnist(&test);
    ac_cleanup();

    return 0;
}
```

---

## Step 4: Compile & Run

```bash
gcc -O3 -mavx2 mnist_train.c -I./include -lm -o mnist_train
./mnist_train
```

**Expected output:**

```
Loading MNIST...
Train: 60000 samples, Test: 10000 samples
Epoch  1 | Loss: 0.4523 | Time: 2.34s
Epoch  2 | Loss: 0.1876 | Time: 2.31s
Epoch  3 | Loss: 0.1245 | Time: 2.28s
...
Epoch 10 | Loss: 0.0412 | Time: 2.25s

Evaluating on test set...
Accuracy: 97.82% (9782/10000)
Model saved to mnist_model.bin
```

---

## Step 5: Inference Only

Create `mnist_infer.c` for production inference:

```c
#include "aicraft/aicraft.h"
#include "mnist_loader.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <image_index>\n", argv[0]);
        return 1;
    }

    ac_init();

    // Load test data
    MnistData test = load_mnist_images(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    );

    int idx = atoi(argv[1]);
    if (idx < 0 || idx >= test.count) {
        printf("Index out of range\n");
        return 1;
    }

    // Rebuild network and load weights
    AcLayer *net[] = {
        ac_dense(784, 256, AC_RELU),
        ac_dense(256, 128, AC_RELU),
        ac_dense(128, 10,  AC_SOFTMAX)
    };
    ac_load_weights(net, 3, "mnist_model.bin");

    // Single inference
    AcTensor *x = ac_tensor_from_data(
        &test.images[idx * 784],
        (int[]){1, 784}, 2
    );

    AcTensor *y = ac_forward_seq(net, 3, x);

    // Print probabilities
    printf("Image #%d (True label: %d)\n", idx, test.labels[idx]);
    printf("Predictions:\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d: %.2f%%\n", i, y->data[i] * 100);
    }

    free_mnist(&test);
    ac_cleanup();
    return 0;
}
```

---

## Next Steps

- **Improve accuracy**: Add dropout, batch normalisation, or conv layers
- **GPU acceleration**: Enable Vulkan backend for faster training
- **INT8 quantisation**: Reduce model size for edge deployment

See also:
- [Training Guide](/docs/guides/training)
- [Vulkan GPU Guide](/docs/guides/vulkan)
- [Edge Deployment](/docs/guides/edge-deployment)
