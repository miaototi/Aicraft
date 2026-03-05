---
sidebar_position: 2
title: "Tutorial: Autoencoder"
---

# Tutorial: Building an Autoencoder

Learn to compress and reconstruct images using an autoencoder neural network.

## What is an Autoencoder?

An autoencoder is a neural network that learns to:
1. **Encode** input data into a compressed representation (latent space)
2. **Decode** the compressed representation back to the original

```
Input (784) → Encoder → Latent (32) → Decoder → Output (784)
```

Use cases:
- Dimensionality reduction
- Denoising
- Anomaly detection
- Feature learning

---

## Step 1: Network Architecture

```c
#include "aicraft/aicraft.h"
#include <stdio.h>

// Encoder: 784 → 256 → 64 → 32
// Decoder: 32 → 64 → 256 → 784

AcLayer *encoder[] = {
    ac_dense(784, 256, AC_RELU),
    ac_dense(256, 64,  AC_RELU),
    ac_dense(64,  32,  AC_RELU)   // Latent space
};

AcLayer *decoder[] = {
    ac_dense(32,  64,  AC_RELU),
    ac_dense(64,  256, AC_RELU),
    ac_dense(256, 784, AC_SIGMOID)  // Output in [0, 1]
};
```

---

## Step 2: Forward Pass

```c
AcTensor *encode(AcTensor *x) {
    return ac_forward_seq(encoder, 3, x);
}

AcTensor *decode(AcTensor *z) {
    return ac_forward_seq(decoder, 3, z);
}

AcTensor *autoencoder_forward(AcTensor *x) {
    AcTensor *z = encode(x);           // Compress
    AcTensor *reconstructed = decode(z);  // Reconstruct
    return reconstructed;
}
```

---

## Step 3: Training Loop

We minimise the **reconstruction loss** (MSE between input and output):

```c
#define BATCH_SIZE 64
#define EPOCHS     20
#define LR         0.001f

int main(void) {
    ac_init();

    // Combine all layers for optimizer
    AcLayer *all_layers[6];
    for (int i = 0; i < 3; i++) all_layers[i] = encoder[i];
    for (int i = 0; i < 3; i++) all_layers[3 + i] = decoder[i];

    AcOptimizer *opt = ac_adam(all_layers, 6, LR);

    // Load your image data (normalized to [0, 1])
    float *images = load_images();  // Your data loading function
    int num_samples = 60000;
    int num_batches = num_samples / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0;

        for (int b = 0; b < num_batches; b++) {
            ac_mem_checkpoint();

            // Get batch
            int offset = b * BATCH_SIZE * 784;
            AcTensor *x = ac_tensor_from_data(
                &images[offset],
                (int[]){BATCH_SIZE, 784}, 2
            );

            // Forward: encode then decode
            AcTensor *z = ac_forward_seq(encoder, 3, x);
            AcTensor *x_reconstructed = ac_forward_seq(decoder, 3, z);

            // Reconstruction loss (MSE)
            AcTensor *loss = ac_mse(x_reconstructed, x);
            epoch_loss += ac_scalar(loss);

            // Backward through entire network
            ac_backward(loss);

            // Update all weights
            ac_optimizer_step(opt);

            ac_mem_restore();
        }

        printf("Epoch %2d | Reconstruction Loss: %.6f\n",
               epoch + 1, epoch_loss / num_batches);
    }

    ac_cleanup();
    return 0;
}
```

---

## Step 4: Using the Trained Autoencoder

### Compress Images

```c
// Encode an image to 32-dimensional latent vector
AcTensor *image = load_single_image(idx);  // [1, 784]
AcTensor *latent = ac_forward_seq(encoder, 3, image);  // [1, 32]

// latent->data contains the compressed representation
printf("Compressed to %d floats (%.1fx compression)\n",
       latent->size, 784.0f / 32);
```

### Reconstruct Images

```c
// Decode back to image
AcTensor *reconstructed = ac_forward_seq(decoder, 3, latent);  // [1, 784]

// Compare with original
float mse = 0;
for (int i = 0; i < 784; i++) {
    float diff = image->data[i] - reconstructed->data[i];
    mse += diff * diff;
}
mse /= 784;
printf("Reconstruction MSE: %.6f\n", mse);
```

### Generate New Images

Sample from the latent space:

```c
// Random latent vector
AcTensor *z_random = ac_tensor_rand((int[]){1, 32}, 2);

// Scale to match learned distribution (roughly)
for (int i = 0; i < 32; i++) {
    z_random->data[i] = (z_random->data[i] - 0.5f) * 2.0f;
}

// Decode to generate new image
AcTensor *generated = ac_forward_seq(decoder, 3, z_random);
```

---

## Step 5: Denoising Autoencoder

Train to remove noise from images:

```c
for (int b = 0; b < num_batches; b++) {
    ac_mem_checkpoint();

    // Clean images
    AcTensor *x_clean = get_batch(b);

    // Add noise
    AcTensor *noise = ac_tensor_rand((int[]){BATCH_SIZE, 784}, 2);
    AcTensor *x_noisy = ac_tensor_new((int[]){BATCH_SIZE, 784}, 2);
    for (int i = 0; i < x_noisy->size; i++) {
        x_noisy->data[i] = x_clean->data[i] + 0.3f * noise->data[i];
        // Clamp to [0, 1]
        if (x_noisy->data[i] < 0) x_noisy->data[i] = 0;
        if (x_noisy->data[i] > 1) x_noisy->data[i] = 1;
    }

    // Train: input noisy, target clean
    AcTensor *z = ac_forward_seq(encoder, 3, x_noisy);
    AcTensor *x_reconstructed = ac_forward_seq(decoder, 3, z);

    // Loss compares reconstruction to CLEAN image
    AcTensor *loss = ac_mse(x_reconstructed, x_clean);

    ac_backward(loss);
    ac_optimizer_step(opt);

    ac_mem_restore();
}
```

---

## Visualisation

Save reconstructions as PGM images:

```c
void save_pgm(const char *filename, float *data, int w, int h) {
    FILE *f = fopen(filename, "wb");
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        unsigned char pixel = (unsigned char)(data[i] * 255);
        fwrite(&pixel, 1, 1, f);
    }
    fclose(f);
}

// Save original and reconstruction side by side
save_pgm("original.pgm", image->data, 28, 28);
save_pgm("reconstructed.pgm", reconstructed->data, 28, 28);
```

---

## Tips for Better Results

1. **Deeper networks**: Add more layers for complex data
2. **Batch normalisation**: Stabilises training
3. **Learning rate scheduling**: Start high, reduce over time
4. **Variational Autoencoder (VAE)**: Add KL divergence loss for smoother latent space
5. **Convolutional layers**: Better for images (coming in Aicraft v1.1)

---

## Next Steps

- [Training Guide](/docs/guides/training) — More training techniques
- [Performance Tuning](/docs/guides/performance-tuning) — Speed up training
- [Edge Deployment](/docs/guides/edge-deployment) — Deploy compressed models
