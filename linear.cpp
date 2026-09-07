#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "linear.h"

#include <string.h>
#include <math.h>





// Create layer with dynamic allocation
LinearLayer* create_layer(int num_inputs, int num_outputs) {
    LinearLayer *layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    layer->num_inputs = num_inputs;
    layer->num_outputs = num_outputs;

    layer->weights = (float*)malloc(num_inputs * num_outputs * sizeof(float));
    layer->biases = (float*)malloc(num_outputs * sizeof(float));

    for (int i = 0; i < num_inputs * num_outputs; i++) {
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int j = 0; j < num_outputs; j++) {
        layer->biases[j] = 0.0f;
    }

    return layer;
}

void free_layer(LinearLayer *layer) {
    if (layer) {
        free(layer->weights);
        free(layer->biases);
        free(layer);
    }
}

// Forward pass: y_j = sum_i(x_i * w_ij) + b_j
void forward(const LinearLayer *layer, const float *inputs, float *outputs) {
    for (int j = 0; j < layer->num_outputs; j++) {
        outputs[j] = layer->biases[j];
        for (int i = 0; i < layer->num_inputs; i++) {
            outputs[j] += inputs[i] * layer->weights[i * layer->num_outputs + j];
        }
    }
}

// Single training iteration (SGD)
float train_step(LinearLayer *layer, const float *inputs, const float *targets, float learning_rate) {
    int I = layer->num_inputs;
    int O = layer->num_outputs;

    float outputs[O];
    forward(layer, inputs, outputs);

    float errors[O];
    float total_loss = 0.0f;
    for (int j = 0; j < O; j++) {
        errors[j] = outputs[j] - targets[j];
        total_loss += errors[j] * errors[j];
    }

    for (int j = 0; j < O; j++) {
        float error_j = errors[j];
        for (int i = 0; i < I; i++) {
            layer->weights[i * O + j] -= learning_rate * error_j * inputs[i];
        }
        layer->biases[j] -= learning_rate * error_j;
    }

    return total_loss / O;
}

// Save model configuration, weights, and biases to a binary file
int save_layer(const LinearLayer *layer, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        return 0;
    }

    // 1. Write layer dimensions (metadata header)
    fwrite(&layer->num_inputs, sizeof(int), 1, file);
    fwrite(&layer->num_outputs, sizeof(int), 1, file);

    // 2. Write parameter arrays
    int total_weights = layer->num_inputs * layer->num_outputs;
    fwrite(layer->weights, sizeof(float), total_weights, file);
    fwrite(layer->biases, sizeof(float), layer->num_outputs, file);

    fclose(file);
    return 1;
}

// Load model configuration, weights, and biases from a binary file
LinearLayer* load_layer(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        return NULL;
    }

    int num_inputs, num_outputs;

    // 1. Read layer dimensions
    if (fread(&num_inputs, sizeof(int), 1, file) != 1 ||
        fread(&num_outputs, sizeof(int), 1, file) != 1) {
        fclose(file);
        return NULL;
    }

    // 2. Allocate layer dynamically with saved dimensions
    LinearLayer *layer = create_layer(num_inputs, num_outputs);

    // 3. Read parameter arrays into memory
    int total_weights = num_inputs * num_outputs;
    fread(layer->weights, sizeof(float), total_weights, file);
    fread(layer->biases, sizeof(float), num_outputs, file);

    fclose(file);
    return layer;
}

int mainsss() {
    srand((unsigned int)time(NULL));

    const char *model_path = "linear_model.bin";
    int num_inputs = 3;
    int num_outputs = 2;
    float learning_rate = 0.01f;

    // Train a initial network
    LinearLayer *net = create_layer(num_inputs, num_outputs);
    printf("Training network (%d inputs -> %d outputs)...\n", num_inputs, num_outputs);

    for (int epoch = 0; epoch < 10000; epoch++) {
        float x[3] = {
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f,
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f,
            ((float)rand() / RAND_MAX) * 2.0f - 1.0f
        };
        float target[2] = {
             2.0f * x[0] - 1.0f * x[1] + 0.5f * x[2] + 1.0f,
            -0.5f * x[0] + 3.0f * x[1] + 0.0f * x[2] - 2.0f
        };
        train_step(net, x, target, learning_rate);
    }

    // Test input sample before saving
    float test_input[3] = {1.0f, 2.0f, -1.0f};
    float original_output[2];
    forward(net, test_input, original_output);

    // Save weights to disk
    if (save_layer(net, model_path)) {
        printf("Successfully saved trained weights to '%s'\n", model_path);
    }

    // Free the original network from memory
    free_layer(net);

    // Load a new network from the saved file
    LinearLayer *loaded_net = load_layer(model_path);
    if (!loaded_net) {
        printf("Failed to load model.\n");
        return 1;
    }
    printf("Successfully loaded model from '%s'\n", model_path);

    // Verify prediction on loaded network
    float loaded_output[2];
    forward(loaded_net, test_input, loaded_output);

    printf("\n--- Validation Check ---\n");
    printf("Input:             [%.2f, %.2f, %.2f]\n", test_input[0], test_input[1], test_input[2]);
    printf("Original Model:    [%.4f, %.4f]\n", original_output[0], original_output[1]);
    printf("Loaded Model:      [%.4f, %.4f]\n", loaded_output[0], loaded_output[1]);

    free_layer(loaded_net);
    return 0;
}
