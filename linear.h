#ifndef LINEAR_H_INCLUDED
#define LINEAR_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int num_inputs;
    int num_outputs;
    float *weights; // Size: num_inputs * num_outputs
    float *biases;  // Size: num_outputs
} LinearLayer;

// Create layer with dynamic allocation
LinearLayer* create_layer(int num_inputs, int num_outputs);

void free_layer(LinearLayer *layer);

// Forward pass: y_j = sum_i(x_i * w_ij) + b_j
void forward(const LinearLayer *layer, const float *inputs, float *outputs);

// Single training iteration (SGD)
float train_step(LinearLayer *layer, const float *inputs, const float *targets, float learning_rate);

// Save model configuration, weights, and biases to a binary file
int save_layer(const LinearLayer *layer, const char *filename);

// Load model configuration, weights, and biases from a binary file
LinearLayer* load_layer(const char *filename);

int mainsss();
#endif
