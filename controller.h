#ifndef CONTROLLER_H_INCLUDED
#define CONTROLLER_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LATENT_DIM 32
#define RNN_HIDDEN_DIM 256
// --- Controller Training Parameters ---
#define POPULATION_SIZE 32
#define ROLLOUT_STEPS 50
#define NUM_GENERATIONS 50
#define NOISE_SIGMA 0.05f
#define LEARNING_RATE 0.01f

#include "mdn.h"


// 1. Controller Forward Pass: [z, h] -> action (4 outputs)
void controller_forward(Controller *c, const float *z, const float *h, float *action);

// 2. Evaluate Controller inside the Dream (Single Episode Rollout)
float rollout_dream(
    Controller *c,
    MDNRNN *net,
    int max_steps,
    float temperature
);


void train_controller(Controller *ctrl, const MDNRNN *world_model);

Controller* controller_create(int z_dim, int h_dim, int action_dim);

void controller_free(Controller *c);

// Compute action: a_t = tanh(W * [z_t, h_t])
void controller_forward(const Controller *c, const float *z_t, int z_dim, const float *h_t, int h_dim, float *action_out);

#endif // CONTROLLER_H_INCLUDED
