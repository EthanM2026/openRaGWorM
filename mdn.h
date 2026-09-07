#ifndef MDN_H_INCLUDED
#define MDN_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979323846f
#define EPSILON 1e-7f
#define LOG_SIGMA_MIN -3.0f // Prevents variance collapse (sigma >= 0.0067)
#define LOG_SIGMA_MAX 3.0f  // Prevents variance explosion (sigma <= 20.085)
#define MAX_GRAD_NORM 1.0f  // Gradient clipping threshold

typedef struct {
    int input_dim;        // Size of input (latent z_t + action a_t)
    int hidden_dim;       // Size of RNN hidden state h_t
    int output_dim;       // Size of predicted state (latent z_{t+1} + reward r_{t+1})
    int num_gaussians;    // Number of mixture components K

    // RNN parameters
    float *W_x;           // [hidden_dim x input_dim]
    float *W_h;           // [hidden_dim x hidden_dim]
    float *b_h;           // [hidden_dim]

    // MDN Head parameters
    float *W_alpha;       // [num_gaussians x hidden_dim]
    float *b_alpha;       // [num_gaussians]

    float *W_mu;          // [(num_gaussians * output_dim) x hidden_dim]
    float *b_mu;          // [num_gaussians * output_dim]

    float *W_sigma;       // [(num_gaussians * output_dim) x hidden_dim]
    float *b_sigma;       // [num_gaussians * output_dim]

    // Gradient accumulators
    float *dW_x, *dW_h, *db_h;
    float *dW_alpha, *db_alpha;
    float *dW_mu, *db_mu;
    float *dW_sigma, *db_sigma;
} MDNRNN;

#pragma pack(push, 1)
typedef struct {
    float *sampled_output; // Size: [output_dim] (e.g., predicted z_{t+1} and reward)
    float reward;          // Extracted reward component
    int selected_component;// The Gaussian component index chosen by mixture sampling
} MDNSample;
#pragma pack(pop)

// Utility functions
static float rand_uniform(float min, float max);

static float rand_gaussian(void);

float* allocate_array(size_t size);

// Initialization
MDNRNN* mdn_rnn_create(int input_dim, int hidden_dim, int output_dim, int num_gaussians);

void mdn_rnn_free(MDNRNN *net);

// Single RNN hidden step update: h_t = tanh(W_x * x_t + W_h * h_{t-1} + b_h)
void mdn_rnn_step(const MDNRNN *net, const float *x_t, const float *h_prev, float *h_next);

// Compute MDN outputs given hidden state h_t
void mdn_forward_heads(const MDNRNN *net, const float *h_t, float *alpha_logits, float *mu, float *log_sigma);

// MDN Temperature Sampling Function
void mdn_rnn_sample(const MDNRNN *net, const float *h_t, float temperature, MDNSample *out_sample);

// BPTT Training Step
float mdn_rnn_train_sequence(MDNRNN *net, const float *X, const float *Y, int seq_len, float learning_rate);

// Model Save and Load
int mdn_rnn_save(const MDNRNN *net, const char *filename);

MDNRNN* mdn_rnn_load(const char *filename);

// Simulated Controller Structure for Recurrent World Model Integration
typedef struct {
    int action_dim;
    float *weights; // Linear policy over [z_t, h_t] -> a_t

    float *z_t;
    float *h_t;
    float *a_t;
    float *x_t;
    float *h_next;

    float temperature;
    float total_reward;
} Controller;

void controller_get_action(const Controller *ctrl, const float *z_t, int z_dim, const float *h_t, int h_dim, float *action);

void pack_mdn_input(const float *z, int z_dim, const float *a, int a_dim, float *out_X);

#endif
