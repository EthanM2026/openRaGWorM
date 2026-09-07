#ifndef ADAM_H_INCLUDED
#define ADAM_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mdn.h"

typedef struct {
    float lr;         // Learning rate (e.g., 0.001f or 0.0003f)
    float beta1;      // Exponential decay rate for 1st moment (default: 0.9f)
    float beta2;      // Exponential decay rate for 2nd moment (default: 0.999f)
    float epsilon;    // Term for numerical stability (default: 1e-8f)
    int t;            // Timestep counter

    // 1st Moment Vectors (m)
    float *m_W_x, *m_W_h, *m_b_h;
    float *m_W_alpha, *m_b_alpha;
    float *m_W_mu, *m_b_mu;
    float *m_W_sigma, *m_b_sigma;

    // 2nd Moment Vectors (v)
    float *v_W_x, *v_W_h, *v_b_h;
    float *v_W_alpha, *v_b_alpha;
    float *v_W_mu, *v_b_mu;
    float *v_W_sigma, *v_b_sigma;
} AdamOptimizer;

// Helper to allocate zero-initialized float memory
static float* allocate_zero_array(size_t size);

AdamOptimizer* adam_create(const MDNRNN *net, float lr, float beta1, float beta2, float epsilon);

void adam_free(AdamOptimizer *opt);

// Vectorized Adam parameter update for a single array
static inline void adam_update_tensor(float *param, const float *grad, float *m, float *v,
                                      int size, float alpha_t, float beta1, float beta2, float eps);

// Perform Adam step over all MDN-RNN parameters
void adam_step(AdamOptimizer *opt, MDNRNN *net);

#endif // ADAM_H_INCLUDED
