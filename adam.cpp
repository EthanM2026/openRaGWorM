#include "adam.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper to allocate zero-initialized float memory
static float* allocate_zero_array(size_t size) {
    float *ptr = (float*)calloc(size, sizeof(float));
    if (!ptr) {
        fprintf(stderr, "Allocation failed in Adam Optimizer.\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

AdamOptimizer* adam_create(const MDNRNN *net, float lr, float beta1, float beta2, float epsilon) {
    AdamOptimizer *opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->epsilon = epsilon;
    opt->t = 0;

    int H = net->hidden_dim;
    int I = net->input_dim;
    int K = net->num_gaussians;
    int D = net->output_dim;
    int num_out = K * D;

    // Allocate 1st moment state buffers
    opt->m_W_x = allocate_zero_array(H * I);
    opt->m_W_h = allocate_zero_array(H * H);
    opt->m_b_h = allocate_zero_array(H);

    opt->m_W_alpha = allocate_zero_array(K * H);
    opt->m_b_alpha = allocate_zero_array(K);

    opt->m_W_mu = allocate_zero_array(num_out * H);
    opt->m_b_mu = allocate_zero_array(num_out);

    opt->m_W_sigma = allocate_zero_array(num_out * H);
    opt->m_b_sigma = allocate_zero_array(num_out);

    // Allocate 2nd moment state buffers
    opt->v_W_x = allocate_zero_array(H * I);
    opt->v_W_h = allocate_zero_array(H * H);
    opt->v_b_h = allocate_zero_array(H);

    opt->v_W_alpha = allocate_zero_array(K * H);
    opt->v_b_alpha = allocate_zero_array(K);

    opt->v_W_mu = allocate_zero_array(num_out * H);
    opt->v_b_mu = allocate_zero_array(num_out);

    opt->v_W_sigma = allocate_zero_array(num_out * H);
    opt->v_b_sigma = allocate_zero_array(num_out);

    return opt;
}

void adam_free(AdamOptimizer *opt) {
    if (!opt) return;

    free(opt->m_W_x); free(opt->m_W_h); free(opt->m_b_h);
    free(opt->m_W_alpha); free(opt->m_b_alpha);
    free(opt->m_W_mu); free(opt->m_b_mu);
    free(opt->m_W_sigma); free(opt->m_b_sigma);

    free(opt->v_W_x); free(opt->v_W_h); free(opt->v_b_h);
    free(opt->v_W_alpha); free(opt->v_b_alpha);
    free(opt->v_W_mu); free(opt->v_b_mu);
    free(opt->v_W_sigma); free(opt->v_b_sigma);

    free(opt);
}

// Vectorized Adam parameter update for a single array
static inline void adam_update_tensor(float *param, const float *grad, float *m, float *v,
                                      int size, float alpha_t, float beta1, float beta2, float eps) {
    for (int i = 0; i < size; i++) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);
        param[i] -= alpha_t * m[i] / (sqrtf(v[i]) + eps);
    }
}

// Perform Adam step over all MDN-RNN parameters
void adam_step(AdamOptimizer *opt, MDNRNN *net) {
    opt->t += 1;

    // Precompute bias-corrected step size: alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
    float correction1 = 1.0f - powf(opt->beta1, (float)opt->t);
    float correction2 = 1.0f - powf(opt->beta2, (float)opt->t);
    float alpha_t = opt->lr * sqrtf(correction2) / correction1;

    int H = net->hidden_dim;
    int I = net->input_dim;
    int K = net->num_gaussians;
    int D = net->output_dim;
    int num_out = K * D;

    // RNN parameters
    adam_update_tensor(net->W_x, net->dW_x, opt->m_W_x, opt->v_W_x, H * I, alpha_t, opt->beta1, opt->beta2, opt->epsilon);
    adam_update_tensor(net->W_h, net->dW_h, opt->m_W_h, opt->v_W_h, H * H, alpha_t, opt->beta1, opt->beta2, opt->epsilon);
    adam_update_tensor(net->b_h, net->db_h, opt->m_b_h, opt->v_b_h, H,     alpha_t, opt->beta1, opt->beta2, opt->epsilon);

    // MDN Alpha head
    adam_update_tensor(net->W_alpha, net->dW_alpha, opt->m_W_alpha, opt->v_W_alpha, K * H, alpha_t, opt->beta1, opt->beta2, opt->epsilon);
    adam_update_tensor(net->b_alpha, net->db_alpha, opt->m_b_alpha, opt->v_b_alpha, K,     alpha_t, opt->beta1, opt->beta2, opt->epsilon);

    // MDN Mu head
    adam_update_tensor(net->W_mu, net->dW_mu, opt->m_W_mu, opt->v_W_mu, num_out * H, alpha_t, opt->beta1, opt->beta2, opt->epsilon);
    adam_update_tensor(net->b_mu, net->db_mu, opt->m_b_mu, opt->v_b_mu, num_out,     alpha_t, opt->beta1, opt->beta2, opt->epsilon);

    // MDN Sigma head
    adam_update_tensor(net->W_sigma, net->dW_sigma, opt->m_W_sigma, opt->v_W_sigma, num_out * H, alpha_t, opt->beta1, opt->beta2, opt->epsilon);
    adam_update_tensor(net->b_sigma, net->db_sigma, opt->m_b_sigma, opt->v_b_sigma, num_out,     alpha_t, opt->beta1, opt->beta2, opt->epsilon);
}
