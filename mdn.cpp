#include "mdn.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979323846f
#define EPSILON 1e-7f
#define MIN_LOG_SIGMA -1.0f    // Min sigma ~ 0.0183
#define MAX_LOG_SIGMA 1.0f     // Max sigma ~ 54.598
#define MIN_VARIANCE 1e-3f     // Hard floor for sigma^2 in denominators
#define GRAD_CLIP_VAL 1.0f     // Gradient clipping threshold
#define DIFF_CLAMP_VAL 50.0f   // Clamp target error differences

void pack_mdn_input(const float *z, int z_dim, const float *a, int a_dim, float *out_X) {
    // 1. Copy z_t into indices [0 ... z_dim - 1]
    memcpy(out_X, z, z_dim * sizeof(float));

    // 2. Copy a_t into indices [z_dim ... z_dim + a_dim - 1]
    memcpy(out_X + z_dim, a, a_dim * sizeof(float));
}


// Utility functions
static float rand_uniform(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

static float rand_gaussian(void) {
    float u1 = rand_uniform(EPSILON, 1.0f);
    float u2 = rand_uniform(0.0f, 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
}

static float clamp_val(float val, float min_val, float max_val) {
    if (isnan(val)) return 0.0f;
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

static void clip_gradients(float *arr, int size, float max_val) {
    for (int i = 0; i < size; i++) {
        if (isnan(arr[i])) {
            arr[i] = 0.0f; // Replace NaN gradients with 0
        } else if (arr[i] > max_val) {
            arr[i] = max_val;
        } else if (arr[i] < -max_val) {
            arr[i] = -max_val;
        }
    }
}

float* allocate_array(size_t size) {
    float *ptr = (float*)calloc(size, sizeof(float));
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Model Lifecycle
MDNRNN* mdn_rnn_create(int input_dim, int hidden_dim, int output_dim, int num_gaussians) {
    MDNRNN *net = (MDNRNN*)malloc(sizeof(MDNRNN));
    net->input_dim = input_dim;
    net->hidden_dim = hidden_dim;
    net->output_dim = output_dim;
    net->num_gaussians = num_gaussians;

    int num_out_params = num_gaussians * output_dim;

    net->W_x = allocate_array(hidden_dim * input_dim);
    net->W_h = allocate_array(hidden_dim * hidden_dim);
    net->b_h = allocate_array(hidden_dim);

    net->W_alpha = allocate_array(num_gaussians * hidden_dim);
    net->b_alpha = allocate_array(num_gaussians);

    net->W_mu = allocate_array(num_out_params * hidden_dim);
    net->b_mu = allocate_array(num_out_params);

    net->W_sigma = allocate_array(num_out_params * hidden_dim);
    net->b_sigma = allocate_array(num_out_params);

    net->dW_x = allocate_array(hidden_dim * input_dim);
    net->dW_h = allocate_array(hidden_dim * hidden_dim);
    net->db_h = allocate_array(hidden_dim);

    net->dW_alpha = allocate_array(num_gaussians * hidden_dim);
    net->db_alpha = allocate_array(num_gaussians);

    net->dW_mu = allocate_array(num_out_params * hidden_dim);
    net->db_mu = allocate_array(num_out_params);

    net->dW_sigma = allocate_array(num_out_params * hidden_dim);
    net->db_sigma = allocate_array(num_out_params);

    // Glorot Initialization
    float scale_x = sqrtf(2.0f / (input_dim + hidden_dim));
    for (int i = 0; i < hidden_dim * input_dim; i++) net->W_x[i] = rand_uniform(-scale_x, scale_x);

    float scale_h = sqrtf(2.0f / (hidden_dim + hidden_dim));
    for (int i = 0; i < hidden_dim * hidden_dim; i++) net->W_h[i] = rand_uniform(-scale_h, scale_h);

    float scale_m = sqrtf(2.0f / (hidden_dim + num_gaussians));
    for (int i = 0; i < num_gaussians * hidden_dim; i++) net->W_alpha[i] = rand_uniform(-scale_m, scale_m);

    float scale_o = sqrtf(2.0f / (hidden_dim + num_out_params));
    for (int i = 0; i < num_out_params * hidden_dim; i++) {
        net->W_mu[i] = rand_uniform(-scale_o, scale_o);
        net->W_sigma[i] = rand_uniform(-scale_o, scale_o);
    }

    return net;
}

void mdn_rnn_free(MDNRNN *net) {
    if (!net) return;
    free(net->W_x); free(net->W_h); free(net->b_h);
    free(net->W_alpha); free(net->b_alpha);
    free(net->W_mu); free(net->b_mu);
    free(net->W_sigma); free(net->b_sigma);

    free(net->dW_x); free(net->dW_h); free(net->db_h);
    free(net->dW_alpha); free(net->db_alpha);
    free(net->dW_mu); free(net->db_mu);
    free(net->dW_sigma); free(net->db_sigma);
    free(net);
}

// Forward Propagation
void mdn_rnn_step(const MDNRNN *net, const float *x_t, const float *h_prev, float *h_next) {
    for (int i = 0; i < net->hidden_dim; i++) {
        float sum = net->b_h[i];
        for (int j = 0; j < net->input_dim; j++) {
            sum += net->W_x[i * net->input_dim + j] * x_t[j];
        }
        for (int j = 0; j < net->hidden_dim; j++) {
            sum += net->W_h[i * net->hidden_dim + j] * h_prev[j];
        }
        // NaN Guard & Tanh activation
        h_next[i] = isnan(sum) ? 0.0f : tanhf(sum);
    }
}

void mdn_forward_heads(const MDNRNN *net, const float *h_t, float *alpha_logits, float *mu, float *log_sigma) {
    int K = net->num_gaussians;
    int D = net->output_dim;
    int H = net->hidden_dim;

    for (int k = 0; k < K; k++) {
        float sum = net->b_alpha[k];
        for (int j = 0; j < H; j++) {
            sum += net->W_alpha[k * H + j] * h_t[j];
        }
        alpha_logits[k] = isnan(sum) ? 0.0f : sum;
    }

    for (int i = 0; i < K * D; i++) {
        float sum_m = net->b_mu[i];
        float sum_s = net->b_sigma[i];
        for (int j = 0; j < H; j++) {
            sum_m += net->W_mu[i * H + j] * h_t[j];
            sum_s += net->W_sigma[i * H + j] * h_t[j];
        }
        mu[i] = isnan(sum_m) ? 0.0f : sum_m;

        // Strict Bound [-4.0, 4.0] and NaN Guard
        log_sigma[i] = clamp_val(sum_s, MIN_LOG_SIGMA, MAX_LOG_SIGMA);
    }
}

// MDN Temperature Sampling
void mdn_rnn_sample(const MDNRNN *net, const float *h_t, float temperature, MDNSample *out_sample) {
    int K = net->num_gaussians;
    int D = net->output_dim;

    if (temperature < 0.01f) temperature = 0.01f;

    float *alpha_logits = allocate_array(K);
    float *mu = allocate_array(K * D);
    float *log_sigma = allocate_array(K * D);

    mdn_forward_heads(net, h_t, alpha_logits, mu, log_sigma);

    float *pi = allocate_array(K);
    float max_logit = -1e9f;
    for (int k = 0; k < K; k++) {
        alpha_logits[k] /= temperature;
        if (alpha_logits[k] > max_logit) max_logit = alpha_logits[k];
    }

    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++) {
        pi[k] = expf(alpha_logits[k] - max_logit);
        sum_exp += pi[k];
    }
    for (int k = 0; k < K; k++) pi[k] /= (sum_exp + EPSILON);

    float r = rand_uniform(0.0f, 1.0f);
    int selected_k = 0;
    float cumulative = 0.0f;
    for (int k = 0; k < K; k++) {
        cumulative += pi[k];
        if (r <= cumulative) {
            selected_k = k;
            break;
        }
    }
    if (selected_k >= K) selected_k = K - 1;

    // Assign directly into caller-owned memory
    out_sample->selected_component = selected_k;
    out_sample->sampled_output = allocate_array(D);

    float temp_scale = sqrtf(temperature);
    for (int d = 0; d < D; d++) {
        int idx = selected_k * D + d;
        float sigma = expf(log_sigma[idx]) * temp_scale;
        out_sample->sampled_output[d] = mu[idx] + sigma * rand_gaussian();
    }

    out_sample->reward = out_sample->sampled_output[D - 1];

    free(alpha_logits);
    free(mu);
    free(log_sigma);
    free(pi);
}

// Robust Backpropagation Through Time (BPTT) with Hardened Stability
float mdn_rnn_train_sequence(MDNRNN *net, const float *X, const float *Y, int seq_len, float learning_rate) {
    int I = net->input_dim;
    int H = net->hidden_dim;
    int D = net->output_dim;
    int K = net->num_gaussians;

    float *h = allocate_array((seq_len + 1) * H);
    float *alpha_logits = allocate_array(seq_len * K);
    float *mu = allocate_array(seq_len * K * D);
    float *log_sigma = allocate_array(seq_len * K * D);
    float *pi = allocate_array(seq_len * K);
    float *log_pi = allocate_array(seq_len * K);

    float total_loss = 0.0f;

    // 1. Forward Pass
    for (int t = 0; t < seq_len; t++) {
        const float *x_t = &X[t * I];
        const float *h_prev = &h[t * H];
        float *h_curr = &h[(t + 1) * H];

        mdn_rnn_step(net, x_t, h_prev, h_curr);
        mdn_forward_heads(net, h_curr, &alpha_logits[t * K], &mu[t * K * D], &log_sigma[t * K * D]);

        // Numerically stable Softmax & Log-Softmax
        float max_l = -1e9f;
        for (int k = 0; k < K; k++) {
            if (alpha_logits[t * K + k] > max_l) max_l = alpha_logits[t * K + k];
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            float e = expf(alpha_logits[t * K + k] - max_l);
            pi[t * K + k] = e;
            sum_exp += e;
        }
        float log_sum_exp_alpha = max_l + logf(sum_exp + EPSILON);

        for (int k = 0; k < K; k++) {
            pi[t * K + k] /= (sum_exp + EPSILON);
            log_pi[t * K + k] = alpha_logits[t * K + k] - log_sum_exp_alpha;
        }

        // Log-Sum-Exp computation for Loss with Variance Safety Floor
        const float *y_t = &Y[t * D];
        float log_joint[K];
        float max_log_joint = -1e9f;

        for (int k = 0; k < K; k++) {
            float log_g_prob = 0.0f;
            for (int d = 0; d < D; d++) {
                int idx = t * K * D + k * D + d;
                float l_sig = log_sigma[idx];

                // Variance floor check
                float sig_sq = expf(2.0f * l_sig);
                if (sig_sq < MIN_VARIANCE) sig_sq = MIN_VARIANCE;

                float diff = clamp_val(y_t[d] - mu[idx], -DIFF_CLAMP_VAL, DIFF_CLAMP_VAL);
                log_g_prob += -l_sig - 0.5f * logf(2.0f * PI) - 0.5f * (diff * diff) / sig_sq;
            }
            log_joint[k] = log_pi[t * K + k] + log_g_prob;
            if (isnan(log_joint[k])) log_joint[k] = -1e5f;
            if (log_joint[k] > max_log_joint) max_log_joint = log_joint[k];
        }

        float sum_exp_joint = 0.0f;
        for (int k = 0; k < K; k++) {
            sum_exp_joint += expf(log_joint[k] - max_log_joint);
        }

        // Explicit NaN Fallback Guard for Loss
        float log_likelihood_t = 0.0f;
        if (isnan(sum_exp_joint) || sum_exp_joint <= 0.0f) {
            log_likelihood_t = -100.0f; // Fallback bound
        } else {
            log_likelihood_t = max_log_joint + logf(sum_exp_joint);
        }

        total_loss -= log_likelihood_t;
    }

    // Zero out gradient accumulators
    memset(net->dW_x, 0, sizeof(float) * H * I);
    memset(net->dW_h, 0, sizeof(float) * H * H);
    memset(net->db_h, 0, sizeof(float) * H);
    memset(net->dW_alpha, 0, sizeof(float) * K * H);
    memset(net->db_alpha, 0, sizeof(float) * K);
    memset(net->dW_mu, 0, sizeof(float) * K * D * H);
    memset(net->db_mu, 0, sizeof(float) * K * D);
    memset(net->dW_sigma, 0, sizeof(float) * K * D * H);
    memset(net->db_sigma, 0, sizeof(float) * K * D);

    // 2. Backpropagation Through Time
    float *dh_next = allocate_array(H);

    for (int t = seq_len - 1; t >= 0; t--) {
        const float *y_t = &Y[t * D];
        const float *h_curr = &h[(t + 1) * H];
        const float *h_prev = &h[t * H];
        const float *x_t = &X[t * I];

        // Responsibilities gamma_k in Log-Space with Safety Floors
        float log_joint[K];
        float max_log_joint = -1e9f;

        for (int k = 0; k < K; k++) {
            float log_g_prob = 0.0f;
            for (int d = 0; d < D; d++) {
                int idx = t * K * D + k * D + d;
                float l_sig = log_sigma[idx];

                float sig_sq = expf(2.0f * l_sig);
                if (sig_sq < MIN_VARIANCE) sig_sq = MIN_VARIANCE;

                float diff = clamp_val(y_t[d] - mu[idx], -DIFF_CLAMP_VAL, DIFF_CLAMP_VAL);
                log_g_prob += -l_sig - 0.5f * logf(2.0f * PI) - 0.5f * (diff * diff) / sig_sq;
            }
            log_joint[k] = log_pi[t * K + k] + log_g_prob;
            if (isnan(log_joint[k])) log_joint[k] = -1e5f;
            if (log_joint[k] > max_log_joint) max_log_joint = log_joint[k];
        }

        float gamma[K];
        float sum_exp_gamma = 0.0f;
        for (int k = 0; k < K; k++) {
            gamma[k] = expf(log_joint[k] - max_log_joint);
            sum_exp_gamma += gamma[k];
        }
        for (int k = 0; k < K; k++) {
            gamma[k] /= (sum_exp_gamma + EPSILON);
            if (isnan(gamma[k])) gamma[k] = 1.0f / K; // Uniform fallback
        }

        // Gradients w.r.t pre-activations
        float d_alpha[K];
        float *d_mu = allocate_array(K * D);
        float *d_sigma = allocate_array(K * D);

        for (int k = 0; k < K; k++) {
            d_alpha[k] = pi[t * K + k] - gamma[k];
            for (int d = 0; d < D; d++) {
                int idx = k * D + d;
                int global_idx = t * K * D + idx;
                float l_sig = log_sigma[global_idx];

                float sig_sq = expf(2.0f * l_sig);
                if (sig_sq < MIN_VARIANCE) sig_sq = MIN_VARIANCE;

                float diff = clamp_val(y_t[d] - mu[global_idx], -DIFF_CLAMP_VAL, DIFF_CLAMP_VAL);

                // Floor-guarded gradients
                d_mu[idx] = gamma[k] * (-diff / sig_sq);
                d_sigma[idx] = gamma[k] * (1.0f - (diff * diff) / sig_sq);

                d_mu[idx] = clamp_val(d_mu[idx], -GRAD_CLIP_VAL, GRAD_CLIP_VAL);
                d_sigma[idx] = clamp_val(d_sigma[idx], -GRAD_CLIP_VAL, GRAD_CLIP_VAL);
            }
        }

        // Head Parameter Gradients
        float *dh_mdn = allocate_array(H);

        for (int k = 0; k < K; k++) {
            net->db_alpha[k] += d_alpha[k];
            for (int j = 0; j < H; j++) {
                net->dW_alpha[k * H + j] += d_alpha[k] * h_curr[j];
                dh_mdn[j] += net->W_alpha[k * H + j] * d_alpha[k];
            }
        }

        for (int i = 0; i < K * D; i++) {
            net->db_mu[i] += d_mu[i];
            net->db_sigma[i] += d_sigma[i];
            for (int j = 0; j < H; j++) {
                net->dW_mu[i * H + j] += d_mu[i] * h_curr[j];
                net->dW_sigma[i * H + j] += d_sigma[i] * h_curr[j];
                dh_mdn[j] += net->W_mu[i * H + j] * d_mu[i] + net->W_sigma[i * H + j] * d_sigma[i];
            }
        }

        // Recurrent Cell Gradient Backprop with Saturating Tanh Protection
        float *dh_total = allocate_array(H);
        for (int j = 0; j < H; j++) {
            dh_total[j] = dh_next[j] + dh_mdn[j];
            dh_total[j] = clamp_val(dh_total[j], -GRAD_CLIP_VAL, GRAD_CLIP_VAL);

            // Tanh derivative safety floor
            float dtanh_factor = (1.0f - h_curr[j] * h_curr[j]);
            if (dtanh_factor < 1e-4f) dtanh_factor = 1e-4f;

            float dtanh = dtanh_factor * dh_total[j];

            net->db_h[j] += dtanh;
            for (int k = 0; k < I; k++) net->dW_x[j * I + k] += dtanh * x_t[k];
            for (int k = 0; k < H; k++) {
                net->dW_h[j * H + k] += dtanh * h_prev[k];
                dh_next[k] = net->W_h[j * H + k] * dtanh;
            }
        }

        free(d_mu); free(d_sigma); free(dh_mdn); free(dh_total);
    }

    // Global Gradient Clipping & NaN Erasure
    clip_gradients(net->dW_x, H * I, GRAD_CLIP_VAL);
    clip_gradients(net->dW_h, H * H, GRAD_CLIP_VAL);
    clip_gradients(net->db_h, H, GRAD_CLIP_VAL);

    clip_gradients(net->dW_alpha, K * H, GRAD_CLIP_VAL);
    clip_gradients(net->db_alpha, K, GRAD_CLIP_VAL);

    int num_out = K * D;
    clip_gradients(net->dW_mu, num_out * H, GRAD_CLIP_VAL);
    clip_gradients(net->db_mu, num_out, GRAD_CLIP_VAL);

    clip_gradients(net->dW_sigma, num_out * H, GRAD_CLIP_VAL);
    clip_gradients(net->db_sigma, num_out, GRAD_CLIP_VAL);

    // Apply Gradient Descent Updates
    for (int i = 0; i < H * I; i++) net->W_x[i] -= learning_rate * net->dW_x[i];
    for (int i = 0; i < H * H; i++) net->W_h[i] -= learning_rate * net->dW_h[i];
    for (int i = 0; i < H; i++) net->b_h[i] -= learning_rate * net->db_h[i];

    for (int i = 0; i < K * H; i++) net->W_alpha[i] -= learning_rate * net->dW_alpha[i];
    for (int i = 0; i < K; i++) net->b_alpha[i] -= learning_rate * net->db_alpha[i];

    for (int i = 0; i < num_out * H; i++) {
        net->W_mu[i] -= learning_rate * net->dW_mu[i];
        net->W_sigma[i] -= learning_rate * net->dW_sigma[i];
    }
    for (int i = 0; i < num_out; i++) {
        net->b_mu[i] -= learning_rate * net->db_mu[i];
        net->b_sigma[i] -= learning_rate * net->db_sigma[i];
    }

    free(h); free(alpha_logits); free(mu); free(log_sigma); free(pi); free(log_pi); free(dh_next);

    float avg_loss = total_loss / seq_len;
    return isnan(avg_loss) ? 100.0f : avg_loss;
}

// Serialization
int mdn_rnn_save(const MDNRNN *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return 0;

    fwrite(&net->input_dim, sizeof(int), 1, f);
    fwrite(&net->hidden_dim, sizeof(int), 1, f);
    fwrite(&net->output_dim, sizeof(int), 1, f);
    fwrite(&net->num_gaussians, sizeof(int), 1, f);

    int H = net->hidden_dim, I = net->input_dim, K = net->num_gaussians, D = net->output_dim;

    fwrite(net->W_x, sizeof(float), H * I, f);
    fwrite(net->W_h, sizeof(float), H * H, f);
    fwrite(net->b_h, sizeof(float), H, f);

    fwrite(net->W_alpha, sizeof(float), K * H, f);
    fwrite(net->b_alpha, sizeof(float), K, f);

    fwrite(net->W_mu, sizeof(float), K * D * H, f);
    fwrite(net->b_mu, sizeof(float), K * D, f);

    fwrite(net->W_sigma, sizeof(float), K * D * H, f);
    fwrite(net->b_sigma, sizeof(float), K * D, f);

    fclose(f);
    return 1;
}

MDNRNN* mdn_rnn_load(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;

    int I, H, D, K;
    fread(&I, sizeof(int), 1, f);
    fread(&H, sizeof(int), 1, f);
    fread(&D, sizeof(int), 1, f);
    fread(&K, sizeof(int), 1, f);

    MDNRNN *net = mdn_rnn_create(I, H, D, K);

    fread(net->W_x, sizeof(float), H * I, f);
    fread(net->W_h, sizeof(float), H * H, f);
    fread(net->b_h, sizeof(float), H, f);

    fread(net->W_alpha, sizeof(float), K * H, f);
    fread(net->b_alpha, sizeof(float), K, f);

    fread(net->W_mu, sizeof(float), K * D * H, f);
    fread(net->b_mu, sizeof(float), K * D, f);

    fread(net->W_sigma, sizeof(float), K * D * H, f);
    fread(net->b_sigma, sizeof(float), K * D, f);

    fclose(f);
    return net;
}


void controller_get_action(const Controller *ctrl, const float *z_t, int z_dim, const float *h_t, int h_dim, float *action) {
    for (int a = 0; a < ctrl->action_dim; a++) {
       float sum = 0.0f;
        for (int i = 0; i < z_dim; i++) sum += ctrl->weights[a * (z_dim + h_dim) + i] * z_t[i];
        for (int j = 0; j < h_dim; j++) sum += ctrl->weights[a * (z_dim + h_dim) + z_dim + j] * h_t[j];
        action[a] = tanhf(sum);
    }
}
