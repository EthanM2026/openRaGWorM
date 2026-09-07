#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include "vae.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 1. BMP FILE I/O IMPLEMENTATION
// ============================================================================


float* load_bmp_normalized(const char* filename, int req_w, int req_h) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    BITMAPFILEHEADER fileHeader;
    BITMAPINFOHEADER infoHeader;

    if (fread(&fileHeader, sizeof(BITMAPFILEHEADER), 1, f) != 1 ||
        fread(&infoHeader, sizeof(BITMAPINFOHEADER), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    if (fileHeader.bfType != 0x4D42 || infoHeader.biBitCount != 24 || infoHeader.biCompression != 0) {
        fclose(f);
        return NULL;
    }

    int src_w = infoHeader.biWidth;
    int src_h = abs(infoHeader.biHeight);
    int row_padded = (src_w * 3 + 3) & (~3);

    fseek(f, fileHeader.bfOffBits, SEEK_SET);
    unsigned char* raw_data = (unsigned char*)malloc(row_padded * src_h);
    if (fread(raw_data, 1, row_padded * src_h, f) != (size_t)(row_padded * src_h)) {
        free(raw_data);
        fclose(f);
        return NULL;
    }
    fclose(f);

    float* img = (float*)malloc(req_w * req_h * 3 * sizeof(float));

    for (int y = 0; y < req_h; y++) {
        int src_y = (y * src_h) / req_h;
        if (infoHeader.biHeight > 0) src_y = src_h - 1 - src_y;

        for (int x = 0; x < req_w; x++) {
            int src_x = (x * src_w) / req_w;
            int src_idx = src_y * row_padded + src_x * 3;
            int dest_idx = (y * req_w + x) * 3;

            img[dest_idx + 0] = raw_data[src_idx + 2] / 255.0f; // Red
            img[dest_idx + 1] = raw_data[src_idx + 1] / 255.0f; // Green
            img[dest_idx + 2] = raw_data[src_idx + 0] / 255.0f; // Blue
        }
    }

    free(raw_data);
    return img;
}

void write_bmp_normalized(VAE* vae, const char* filename, const float* img, int width, int height) {
    unsigned char* data = (unsigned char*)calloc(1, (width * height*3));

 // 1. Find min and max across the current frame
float min_v = 1e9f, max_v = -1e9f;
int total_pixels = width * height * 3;

for (int j = 0; j < total_pixels; j++) {
    if (img[j] < min_v) min_v = img[j];
    if (img[j] > max_v) max_v = img[j];
}

float range = max_v - min_v;
if (range < 1e-5f) range = 1.0f; // Guard against divide-by-zero

// 2. Scale min_v -> 0 and max_v -> 255
for (int j = 0; j < total_pixels; j++) {
    float normalized = (img[j] - min_v) / range;
    data[j] = (unsigned char)fminf(fmaxf(normalized * 255.0f, 0.0f), 255.0f);
}

    vae->bmpdata = data;

   // BITMAPFILEHEADER fh;
  //  fh.bfType = 0x4D42; // "BM"
  //  fh.bfSize = 54 + (width * height);
  //  fh.bfReserved1 = 0;
  //  fh.bfReserved2 = 0;
   // fh.bfOffBits = 54;

 //   BITMAPINFOHEADER ih;
 //   ih.biSize = 40;
 //   ih.biWidth = width;
 //   ih.biHeight = height; // Positive means bottom-up row order
  //  ih.biPlanes = 1;
  //  ih.biBitCount = 24;
  //  ih.biCompression = 0;
 //   ih.biSizeImage = (width * height);
 //   ih.biXPelsPerMeter = 2835;
 //   ih.biYPelsPerMeter = 2835;
 //   ih.biClrUsed = 0;
  //  ih.biClrImportant = 0;

 //   FILE* f = fopen(filename, "wb");
 //   if (f) {
  //      fwrite(&fh, sizeof(fh), 1, f);
  //      fwrite(&ih, sizeof(ih), 1, f);
  //      fwrite(data, 1, row_padded * height, f);
   //     fclose(f);
   // }
    free(data);
}

// ============================================================================
// 2. MATH & ADAM OPTIMIZER IMPLEMENTATION
// ============================================================================

float rand_gaussian(float mean, float std) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 <= 1e-7f) u1 = 1e-7f;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return mean + z * std;
}

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }



MLP* create_mlp(const int* sizes, int num_layers) {
    MLP* net = (MLP*)malloc(sizeof(MLP));
    net->num_layers = num_layers;
    net->sizes = (int*)malloc(num_layers * sizeof(int));
    memcpy(net->sizes, sizes, num_layers * sizeof(int));

    net->weights = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->biases = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->weight_grads = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->bias_grads = (float**)malloc((num_layers - 1) * sizeof(float*));

    net->m_w = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->v_w = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->m_b = (float**)malloc((num_layers - 1) * sizeof(float*));
    net->v_b = (float**)malloc((num_layers - 1) * sizeof(float*));

    net->activations = (float**)malloc(num_layers * sizeof(float*));
    net->z_values = (float**)malloc(num_layers * sizeof(float*));

    for (int i = 0; i < num_layers; i++) {
        net->activations[i] = (float*)calloc(sizes[i], sizeof(float));
        net->z_values[i] = (float*)calloc(sizes[i], sizeof(float));
    }

    for (int l = 0; l < num_layers - 1; l++) {
        int in = sizes[l];
        int out = sizes[l + 1];
        net->weights[l] = (float*)malloc(in * out * sizeof(float));
        net->biases[l] = (float*)calloc(out, sizeof(float));
        net->weight_grads[l] = (float*)calloc(in * out, sizeof(float));
        net->bias_grads[l] = (float*)calloc(out, sizeof(float));

        net->m_w[l] = (float*)calloc(in * out, sizeof(float));
        net->v_w[l] = (float*)calloc(in * out, sizeof(float));
        net->m_b[l] = (float*)calloc(out, sizeof(float));
        net->v_b[l] = (float*)calloc(out, sizeof(float));

        // Xavier / Glorot Initialization
        float scale = sqrtf(2.0f / (in + out));
        for (int i = 0; i < in * out; i++) {
            net->weights[l][i] = rand_gaussian(0.0f, scale);
        }
    }
    return net;
}

void free_mlp(MLP* net) {
    if (!net) return;
    for (int i = 0; i < net->num_layers; i++) {
        free(net->activations[i]);
        free(net->z_values[i]);
    }
    for (int l = 0; l < net->num_layers - 1; l++) {
        free(net->weights[l]); free(net->biases[l]);
        free(net->weight_grads[l]); free(net->bias_grads[l]);
        free(net->m_w[l]); free(net->v_w[l]);
        free(net->m_b[l]); free(net->v_b[l]);
    }
    free(net->weights); free(net->biases);
    free(net->weight_grads); free(net->bias_grads);
    free(net->m_w); free(net->v_w); free(net->m_b); free(net->v_b);
    free(net->activations); free(net->z_values);
    free(net->sizes); free(net);
}

void forward_mlp(MLP* net, const float* input, int is_decoder) {
    memcpy(net->activations[0], input, net->sizes[0] * sizeof(float));

    for (int l = 0; l < net->num_layers - 1; l++) {
        int in_dim = net->sizes[l];
        int out_dim = net->sizes[l + 1];

        for (int j = 0; j < out_dim; j++) {
            float sum = net->biases[l][j];
            for (int i = 0; i < in_dim; i++) {
                sum += net->activations[l][i] * net->weights[l][i * out_dim + j];
            }
            net->z_values[l + 1][j] = sum;

            if (l == net->num_layers - 2) {
                // Sigmoid on final decoder layer keeps pixel outputs safely in [0, 1]
                net->activations[l + 1][j] = is_decoder ? sigmoid(sum) : sum;
            } else {
                // LeakyReLU for hidden layers
                net->activations[l + 1][j] = sum > 0.0f ? sum : 0.01f * sum;
            }
        }
    }
}

void zero_mlp_grads(MLP* net) {
    for (int l = 0; l < net->num_layers - 1; l++) {
        memset(net->weight_grads[l], 0, net->sizes[l] * net->sizes[l + 1] * sizeof(float));
        memset(net->bias_grads[l], 0, net->sizes[l + 1] * sizeof(float));
    }
}

// ADAM OPTIMIZER STEP (With Gradient Clipping)
void apply_adam_grads(MLP* net, float lr, int t) {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float max_grad = 1.0f; // Clip threshold prevents exploding gradients

    float beta1_t = powf(beta1, (float)t);
    float beta2_t = powf(beta2, (float)t);

    for (int l = 0; l < net->num_layers - 1; l++) {
        int in = net->sizes[l];
        int out = net->sizes[l + 1];

        // Update Weights
        for (int i = 0; i < in * out; i++) {
            float g = net->weight_grads[l][i];
            if (g > max_grad) g = max_grad;
            if (g < -max_grad) g = -max_grad;

            net->m_w[l][i] = beta1 * net->m_w[l][i] + (1.0f - beta1) * g;
            net->v_w[l][i] = beta2 * net->v_w[l][i] + (1.0f - beta2) * (g * g);

            float m_hat = net->m_w[l][i] / (1.0f - beta1_t);
            float v_hat = net->v_w[l][i] / (1.0f - beta2_t);

            net->weights[l][i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }

        // Update Biases
        for (int j = 0; j < out; j++) {
            float g = net->bias_grads[l][j];
            if (g > max_grad) g = max_grad;
            if (g < -max_grad) g = -max_grad;

            net->m_b[l][j] = beta1 * net->m_b[l][j] + (1.0f - beta1) * g;
            net->v_b[l][j] = beta2 * net->v_b[l][j] + (1.0f - beta2) * (g * g);

            float m_hat = net->m_b[l][j] / (1.0f - beta1_t);
            float v_hat = net->v_b[l][j] / (1.0f - beta2_t);

            net->biases[l][j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}


VAE* create_vae(int in_dim, const int* enc_hiddens, int num_enc_hiddens,
               int latent_dim, const int* dec_hiddens, int num_dec_hiddens, int out_dim) {
    VAE* vae = (VAE*)malloc(sizeof(VAE));
    vae->latent_dim = latent_dim;

    int* enc_sizes = (int*)malloc((num_enc_hiddens + 2) * sizeof(int));
    enc_sizes[0] = in_dim;
    for (int i = 0; i < num_enc_hiddens; i++) enc_sizes[i + 1] = enc_hiddens[i];
    enc_sizes[num_enc_hiddens + 1] = latent_dim * 2;
    vae->encoder = create_mlp(enc_sizes, num_enc_hiddens + 2);
    free(enc_sizes);

    int* dec_sizes = (int*)malloc((num_dec_hiddens + 2) * sizeof(int));
    dec_sizes[0] = latent_dim;
    for (int i = 0; i < num_dec_hiddens; i++) dec_sizes[i + 1] = dec_hiddens[i];
    dec_sizes[num_dec_hiddens + 1] = out_dim;
    vae->decoder = create_mlp(dec_sizes, num_dec_hiddens + 2);
    free(dec_sizes);

    return vae;
}

void free_vae(VAE* vae) {
    if (!vae) return;
    free_mlp(vae->encoder);
    free_mlp(vae->decoder);
    free(vae);
}

// ============================================================================
// 3. WEIGHT SERIALIZATION
// ============================================================================

static void save_mlp_weights(FILE* f, MLP* net) {
    fwrite(&net->num_layers, sizeof(int), 1, f);
    fwrite(net->sizes, sizeof(int), net->num_layers, f);

    for (int l = 0; l < net->num_layers - 1; l++) {
        int w_count = net->sizes[l] * net->sizes[l + 1];
        int b_count = net->sizes[l + 1];
        fwrite(net->weights[l], sizeof(float), w_count, f);
        fwrite(net->biases[l], sizeof(float), b_count, f);
    }
}

static MLP* load_mlp_weights(FILE* f) {
    int num_layers;
    if (fread(&num_layers, sizeof(int), 1, f) != 1) return NULL;

    int* sizes = (int*)malloc(num_layers * sizeof(int));
    if (fread(sizes, sizeof(int), num_layers, f) != (size_t)num_layers) {
        free(sizes);
        return NULL;
    }

    MLP* net = create_mlp(sizes, num_layers);
    free(sizes);

    for (int l = 0; l < net->num_layers - 1; l++) {
        int w_count = net->sizes[l] * net->sizes[l + 1];
        int b_count = net->sizes[l + 1];
        if (fread(net->weights[l], sizeof(float), w_count, f) != (size_t)w_count ||
            fread(net->biases[l], sizeof(float), b_count, f) != (size_t)b_count) {
            free_mlp(net);
            return NULL;
        }
    }
    return net;
}

int save_vae_weights(const char* filepath, VAE* vae) {
    FILE* f = fopen(filepath, "wb");
    if (!f) return 0;

    fwrite(&vae->latent_dim, sizeof(int), 1, f);
    save_mlp_weights(f, vae->encoder);
    save_mlp_weights(f, vae->decoder);

    fclose(f);
    return 1;
}

VAE* load_vae_weights(const char* filepath) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    VAE* vae = (VAE*)malloc(sizeof(VAE));
    if (fread(&vae->latent_dim, sizeof(int), 1, f) != 1) {
        fclose(f);
        free(vae);
        return NULL;
    }

    vae->encoder = load_mlp_weights(f);
    vae->decoder = load_mlp_weights(f);

    fclose(f);

    if (!vae->encoder || !vae->decoder) {
        free_vae(vae);
        return NULL;
    }
    return vae;
}

// ============================================================================
// 4. VAE TRAINING ENGINE (STABLE MSE + ADAM)
// ============================================================================

float train_step_vae(VAE* vae, const float* input_img, const float* target_img, float* eps) {
    int in_dim = vae->encoder->sizes[0];
    int out_dim = vae->decoder->sizes[vae->decoder->num_layers - 1];
    int latent_dim = vae->latent_dim;

    // 1. Encoder Forward
    forward_mlp(vae->encoder, input_img, 0);

    float* enc_out = vae->encoder->activations[vae->encoder->num_layers - 1];
    float* mu = enc_out;
    float* log_var = enc_out + latent_dim;

    // 2. Reparameterization Trick with Hard Clamp on log_var
    float* z = (float*)malloc(latent_dim * sizeof(float));
    float* std_dev = (float*)malloc(latent_dim * sizeof(float));
    for (int i = 0; i < latent_dim; i++) {
        float clipped_log_var = fminf(fmaxf(log_var[i], -8.0f), 8.0f);
        eps[i] = rand_gaussian(0.0f, 1.0f);
        std_dev[i] = expf(0.5f * clipped_log_var);
        z[i] = mu[i] + std_dev[i] * eps[i];
    }

    // 3. Decoder Forward
    forward_mlp(vae->decoder, z, 1);
    float* recon = vae->decoder->activations[vae->decoder->num_layers - 1];

    // 4. Normalized MSE Loss
    float recon_loss = 0.0f;
    for (int i = 0; i < out_dim; i++) {
        float diff = recon[i] - target_img[i];
        recon_loss += diff * diff;
    }
    recon_loss /= (float)out_dim;

    // KL Loss
    float kl_loss = 0.0f;
    for (int i = 0; i < latent_dim; i++) {
        float clipped_log_var = fminf(fmaxf(log_var[i], -8.0f), 8.0f);
        kl_loss += -0.5f * (1.0f + clipped_log_var - mu[i] * mu[i] - expf(clipped_log_var));
    }
    kl_loss /= (float)out_dim;

    float beta = 0.0001f; // Scaled KL weight
    float total_loss = recon_loss + beta * kl_loss;

    // 5. Backpropagate Decoder (MSE Loss + Sigmoid Output)
    float** d_dec = (float**)malloc(vae->decoder->num_layers * sizeof(float*));
    for (int l = 0; l < vae->decoder->num_layers; l++) {
        d_dec[l] = (float*)calloc(vae->decoder->sizes[l], sizeof(float));
    }

    for (int i = 0; i < out_dim; i++) {
        float diff = recon[i] - target_img[i];
        // Sigmoid derivative is safely combined with MSE derivative
        float sig_grad = fmaxf(recon[i] * (1.0f - recon[i]), 1e-4f);
        d_dec[vae->decoder->num_layers - 1][i] = (2.0f * diff / (float)out_dim) * sig_grad;
    }

    for (int l = vae->decoder->num_layers - 2; l >= 0; l--) {
        int in = vae->decoder->sizes[l];
        int out = vae->decoder->sizes[l + 1];

        for (int j = 0; j < out; j++) {
            float delta = d_dec[l + 1][j];
            vae->decoder->bias_grads[l][j] += delta;
            for (int i = 0; i < in; i++) {
                vae->decoder->weight_grads[l][i * out + j] += delta * vae->decoder->activations[l][i];
                d_dec[l][i] += delta * vae->decoder->weights[l][i * out + j];
            }
        }

        if (l > 0) {
            for (int i = 0; i < in; i++) {
                d_dec[l][i] *= (vae->decoder->z_values[l][i] > 0.0f ? 1.0f : 0.01f);
            }
        }
    }

    float* d_z = d_dec[0];

    // 6. Backpropagate Encoder
    float** d_enc = (float**)malloc(vae->encoder->num_layers * sizeof(float*));
    for (int l = 0; l < vae->encoder->num_layers; l++) {
        d_enc[l] = (float*)calloc(vae->encoder->sizes[l], sizeof(float));
    }

    float* d_enc_last = d_enc[vae->encoder->num_layers - 1];
    for (int i = 0; i < latent_dim; i++) {
        float clipped_log_var = fminf(fmaxf(log_var[i], -8.0f), 8.0f);
        d_enc_last[i] = d_z[i] + beta * (mu[i] / (float)out_dim);
        d_enc_last[latent_dim + i] = d_z[i] * 0.5f * std_dev[i] * eps[i] +
                                     beta * (0.5f * (expf(clipped_log_var) - 1.0f) / (float)out_dim);
    }

    for (int l = vae->encoder->num_layers - 2; l >= 0; l--) {
        int in = vae->encoder->sizes[l];
        int out = vae->encoder->sizes[l + 1];

        for (int j = 0; j < out; j++) {
            float delta = d_enc[l + 1][j];
            vae->encoder->bias_grads[l][j] += delta;
            for (int i = 0; i < in; i++) {
                vae->encoder->weight_grads[l][i * out + j] += delta * vae->encoder->activations[l][i];
                d_enc[l][i] += delta * vae->encoder->weights[l][i * out + j];
            }
        }

        if (l > 0) {
            for (int i = 0; i < in; i++) {
                d_enc[l][i] *= (vae->encoder->z_values[l][i] > 0.0f ? 1.0f : 0.01f);
            }
        }
    }

    for (int l = 0; l < vae->decoder->num_layers; l++) free(d_dec[l]);
    for (int l = 0; l < vae->encoder->num_layers; l++) free(d_enc[l]);
    free(d_dec); free(d_enc); free(z); free(std_dev);

    return total_loss;
}

void generate_vae_sample(VAE* vae, const char* out_bmp_filename, int out_w, int out_h) {
    float* z = (float*)malloc(vae->latent_dim * sizeof(float));
    for (int i = 0; i < vae->latent_dim; i++) {
        z[i] = rand_gaussian(0.0f, 1.0f);
    }

    forward_mlp(vae->decoder, z, 1);
    vae->gen_img = vae->decoder->activations[vae->decoder->num_layers - 1];

    write_bmp_normalized(vae, out_bmp_filename, vae->gen_img, out_w, out_h);
    free(z);
}

// Extract latent representation for a given input image
void extract_latent_vector(VAE* vae, const float* input_img, float* out_mu, float* out_z) {
    int latent_dim = vae->latent_dim;

    // 1. Run Forward Pass on Encoder ONLY
    forward_mlp(vae->encoder, input_img, 0);

    // 2. Read final layer activations
    float* enc_out = vae->encoder->activations[vae->encoder->num_layers - 1];
    float* mu = enc_out;
    float* log_var = enc_out + latent_dim;

    // Copy out the mean vector
    if (out_mu) {
        memcpy(out_mu, mu, latent_dim * sizeof(float));
    }

    // Compute sampled z vector
    if (out_z) {
        for (int i = 0; i < latent_dim; i++) {
            float clipped_log_var = fminf(fmaxf(log_var[i], -8.0f), 8.0f);
            float eps = rand_gaussian(0.0f, 1.0f);
            float std_dev = expf(0.5f * clipped_log_var);
            out_z[i] = mu[i] + std_dev * eps;
        }
    }
}



// Reconstruct or generate an image given a latent vector z
void decode_latent_vector(VAE* vae, const float* z, const char* out_bmp_filename, int out_w, int out_h) {
    // 1. Run Forward Pass on Decoder ONLY
    forward_mlp(vae->decoder, z, 1);

    // 2. Read final layer output image pixels
    float* gen_img = vae->decoder->activations[vae->decoder->num_layers - 1];

    // 3. Save as BMP image
    write_bmp_normalized(vae, out_bmp_filename, gen_img, out_w, out_h);
}
