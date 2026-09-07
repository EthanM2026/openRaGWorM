#ifndef VAE_H_INCLUDED
#define VAE_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <dirent.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 1. BMP FILE I/O IMPLEMENTATION
// ============================================================================

#pragma pack(push, 1)
typedef struct {
    unsigned short bfType;
    unsigned int   bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int   bfOffBits;
} BITMAPFILEHEADER;

typedef struct {
    unsigned int   biSize;
    int            biWidth;
    int            biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    int            biXPelsPerMeter;
    int            biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

typedef struct {
    int num_layers;
    int* sizes;
    float** weights;
    float** biases;
    float** weight_grads;
    float** bias_grads;

    // Adam Optimizer State
    float** m_w;
    float** v_w;
    float** m_b;
    float** v_b;

    float** activations;
    float** z_values;
} MLP;

typedef struct {
    MLP* encoder;
    MLP* decoder;
    int latent_dim;

    float* gen_img;
    unsigned char* bmpdata;

} VAE;

float* load_bmp_normalized(const char* filename, int req_w, int req_h);

void write_bmp_normalized(VAE* vae, const char* filename, const float* img, int width, int height);

// ============================================================================
// 2. MATH & ADAM OPTIMIZER IMPLEMENTATION
// ============================================================================

float rand_gaussian(float mean, float std);

float sigmoid(float x);



MLP* create_mlp(const int* sizes, int num_layers);

void free_mlp(MLP* net);

void forward_mlp(MLP* net, const float* input, int is_decoder);

void zero_mlp_grads(MLP* net);

// ADAM OPTIMIZER STEP (With Gradient Clipping)
void apply_adam_grads(MLP* net, float lr, int t);



VAE* create_vae(int in_dim, const int* enc_hiddens, int num_enc_hiddens,
               int latent_dim, const int* dec_hiddens, int num_dec_hiddens, int out_dim);

void free_vae(VAE* vae);

// ============================================================================
// 3. WEIGHT SERIALIZATION
// ============================================================================

static void save_mlp_weights(FILE* f, MLP* net);

static MLP* load_mlp_weights(FILE* f);

int save_vae_weights(const char* filepath, VAE* vae);

VAE* load_vae_weights(const char* filepath);

// ============================================================================
// 4. VAE TRAINING ENGINE (STABLE MSE + ADAM)
// ============================================================================

float train_step_vae(VAE* vae, const float* input_img, const float* target_img, float* eps);

void generate_vae_sample(VAE* vae, const char* out_bmp_filename, int out_w, int out_h);

// ============================================================================
// 5. MAIN EXECUTION
// ============================================================================
void extract_latent_vector(VAE* vae, const float* input_img, float* out_mu, float* out_z);
void decode_latent_vector(VAE* vae, const float* z, const char* out_bmp_filename, int out_w, int out_h);

#endif
