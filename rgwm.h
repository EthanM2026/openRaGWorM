#ifndef RGWM_H_INCLUDED
#define RGWM_H_INCLUDED

#define WIDTH 96
#define HEIGHT 54

#define LATENT_VECTOR_SIZE 16

#define CONTROLLER_OUTPUT_SIZE 4

#define CURRENT_STATE_TRAIN_AUTOENCODER 1

#define CURRENT_STATE_TRAIN_MDNRNN 2

#define CURRENT_STATE_TRAIN_CONTROLLER 3

#define CURRENT_STATE_TRAIN_FROM_DREAM 4

#include "mdn.h"

#include "vae.h"
#include "linear.h"
#include "controller.h"
#include "adam.h"


// Diagonal CMA-ES Optimizer
typedef struct {
    int N;            // Parameter dimension
    int lambda;       // Population size
    int mu;           // Number of elite candidates used for recombination
    float *weights;   // [mu] Normalized recombination weights

    float *mean;      // [N] Mean parameter vector
    float *sigma;     // [N] Individual parameter standard deviations
    float **pop;      // [lambda x N] Sampled population
    float *fitness;   // [lambda] Reward scores for each candidate

    float ccov;       // Covariance learning rate
} SepCMAES;

// Helper: Comparator for sorting candidate indices by fitness descending (higher reward is better)
typedef struct {
    int index;
    float fitness;
} Candidate;



//

struct _Recurrent_Generative_World_Model
{
    int Train_Or_Run_Mode;

    VAE* vae;
    MDNRNN* net;
    LinearLayer* LLnet;

    const char* model_path = "vae_weights.bin";
    const char* folder_path = "./dataset";
    int in_w = WIDTH, in_h = HEIGHT;
    int out_w = WIDTH, out_h = HEIGHT;
    int latent_dim = 16;
    int epochs = 200;
    float learning_rate = 0.05f; // Ideal learning rate for Adam

    int in_dim = in_w * in_h * 3;
    int out_dim = out_w * out_h * 3;

    struct dirent* entry;
    int max_images = 500;
    float** dataset;
    int img_count = 0;

    float** target_dataset;
    float* eps;

    int adam_step;
    int epoch;

    int Current_State;

    float* mu;

    float** X;
    float** Y;


    float** recorded_z;
    float** recorded_a;

    float** extracted_h;

    int RNNinput_dim;
    int RNNhidden_dim;
    int RNNoutput_dim;
    int RNNnum_mixtures;
    int RNNseq_len;
    float RNNlr;
    float RNNtemperature;

    const char* RNNmodel_path;

    float *predicted_h;
    float *sample_output;

    Controller controller;

    AdamOptimizer *optimizer;
};

struct _Recurrent_Generative_World_Model* Create_RGWM();
void Initialize_RGWM(struct _Recurrent_Generative_World_Model* Model);

void Initialize_VAE(struct _Recurrent_Generative_World_Model* Model);
void Initialize_RNN(struct _Recurrent_Generative_World_Model* Model);
void Initialize_Controller(struct _Recurrent_Generative_World_Model* Model);

void Train_RGWM(struct _Recurrent_Generative_World_Model* Model);

void Train_VAE(struct _Recurrent_Generative_World_Model* Model);
void Train_RNN(struct _Recurrent_Generative_World_Model* Model);
void Train_Controller(struct _Recurrent_Generative_World_Model* Model);

void RNN_Epoch(struct _Recurrent_Generative_World_Model* Model);

void VAE_Epoch(struct _Recurrent_Generative_World_Model* Model,int img_count,float** dataset,float** target_dataset,float* eps,float learning_rate,int adam_step,int epoch,int epochs,int out_w,int out_h);

float* allocate_array(size_t size);

#endif // RGWM_H_INCLUDED
