#include "rgwm.h"
#include "mdn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979323846f
#define EPSILON 1e-7f

struct _Recurrent_Generative_World_Model* Create_RGWM()
{

}

void Initialize_RGWM(struct _Recurrent_Generative_World_Model* Model)
{
    Model->predicted_h = (float *)calloc(LATENT_VECTOR_SIZE*8, sizeof(float));
    Model->sample_output = (float *)malloc(LATENT_VECTOR_SIZE * sizeof(float));

    // Allocate 2D array for recorded_z: [seq_len x z_dim]
    Model->recorded_z = (float **)malloc(2 * sizeof(float *));
    for (int t = 0; t < 2; t++) {
        Model->recorded_z[t] = (float *)malloc(LATENT_VECTOR_SIZE * sizeof(float));
    }

    // Allocate 2D array for recorded_a: [seq_len x a_dim]
    Model->recorded_a = (float **)calloc(2, sizeof(float *));
    for (int t = 0; t < 2; t++) {
    Model->recorded_a[t] = (float *)calloc(CONTROLLER_OUTPUT_SIZE, sizeof(float));
}

    Model->extracted_h = (float **)malloc((1 + 1) * sizeof(float *));
    for (int t = 0; t <= 1; t++) {
        Model->extracted_h[t] = (float *)malloc(LATENT_VECTOR_SIZE*8 * sizeof(float)); //Num of hidden values
    }

    Model->mu = (float*)malloc(LATENT_VECTOR_SIZE * sizeof(float));
}


void VAE_Epoch(struct _Recurrent_Generative_World_Model* Model,int which_image,float** dataset,float** target_dataset,float* eps,float learning_rate,int adam_step,int epoch,int epochs,int out_w,int out_h)
{
    zero_mlp_grads(Model->vae->encoder);
            zero_mlp_grads(Model->vae->decoder);

            float epoch_loss = 0.0f;
                float loss = train_step_vae(Model->vae, dataset[which_image], target_dataset[which_image], eps);
                epoch_loss += loss;


            // Apply Adam Optimization updates
            apply_adam_grads(Model->vae->encoder, learning_rate, adam_step);
            apply_adam_grads(Model->vae->decoder, learning_rate, adam_step);
            adam_step++;
            printf("Epoch [%3d/%3d] - Average Loss: %.6f\n", epoch, epochs, epoch_loss / 1);
            generate_vae_sample(Model->vae, "generated_sample.bmp", out_w, out_h);
}


void Initialize_VAE(struct _Recurrent_Generative_World_Model* Model)
{
    srand((unsigned int)time(NULL));

    Model->model_path = "vae_weights.bin";
    Model->folder_path = "./dataset";
    Model->in_w = WIDTH;
    Model->in_h = HEIGHT;
    Model->out_w = WIDTH;
    Model->out_h = HEIGHT;
    Model->latent_dim = LATENT_VECTOR_SIZE;
    Model->epochs = 200;
    Model->learning_rate = 0.0005f; // Ideal learning rate for Adam

    Model->in_dim = Model->in_w * Model->in_h * 3;
    Model->out_dim = Model->out_w * Model->out_h * 3;

    FILE* check_file = fopen(Model->model_path, "rb");
    if (check_file && Model->Current_State != CURRENT_STATE_TRAIN_AUTOENCODER) {
        fclose(check_file);
        //printf("Found existing model '%s'. Loading weights...\n", Model->model_path);
        Model->vae = load_vae_weights(Model->model_path);

        DIR* dir = opendir(Model->folder_path);
        if (!dir) {
            //printf("Error: Directory '%s' not found. Please create it and add .bmp images.\n", Model->folder_path);
            return;
        }

        Model->max_images = 500;
        Model->dataset = (float**)malloc(Model->max_images * sizeof(float*));
        Model->img_count = 0;

        //printf("Loading .bmp images from %s...\n", Model->folder_path);
        while ((Model->entry = readdir(dir)) != NULL && Model->img_count < Model->max_images) {
            if (strstr(Model->entry->d_name, ".bmp") || strstr(Model->entry->d_name, ".BMP")) {
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", Model->folder_path, Model->entry->d_name);

                float* img = load_bmp_normalized(filepath, Model->in_w, Model->in_h);
                if (img) {
                    Model->dataset[Model->img_count++] = img;
                    //printf(" Loaded [%d]: %s\n", Model->img_count, Model->entry->d_name);
                }
            }
        }
        closedir(dir);

        if (Model->img_count == 0) {
            //printf("No valid 24-bit .bmp files found. Exiting.\n");
            free(Model->dataset);
            return;
        }
    }

    if (!Model->vae || Model->Current_State == CURRENT_STATE_TRAIN_AUTOENCODER) {
        int enc_hiddens[] = {512,128,32};

        //1024 → 256 → 64 → 16
        int num_enc_hiddens = sizeof(enc_hiddens) / sizeof(enc_hiddens[0]);

        int dec_hiddens[] = {32,128,512};
        int num_dec_hiddens = sizeof(dec_hiddens) / sizeof(dec_hiddens[0]);

        DIR* dir = opendir(Model->folder_path);
        if (!dir) {
            //printf("Error: Directory '%s' not found. Please create it and add .bmp images.\n", Model->folder_path);
            return;
        }

        Model->max_images = 500;
        Model->dataset = (float**)malloc(Model->max_images * sizeof(float*));
        Model->img_count = 0;

        //printf("Loading .bmp images from %s...\n", Model->folder_path);
        while ((Model->entry = readdir(dir)) != NULL && Model->img_count < Model->max_images) {
            if (strstr(Model->entry->d_name, ".bmp") || strstr(Model->entry->d_name, ".BMP")) {
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", Model->folder_path, Model->entry->d_name);

                float* img = load_bmp_normalized(filepath, Model->in_w, Model->in_h);
                if (img) {
                    Model->dataset[Model->img_count++] = img;
                    //printf(" Loaded [%d]: %s\n", Model->img_count, Model->entry->d_name);
                }
            }
        }
        closedir(dir);

        if (Model->img_count == 0) {
            //printf("No valid 24-bit .bmp files found. Exiting.\n");
            free(Model->dataset);
            return;
        }

        Model->vae = create_vae(Model->in_dim, enc_hiddens, num_enc_hiddens, Model->latent_dim, dec_hiddens, num_dec_hiddens, Model->out_dim);

        Model->eps = (float*)malloc(Model->latent_dim * sizeof(float));

        Model->target_dataset = (float**)malloc(Model->img_count * sizeof(float*));
        for (int i = 0; i < Model->img_count; i++) {
            Model->target_dataset[i] = (float*)malloc(Model->out_dim * sizeof(float));
            for (int y = 0; y < Model->out_h; y++) {
                for (int x = 0; x < Model->out_w; x++) {
                    int src_x = (x * Model->in_w) / Model->out_w;
                    int src_y = (y * Model->in_h) / Model->out_h;
                    for (int c = 0; c < 3; c++) {
                        Model->target_dataset[i][(y * Model->out_w + x) * 3 + c] = Model->dataset[i][(src_y * Model->in_w + src_x) * 3 + c];
                    }
                }
            }
        }

        //printf("\nStarting Training with Adam Optimizer...\n");
        Model->adam_step = 1;
       // for (Model->epoch = 1; Model->epoch <= Model->epochs; Model->epoch++) {
          //  VAE_Epoch(Model,Model->img_count,Model->dataset,Model->target_dataset,Model->eps,Model->learning_rate,Model->adam_step,Model->epoch,Model->epochs,Model->out_w,Model->out_h);

       // }

      //  //printf("Saving trained weights to '%s'...\n", Model->model_path);
       // save_vae_weights(Model->model_path, Model->vae);

      //  free(Model->eps);
      //  for (int i = 0; i < Model->img_count; i++) {
       //     free(Model->dataset[i]);
       //     free(Model->target_dataset[i]);
      //  }
       // free(Model->dataset);
       // free(Model->target_dataset);
    }

    //const char* output_bmp = "generated_sample.bmp";
   // //printf("\nGenerating sample image to '%s'...\n", output_bmp);
   // generate_vae_sample(Model->vae, output_bmp, Model->out_w, Model->out_h);

   // free_vae(Model->vae);
}

void RNN_Epoch(struct _Recurrent_Generative_World_Model* Model)
{
    for (int t = 0; t < Model->RNNseq_len; t++) {
        //LATER: pack_mdn_input(Model->recorded_z[t], LATENT_VECTOR_SIZE, *Model->recorded_a, CONTROLLER_OUTPUT_SIZE, Model->X[t]);
        memcpy(Model->Y[t], Model->recorded_z[t + 1], LATENT_VECTOR_SIZE * sizeof(float));
    }
    //printf("Pack");
    //LATER: zero_grad(Model->net);
    float loss = 1;//LATER: train_bptt(Model->net, Model->X, Model->Y, Model->RNNseq_len, Model->RNNtemperature);
    //printf("loss: %f\n", loss);
    //LATER: adam_step(Model->net);
    //save_model(Model->net, "rnn_mdn_weights.bin");
}

static float rand_uniform(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

static float rand_gaussian(void) {
    float u1 = rand_uniform(EPSILON, 1.0f);
    float u2 = rand_uniform(0.0f, 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
}




void Initialize_RNN(struct _Recurrent_Generative_World_Model* Model)
{
    const char *model_path = "mdn_rnn_model.bin";

    int latent_dim = LATENT_VECTOR_SIZE;
    int action_dim = CONTROLLER_OUTPUT_SIZE;
    int hidden_dim = LATENT_VECTOR_SIZE * 1;
    int num_gaussians = CONTROLLER_OUTPUT_SIZE;

    int seq_len = 1;

    int input_dim = LATENT_VECTOR_SIZE + CONTROLLER_OUTPUT_SIZE;
    int output_dim = CONTROLLER_OUTPUT_SIZE * (1+(2*LATENT_VECTOR_SIZE));         // [z_{t+1}, reward_{t+1}]

    Model->X = (float **)malloc(seq_len * sizeof(float *));
    Model->Y = (float **)malloc(seq_len * sizeof(float *));
    for (int t = 0; t < seq_len; t++) {
        Model->X[t] = (float *)malloc(input_dim * sizeof(float));
        Model->Y[t] = (float *)malloc(output_dim * sizeof(float));
    }

    if(Model->Current_State != CURRENT_STATE_TRAIN_MDNRNN)
    {
        Model->net = mdn_rnn_load(model_path);
        Model->optimizer = adam_create(Model->net, 0.0003f, 0.9f, 0.999f, 1e-8f);
        //printf("\nModel successfully loaded from '%s'.\n", model_path);
        return;
    }

    else if(Model->Current_State == CURRENT_STATE_TRAIN_MDNRNN)
    {
        //printf("Training new world model");
        Model->net = mdn_rnn_create(input_dim, hidden_dim, output_dim, num_gaussians);
        Model->optimizer = adam_create(Model->net, 0.0003f, 0.9f, 0.999f, 1e-8f);

    }
}

void Initialize_Controller(struct _Recurrent_Generative_World_Model* Model)
{
    Model->controller.action_dim = CONTROLLER_OUTPUT_SIZE;
    Model->controller.weights = allocate_array(CONTROLLER_OUTPUT_SIZE * (LATENT_VECTOR_SIZE + LATENT_VECTOR_SIZE * 1));
    for (int i = 0; i < CONTROLLER_OUTPUT_SIZE * (LATENT_VECTOR_SIZE + LATENT_VECTOR_SIZE * 1); i++) {
        Model->controller.weights[i] = rand_uniform(-0.5f, 0.5f);
    }

    Model->controller.z_t = allocate_array(LATENT_VECTOR_SIZE);
    Model->controller.h_t = allocate_array(LATENT_VECTOR_SIZE * 1);
    Model->controller.a_t = allocate_array(CONTROLLER_OUTPUT_SIZE);
    Model->controller.x_t = allocate_array(LATENT_VECTOR_SIZE + CONTROLLER_OUTPUT_SIZE);
    Model->controller.h_next = allocate_array(LATENT_VECTOR_SIZE * 1);

    Model->controller.temperature = 0.8f;
    Model->controller.total_reward = 0.0f;
}


void Train_RGWM(struct _Recurrent_Generative_World_Model* Model)
{

}


void Train_VAE(struct _Recurrent_Generative_World_Model* Model)
{

}

void Train_RNN(struct _Recurrent_Generative_World_Model* Model)
{

}

void Train_Controller(struct _Recurrent_Generative_World_Model* Model)
{

}

