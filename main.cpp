#include "SDL2/SDL.h"
#include "SDL2/SDL_mixer.h"
#include <GL/glew.h>"
#include "GLFW/glfw3.h"
#include "linear.h"
#include "mdn.h"
#include "vae.h"
#include "rgwm.h"
#include "image.h"
#include "environment.h"
#include "controller.h"

float MyController[4];
float temperature_level = 0.8;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{


}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width,height);
}

int main(int argc, char** argv)
{
    Environment Env;

    struct _Recurrent_Generative_World_Model* Model = (struct _Recurrent_Generative_World_Model*) calloc(1,sizeof(struct _Recurrent_Generative_World_Model));
    Initialize_RGWM(Model);

    glewInit();

    Model->Current_State = CURRENT_STATE_TRAIN_FROM_DREAM;

    SDL_Init( SDL_INIT_VIDEO | SDL_INIT_AUDIO );
    Mix_OpenAudio( 44100, MIX_DEFAULT_FORMAT, 2, 2048 );
    glfwInit();
    GLFWwindow* W;
    W = glfwCreateWindow(96,54, "OpenRaGWorM", NULL, NULL); //SCREEN SIZE
    glfwMakeContextCurrent(W);
    glfwSwapInterval(1); //
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
    glViewport(0, 0, 96,54); //MAGNIFICATION
    glfwSetWindowSizeCallback(W, window_size_callback);
    glfwSetKeyCallback(W, key_callback);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 96,54, 0.0, 1.0, -1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.f,0.f, 0.f, 1.f );
    glClear(GL_COLOR_BUFFER_BIT);

    Initialize_VAE(Model);
    Initialize_RNN(Model);
    Initialize_Controller(Model);

    Env.Number_Of_Frames = Model->img_count;
    //printf("There are %d frames.\n", Env.Number_Of_Frames);

    Env.Current_Frame = -1;
    //printf("Current frame: %d\n", Env.Current_Frame);

    struct _Image* Image = Create_Image();
    Load_Image(Image,"generated_sample2.bmp");

    while (!glfwWindowShouldClose(W))
    {
        Env.Current_Frame += 1;
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
        switch(Model->Current_State)
        {
            case CURRENT_STATE_TRAIN_AUTOENCODER:
                for(int j = 0; j < Model->img_count; j++)
                {
                    VAE_Epoch(Model,j,Model->dataset,Model->target_dataset,Model->eps,Model->learning_rate,Model->adam_step,Model->epoch,Model->epochs,Model->out_w,Model->out_h);
                    Model->epoch += 1;
                    save_vae_weights(Model->model_path, Model->vae);
                    extract_latent_vector(Model->vae, Model->dataset[j], Model->mu, Model->recorded_z[0]);
                    decode_latent_vector(Model->vae, Model->recorded_z[0], "none", 96, 54);
                }
                                    memcpy(Image->RGB_Canvas,Model->vae->bmpdata,96*54*3);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 96, 54, 0, GL_RGB, GL_UNSIGNED_BYTE, Image->RGB_Canvas);
                    Render_Image(Image,0,0,1);
            break;

            case CURRENT_STATE_TRAIN_MDNRNN:
                for(int j = 0; j < Model->img_count; j++)
                {
                float a_t[4];
                float loss;
                a_t[0] = j/0.1;
                a_t[1] = 0;
                a_t[2] = 0;
                a_t[3] = 0;

                Model->recorded_a[0] = a_t;
                extract_latent_vector(Model->vae, Model->dataset[j], Model->mu, Model->recorded_z[0]);
                pack_mdn_input(Model->recorded_z[0], LATENT_VECTOR_SIZE, *Model->recorded_a, CONTROLLER_OUTPUT_SIZE, Model->X[0]);
                memcpy(Model->Y[0], Model->recorded_z[0 + 1], LATENT_VECTOR_SIZE * sizeof(float));

                ////printf("Training MDN-RNN World Model...\n");
                loss = mdn_rnn_train_sequence(Model->net, Model->X[0], Model->Y[0], 1, 0.0005);
                printf("Loss: %lf\n", loss);
                adam_step(Model->optimizer, Model->net);
                mdn_rnn_save(Model->net, "mdn_rnn_model.bin");

                MDNSample sample;
                mdn_rnn_sample(Model->net, Model->controller.h_next, 0.01, &sample);
                memcpy(Model->recorded_z[0], sample.sampled_output, sizeof(float) * LATENT_VECTOR_SIZE);
                decode_latent_vector(Model->vae, Model->recorded_z[0], "none", 96, 54);

                memcpy(Image->RGB_Canvas,Model->vae->bmpdata,96*54*3);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 96, 54, 0, GL_RGB, GL_UNSIGNED_BYTE, Image->RGB_Canvas);
                Render_Image(Image,0,0,1);
                }
            break;

            case CURRENT_STATE_TRAIN_CONTROLLER:

            break;

            case CURRENT_STATE_TRAIN_FROM_DREAM:


                //Model->controller.z_t = allocate_array(LATENT_VECTOR_SIZE);
               // Model->controller.h_t = allocate_array(LATENT_VECTOR_SIZE * 1);
               // Model->controller.a_t = allocate_array(CONTROLLER_OUTPUT_SIZE);
               // Model->controller.x_t = allocate_array(LATENT_VECTOR_SIZE + CONTROLLER_OUTPUT_SIZE);
              //  Model->controller.h_next = allocate_array(LATENT_VECTOR_SIZE * 1);

              //void *memcpy(void *dest, const void *src, size_t n);

              float a_t[4];
                float loss;
                a_t[0] = 1;
                a_t[1] = 1;
                a_t[2] = 1;
                a_t[3] = 1;

                Model->recorded_a[0] = MyController;


                //controller_get_action(&Model->controller, Model->recorded_z[0], LATENT_VECTOR_SIZE, Model->predicted_h, LATENT_VECTOR_SIZE * 1, Model->recorded_a[0]);

                memcpy(Model->controller.x_t, Model->recorded_z[0], sizeof(float) * LATENT_VECTOR_SIZE);
                memcpy(&Model->controller.x_t[LATENT_VECTOR_SIZE], Model->recorded_a[0], sizeof(float) * CONTROLLER_OUTPUT_SIZE);

                mdn_rnn_step(Model->net, Model->controller.x_t, Model->predicted_h, Model->controller.h_next);

                MDNSample sample;
                mdn_rnn_sample(Model->net, Model->controller.h_next, temperature_level, &sample);
                memcpy(Model->recorded_z[0], sample.sampled_output, sizeof(float) * LATENT_VECTOR_SIZE);
                //Model->controller.total_reward += sample.reward;

                printf("Inputs: [%.2f, %.2f, %.2f, %.2f] | Reward: %.4f | Choice: %d\n",
                       Model->recorded_a[0][0], Model->recorded_a[0][1], Model->recorded_a[0][2], Model->recorded_a[0][3], sample.reward, sample.selected_component);
                       printf("The temperature is %f\n", temperature_level);

                memcpy(Model->predicted_h, Model->controller.h_next, sizeof(float) * LATENT_VECTOR_SIZE * 1);
                free(sample.sampled_output);
                decode_latent_vector(Model->vae, Model->recorded_z[0], "none", 96, 54);
                memcpy(Image->RGB_Canvas,Model->vae->bmpdata,96*54*3);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 96, 54, 0, GL_RGB, GL_UNSIGNED_BYTE, Image->RGB_Canvas);
                Render_Image(Image,0,0,1);

            break;

        }


           // //printf("Current frame: %d\n", Env.Current_Frame);
            if(Env.Current_Frame == Env.Number_Of_Frames-1)
            {
                Env.Current_Frame = 0;
                ////printf("Loop\n");
            }

        glfwSwapBuffers(W);
        glfwPollEvents();

        if (glfwGetKey(W, GLFW_KEY_A) == GLFW_PRESS)
        {
            temperature_level += 0.1;
        }

        if (glfwGetKey(W, GLFW_KEY_D) == GLFW_PRESS)
        {
            temperature_level -= 0.1;
        }

        if (glfwGetKey(W, GLFW_KEY_UP) == GLFW_PRESS)
        {
            MyController[0] += 0.1;
        }

        if (glfwGetKey(W, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            MyController[0] -= 0.1;
        }


        if (glfwGetKey(W, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            MyController[1] += 0.1;
        }

        if (glfwGetKey(W, GLFW_KEY_LEFT) == GLFW_PRESS)
        {
            MyController[1] -= 0.1;
        }
    }
    Mix_Quit();
    SDL_Quit();
    return 0;
}
