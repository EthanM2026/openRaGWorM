#ifndef ENVIRONMENT_H_INCLUDED
#define ENVIRONMENT_H_INCLUDED

typedef struct {
    // True internal state (NOT directly seen by Controller or VAE)
    int Number_Of_Frames;
    int Current_Frame;
    float My_x;
    float My_y;
    float My_z;
    float Current_Instructions[4];
} Environment;

int environment_step(Environment *env,
                     const float *action,
                     unsigned char *frame_buffer_out,
                     float *reward_out) {


}

#endif // ENVIRONMENT_H_INCLUDED
