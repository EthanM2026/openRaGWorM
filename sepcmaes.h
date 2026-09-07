#ifndef SEPCMAES_H_INCLUDED
#define SEPCMAES_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "mdn.h"
#include "rgwm.h"



// Helper: Box-Muller standard normal Gaussian generator N(0, 1)
float sample_gaussian();

// 1. Convert Controller struct <-> 1D Flat Parameter Vector
void controller_to_vector(const Controller *c, float *theta);

void vector_to_controller(const float *theta, Controller *c);

// 2. Initialize Sep-CMA-ES
SepCMAES *create_cmaes(int N, float initial_sigma);

// 3. Sample a population of offspring vectors theta_k ~ N(mean, diag(sigma^2))
void cmaes_sample(SepCMAES *cma);



int compare_candidates(const void *a, const void *b);

// 4. Update Mean and Variances based on Elite Candidate Rewards
void cmaes_update(SepCMAES *cma);

void free_cmaes(SepCMAES *cma);



float evaluate_controller_in_dream(RNN_MDN *mdn_net, Controller *ctrl, int max_steps);

int mainabc();


void New_Initialize_Controller(struct _Recurrent_Generative_World_Model* Model);

/**
 * Packs latent vector z_t and RNN hidden state h_t into controller input buffer
 *
 * @param z       Pointer to latent vector z_t [size: z_dim]
 * @param z_dim   VAE latent dimension (e.g., 32)
 * @param h       Pointer to MDN-RNN hidden state h_t [size: h_dim]
 * @param h_dim   RNN hidden state dimension (e.g., 64 or 256)
 * @param out_c   Destination array for Controller input [size: z_dim + h_dim]
 */
void pack_controller_input(const float *z, int z_dim, const float *h, int h_dim, float *out_c);


// Computes a_t = tanh(W * [z_t, h_t] + b)
void controller_forward(Controller *c, const float *z_t, int z_dim, const float *h_t, int h_dim, float *out_a);

#endif // SEPCMAES_H_INCLUDED
