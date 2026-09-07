#include "sepcmaes.h"



// Helper: Box-Muller standard normal Gaussian generator N(0, 1)
float sample_gaussian() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(2.0f * (float)M_PI * u2);
}

// 1. Convert Controller struct <-> 1D Flat Parameter Vector
void controller_to_vector(const Controller *c, float *theta) {
    int in_dim = c->z_dim + c->h_dim;
    int w_size = c->action_dim * in_dim;
    memcpy(theta, c->W, w_size * sizeof(float));
    memcpy(theta + w_size, c->b, c->action_dim * sizeof(float));
}

void vector_to_controller(const float *theta, Controller *c) {
    int in_dim = c->z_dim + c->h_dim;
    int w_size = c->action_dim * in_dim;
    memcpy(c->W, theta, w_size * sizeof(float));
    memcpy(c->b, theta + w_size, c->action_dim * sizeof(float));
}

// 2. Initialize Sep-CMA-ES
SepCMAES *create_cmaes(int N, float initial_sigma) {
    SepCMAES *cma = (SepCMAES *)malloc(sizeof(SepCMAES));
    cma->N = N;

    // Default population size rule of thumb: lambda = 4 + floor(3 * log(N))
    cma->lambda = 4 + (int)(3.0f * logf((float)N));
    if (cma->lambda < 16) cma->lambda = 16; // Minimum size for stability
    cma->mu = cma->lambda / 2;

    cma->mean = (float *)calloc(N, sizeof(float));
    cma->sigma = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) cma->sigma[i] = initial_sigma;

    // Log-linear recombination weights
    cma->weights = (float *)malloc(cma->mu * sizeof(float));
    float sum_w = 0.0f;
    for (int i = 0; i < cma->mu; i++) {
        cma->weights[i] = logf(cma->mu + 0.5f) - logf(i + 1.0f);
        sum_w += cma->weights[i];
    }
    for (int i = 0; i < cma->mu; i++) cma->weights[i] /= sum_w; // Normalize

    // Learning rate for variance adaptation
    cma->ccov = (1.0f / (float)N) * 0.5f;

    // Allocate population matrices
    cma->pop = (float **)malloc(cma->lambda * sizeof(float *));
    for (int k = 0; k < cma->lambda; k++) {
        cma->pop[k] = (float *)malloc(N * sizeof(float));
    }
    cma->fitness = (float *)malloc(cma->lambda * sizeof(float));

    return cma;
}

// 3. Sample a population of offspring vectors theta_k ~ N(mean, diag(sigma^2))
void cmaes_sample(SepCMAES *cma) {
    for (int k = 0; k < cma->lambda; k++) {
        for (int i = 0; i < cma->N; i++) {
            cma->pop[k][i] = cma->mean[i] + cma->sigma[i] * sample_gaussian();
        }
    }
}



int compare_candidates(const void *a, const void *b) {
    float diff = ((Candidate *)b)->fitness - ((Candidate *)a)->fitness;
    return (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);
}

// 4. Update Mean and Variances based on Elite Candidate Rewards
void cmaes_update(SepCMAES *cma) {
    // Sort candidates by fitness (descending order)
    Candidate *rankings = (Candidate *)malloc(cma->lambda * sizeof(Candidate));
    for (int k = 0; k < cma->lambda; k++) {
        rankings[k].index = k;
        rankings[k].fitness = cma->fitness[k];
    }
    qsort(rankings, cma->lambda, sizeof(Candidate), compare_candidates);

    // Store previous mean for variance step computation
    float *old_mean = (float *)malloc(cma->N * sizeof(float));
    memcpy(old_mean, cma->mean, cma->N * sizeof(float));

    // Update Mean: weighted recombination of top mu elites
    memset(cma->mean, 0, cma->N * sizeof(float));
    for (int i = 0; i < cma->mu; i++) {
        int elite_idx = rankings[i].index;
        float w = cma->weights[i];
        for (int j = 0; j < cma->N; j++) {
            cma->mean[j] += w * cma->pop[elite_idx][j];
        }
    }

    // Update Diagonal Covariance / Variances (sigma)
    for (int j = 0; j < cma->N; j++) {
        float var_update = 0.0f;
        for (int i = 0; i < cma->mu; i++) {
            int elite_idx = rankings[i].index;
            float diff = (cma->pop[elite_idx][j] - old_mean[j]) / cma->sigma[j];
            var_update += cma->weights[i] * (diff * diff);
        }
        // Adapt per-parameter variance
        cma->sigma[j] = cma->sigma[j] * sqrtf((1.0f - cma->ccov) + cma->ccov * var_update);
    }

    free(rankings);
    free(old_mean);
}

void free_cmaes(SepCMAES *cma) {
    free(cma->mean);
    free(cma->sigma);
    free(cma->weights);
    for (int k = 0; k < cma->lambda; k++) free(cma->pop[k]);
    free(cma->pop);
    free(cma->fitness);
    free(cma);
}

// Simulated Rollout Function: Runs an episode and returns total cumulative reward
float evaluate_controller_in_dream(RNN_MDN *mdn_net, Controller *ctrl, int max_steps) {
    float cumulative_reward = 0.0f;
    int z_dim = ctrl->z_dim;
    int h_dim = ctrl->h_dim;
    int a_dim = ctrl->action_dim;

    float *z_t = (float *)calloc(z_dim, sizeof(float));
    float *h_t = (float *)calloc(h_dim, sizeof(float));
    float *a_t = (float *)malloc(a_dim * sizeof(float));
    float *mdn_in = (float *)malloc((z_dim + a_dim) * sizeof(float));

    for (int step = 0; step < max_steps; step++) {
        // 1. Controller computes action based on hallucinated state [z_t, h_t]
        controller_forward(ctrl, z_t, z_dim, h_t, h_dim, a_t);

        // 2. Feed [z_t, a_t] into MDN-RNN to step the dream forward
        pack_mdn_input(z_t, z_dim, a_t, a_dim, mdn_in);

        // Predict next latent z_{t+1} and update hidden state h_{t+1}
        sample_mdn(mdn_net, h_t, z_t, 1.0f /* temperature */);

        // Accumulate fitness (e.g., surrogate reward based on predicted state features)
        cumulative_reward += z_t[0];
    }

    free(z_t); free(h_t); free(a_t); free(mdn_in);
    return cumulative_reward;
}

void New_Initialize_Controller(struct _Recurrent_Generative_World_Model* Model)
{
    srand(42);

    int z_dim = LATENT_VECTOR_SIZE;
    int h_dim = Model->RNNhidden_dim;
    int a_dim = CONTROLLER_OUTPUT_SIZE;
    int N = (z_dim + h_dim) * a_dim + a_dim; // Total Controller weights: 291

    Model->Controller_Inst.z_dim = z_dim;
    Model->Controller_Inst.h_dim = h_dim;
    Model->Controller_Inst.action_dim = a_dim;
    Model->Controller_Inst.total_params = N;
    Model->Controller_Inst.W = (float *)malloc(a_dim * (z_dim + h_dim) * sizeof(float));
    Model->Controller_Inst.b = (float *)malloc(a_dim * sizeof(float));

    // Initialize Optimizer with initial variance 0.1f
    Model->SEPCMAES_Inst = create_cmaes(N, 0.1f);
    printf("Sep-CMA-ES initialized for Controller (%d parameters, population size %d)\n", N, Model->SEPCMAES_Inst->lambda);

    // Mock MDN-RNN network pointer

    // --- CMA-ES Optimization Generations ---
    for (int gen = 1; gen <= 50; gen++) {
        // 1. Sample population
        cmaes_sample(Model->SEPCMAES_Inst);

        // 2. Evaluate fitness for each candidate controller in the population
        for (int k = 0; k < Model->SEPCMAES_Inst->lambda; k++) {
            // Load candidate weights into Controller struct
            vector_to_controller(Model->SEPCMAES_Inst->pop[k], &Model->Controller_Inst);

            // Run rollout and save score
            Model->SEPCMAES_Inst->fitness[k] = evaluate_controller_in_dream(Model->net, &Model->Controller_Inst, 100);
        }

        // 3. Update mean distribution and variance based on top elites
        cmaes_update(Model->SEPCMAES_Inst);

        // Find best fitness in generation
        float max_fit = -1e9f;
        for (int k = 0; k < Model->SEPCMAES_Inst->lambda; k++) {
            if (Model->SEPCMAES_Inst->fitness[k] > max_fit) max_fit = Model->SEPCMAES_Inst->fitness[k];
        }

        printf("Generation %2d | Best Reward: %.4f | Avg Sigma: %.5f\n",
               gen, max_fit, Model->SEPCMAES_Inst->sigma[0]);
    }

    // Assign final optimal mean parameters to controller
    vector_to_controller(Model->SEPCMAES_Inst->mean, &Model->Controller_Inst);
    printf("\nController successfully optimized.\n");

   // free(Model->Controller_Inst.W); free(Model->Controller_Inst.b);
   // free_cmaes(Model->SEPCMAES_Inst);
    //free_rnn_mdn(Model->net);
}

int mainabc() {

}



/**
 * Packs latent vector z_t and RNN hidden state h_t into controller input buffer
 *
 * @param z       Pointer to latent vector z_t [size: z_dim]
 * @param z_dim   VAE latent dimension (e.g., 32)
 * @param h       Pointer to MDN-RNN hidden state h_t [size: h_dim]
 * @param h_dim   RNN hidden state dimension (e.g., 64 or 256)
 * @param out_c   Destination array for Controller input [size: z_dim + h_dim]
 */
void pack_controller_input(const float *z, int z_dim, const float *h, int h_dim, float *out_c) {
    // 1. Copy z_t into indices [0 ... z_dim - 1]
    memcpy(out_c, z, z_dim * sizeof(float));

    // 2. Copy h_t into indices [z_dim ... z_dim + h_dim - 1]
    memcpy(out_c + z_dim, h, h_dim * sizeof(float));
}


// Computes a_t = tanh(W * [z_t, h_t] + b)
void controller_forward(Controller *c, const float *z_t, int z_dim, const float *h_t, int h_dim, float *out_a) {
    int in_dim = z_dim + h_dim;
    float *c_in = (float *)malloc(in_dim * sizeof(float));

    // 1. Pack [z_t, h_t]
    pack_controller_input(z_t, z_dim, h_t, h_dim, c_in);

    // 2. Linear layer + tanh activation
    for (int i = 0; i < c->action_dim; i++) {
        float sum = c->b[i];
        for (int j = 0; j < in_dim; j++) {
            sum += c->W[i * in_dim + j] * c_in[j];
        }
        out_a[i] = tanhf(sum); // Bounded actions between [-1.0, 1.0]
    }

    free(c_in);
}
