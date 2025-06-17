#include "params.h"
#include <curand_kernel.h> 
#include <cuda_runtime.h>


void update(const float temp, int *red_grid, int *black_grid, curandState *d_state);
float calculate(int *red_grid, int *black_grid, int *M_max);
void init_state();
void initialize_random_states(curandState **d_state, unsigned long seed); 
