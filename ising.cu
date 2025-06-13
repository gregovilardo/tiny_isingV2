#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ising.h"
#include "params.h"
#include "xoshiro256plus.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>


#define N (L * L)

__constant__ float *d_exp_table;
float h_exp_table[32];
static void init_exp_table(const float temp) {
  h_exp_table[(-8) + 8] = expf(-8.0f / temp);
  h_exp_table[(-4) + 8] = expf(-4.0f / temp);
  h_exp_table[(0) + 8] = expf(0.0f / temp);
  h_exp_table[(4) + 8] = expf(4.0f / temp);
  h_exp_table[(8) + 8] = expf(8.0f / temp);
  cudaMemcpyToSymbol(d_exp_table, h_exp_table, sizeof(h_exp_table));
}

// Device pointer to per-thread states
__device__ curandState_t *devStates;

// Kernel to initialize each thread's state
__global__ void init_kernel(unsigned long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    curand_init(seed, idx, 0, &devStates[idx]);
  }
}

/* printf("i: %d, color: %d\n", i, color); */
/* printf("f_i    f_in    f_is    in    is    jw    je\n"); */
/* printf("%d      %d      %d     %d    %d    %d    %d\n", full_i,
 * full_in, */
/*        full_is, in, is, jw, je); */
/* printf("\n\n\n"); */


static int calculate_red_black_grid(const int *write, const int *read,
                                    int *M_max, int color) {
  int E = 0;
  *M_max = 0;

  int other = 1 - color;

  for (size_t i = 0; i < L / 2; ++i) {
    for (size_t j = 0; j < L; ++j) {
      size_t full_i = 2 * i + color;
      size_t full_in = (full_i + L - 1) % L; // North neighbor row
      size_t full_is = (full_i + 1) % L;     // South neighbor row

      int spin = write[i * L + j];

      size_t in = ((full_in - other + L) % L) / 2; // North checkerboard row
      size_t is = ((full_is - other + L) % L) / 2; // South checkerboard row
      size_t jw = (j + L - 1) % L;                 // West column
      size_t je = (j + 1) % L;                     // East column

      int spin_n = read[in * L + j]; // North: in_row * L + j
      int spin_s = read[is * L + j]; // South: is_row * L + j
      int spin_w = read[i * L + jw]; // West: same row i, column jw
      int spin_e = read[i * L + je]; // East: same row i, column je

      E += spin * (spin_n + spin_e + spin_w + spin_s);

      *M_max += spin;
    }
  }

  return E;
}


// Kernel to update one color of the grid (red or black)
// write: output grid (flattened 1D of size N=L*L/2)
// read: input grid (flattened 1D of size N)
// color: 0 for red, 1 for black
// L: grid width
__global__ void update_red_black_kernel(int *write, const int *read,
                                        int color) {
  printf("Hola  ");

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = (L * L) / 2; // N
  int stride = blockDim.x * gridDim.x;
  int other = 1 - color;
  printf("Hola  soy el tid: %d\n", idx);

  for (int linear = idx; linear < half; linear += stride) {
    // linear index corresponds to (i,j) in the half-grid
    int i = linear / L;
    int j = linear % L;
    int full_i = 2 * i + color;
    int full_in = (full_i + L - 1) % L;
    int full_is = (full_i + 1) % L;

    // Map neighbors back to half-grid indices
    int in = ((full_in - other + L) % L) / 2;
    int is = ((full_is - other + L) % L) / 2;
    int jw = (j + L - 1) % L;
    int je = (j + 1) % L;

    int idx_n = in * L + j;
    int idx_s = is * L + j;
    int idx_w = i * L + jw;
    int idx_e = i * L + je;

    int spin_old = write[linear];
    int spin_n = read[idx_n];
    int spin_s = read[idx_s];
    int spin_w = read[idx_w];
    int spin_e = read[idx_e];

    int delta_E = 2 * spin_old * (spin_n + spin_e + spin_w + spin_s);

    // Get thread-specific RNG state
    float p = curand_uniform(&devStates[idx]);
    printf("p es: %f\n", p);

    // Metropolis criterion
    if (delta_E <= 0 || p <= d_exp_table[-delta_E + 8]) {
      write[linear] = -spin_old;
    }
  }
}

void init_state() {
  printf("dentor de init\n");
  const int THREADS = 256;
  const int BLOCKS = (N + THREADS - 1) / THREADS;

  printf("Launching kernel with %d blocks and %d threads/block\n", BLOCKS,
         THREADS);

  cudaMalloc((void **)&devStates, N * sizeof(curandState_t));
  init_kernel<<<BLOCKS, THREADS>>>(42);
  cudaDeviceSynchronize(); // Ensure init completes
  printf("termine de init\n");
}


void update(float temp, int *d_red, int *d_black) {

  const int THREADS = 256;
  const int BLOCKS = (N / 2 + THREADS - 1) / THREADS;
  static float last_temp = -99.99f;

  if (temp != last_temp) {
    init_exp_table(temp);
    last_temp = temp;
  }

  // Update red (color=0)
  update_red_black_kernel<<<BLOCKS, THREADS>>>(d_red, d_black, 0);
  cudaDeviceSynchronize();

  // Update black (color=1)
  update_red_black_kernel<<<BLOCKS, THREADS>>>(d_black, d_red, 1);
  cudaDeviceSynchronize();
}

// Example of flattening 2D half-grids into 1D:
// index = i * L + j, where i in [0, L/2) and j in [0, L)

// Note: RNG and exp_table initialization omitted for brevity.

float calculate(int *red_grid, int *black_grid, int *M_max) {
  int E = 0;

  E += calculate_red_black_grid(red_grid, black_grid, M_max, 0);
  E += calculate_red_black_grid(black_grid, red_grid, M_max, 1);

  return -((float)E / 2.0f);
}
