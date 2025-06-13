#include "ising.h"
#include "xoshiro256plus.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <stdio.h>

static float exp_table[32];

static void init_exp_table(const float temp) {
  exp_table[(-8) + 8] = expf(-8.0f / temp);
  exp_table[(-4) + 8] = expf(-4.0f / temp);
  exp_table[(0) + 8] = expf(0.0f / temp);
  exp_table[(4) + 8] = expf(4.0f / temp);
  exp_table[(8) + 8] = expf(8.0f / temp);
}
/* printf("i: %d, color: %d\n", i, color); */
/* printf("f_i    f_in    f_is    in    is    jw    je\n"); */
/* printf("%d      %d      %d     %d    %d    %d    %d\n", full_i,
 * full_in, */
/*        full_is, in, is, jw, je); */
/* printf("\n\n\n"); */

static void update_red_black_grid(int (*write)[L], const int (*read)[L],
                                  int color) {
  const int other = 1 - color;

  for (size_t i = 0; i < L / 2; ++i) {
    for (size_t j = 0; j < L; ++j) {
      size_t full_i = 2 * i + color;
      size_t full_in = (full_i + L - 1) % L;
      size_t full_is = (full_i + 1) % L;

      int spin_old = write[i][j];

      size_t in = ((full_in - other + L) % L) / 2;
      size_t is = ((full_is - other + L) % L) / 2;
      size_t jw = (j + L - 1) % L;
      size_t je = (j + 1) % L;

      int spin_n = read[in][j];
      int spin_s = read[is][j];
      int spin_w = read[i][jw];
      int spin_e = read[i][je];

      int delta_E = 2 * spin_old * (spin_n + spin_e + spin_w + spin_s);

      float p = optimized_random_probability();
      if (delta_E <= 0 || p <= exp_table[-delta_E + 8]) {
        write[i][j] = -spin_old;
      }
    }
  }
}

static int calculate_red_black_grid(const int (*write)[L], const int (*read)[L],
                                    int *M_max, int color) {
  int E = 0;
  *M_max = 0;

  int other = 1 - color;

  for (size_t i = 0; i < L / 2; ++i) {
    for (size_t j = 0; j < L; ++j) {
      size_t full_i = 2 * i + color;
      size_t full_in = (full_i + L - 1) % L;
      size_t full_is = (full_i + 1) % L;

      int spin = write[i][j];

      size_t in = ((full_in - other + L) % L) / 2;
      size_t is = ((full_is - other + L) % L) / 2;
      size_t jw = (j + L - 1) % L;
      size_t je = (j + 1) % L;


      int spin_n = read[in][j];
      int spin_s = read[is][j];
      int spin_w = read[i][jw];
      int spin_e = read[i][je];

      // accumulate energy: ½ factor omitted if you divide later
      E += spin * (spin_n + spin_e + spin_w + spin_s);

      // accumulate magnetization
      *M_max += spin;
    }
  }

  return E;
}

void update(const float temp, int (*red_grid)[L], int (*black_grid)[L]) {

  // Only initialized on first call
  static float last_temp = -99.99f;

  if (temp != last_temp) {
    init_exp_table(temp);
    last_temp = temp;
  }

  update_red_black_grid(red_grid, black_grid, 0);
  update_red_black_grid(black_grid, red_grid, 1);
}


// Kernel to update one color of the grid (red or black)
// write: output grid (flattened 1D of size N=L*L/2)
// read: input grid (flattened 1D of size N)
// color: 0 for red, 1 for black
// L: grid width
__global__ void update_red_black_kernel(int *write, const int *read, int color,
                                        const float *exp_table) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half = (L * L) / 2; // N
  int stride = blockDim.x * gridDim.x;
  int other = 1 - color;

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
    float p = /* optimized_random_probability() can be replaced by CURAND or
                 precomputed RNG */
        /* here just a placeholder*/ 0.0f;

    // Metropolis criterion
    if (delta_E <= 0 || p <= exp_table[-delta_E + 8]) {
      write[linear] = -spin_old;
    }
  }
}

// Host wrapper to launch kernels for both colors
void update_gpu(float temp, int *d_red, int *d_black,
                const float *d_exp_table) {
  // Assuming exp_table initialized on device
  // Determine execution configuration
  int half = (L * L) / 2;
  int threads = 256;
  int blocks = (half + threads - 1) / threads;

  // Update red (color=0)
  update_red_black_kernel<<<blocks, threads>>>(d_red, d_black, 0, d_exp_table);
  cudaDeviceSynchronize();

  // Update black (color=1)
  update_red_black_kernel<<<blocks, threads>>>(d_black, d_red, 1, d_exp_table);
  cudaDeviceSynchronize();
}

// Example of flattening 2D half-grids into 1D:
// index = i * L + j, where i in [0, L/2) and j in [0, L)

// Note: RNG and exp_table initialization omitted for brevity.

float calculate(int (*red_grid)[L], int (*black_grid)[L], int *M_max) {
  int E = 0;

  E += calculate_red_black_grid(red_grid, black_grid, M_max, 0);
  E += calculate_red_black_grid(black_grid, red_grid, M_max, 1);

  return -((float)E / 2.0f);
}
