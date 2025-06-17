/*
 * Tiny Ising model.
 * Loosely based on  "q-state Potts model metastability
 * study using optimized GPU-based Monte Carlo algorithms",
 * Ezequiel E. Ferrero, Juan Pablo De Francesco, Nicolás Wolovick,
 * Sergio A. Cannas
 * http://arxiv.org/abs/1101.0876
 *
 * Debugging: Ezequiel Ferrero
 */

#include "ising.h"
#include "params.h"

#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h> // UINT_MAX
#include <stdint.h>
#include <stdio.h>  // printf()
#include <stdlib.h> // abs()


// Internal definitions and functions
// out vector size, it is +1 since we reach TEMP_
#define NPOINTS (1 + (int)((TEMP_FINAL - TEMP_INITIAL) / TEMP_DELTA))
#define N (L_SIZE * L_SIZE) // system size
#define ROWS (L_SIZE / 2)
#define COLS L_SIZE
#define SEED (0xCAFEUL)

// temperature, E, E^2, E^4, M, M^2, M^4
struct statpoint {
  float t;
  float e;
  float e2;
  float e4;
  float m;
  float m2;
  float m4;
};

static void cycle(int *black_grid, int *red_grid, const float min,
                  const float max, const float step,
                  const unsigned int calc_step, struct statpoint stats[],
                  curandState *d_state) {

  assert((0.0f < step && min <= max) || (step < 0.0f && max <= min));
  int modifier = (0.0f < step) ? 1 : -1;

  size_t index = 0;
  for (float temp = min; modifier * temp <= modifier * max; temp += step) {

    // equilibrium phase
    for (size_t j = 0; j < TRAN; ++j) {
      update(temp, black_grid, red_grid, d_state);
    }
    // measurement phase
    unsigned int measurements = 0;
    float e = 0.0, e2 = 0.0, e4 = 0.0, m = 0.0, m2 = 0.0, m4 = 0.0;
    for (size_t j = 0; j < TMAX; ++j) {
      update(temp, black_grid, red_grid, d_state);
      if (j % calc_step == 0) {
        float energy = 0.0, mag = 0.0;
        int M_max = 0;
        energy = calculate(black_grid, red_grid, &M_max);
        mag = abs(M_max) / (float)N;
        e += energy;
        e2 += energy * energy;
        e4 += energy * energy * energy * energy;
        m += mag;
        m2 += mag * mag;
        m4 += mag * mag * mag * mag;
        ++measurements;
      }
    }
    assert(index < NPOINTS);
    stats[index].t = temp;
    stats[index].e += e / measurements;
    stats[index].e2 += e2 / measurements;
    stats[index].e4 += e4 / measurements;
    stats[index].m += m / measurements;
    stats[index].m2 += m2 / measurements;
    stats[index].m4 += m4 / measurements;
    ++index;
  }
}

__global__ void init(int *array) {
  size_t i = blockIdx.y * blockDim.y + threadIdx.y;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ROWS && j < COLS) {
    size_t idx = i * COLS + j;
    array[idx] = 1;
  }
}


int main(void) {
  // parameter checking
  static_assert(TEMP_DELTA != 0, "Invalid temperature step");
  static_assert(((TEMP_DELTA > 0) && (TEMP_INITIAL <= TEMP_FINAL)) ||
                    ((TEMP_DELTA < 0) && (TEMP_INITIAL >= TEMP_FINAL)),
                "Invalid temperature range+step");
  static_assert(
      TMAX % DELTA_T == 0,
      "Measurements must be equidistant"); // take equidistant calculate()
  static_assert(
      (L_SIZE * L_SIZE / 2) * 4ULL < UINT_MAX,
      "L too large for uint indices"); // max energy, that is all spins are the
                                       // same, fits into a ulong

  // the stats
  struct statpoint stat[NPOINTS];
  for (size_t i = 0; i < NPOINTS; ++i) {
    stat[i].t = 0.0f;
    stat[i].e = stat[i].e2 = stat[i].e4 = 0.0f;
    stat[i].m = stat[i].m2 = stat[i].m4 = 0.0f;
  }

  // print header
  printf("#L_SIZE: %i\n",L_SIZE);
  printf("# Minimum Temperature: %f\n", TEMP_INITIAL);
  printf("# Maximum Temperature: %f\n", TEMP_FINAL);
  printf("# Temperature Step: %.12f\n", TEMP_DELTA);
  printf("# Equilibration Time: %i\n", TRAN);
  printf("# Measurement Time: %i\n", TMAX);
  printf("# Data Acquiring Step: %i\n", DELTA_T);
  printf("# Number of Points: %i\n", NPOINTS);

  // start timer
  const auto start = std::chrono::high_resolution_clock::now();
  curandState *d_state;
  int *d_black, *d_red; // TODO: probar si funciona mejor con int8_t/char
  cudaMalloc(&d_black, ROWS * COLS * sizeof(int));
  cudaMalloc(&d_red, ROWS * COLS * sizeof(int));
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Error allocating memory on device\n");
    return -1;
  }
  // initialize the grids
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((ROWS + 15) / 16, (COLS + 15) / 16);
  init<<<numBlocks, threadsPerBlock>>>(d_black);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Error initializing black grid\n");
    return -1;
  }
  init<<<numBlocks, threadsPerBlock>>>(d_red);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Error initializing red grid\n");
    return -1;
  }

  // Initialize random states once
  initialize_random_states(&d_state, SEED); // or any seed

  // temperature increasing cycle
  cycle(d_black, d_red, TEMP_INITIAL, TEMP_FINAL, TEMP_DELTA, DELTA_T, stat,
        d_state);

  // stop timer
  const auto elapsed = std::chrono::high_resolution_clock::now() - start;

  // Convert to seconds for printing
  auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsed);
  printf("# Total Simulation Time (sec): %lf\n", elapsed_seconds.count());

  // Convert to milliseconds for spins calculation
  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
  printf("# Spins/ms: %lf\n", (double)N / elapsed_ms.count());

  printf("# Temp\tE\tE^2\tE^4\tM\tM^2\tM^4\n");
  for (size_t i = 0; i < NPOINTS; ++i) {
    printf("%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n", stat[i].t,
           stat[i].e / ((float)N), stat[i].e2 / ((float)N * N),
           stat[i].e4 / ((float)N * N * N * N), stat[i].m, stat[i].m2,
           stat[i].m4);
  }

  // free memory
  cudaFree(d_black);
  cudaFree(d_red);
  cudaFree(d_state);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Error freeing memory on device\n");
    return -1;
  }

  return 0;
}

