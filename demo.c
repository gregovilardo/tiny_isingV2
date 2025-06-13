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

#include "colormap.h"
#include "gl2d.h"
#include "ising.h"
#include "params.h"
#include "xoshiro256plus.h"

#include <assert.h>
#include <limits.h> // UINT_MAX
#include <omp.h>    // omp_get_wtime()
#include <stdio.h>  // printf()
#include <stdlib.h> // rand()
#include <string.h>
#include <time.h> // time()

#define MAXFPS 60
#define N (L * L)   // system size
#define SEED 0xC4FE //(time(NULL)) // random seed

#define ROWS (L / 2)
#define COLS L

/**
 * GL output
 */
static void draw(gl2d_t gl2d, float t_now, float t_min, float t_max,
                 int *black_grid, int *red_grid) {
  static double last_frame = 0.0;

  double current_time = omp_get_wtime();
  if (current_time - last_frame < 1.0 / MAXFPS) {
    return;
  }
  last_frame = current_time;

  float row[L * 3];
  float color[3];
  colormap_rgbf(COLORMAP_VIRIDIS, t_now, t_min, t_max, &color[0], &color[1],
                &color[2]);
  for (int i = 0; i < L; ++i) {
    memset(row, 0, sizeof(row));
    for (int j = 0; j < L; ++j) {
      int cell = (i % 2 == 0) ? black_grid[(i / 2) * L + j]
                              : red_grid[(i / 2) * L + j];
      if (cell > 0) {
        row[j * 3] = color[0];
        row[j * 3 + 1] = color[1];
        row[j * 3 + 2] = color[2];
      }
    }
    gl2d_draw_rgbf(gl2d, 0, i, L, 1, row);
  }
  gl2d_display(gl2d);
}


static void cycle(gl2d_t gl2d, const float initial, const float final,
                  const float step, int *black_grid, int *red_grid) {
  assert((0.0f < step && initial <= final) ||
         (step < 0.0f && final <= initial));
  int modifier = (0.0f < step) ? 1 : -1;

  for (float temp = initial; modifier * temp <= modifier * final;
       temp += step) {
    printf("Temp: %f\n", temp);
    for (size_t j = 0; j < TRAN + TMAX; ++j) {
      update(temp, black_grid, red_grid);
      draw(gl2d, temp, initial < final ? initial : final,
           initial < final ? final : initial, black_grid, red_grid);
    }
  }
}


static void init(int *grid) {
  for (size_t i = 0; i < L / 2; ++i) {
    for (size_t j = 0; j < L; ++j) {
      grid[i * L + j] = (rand() / (float)RAND_MAX) < 0.5f ? -1 : 1;
    }
  }
}


int main(void) {
  // parameter checking
  static_assert(TEMP_DELTA != 0, "Invalid temperature step");
  static_assert(((TEMP_DELTA > 0) && (TEMP_INITIAL <= TEMP_FINAL)) ||
                    ((TEMP_DELTA < 0) && (TEMP_INITIAL >= TEMP_FINAL)),
                "Invalid temperature range+step");
  static_assert(TRAN + TMAX > 0, "Invalid times");
  static_assert(
      (L * L / 2) * 4ULL < UINT_MAX,
      "L too large for uint indices"); // max energy, that is all spins are the
                                       // same, fits into a ulong

  // print header
  printf("# L: %i\n", L);
  printf("# Minimum Temperature: %f\n", TEMP_INITIAL);
  printf("# Maximum Temperature: %f\n", TEMP_FINAL);
  printf("# Temperature Step: %.12f\n", TEMP_DELTA);
  printf("# Equilibration Time: %i\n", TRAN);
  printf("# Measurement Time: %i\n", TMAX);

  // Seed the generator
  seed(SEED);

  gl2d_t gl2d = gl2d_init("tiny_ising", L, L);

  // start timer
  double start = omp_get_wtime();

  int *d_black = (int *)malloc(ROWS * COLS * sizeof(int));
  int *d_red = (int *)malloc(ROWS * COLS * sizeof(int));
  init(d_black);
  init(d_red);


  // temperature increasing cycle
  cycle(gl2d, TEMP_INITIAL, TEMP_FINAL, TEMP_DELTA, d_black, d_red);

  // stop timer
  double elapsed = omp_get_wtime() - start;
  printf("# Total Simulation Time (sec): %lf\n", elapsed);

  gl2d_destroy(gl2d);

  return 0;
}
