/*
 * Tiny Ising model with CUDA and OpenGL visualization.
 * Loosely based on  "q-state Potts model metastability
 * study using optimized GPU-based Monte Carlo algorithms",
 * Ezequiel E. Ferrero, Juan Pablo De Francesco, Nicolás Wolovick,
 * Sergio A. Cannas
 * http://arxiv.org/abs/1101.0876
 *
 * Converted to CUDA with visualization
 */

#include "colormap.h"
#include "gl2d.h"
#include "ising.h"
#include "params.h"
#include "wtime.h"    // wtime()

#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h> // UINT_MAX
#include <stdio.h>  // printf()
#include <stdlib.h> // rand()
#include <string.h>
#include <time.h> // time()

#define MAXFPS 60
#define N (L_SIZE * L_SIZE)   // system size
#define SEED 0xCAFE //(time(NULL)) // random seed
#define ROWS (L_SIZE / 2)
#define COLS L_SIZE

// CUDA kernel for random initialization
__global__ void init_kernel(int *grid, unsigned long seed, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < grid_size) {
        // Simple pseudo-random initialization
        // Each thread uses a different seed based on idx
        unsigned long local_seed = seed + idx;
        local_seed = local_seed * 1103515245 + 12345; // Linear congruential generator
        grid[idx] = ((local_seed / 65536) % 2 == 0) ? -1 : 1;
    }
}

// CUDA kernel to copy data from device to host for visualization
__global__ void copy_for_display_kernel(int *d_black, int *d_red, int *d_display, int L) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < L && j < L) {
        int cell;
        if (i % 2 == 0) {
            // Even rows use black grid
            cell = d_black[(i / 2) * L + j];
        } else {
            // Odd rows use red grid
            cell = d_red[(i / 2) * L + j];
        }
        d_display[i * L + j] = cell;
    }
}

/**
 * GL output with CUDA data
 */
static void draw(gl2d_t gl2d, float t_now, float t_min, float t_max,
                 int *d_black_grid, int *d_red_grid) {
    static double last_frame = 0.0;
    double current_time = wtime();
    if (current_time - last_frame < 1.0 / MAXFPS) {
        return;
    }
    last_frame = current_time;
    
    // Allocate temporary device memory for display
    int *d_display;
    cudaMalloc(&d_display, L_SIZE * L_SIZE * sizeof(int));
    
    // Copy and reorganize data for display
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((L_SIZE + 15) / 16, (L_SIZE + 15) / 16);
    copy_for_display_kernel<<<numBlocks, threadsPerBlock>>>(d_black_grid, d_red_grid, d_display, L_SIZE);
    
    // Copy to host for OpenGL rendering
    int *h_display = (int*)malloc(L_SIZE * L_SIZE * sizeof(int));
    cudaMemcpy(h_display, d_display, L_SIZE * L_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    
    float row[L_SIZE * 3];
    float color[3];
    colormap_rgbf(COLORMAP_VIRIDIS, t_now, t_min, t_max, &color[0], &color[1], &color[2]);
    
    for (int i = 0; i < L_SIZE; ++i) {
        memset(row, 0, sizeof(row));
        for (int j = 0; j < L_SIZE; ++j) {
            int cell = h_display[i * L_SIZE + j];
            if (cell > 0) {
                row[j * 3] = color[0];
                row[j * 3 + 1] = color[1];
                row[j * 3 + 2] = color[2];
            }
        }
        gl2d_draw_rgbf(gl2d, 0, i, L_SIZE, 1, row);
    }
    gl2d_display(gl2d);
    
    // Cleanup
    free(h_display);
    cudaFree(d_display);
}

static void cycle(gl2d_t gl2d, const float initial, const float final,
                  const float step, int *d_black_grid, int *d_red_grid, 
                  curandState *d_state) {
    assert((0.0f < step && initial <= final) ||
           (step < 0.0f && final <= initial));
    int modifier = (0.0f < step) ? 1 : -1;
    
    for (float temp = initial; modifier * temp <= modifier * final; temp += step) {
        printf("Temp: %f\n", temp);
        
        // Equilibration phase
        for (size_t j = 0; j < TRAN; ++j) {
            update(temp, d_red_grid, d_black_grid, d_state);
        }
        
        // Measurement/visualization phase
        for (size_t j = 0; j < TMAX; ++j) {
            update(temp, d_red_grid, d_black_grid, d_state);
            
            // Draw every few steps to maintain reasonable frame rate
            if (j % 10 == 0) {
                draw(gl2d, temp, initial < final ? initial : final,
                     initial < final ? final : initial, d_black_grid, d_red_grid);
            }
        }
    }
}

static void init_cuda_grids(int *d_black, int *d_red) {
    int grid_size = ROWS * COLS;
    int threadsPerBlock = 256;
    int numBlocks = (grid_size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Initialize both grids with random values
    init_kernel<<<numBlocks, threadsPerBlock>>>(d_black, SEED, grid_size);
    cudaDeviceSynchronize();
    
    init_kernel<<<numBlocks, threadsPerBlock>>>(d_red, SEED + 12345, grid_size);
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in grid initialization: %s\n", cudaGetErrorString(err));
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
        (L_SIZE * L_SIZE / 2) * 4ULL < UINT_MAX,
        "L too large for uint indices"); // max energy, that is all spins are the
                                         // same, fits into a ulong

    // print header
    printf("# L_SIZE: %i\n", L_SIZE);
    printf("# Minimum Temperature: %f\n", TEMP_INITIAL);
    printf("# Maximum Temperature: %f\n", TEMP_FINAL);
    printf("# Temperature Step: %.12f\n", TEMP_DELTA);
    printf("# Equilibration Time: %i\n", TRAN);
    printf("# Measurement Time: %i\n", TMAX);

    // Initialize OpenGL context
    gl2d_t gl2d = gl2d_init("tiny_ising_cuda", L_SIZE, L_SIZE);

    // start timer
    const auto start = std::chrono::high_resolution_clock::now();

    // Allocate CUDA memory
    curandState *d_state;
    int *d_black, *d_red;
    cudaMalloc(&d_black, ROWS * COLS * sizeof(int));
    cudaMalloc(&d_red, ROWS * COLS * sizeof(int));
    
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error allocating memory on device\n");
        return -1;
    }

    // Initialize grids and random states
    init_cuda_grids(d_black, d_red);
    initialize_random_states(&d_state, SEED);

    // temperature cycle with visualization
    cycle(gl2d, TEMP_INITIAL, TEMP_FINAL, TEMP_DELTA, d_black, d_red, d_state);

    // stop timer
    const auto elapsed = std::chrono::high_resolution_clock::now() - start;
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed);
    printf("# Total Simulation Time (sec): %lf\n", elapsed_seconds.count());

    // Cleanup
    gl2d_destroy(gl2d);
    cudaFree(d_black);
    cudaFree(d_red);
    cudaFree(d_state);
    
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "Error freeing memory on device\n");
        return -1;
    }

    return 0;
}

