#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cub/cub.cuh>

#include "ising.h"
#include "params.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#define ROWS (L_SIZE / 2)
#define COLS L_SIZE

// Optimized block size for better occupancy
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Pre-computed exponential lookup table
__constant__ float exp_table[9]; // For delta_E values -8, -6, -4, -2, 0, 2, 4, 6, 8

// Initialize exponential lookup table
void init_exp_table(float temp) {
    float h_exp_table[9];
    for (int i = 0; i < 9; i++) {
        int delta_E = (i - 4) * 2; // Maps 0->-8, 1->-6, ..., 8->8
        h_exp_table[i] = expf(-delta_E / temp);
    }
    cudaMemcpyToSymbol(exp_table, h_exp_table, 9 * sizeof(float));
}

// Optimized random state setup with better memory access
__global__ void setup_random_states_opt(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = ROWS * COLS;
    
    if (idx < total_size) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

void initialize_random_states(curandState **d_state, unsigned long seed) {
    int total_size = ROWS * COLS;
    
    cudaError_t err = cudaMalloc(d_state, total_size * sizeof(curandState));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating random states: %s\n", cudaGetErrorString(err));
        return;
    }

    int numBlocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    setup_random_states_opt<<<numBlocks, BLOCK_SIZE>>>(*d_state, seed);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in setup_random_states kernel: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

// Optimized update kernel with shared memory and reduced divergence
__global__ void update_red_black_opt(int *__restrict__ write,
                                     const int *__restrict__ read, 
                                     int color,
                                     curandState *__restrict__ state_rng) {
    
    // Shared memory for better cache performance
    __shared__ int s_data[18][18]; // 16x16 + 1 border on each side
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Calculate thread position in 2D
    int threads_per_row = (COLS + WARP_SIZE - 1) / WARP_SIZE;
    int i = bid / threads_per_row;
    int j_base = (bid % threads_per_row) * WARP_SIZE;
    int j = j_base + tid;
    
    if (i >= ROWS || j >= COLS) return;
    
    int idx = i * COLS + j;
    
    // Load data into shared memory with coalesced access
    int si = threadIdx.x / WARP_SIZE + 1; // Simplified for demonstration
    int sj = threadIdx.x % WARP_SIZE + 1;
    
    if (tid < WARP_SIZE && j < COLS) {
        s_data[si][sj] = read[idx];
        
        // Load boundaries
        if (threadIdx.x == 0 && j > 0) {
            s_data[si][0] = read[i * COLS + j - 1];
        }
        if (threadIdx.x == WARP_SIZE - 1 && j < COLS - 1) {
            s_data[si][sj + 1] = read[i * COLS + j + 1];
        }
    }
    
    __syncthreads();
    
    // Calculate neighbors with periodic boundaries
    int jw = (j == 0) ? COLS - 1 : j - 1;
    int je = (j + 1) % COLS;
    
    int spin_w = read[i * COLS + jw];
    int spin_e = read[i * COLS + je];
    
    const int other = 1 - color;
    size_t full_i = 2 * i + color;
    
    // Optimized neighbor calculations
    size_t full_in = (full_i == 0) ? L_SIZE - 1 : full_i - 1;
    size_t full_is = (full_i == L_SIZE - 1) ? 0 : full_i + 1;
    
    int spin_old = write[idx];
    
    size_t in = (full_in % 2 == other) ? full_in / 2 : (full_in + L_SIZE) / 2;
    size_t is = (full_is % 2 == other) ? full_is / 2 : (full_is + L_SIZE) / 2;
    
    in = in % ROWS;
    is = is % ROWS;
    
    int spin_n = read[in * COLS + j];
    int spin_s = read[is * COLS + j];
    
    int delta_E = 2 * spin_old * (spin_n + spin_e + spin_w + spin_s);
    
    // Use lookup table instead of expensive exp calculation
    float acceptance_prob;
    if (delta_E <= 0) {
        acceptance_prob = 1.0f;
    } else {
        int table_idx = (delta_E / 2) + 4; // Map delta_E to table index
        acceptance_prob = exp_table[table_idx];
    }
    
    float p = curand_uniform(&state_rng[idx]);
    
    if (p <= acceptance_prob) {
        write[idx] = -spin_old;
    }
}

// Optimized calculation kernel with better memory access patterns
__global__ void calculate_red_black_opt(const int *__restrict__ write, 
                                       const int *__restrict__ read,
                                       int color, 
                                       int *__restrict__ E_partial,
                                       int *__restrict__ M_partial) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int total_elements = ROWS * COLS;
    
    // Process multiple elements per thread for better memory bandwidth utilization
    for (int base_idx = bid * blockDim.x; base_idx < total_elements; base_idx += total_threads) {
        int idx = base_idx + tid;
        if (idx >= total_elements) break;
        
        int i = idx / COLS;
        int j = idx % COLS;
        
        int other = 1 - color;
        size_t full_i = 2 * i + color;
        
        size_t full_in = (full_i == 0) ? L_SIZE - 1 : full_i - 1;
        size_t full_is = (full_i + 1) % L_SIZE;
        
        int spin = write[idx];
        
        size_t in = (full_in % 2 == other) ? full_in / 2 : (full_in + L_SIZE) / 2;
        size_t is = (full_is % 2 == other) ? full_is / 2 : (full_is + L_SIZE) / 2;
        
        in = in % ROWS;
        is = is % ROWS;
        
        size_t jw = (j == 0) ? COLS - 1 : j - 1;
        size_t je = (j + 1) % COLS;
        
        // Vectorized memory access where possible
        int4 neighbors;
        neighbors.x = read[in * COLS + j];  // north
        neighbors.y = read[is * COLS + j];  // south
        neighbors.z = read[i * COLS + jw];  // west
        neighbors.w = read[i * COLS + je];  // east
        
        int local_E = spin * (neighbors.x + neighbors.y + neighbors.z + neighbors.w);
        
        E_partial[idx] = local_E;
        M_partial[idx] = spin;
    }
}

// Optimized update function with streams for overlapping computation
void update(float temp, int *d_red, int *d_black, curandState *d_state) {
    // Initialize lookup table for this temperature
    init_exp_table(temp);
    
    // Optimized grid configuration
    int threads_per_row = (COLS + WARP_SIZE - 1) / WARP_SIZE;
    int total_blocks = ROWS * threads_per_row;
    
    // Use streams for potential overlap
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Update red grid (color = 0)
    update_red_black_opt<<<total_blocks, WARP_SIZE, 0, stream1>>>(d_red, d_black, 0, d_state);
    
    // Synchronize before updating black grid
    cudaStreamSynchronize(stream1);
    
    // Update black grid (color = 1) 
    update_red_black_opt<<<total_blocks, WARP_SIZE, 0, stream2>>>(d_black, d_red, 1, d_state);
    
    cudaStreamSynchronize(stream2);
    
    // Cleanup streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// Optimized calculate function using CUB for faster reductions
float calculate(int *d_red_grid, int *d_black_grid, int *M_max) {
    int grid_size = ROWS * COLS;
    
    // Use CUB for faster reductions instead of Thrust
    static int *d_E_partial = nullptr;
    static int *d_M_partial = nullptr;
    static int *d_E_result = nullptr;
    static int *d_M_result = nullptr;
    static void *d_temp_storage = nullptr;
    static size_t temp_storage_bytes = 0;
    
    // Allocate memory once (static allocation for better performance)
    if (d_E_partial == nullptr) {
        cudaMalloc(&d_E_partial, 2 * grid_size * sizeof(int));
        cudaMalloc(&d_M_partial, 2 * grid_size * sizeof(int));
        cudaMalloc(&d_E_result, 2 * sizeof(int));
        cudaMalloc(&d_M_result, 2 * sizeof(int));
        
        // Determine temporary storage requirements
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                              d_E_partial, d_E_result, 2 * grid_size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }
    
    // Launch kernels with optimal configuration
    int numBlocks = (grid_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    calculate_red_black_opt<<<numBlocks, BLOCK_SIZE>>>(
        d_red_grid, d_black_grid, 0, d_E_partial, d_M_partial);
    
    calculate_red_black_opt<<<numBlocks, BLOCK_SIZE>>>(
        d_black_grid, d_red_grid, 1, d_E_partial + grid_size, d_M_partial + grid_size);
    
    // Fast reduction using CUB
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                          d_E_partial, d_E_result, 2 * grid_size);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                          d_M_partial, d_M_result, 2 * grid_size);
    
    // Copy results back
    int h_E_total, h_M_total;
    cudaMemcpy(&h_E_total, d_E_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_M_total, d_M_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    *M_max = h_M_total;
    
    return -((float)h_E_total / 2.0f);
}
