#include "gol.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define INDEX(x, y, dim) ((y) * (dim) + (x))

__global__ void gol_kernel(const uint32_t* input, uint32_t* output, int dim) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim || y >= dim) return;

    int count = 0;
    
    int x_start = (x == 0) ? 0 : -1;
    int x_end = (x == dim - 1) ? 0 : 1;
    int y_start = (y == 0) ? 0 : -1;
    int y_end = (y == dim - 1) ? 0 : 1;
    
    for (int dx = x_start; dx <= x_end; dx++) {
        for (int dy = y_start; dy <= y_end; dy++) {
            
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            count += input[INDEX(nx, ny, dim)];
        }
    }

    int current = input[INDEX(x, y, dim)];
    int new_state = (count == 3 || (current && count == 2)) ? 1 : 0;

    output[INDEX(x, y, dim)] = new_state;
}

void run_game_of_life(const std::uint32_t* input, std::uint32_t* output, int grid_dimensions) {
    uint32_t *d_input, *d_output;
    size_t size = grid_dimensions * grid_dimensions * sizeof(uint32_t);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((grid_dimensions) / 16, (grid_dimensions) / 16);

    gol_kernel<<<gridSize, blockSize>>>(d_input, d_output, grid_dimensions);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
