#include "gol.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include "tile-macro.hpp"

#define WORD_TYPE uint64_t

__device__ WORD_TYPE load_word(const WORD_TYPE* input, int x_word, int y_word, int x_dim, int y_dim) {
    if (x_word < 0 || x_word >= x_dim || y_word < 0 || y_word >= y_dim) {
        return 0;
    }
    return input[y_word * x_dim + x_word];
}

__global__ void gol_kernel(const WORD_TYPE* input, WORD_TYPE* output, int dim) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int x_dim = dim / 8;
    int y_dim = dim / 8;
    
    if (x >= x_dim || y >= y_dim) return;

    WORD_TYPE tl, tc, tr;
    WORD_TYPE cl, cc, cr;
    WORD_TYPE bl, bc, br;

    tl = load_word(input, x - 1, y - 1, x_dim, y_dim);
    tc = load_word(input, x,     y - 1, x_dim, y_dim);
    tr = load_word(input, x + 1, y - 1, x_dim, y_dim);
    cl = load_word(input, x - 1, y,     x_dim, y_dim);
    cc = load_word(input, x,     y,     x_dim, y_dim);
    cr = load_word(input, x + 1, y,     x_dim, y_dim);
    bl = load_word(input, x - 1, y + 1, x_dim, y_dim);
    bc = load_word(input, x,     y + 1, x_dim, y_dim);
    br = load_word(input, x + 1, y + 1, x_dim, y_dim);

    WORD_TYPE result = GOL_COMPUTE_TILED(
        tl, tc, tr,
        cl, cc, cr,
        bl, bc, br);

    output[y * x_dim + x] = result;
}

void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    int x_dim = grid_dimensions / 8;
    int y_dim = grid_dimensions / 8;

    dim3 blockSize(16, 16);
    dim3 gridSize((x_dim + blockSize.x - 1) / blockSize.x, (y_dim + blockSize.y - 1) / blockSize.y);

    gol_kernel<<<gridSize, blockSize>>>(input, output, grid_dimensions);
}

void run_game_of_life(const std::uint32_t* input, std::uint32_t* output, int grid_dimensions) {}