#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "gol.cuh"
#include <cuda_runtime.h>

#define seed 42
#define warm_up_iterations 3
#define hot_runs 10
#define check_sum_parts 16
#define ROW_TYPE std::uint64_t

#define LOG std::cerr
#define RESULT std::cout

void print_grid(const std::vector<std::uint32_t>& grid, int dim) {
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            std::cout << (grid[y * dim + x] ? '#' : '.') << ' ';
        }
        std::cout << +'\n';
    }
    std::cout << "-----------------" << std::endl;
}

void parse_args(int argc, char* argv[], bool& use_bit_packing, int& dim, int& iterations) {
    if (argc > 1) {
        use_bit_packing = std::atoi(argv[1]);
    }
    if (argc > 2) {
        dim = std::atoi(argv[2]);
    }
    if (argc > 3) {
        iterations = std::atoi(argv[3]);
    }
}

void init_grid(std::vector<std::uint32_t>& grid, int dim) {
    std::srand(seed);
    for (auto& cell : grid) {
        cell = std::rand() % 2;
    }
}

std::vector<ROW_TYPE> to_bitpacked_rows(const std::vector<std::uint32_t>& grid, int dim) {
    std::vector<ROW_TYPE> new_grid(dim * dim / sizeof(ROW_TYPE), 0);
 
    for (std::size_t i = 0; i < grid.size(); i += sizeof(ROW_TYPE) * 8) {
        ROW_TYPE row = 0;

        for (std::size_t j = 0; j < sizeof(ROW_TYPE) * 8; ++j) {
            if (grid[i + j]) {
                row |= 1ULL << j;
            }
        }

        new_grid[i / (sizeof(ROW_TYPE) * 8)] = row;
    }
 
    return new_grid;
}

std::vector<std::uint32_t> from_bitpacked_rows(const std::vector<ROW_TYPE>& grid, int dim) {
    std::vector<std::uint32_t> new_grid(dim * dim, 0);

    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < sizeof(ROW_TYPE) * 8; ++j) {
            new_grid[i * sizeof(ROW_TYPE) * 8 + j] = (grid[i] >> j) & 1;
        }
    }

    return new_grid;
}

template<typename T>
void init_gpu_mem(const std::vector<T>& grid, T** d_grid, T** d_new_grid) {
    cudaMalloc(d_grid, grid.size() * sizeof(T));
    cudaMalloc(d_new_grid, grid.size() * sizeof(T));

    cudaMemcpy(*d_grid, grid.data(), grid.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void free_gpu_mem(T* d_grid, T* d_new_grid) {
    cudaFree(d_grid);
    cudaFree(d_new_grid);
}

std::string get_check_sums(const std::vector<std::uint32_t>& grid) {
    std::string output = "";

    for (std::size_t i = 0; i < check_sum_parts; ++i) {
        std::size_t check_sum = 0;

        for (std::size_t j = 0; j < grid.size() / check_sum_parts; ++j) {
            check_sum += grid[i * grid.size() / check_sum_parts + j];
        }

        if (i == 0) {
            output += std::to_string(check_sum);
        } else {
            output += "-" + std::to_string(check_sum);
        }
    }

    return output;
}

void start_cuda_timer(cudaEvent_t& start) {
    cudaEventCreate(&start);
    cudaEventRecord(start);
}

float stop_cuda_timer(cudaEvent_t& start) {
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

void perform_gol(const std::vector<std::uint32_t>& grid, std::vector<std::uint32_t>& out_grid, int dim, int iterations) {
    std::uint32_t* d_grid;
    std::uint32_t* d_new_grid;

    init_gpu_mem(grid, &d_grid, &d_new_grid);

    cudaEvent_t start;
    start_cuda_timer(start);

    for (int i = 0; i < iterations; ++i) {
        run_game_of_life(d_grid, d_new_grid, dim);
        std::swap(d_grid, d_new_grid);
    }

    cudaMemcpy(out_grid.data(), d_grid, out_grid.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    free_gpu_mem(d_grid, d_new_grid);

    RESULT << iterations << ";" << dim << ";" << stop_cuda_timer(start) << ";" << get_check_sums(out_grid) << seed << std::endl;
}

void perform_gol_row_packed(const std::vector<std::uint32_t>& grid, std::vector<std::uint32_t>& out_grid, int dim, int iterations) {

    std::vector<ROW_TYPE> row_packed_grid = to_bitpacked_rows(grid, dim);
    std::vector<ROW_TYPE> row_packed_out_grid(dim * dim / sizeof(ROW_TYPE), 0);

    ROW_TYPE* d_grid;
    ROW_TYPE* d_new_grid;

    init_gpu_mem(row_packed_grid, &d_grid, &d_new_grid);

    cudaEvent_t start;
    start_cuda_timer(start);

    for (int i = 0; i < iterations; ++i) {
        run_game_of_life(d_grid, d_new_grid, dim);
        std::swap(d_grid, d_new_grid);
    }

    float milliseconds = stop_cuda_timer(start);

    cudaMemcpy(row_packed_out_grid.data(), d_grid, row_packed_out_grid.size() * sizeof(ROW_TYPE), cudaMemcpyDeviceToHost);
    free_gpu_mem(d_grid, d_new_grid);

    out_grid = from_bitpacked_rows(row_packed_out_grid, dim);

    RESULT << iterations << ";" << dim << ";" << milliseconds << ";" << get_check_sums(out_grid) << seed << std::endl;
}

int main(int argc, char* argv[]) {

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <use bit packing> <grid size> <iterations>\n";
        return 1;
    }

    bool use_bit_packing;
    int dim, iterations;

    parse_args(argc, argv, use_bit_packing, dim, iterations);
    
    LOG << "Initializing grid" << std::endl;

    std::vector<std::uint32_t> grid(dim * dim, 0);
    std::vector<std::uint32_t> out_grid(dim * dim, 0);

    init_grid(grid, dim);

    LOG << "Running" << std::endl;

    RESULT << "iterations;dim;time;checksum;seed" << std::endl;

    for (int i = 0; i < warm_up_iterations + hot_runs; ++i) {
        LOG << " -- Iteration " << i << std::endl;

        if (use_bit_packing) {
            perform_gol_row_packed(grid, out_grid, dim, iterations);
        } else {
            perform_gol(grid, out_grid, dim, iterations);
        }
    }
}
