#include <iostream>
#include <thread>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "gol.cuh"

void print_grid(const std::vector<std::uint32_t>& grid, int dim) {
    for (int y = 0; y < dim; ++y) {
        for (int x = 0; x < dim; ++x) {
            std::cout << (grid[y * dim + x] ? '#' : '.') << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "-----------------" << std::endl;
}

int main() {
    constexpr int dim = 16;
    std::vector<std::uint32_t> grid(dim * dim, 0);
    std::vector<std::uint32_t> new_grid(dim * dim, 0);

    // std::srand(std::time(nullptr));
    // for (auto& cell : grid) {
    //     cell = std::rand() % 2;
    // }

    // load one glider
    grid[1 * dim + 2] = 1;
    grid[2 * dim + 3] = 1;
    grid[3 * dim + 1] = 1;
    grid[3 * dim + 2] = 1;
    grid[3 * dim + 3] = 1;


    std::cout << "Initial Grid:\n";
    print_grid(grid, dim);

    
    for (int i = 0; i < 10; ++i) {
        run_game_of_life(grid.data(), new_grid.data(), dim);

        std::cout << "Next Generation:\n";
        print_grid(new_grid, dim);

        std::swap(grid, new_grid);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }


    return 0;
}
