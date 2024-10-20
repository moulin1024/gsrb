#include "csr_matrix.h"
#include "gauss_seidel_red_black.h"
#include "switch_gpu_backend.h"
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>


// Helper function for CPU timing
template<typename F, typename... Args>
double measure_time(F func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    // Matrix size and parameters
    int size = 16392;
    double density = 0.5;
    const double min_value = -1.0;
    const double max_value = 1.0;

    // Generate sparse matrix
    CSRMatrix sparse_matrix = generate_sparse_diagonal_matrix(size, density, min_value, max_value);
    // print_csr_matrix(sparse_matrix);

    // Generate random vector b
    std::vector<double> b = generate_random_vector(sparse_matrix.rows, -10.0, 10.0);
    // print_vector(b);

    // Solve Ax = b using Gauss-Seidel method
    int max_iterations = 1000;
    double tolerance = 1e-9;

    // CPU Gauss-Seidel
    std::vector<double> solution_cpu;
    double cpu_solve_time = measure_time([&]() {
        solution_cpu = gauss_seidel_red_black_cpu(sparse_matrix, b, max_iterations, tolerance);
    });

    // GPU Gauss-Seidel
    std::vector<double> solution_gpu;
    
    // Run GPU Gauss-Seidel
    solution_gpu = gauss_seidel_red_black_gpu(sparse_matrix, b, max_iterations, tolerance);

    // Compute residuals
    double residual_cpu, residual_gpu;
    double cpu_residual_time = measure_time([&]() {
        residual_cpu = compute_residual_cpu(sparse_matrix, solution_cpu, b);
    });

    double gpu_residual_time = measure_time([&]() {
        residual_gpu = compute_residual_gpu(sparse_matrix, solution_gpu, b);
    });

    // Print results
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "CPU Red-Black Gauss-Seidel:" << std::endl;
    std::cout << "  Solve time: " << cpu_solve_time*1000 << " milliseconds" << std::endl;
    std::cout << "  Residual ||Ax - b|| = " << residual_cpu << std::endl;

    std::cout << "\nGPU Red-Black Gauss-Seidel:" << std::endl;
    std::cout << "  Residual ||Ax - b|| = " << residual_gpu << std::endl;

    // Compare solutions
    double max_diff = 0.0;
    for (size_t i = 0; i < solution_cpu.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(solution_cpu[i] - solution_gpu[i]));
    }
    std::cout << "\nMaximum difference between CPU and GPU solutions: " << max_diff << std::endl;

    return 0;
}
