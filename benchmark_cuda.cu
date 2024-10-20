#include "csr_matrix.h"
#include "gauss_seidel.h"
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Helper function for CPU timing
template<typename F, typename... Args>
double measure_time(F func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// CUDA error checking wrapper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // Matrix size and parameters
    int size = 10000;
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
        solution_cpu = gauss_seidel_red_black(sparse_matrix, b, max_iterations, tolerance);
    });

    // GPU Gauss-Seidel
    std::vector<double> solution_gpu;
    float gpu_solve_time;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Run GPU Gauss-Seidel
    solution_gpu = gauss_seidel_red_black_cuda(sparse_matrix, b, max_iterations, tolerance);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    CUDA_CHECK(cudaEventElapsedTime(&gpu_solve_time, start, stop));
    gpu_solve_time /= 1000.0f; // Convert ms to seconds

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Compute residuals
    double residual_cpu, residual_gpu;
    double cpu_residual_time = measure_time([&]() {
        residual_cpu = compute_residual(sparse_matrix, solution_cpu, b);
    });

    double gpu_residual_time = measure_time([&]() {
        residual_gpu = compute_residual_cuda(sparse_matrix, solution_gpu, b);
    });

    // Print results
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "CPU Red-Black Gauss-Seidel:" << std::endl;
    std::cout << "  Solve time: " << cpu_solve_time << " seconds" << std::endl;
    std::cout << "  Residual ||Ax - b|| = " << residual_cpu << std::endl;

    std::cout << "\nGPU Red-Black Gauss-Seidel:" << std::endl;
    std::cout << "  Solve time: " << gpu_solve_time << " seconds" << std::endl;
    std::cout << "  Residual ||Ax - b|| = " << residual_gpu << std::endl;

    // Compare solutions
    double max_diff = 0.0;
    for (size_t i = 0; i < solution_cpu.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(solution_cpu[i] - solution_gpu[i]));
    }
    std::cout << "\nMaximum difference between CPU and GPU solutions: " << max_diff << std::endl;

    // Print speedup
    double cpu_total_time = cpu_solve_time;
    double gpu_total_time = gpu_solve_time;
    double speedup = cpu_total_time / gpu_total_time;
    std::cout << "\nGPU Speedup: " << speedup << "x" << std::endl;

    return 0;
}
