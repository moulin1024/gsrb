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

// Add this function to compare results
void compare_results(const std::vector<double>& x1, const std::vector<double>& x2, double tolerance) {
    if (x1.size() != x2.size()) {
        std::cout << "Error: Solutions have different sizes." << std::endl;
        return;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        double diff = std::abs(x1[i] - x2[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Maximum difference between solutions: " << max_diff << std::endl;
    if (max_diff < tolerance) {
        std::cout << "Solutions are consistent within tolerance." << std::endl;
    } else {
        std::cout << "Solutions differ more than the specified tolerance." << std::endl;
    }
}

int main() {
    // Matrix size and parameters
    int size = 8192;
    double density = 0.5;
    const double min_value = -1.0;
    const double max_value = 1.0;

    // Generate sparse matrix
    CSRMatrix sparse_matrix = generate_sparse_diagonal_matrix(size, density, min_value, max_value);

    // Generate random vector b
    std::vector<double> b = generate_random_vector(sparse_matrix.rows, -10.0, 10.0);

    // Solve Ax = b using Gauss-Seidel method
    int max_iterations = 1000;
    double tolerance = 1e-9;

    // Original GPU Gauss-Seidel
    std::vector<double> solution_gpu_original;
    
    // Run original GPU Gauss-Seidel
    auto start_original = std::chrono::high_resolution_clock::now();
    solution_gpu_original = gauss_seidel_red_black_gpu(sparse_matrix, b, max_iterations, tolerance);
    auto end_original = std::chrono::high_resolution_clock::now();
    auto duration_original = std::chrono::duration_cast<std::chrono::milliseconds>(end_original - start_original);

    // New GPU Gauss-Seidel
    std::vector<double> solution_gpu_new;
    
    // Run new GPU Gauss-Seidel
    auto start_new = std::chrono::high_resolution_clock::now();
    solution_gpu_new = gauss_seidel_red_black_gpu_new(sparse_matrix, b, max_iterations, tolerance);
    auto end_new = std::chrono::high_resolution_clock::now();
    auto duration_new = std::chrono::duration_cast<std::chrono::milliseconds>(end_new - start_new);

    // Compute residuals
    double residual_gpu_original = compute_residual_gpu(sparse_matrix, solution_gpu_original, b);
    double residual_gpu_new = compute_residual_gpu(sparse_matrix, solution_gpu_new, b);

    // Check results
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "\nOriginal GPU Red-Black Gauss-Seidel:" << std::endl;
    std::cout << "  Residual ||Ax - b|| = " << residual_gpu_original << std::endl;
    std::cout << "  Solving time: " << duration_original.count() << " ms" << std::endl;

    std::cout << "\nNew GPU Red-Black Gauss-Seidel:" << std::endl;
    std::cout << "  Residual ||Ax - b|| = " << residual_gpu_new << std::endl;
    std::cout << "  Solving time: " << duration_new.count() << " ms" << std::endl;

    // Compare results
    std::cout << "\nComparing results:" << std::endl;
    compare_results(solution_gpu_original, solution_gpu_new, 1e-6);

    return 0;
}
