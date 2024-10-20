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

    // GPU Gauss-Seidel
    std::vector<double> solution_gpu;
    
    // Run GPU Gauss-Seidel
    solution_gpu = gauss_seidel_red_black_gpu(sparse_matrix, b, max_iterations, tolerance);

    // Compute residuals
    double residual_gpu;

    residual_gpu = compute_residual_gpu(sparse_matrix, solution_gpu, b);

    // Check results
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "\nGPU Red-Black Gauss-Seidel:" << std::endl;
    std::cout << "  Residual ||Ax - b|| = " << residual_gpu << std::endl;

    return 0;
}
