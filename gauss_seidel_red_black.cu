#include "gauss_seidel_red_black.h"
#include "switch_gpu_backend.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

// CUDA kernel for Red-Black Gauss-Seidel iteration
__global__ void gauss_seidel_red_black_kernel(int* row_ptr, int* col_ind, double* values, double* x, double* x_new, double* b, double* diagonal_inv, int n, int offset, double* max_diff) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + offset;
    if (i < n) {
        double sigma = 0.0;

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_ind[j] != i) {
                sigma += values[j] * x[col_ind[j]];
            }
        }
        x_new[i] = (b[i] - sigma) * diagonal_inv[i];  // Use the inverted diagonal element
    }
}

// CUDA kernel for residual computation
__global__ void compute_residual_kernel(int* row_ptr, int* col_ind, double* values, double* x, double* b, double* residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double row_sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            row_sum += values[j] * x[col_ind[j]];
        }
        residual[i] = b[i] - row_sum;
    }
}

std::vector<double> gauss_seidel_red_black_cpu(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance) {
    int n = A.rows;
    std::vector<double> x(n, 0.0); // Initial guess
    std::vector<double> x_new(n, 0.0);

    for (int iter = 0; iter < max_iterations; ++iter) {
        double max_diff = 0.0;

        // Red sweep
        for (int i = 0; i < n; i += 2) {
            double sigma = 0.0;
            double a_ii = 0.0;

            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                int col = A.col_ind[j];
                double val = A.values[j];
                if (col == i) {
                    a_ii = val;
                } else {
                    sigma += val * x[col];
                }
            }

            if (a_ii == 0.0) {
                std::cerr << "Zero diagonal element detected at row " << i << ". Cannot proceed." << std::endl;
                exit(EXIT_FAILURE);
            }

            x_new[i] = (b[i] - sigma) / a_ii;
            double diff = std::abs(x_new[i] - x[i]);
            if (diff > max_diff) max_diff = diff;
        }

        // Update x with red values
        for (int i = 0; i < n; i += 2) {
            x[i] = x_new[i];
        }

        // Black sweep
        for (int i = 1; i < n; i += 2) {
            double sigma = 0.0;
            double a_ii = 0.0;

            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                int col = A.col_ind[j];
                double val = A.values[j];
                if (col == i) {
                    a_ii = val;
                } else {
                    sigma += val * x[col];
                }
            }

            if (a_ii == 0.0) {
                std::cerr << "Zero diagonal element detected at row " << i << ". Cannot proceed." << std::endl;
                exit(EXIT_FAILURE);
            }

            x_new[i] = (b[i] - sigma) / a_ii;
            double diff = std::abs(x_new[i] - x[i]);
            if (diff > max_diff) max_diff = diff;
        }

        // Update x with black values
        for (int i = 1; i < n; i += 2) {
            x[i] = x_new[i];
        }

        // Check for convergence
        if (max_diff < tolerance) {
            std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            return x;
        }
    }

    std::cout << "Reached maximum iterations without full convergence." << std::endl;
    return x;
}

// Host function for CUDA Red-Black Gauss-Seidel solver
std::vector<double> gauss_seidel_red_black_gpu(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance) {
    int n = A.rows;
    std::vector<double> x(n, 0.0); // Initial guess

    // Allocate device memory
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_x, *d_x_new, *d_b, *d_max_diff, *d_diagonal_inv;
    GPU_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_col_ind, A.col_ind.size() * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_values, A.values.size() * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_x_new, n * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_max_diff, sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_diagonal_inv, n * sizeof(double)));  // Allocate memory for inverted diagonal

    // Copy data to device
    GPU_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_col_ind, A.col_ind.data(), A.col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_values, A.values.data(), A.values.size() * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_x_new, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_diagonal_inv, A.diagonal_inv.data(), n * sizeof(double), cudaMemcpyHostToDevice));  // Copy inverted diagonal

    // Set up kernel launch parameters
    int block_size = 64;
    int num_blocks = (n / 2 + block_size - 1) / block_size;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    for (int iter = 0; iter < 10; ++iter) {
        double max_diff = 0.0;
        GPU_CHECK(cudaMemcpy(d_max_diff, &max_diff, sizeof(double), cudaMemcpyHostToDevice));

        // Red sweep
        gauss_seidel_red_black_kernel<<<num_blocks, block_size>>>(d_row_ptr, d_col_ind, d_values, d_x, d_x_new, d_b, d_diagonal_inv, n, 0, d_max_diff);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());

        // Black sweep
        gauss_seidel_red_black_kernel<<<num_blocks, block_size>>>(d_row_ptr, d_col_ind, d_values, d_x, d_x_new, d_b, d_diagonal_inv, n, 1, d_max_diff);
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());

        // Update x
        GPU_CHECK(cudaMemcpy(d_x, d_x_new, n * sizeof(double), cudaMemcpyDeviceToDevice));

        GPU_CHECK(cudaMemcpy(&max_diff, d_max_diff, sizeof(double), cudaMemcpyDeviceToHost));

        // if (max_diff < tolerance) {
        //     std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
        //     break;
        // }
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the solving time
    std::cout << "GPU solving time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    GPU_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    GPU_CHECK(cudaFree(d_row_ptr));
    GPU_CHECK(cudaFree(d_col_ind));
    GPU_CHECK(cudaFree(d_values));
    GPU_CHECK(cudaFree(d_x));
    GPU_CHECK(cudaFree(d_x_new));
    GPU_CHECK(cudaFree(d_b));
    GPU_CHECK(cudaFree(d_max_diff));
    GPU_CHECK(cudaFree(d_diagonal_inv));  // Free the inverted diagonal memory

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return x;
}

double compute_residual_cpu(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> residual(b.size(), 0.0);
    
    for (int i = 0; i < A.rows; ++i) {
        double row_sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += A.values[j] * x[A.col_ind[j]];
        }
        residual[i] = b[i] - row_sum;
    }

    double norm = 0.0;
    for (double r : residual) {
        norm += r * r;
    }
    return std::sqrt(norm);
}

// Host function for CUDA residual computation
double compute_residual_gpu(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    int n = A.rows;
    std::vector<double> residual(n, 0.0);

    // Allocate device memory
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_x, *d_b, *d_residual;
    GPU_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_col_ind, A.col_ind.size() * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_values, A.values.size() * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    GPU_CHECK(cudaMalloc(&d_residual, n * sizeof(double)));

    // Copy data to device
    GPU_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_col_ind, A.col_ind.data(), A.col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_values, A.values.data(), A.values.size() * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Launch kernel to compute residual
    compute_residual_kernel<<<num_blocks, block_size>>>(d_row_ptr, d_col_ind, d_values, d_x, d_b, d_residual, n);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    GPU_CHECK(cudaMemcpy(residual.data(), d_residual, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute norm on CPU
    double norm = 0.0;
    for (double r : residual) {
        norm += r * r;
    }

    // Free device memory
    GPU_CHECK(cudaFree(d_row_ptr));
    GPU_CHECK(cudaFree(d_col_ind));
    GPU_CHECK(cudaFree(d_values));
    GPU_CHECK(cudaFree(d_x));
    GPU_CHECK(cudaFree(d_b));
    GPU_CHECK(cudaFree(d_residual));

    return std::sqrt(norm);
}
