#include "gauss_seidel.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

// CUDA error checking wrapper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)

// CUDA kernel for Red-Black Gauss-Seidel iteration
__global__ void gauss_seidel_red_black_kernel(int* row_ptr, int* col_ind, double* values, double* x, double* x_new, double* b, int n, int offset, double* max_diff) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + offset;
    if (i < n) {
        double sigma = 0.0;
        double a_ii = 0.0;

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int col = col_ind[j];
            double val = values[j];
            if (col == i) {
                a_ii = val;
            } else {
                sigma += val * x[col];
            }
        }

        if (a_ii == 0.0) {
            printf("Zero diagonal element detected at row %d. Cannot proceed.\n", i);
            return;
        }

        x_new[i] = (b[i] - sigma) / a_ii;
        double diff = fabs(x_new[i] - x[i]);
        atomicMax((unsigned long long int*)max_diff, __double_as_longlong(diff));
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

// Host function for CUDA Red-Black Gauss-Seidel solver
std::vector<double> gauss_seidel_red_black_cuda(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance) {
    int n = A.rows;
    std::vector<double> x(n, 0.0); // Initial guess

    // Allocate device memory
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_x, *d_x_new, *d_b, *d_max_diff;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, A.col_ind.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, A.values.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_new, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_max_diff, sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, A.col_ind.data(), A.col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, A.values.data(), A.values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_new, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    int block_size = 256;
    int num_blocks = (n / 2 + block_size - 1) / block_size;

    for (int iter = 0; iter < max_iterations; ++iter) {
        double max_diff = 0.0;
        CUDA_CHECK(cudaMemcpy(d_max_diff, &max_diff, sizeof(double), cudaMemcpyHostToDevice));

        // Red sweep
        gauss_seidel_red_black_kernel<<<num_blocks, block_size>>>(d_row_ptr, d_col_ind, d_values, d_x, d_x_new, d_b, n, 0, d_max_diff);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Black sweep
        gauss_seidel_red_black_kernel<<<num_blocks, block_size>>>(d_row_ptr, d_col_ind, d_values, d_x, d_x_new, d_b, n, 1, d_max_diff);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update x
        CUDA_CHECK(cudaMemcpy(d_x, d_x_new, n * sizeof(double), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemcpy(&max_diff, d_max_diff, sizeof(double), cudaMemcpyDeviceToHost));

        if (max_diff < tolerance) {
            std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_x_new));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_max_diff));

    return x;
}

// Host function for CUDA residual computation
double compute_residual_cuda(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    int n = A.rows;
    std::vector<double> residual(n, 0.0);

    // Allocate device memory
    int *d_row_ptr, *d_col_ind;
    double *d_values, *d_x, *d_b, *d_residual;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, A.col_ind.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, A.values.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_residual, n * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, A.col_ind.data(), A.col_ind.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, A.values.data(), A.values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Set up kernel launch parameters
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Launch kernel to compute residual
    compute_residual_kernel<<<num_blocks, block_size>>>(d_row_ptr, d_col_ind, d_values, d_x, d_b, d_residual, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(residual.data(), d_residual, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute norm on CPU
    double norm = 0.0;
    for (double r : residual) {
        norm += r * r;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_residual));

    return std::sqrt(norm);
}