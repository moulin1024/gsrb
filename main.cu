#include "switch_gpu_backend.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <omp.h> // For OpenMP parallelization in compute_residual

// Function to load binary data from files
void load_binary_data(const char* filename, void* data, size_t size) {
    FILE* file = fopen(filename, "rb");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t read_size = fread(data, 1, size, file);
    if (read_size != size) {
        std::cerr << "Error reading file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    fclose(file);
}

// Modified kernel function with contiguous data access
__global__ void gsrb_subloop_contiguous(
    const int                     row_start,
    const int                     row_end,
    const int*       __restrict__ i,
    const int*       __restrict__ j,
    const double*    __restrict__ val,
    const double*    __restrict__ dinv,
    double*          __restrict__ x,
    const double*    __restrict__ b)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ll = idx + row_start;

    if (ll < row_end) {
        double inner_product = 0.0;
        int row_start_idx = i[ll] - 1;
        int row_end_idx = i[ll + 1] - 1;

        #pragma unroll 5
        for (int mm = row_start_idx; mm < row_end_idx; ++mm) {
            inner_product += val[mm] * x[j[mm] - 1];
        }

        // Update x[ll]
        x[ll] += (b[ll] - inner_product) * dinv[ll];
    }
}

// Function to compute the residual on CPU
double compute_residual(const int* i, const int* j, const double* val, const double* x, const double* b, int n_points) {
    double residual = 0.0;
    #pragma omp parallel for reduction(+:residual)
    for (int row = 0; row < n_points; ++row) {
        double sum = 0.0;
        for (int idx = i[row] - 1; idx < i[row + 1] - 1; ++idx) {
            sum += val[idx] * x[j[idx] - 1];
        }
        double r = b[row] - sum;
        residual += r * r;
    }
    return std::sqrt(residual);
}

int main()
{
    const int blockSize = 256;
    const int nnz = 56825529;
    const int n_points = 11437831;
    const int n_points_red = 5666455;
    const int n_points_black = 5666400;
    const int n_points_edge = n_points - (n_points_red + n_points_black);

    // Declare host variables
    int *h_redblack_indices, *h_i, *h_j;
    double *h_val, *h_dinv, *h_u, *h_b;

    // Allocate host memory
    h_redblack_indices = new int[n_points];
    h_i = new int[n_points + 1];
    h_j = new int[nnz];
    h_val = new double[nnz];
    h_dinv = new double[n_points];
    h_u = new double[n_points];
    h_b = new double[n_points];

    // Load data from binary files
    load_binary_data("mat/redblack_indices.bin", h_redblack_indices, n_points * sizeof(int));
    load_binary_data("mat/i.bin", h_i, (n_points + 1) * sizeof(int));
    load_binary_data("mat/j.bin", h_j, nnz * sizeof(int));
    load_binary_data("mat/val.bin", h_val, nnz * sizeof(double));
    load_binary_data("mat/dinv.bin", h_dinv, n_points * sizeof(double));
    load_binary_data("mat/b.bin", h_b, n_points * sizeof(double));
    std::fill(h_u, h_u + n_points, 0.0);

    // Create permutation mappings
    int* old_to_new = new int[n_points]; // Maps old indices to new indices
    int* new_to_old = new int[n_points]; // Maps new indices to old indices

    int red_count = 0;
    int black_count = n_points_red;
    int edge_count = n_points_red + n_points_black;

    // Process red, black, and edge nodes
    for (int kk = 0; kk < n_points; ++kk) {
        int old_idx = h_redblack_indices[kk] - 1;
        if (kk < n_points_red) {
            // Red nodes
            old_to_new[old_idx] = red_count;
            new_to_old[red_count] = old_idx;
            red_count++;
        } else if (kk < n_points_red + n_points_black) {
            // Black nodes
            old_to_new[old_idx] = black_count;
            new_to_old[black_count] = old_idx;
            black_count++;
        } else {
            // Edge nodes
            old_to_new[old_idx] = edge_count;
            new_to_old[edge_count] = old_idx;
            edge_count++;
        }
    }

    // Verify counts
    if (edge_count != n_points) {
        std::cerr << "Error in permutation mapping: counts do not match total number of points." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Reorder vectors
    double* h_x_reordered = new double[n_points];
    double* h_b_reordered = new double[n_points];
    double* h_dinv_reordered = new double[n_points];

    for (int idx = 0; idx < n_points; ++idx) {
        int old_idx = new_to_old[idx];
        h_x_reordered[idx] = h_u[old_idx];
        h_b_reordered[idx] = h_b[old_idx];
        h_dinv_reordered[idx] = h_dinv[old_idx];
    }

    // Reorder CSR matrix
    int* h_i_reordered = new int[n_points + 1];
    int* h_j_reordered = new int[nnz];
    double* h_val_reordered = new double[nnz];

    int nnz_counter = 0;
    h_i_reordered[0] = 1; // CSR format uses 1-based indexing

    for (int new_row = 0; new_row < n_points; ++new_row) {
        int old_row = new_to_old[new_row];
        int row_start = h_i[old_row] - 1;
        int row_end = h_i[old_row + 1] - 1;

        for (int idx = row_start; idx < row_end; ++idx) {
            int old_col = h_j[idx] - 1;
            int new_col = old_to_new[old_col];

            h_j_reordered[nnz_counter] = new_col + 1;
            h_val_reordered[nnz_counter] = h_val[idx];
            nnz_counter++;
        }

        h_i_reordered[new_row + 1] = nnz_counter + 1;
    }

    // Declare device variables
    int *d_i, *d_j;
    double *d_val, *d_dinv, *d_u, *d_b;

    // Allocate device memory
    cudaMalloc(&d_i, (n_points + 1) * sizeof(int));
    cudaMalloc(&d_j, nnz * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(double));
    cudaMalloc(&d_dinv, n_points * sizeof(double));
    cudaMalloc(&d_u, n_points * sizeof(double));
    cudaMalloc(&d_b, n_points * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_i, h_i_reordered, (n_points + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_j, h_j_reordered, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val_reordered, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dinv, h_dinv_reordered, n_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_x_reordered, n_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b_reordered, n_points * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate device memory for the old solution vector
    double* d_u_old;
    cudaMalloc(&d_u_old, n_points * sizeof(double));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Compute initial residual
    double initial_residual = compute_residual(h_i_reordered, h_j_reordered, h_val_reordered, h_x_reordered, h_b_reordered, n_points);
    std::cout << "Initial residual: " << std::setprecision(12) << initial_residual << std::endl;

    // Start timing
    cudaEventRecord(start);
    int loop_count = 2000;
    double omega = 1.4; // Define relaxation parameter

    for (int iter = 0; iter < loop_count; ++iter) {
        // Copy d_u to d_u_old (device to device)
        cudaMemcpy(d_u_old, d_u, n_points * sizeof(double), cudaMemcpyDeviceToDevice);

        // Red nodes
        int numBlocks = (n_points_red + blockSize - 1) / blockSize;
        gsrb_subloop_contiguous<<<numBlocks, blockSize>>>(
            0,
            n_points_red,
            d_i,
            d_j,
            d_val,
            d_dinv,
            d_u,
            d_b);

        // Black nodes
        numBlocks = (n_points_black + blockSize - 1) / blockSize;
        gsrb_subloop_contiguous<<<numBlocks, blockSize>>>(
            n_points_red,
            n_points_red + n_points_black,
            d_i,
            d_j,
            d_val,
            d_dinv,
            d_u,
            d_b);

        // Edge nodes
        numBlocks = (n_points_edge + blockSize - 1) / blockSize;
        gsrb_subloop_contiguous<<<numBlocks, blockSize>>>(
            n_points_red + n_points_black,
            n_points,
            d_i,
            d_j,
            d_val,
            d_dinv,
            d_u,
            d_b);

        // Apply relaxation
        numBlocks = (n_points + blockSize - 1) / blockSize;
        relaxation_kernel<<<numBlocks, blockSize>>>(
            d_u,
            d_u_old,
            omega,
            n_points);
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU execution time per iteration: " << milliseconds / loop_count << " ms" << std::endl;

    // Calculate and print FLOPs
    double avg_nnz_per_row = static_cast<double>(nnz) / n_points;
    double flops_per_iteration = (2 * avg_nnz_per_row) + 3;
    double total_flops = n_points * flops_per_iteration;
    double gflops = total_flops / 1e9;
    double gflops_per_second = gflops / (milliseconds / 1000.0 / loop_count);

    std::cout << "Estimated GFLOP/s: " << gflops_per_second << std::endl;

    // Copy results back to host
    cudaMemcpy(h_x_reordered, d_u, n_points * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute final residual
    double final_residual = compute_residual(h_i_reordered, h_j_reordered, h_val_reordered, h_x_reordered, h_b_reordered, n_points);
    std::cout << "Final residual: " << std::setprecision(12) << final_residual << std::endl;

    // Free device memory
    cudaFree(d_i);
    cudaFree(d_j);
    cudaFree(d_val);
    cudaFree(d_dinv);
    cudaFree(d_u);
    cudaFree(d_b);
    cudaFree(d_u_old);

    // Free host memory
    delete[] h_redblack_indices;
    delete[] h_i;
    delete[] h_j;
    delete[] h_val;
    delete[] h_dinv;
    delete[] h_u;
    delete[] h_b;

    delete[] old_to_new;
    delete[] new_to_old;

    delete[] h_i_reordered;
    delete[] h_j_reordered;
    delete[] h_val_reordered;
    delete[] h_dinv_reordered;
    delete[] h_x_reordered;
    delete[] h_b_reordered;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}
