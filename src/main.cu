// main.cu

#include "kernels.h"
#include "definition.h"
#include "data_loader.h"
#include "utils.h"
#include "switch_gpu_backend.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <omp.h>       // For OpenMP parallelization in compute_residual
#include <algorithm>   // For std::copy, std::fill

int main()
{
    // Load data for both versions
    int* h_redblack_indices = new int[n_points];
    int* h_i = new int[n_points + 1];
    int* h_j = new int[nnz];
    double* h_val = new double[nnz];
    double* h_dinv = new double[n_points];
    double* h_u = new double[n_points];
    double* h_b = new double[n_points];

    load_binary_data("mat/redblack_indices.bin", h_redblack_indices, n_points * sizeof(int));
    load_binary_data("mat/i.bin", h_i, (n_points + 1) * sizeof(int));
    load_binary_data("mat/j.bin", h_j, nnz * sizeof(int));
    load_binary_data("mat/val.bin", h_val, nnz * sizeof(double));
    load_binary_data("mat/dinv.bin", h_dinv, n_points * sizeof(double));
    load_binary_data("mat/b.bin", h_b, n_points * sizeof(double));
    std::fill(h_u, h_u + n_points, 0.0);

    // Copy data for original version (strided access)
    int* h_redblack_indices_orig = new int[n_points];
    int* h_i_orig = new int[n_points + 1];
    int* h_j_orig = new int[nnz];
    double* h_val_orig = new double[nnz];
    double* h_dinv_orig = new double[n_points];
    double* h_u_orig = new double[n_points];
    double* h_b_orig = new double[n_points];

    std::copy(h_redblack_indices, h_redblack_indices + n_points, h_redblack_indices_orig);
    std::copy(h_i, h_i + n_points + 1, h_i_orig);
    std::copy(h_j, h_j + nnz, h_j_orig);
    std::copy(h_val, h_val + nnz, h_val_orig);
    std::copy(h_dinv, h_dinv + n_points, h_dinv_orig);
    std::copy(h_u, h_u + n_points, h_u_orig);
    std::copy(h_b, h_b + n_points, h_b_orig);

    // Copy data for optimized version (contiguous access)
    int* h_i_reordered = new int[n_points + 1];
    int* h_j_reordered = new int[nnz];
    double* h_val_reordered = new double[nnz];
    double* h_dinv_reordered = new double[n_points];
    double* h_u_reordered = new double[n_points];
    double* h_b_reordered = new double[n_points];

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
    for (int idx = 0; idx < n_points; ++idx) {
        int old_idx = new_to_old[idx];
        h_u_reordered[idx] = h_u[old_idx];
        h_b_reordered[idx] = h_b[old_idx];
        h_dinv_reordered[idx] = h_dinv[old_idx];
    }

    // Reorder CSR matrix
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

    // Variables for timing and performance
    float milliseconds_orig = 0, milliseconds_opt = 0;
    double gflops_per_second_orig = 0, gflops_per_second_opt = 0;
    double final_residual_orig = 0, final_residual_opt = 0;

    // ------------------------
    // Original Version (Strided Access)
    // ------------------------

    {
        std::cout << "\nRunning Original Version (Strided Access)...\n";

        // Declare device variables
        int* d_redblack_indices, * d_i, * d_j;
        double* d_val, * d_dinv, * d_u, * d_b, * d_u_old;

        // Allocate device memory
        cudaMalloc(&d_redblack_indices, n_points * sizeof(int));
        cudaMalloc(&d_i, (n_points + 1) * sizeof(int));
        cudaMalloc(&d_j, nnz * sizeof(int));
        cudaMalloc(&d_val, nnz * sizeof(double));
        cudaMalloc(&d_dinv, n_points * sizeof(double));
        cudaMalloc(&d_u, n_points * sizeof(double));
        cudaMalloc(&d_b, n_points * sizeof(double));
        cudaMalloc(&d_u_old, n_points * sizeof(double));

        // Copy data from host to device
        cudaMemcpy(d_redblack_indices, h_redblack_indices_orig, n_points * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_i, h_i_orig, (n_points + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_j, h_j_orig, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, h_val_orig, nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dinv, h_dinv_orig, n_points * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u, h_u_orig, n_points * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b_orig, n_points * sizeof(double), cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEvent_t start_orig, stop_orig;
        cudaEventCreate(&start_orig);
        cudaEventCreate(&stop_orig);

        // Compute initial residual
        double initial_residual_orig = compute_residual(h_i_orig, h_j_orig, h_val_orig, h_u_orig, h_b_orig, n_points);
        std::cout << "Initial residual (Original): " << std::setprecision(12) << initial_residual_orig << std::endl;

        // Start timing
        cudaEventRecord(start_orig);

        for (int iter = 0; iter < loop_count; ++iter) {
            // Copy d_u to d_u_old (device to device)
            cudaMemcpy(d_u_old, d_u, n_points * sizeof(double), cudaMemcpyDeviceToDevice);

            // Red loop
            int numBlocks = (n_points_red + blockSize - 1) / blockSize;
            gsrb_subloop<<<numBlocks, blockSize>>>(
                1,
                n_points_red,
                d_redblack_indices,
                d_i,
                d_j,
                d_val,
                d_dinv,
                d_u,
                d_b);

            // Black loop
            numBlocks = (n_points_black + blockSize - 1) / blockSize;
            gsrb_subloop<<<numBlocks, blockSize>>>(
                n_points_red,
                n_points_red + n_points_black,
                d_redblack_indices,
                d_i,
                d_j,
                d_val,
                d_dinv,
                d_u,
                d_b);

            // Edge loop
            numBlocks = (n_points_edge + blockSize - 1) / blockSize;
            gsrb_subloop<<<numBlocks, blockSize>>>(
                n_points_red + n_points_black,
                n_points,
                d_redblack_indices,
                d_i,
                d_j,
                d_val,
                d_dinv,
                d_u,
                d_b);
        }

        // Stop timing
        cudaEventRecord(stop_orig);
        cudaEventSynchronize(stop_orig);

        // Calculate elapsed time
        cudaEventElapsedTime(&milliseconds_orig, start_orig, stop_orig);
        std::cout << "GPU execution time per iteration (Original): " << milliseconds_orig / loop_count << " ms" << std::endl;

        // Calculate FLOPs
        double avg_nnz_per_row = static_cast<double>(nnz) / n_points;
        double flops_per_iteration = (2 * avg_nnz_per_row) + 3;
        double total_flops = n_points * flops_per_iteration;
        double gflops = total_flops / 1e9;
        gflops_per_second_orig = gflops / (milliseconds_orig / 1000.0 / loop_count);

        std::cout << "Estimated GFLOP/s (Original): " << gflops_per_second_orig << std::endl;

        // Copy results back to host
        cudaMemcpy(h_u_orig, d_u, n_points * sizeof(double), cudaMemcpyDeviceToHost);

        // Compute final residual
        final_residual_orig = compute_residual(h_i_orig, h_j_orig, h_val_orig, h_u_orig, h_b_orig, n_points);
        std::cout << "Final residual (Original): " << std::setprecision(12) << final_residual_orig << std::endl;

        // Free device memory
        cudaFree(d_redblack_indices);
        cudaFree(d_i);
        cudaFree(d_j);
        cudaFree(d_val);
        cudaFree(d_dinv);
        cudaFree(d_u);
        cudaFree(d_b);
        cudaFree(d_u_old);

        // Destroy CUDA events
        cudaEventDestroy(start_orig);
        cudaEventDestroy(stop_orig);
    }

    // ------------------------
    // Optimized Version (Contiguous Access)
    // ------------------------

    {
        std::cout << "\nRunning Optimized Version (Contiguous Access)...\n";

        // Declare device variables
        int* d_i, * d_j;
        double* d_val, * d_dinv, * d_u, * d_b, * d_u_old;

        // Allocate device memory
        cudaMalloc(&d_i, (n_points + 1) * sizeof(int));
        cudaMalloc(&d_j, nnz * sizeof(int));
        cudaMalloc(&d_val, nnz * sizeof(double));
        cudaMalloc(&d_dinv, n_points * sizeof(double));
        cudaMalloc(&d_u, n_points * sizeof(double));
        cudaMalloc(&d_b, n_points * sizeof(double));
        cudaMalloc(&d_u_old, n_points * sizeof(double));

        // Copy data from host to device
        cudaMemcpy(d_i, h_i_reordered, (n_points + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_j, h_j_reordered, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val, h_val_reordered, nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dinv, h_dinv_reordered, n_points * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_u, h_u_reordered, n_points * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b_reordered, n_points * sizeof(double), cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEvent_t start_opt, stop_opt;
        cudaEventCreate(&start_opt);
        cudaEventCreate(&stop_opt);

        // Compute initial residual
        double initial_residual_opt = compute_residual(h_i_reordered, h_j_reordered, h_val_reordered, h_u_reordered, h_b_reordered, n_points);
        std::cout << "Initial residual (Optimized): " << std::setprecision(12) << initial_residual_opt << std::endl;

        // Start timing
        cudaEventRecord(start_opt);

        for (int iter = 0; iter < loop_count; ++iter) {
            // Copy d_u to d_u_old (device to device)
            cudaMemcpy(d_u_old, d_u, n_points * sizeof(double), cudaMemcpyDeviceToDevice);

            // Red nodes
            int numBlocks = (n_points_red + blockSize - 1) / blockSize;
            gsrb_subloop_contiguous<<<numBlocks, blockSize>>>(
                1,
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
                n_points_red + 1,
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
                n_points_red + n_points_black + 1,
                n_points,
                d_i,
                d_j,
                d_val,
                d_dinv,
                d_u,
                d_b);
        }

        // Stop timing
        cudaEventRecord(stop_opt);
        cudaEventSynchronize(stop_opt);

        // Calculate elapsed time
        cudaEventElapsedTime(&milliseconds_opt, start_opt, stop_opt);
        std::cout << "GPU execution time per iteration (Optimized): " << milliseconds_opt / loop_count << " ms" << std::endl;

        // Calculate FLOPs
        double avg_nnz_per_row = static_cast<double>(nnz) / n_points;
        double flops_per_iteration = (2 * avg_nnz_per_row) + 3;
        double total_flops = n_points * flops_per_iteration;
        double gflops = total_flops / 1e9;
        gflops_per_second_opt = gflops / (milliseconds_opt / 1000.0 / loop_count);

        std::cout << "Estimated GFLOP/s (Optimized): " << gflops_per_second_opt << std::endl;

        // Copy results back to host
        cudaMemcpy(h_u_reordered, d_u, n_points * sizeof(double), cudaMemcpyDeviceToHost);

        // Permute solution back to original order for comparison
        double* h_u_opt_original_order = new double[n_points];
        permute_back(h_u_reordered, h_u_opt_original_order, new_to_old, n_points);

        // Compute final residual
        final_residual_opt = compute_residual(h_i_orig, h_j_orig, h_val_orig, h_u_opt_original_order, h_b_orig, n_points);
        std::cout << "Final residual (Optimized): " << std::setprecision(12) << final_residual_opt << std::endl;

        // Free device memory
        cudaFree(d_i);
        cudaFree(d_j);
        cudaFree(d_val);
        cudaFree(d_dinv);
        cudaFree(d_u);
        cudaFree(d_b);
        cudaFree(d_u_old);

        // Destroy CUDA events
        cudaEventDestroy(start_opt);
        cudaEventDestroy(stop_opt);

        // Free temporary host memory
        delete[] h_u_opt_original_order;
    }

    // ------------------------
    // Comparison Results
    // ------------------------

    std::cout << "\n=== Comparison Results ===\n";
    std::cout << "Execution Time per Iteration:\n";
    std::cout << "Original Version: " << milliseconds_orig / loop_count << " ms\n";
    std::cout << "Optimized Version: " << milliseconds_opt / loop_count << " ms\n";

    std::cout << "\nEstimated GFLOP/s:\n";
    std::cout << "Original Version: " << gflops_per_second_orig << "\n";
    std::cout << "Optimized Version: " << gflops_per_second_opt << "\n";

    std::cout << "\nFinal Residuals:\n";
    std::cout << "Original Version: " << std::setprecision(12) << final_residual_orig << "\n";
    std::cout << "Optimized Version: " << std::setprecision(12) << final_residual_opt << "\n";

    // Compare solutions (difference between the two)
    double solution_diff = 0.0;
    for (int idx = 0; idx < n_points; ++idx) {
        double diff = h_u_orig[idx] - h_u_reordered[old_to_new[idx]];
        solution_diff += diff * diff;
    }
    solution_diff = std::sqrt(solution_diff);
    std::cout << "\nSolution Difference (L2 Norm): " << std::setprecision(12) << solution_diff << "\n";

    // Free host memory
    delete[] h_redblack_indices;
    delete[] h_i;
    delete[] h_j;
    delete[] h_val;
    delete[] h_dinv;
    delete[] h_u;
    delete[] h_b;

    delete[] h_redblack_indices_orig;
    delete[] h_i_orig;
    delete[] h_j_orig;
    delete[] h_val_orig;
    delete[] h_dinv_orig;
    delete[] h_u_orig;
    delete[] h_b_orig;

    delete[] h_i_reordered;
    delete[] h_j_reordered;
    delete[] h_val_reordered;
    delete[] h_dinv_reordered;
    delete[] h_u_reordered;
    delete[] h_b_reordered;

    delete[] old_to_new;
    delete[] new_to_old;

    return EXIT_SUCCESS;
}
