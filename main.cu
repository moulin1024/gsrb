#include "switch_gpu_backend.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>

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

__global__ void gsrb_subloop(
    const int                     kinit,
    const int                     kfinal,
    const int*       __restrict__ redblack_indices,
    const int*       __restrict__ i,
    const int*       __restrict__ j,
    const double*    __restrict__ val,
    const double*    __restrict__ dinv,
    double*          __restrict__ x,
    const double*    __restrict__ b)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int kk = idx + kinit;
    
    if (kk < kfinal) {
        const int ll = redblack_indices[kk] - 1;
        double inner_product = 0.0;
        int row_start = i[ll] - 1;
        int row_end = i[ll + 1] - 1;
        #pragma unroll 5
        for (int mm = row_start; mm < row_end; ++mm) {
            inner_product += val[mm] * x[j[mm] - 1];
        }
        // x[ll] += (b[ll] - inner_product) * dinv[ll];
        atomicAdd(&x[ll], (b[ll] - inner_product) * dinv[ll]);
    }
}

int main()
{
    const int blockSize = 256;
    const int nnz = 56825529;
    const int n_points = 11437831;
    const int n_points_red = 5666455;
    const int n_points_black = 5666400;

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

    // Declare device variables
    int *d_redblack_indices, *d_i, *d_j;
    double *d_val, *d_dinv, *d_u, *d_b;

    // Allocate device memory
    cudaMalloc(&d_redblack_indices, n_points * sizeof(int));
    cudaMalloc(&d_i, (n_points + 1) * sizeof(int));
    cudaMalloc(&d_j, nnz * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(double));
    cudaMalloc(&d_dinv, n_points * sizeof(double));
    cudaMalloc(&d_u, n_points * sizeof(double));
    cudaMalloc(&d_b, n_points * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_redblack_indices, h_redblack_indices, n_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i, h_i, (n_points + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_j, h_j, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dinv, h_dinv, n_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, n_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_points * sizeof(double), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);
    int loop_count = 100;

    for (int i = 0; i < loop_count; ++i) {
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
        numBlocks = (n_points - n_points_black - n_points_red + blockSize - 1) / blockSize;
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
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU execution time: " << milliseconds/loop_count << " ms" << std::endl;

    // Calculate and print FLOPs
    double avg_nnz_per_row = static_cast<double>(nnz) / n_points;
    double flops_per_iteration = (2 * avg_nnz_per_row) + 3;
    double total_flops = n_points * flops_per_iteration;
    double gflops = total_flops / 1e9;
    double gflops_per_second = gflops / (milliseconds / 1000.0 / loop_count);

    std::cout << "Estimated GFLOP/s: " << gflops_per_second << std::endl;

    // Copy results back to host (if needed)
    cudaMemcpy(h_u, d_u, n_points * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_redblack_indices);
    cudaFree(d_i);
    cudaFree(d_j);
    cudaFree(d_val);
    cudaFree(d_dinv);
    cudaFree(d_u);
    cudaFree(d_b);

    // Free host memory
    delete[] h_redblack_indices;
    delete[] h_i;
    delete[] h_j;
    delete[] h_val;
    delete[] h_dinv;
    delete[] h_u;
    delete[] h_b;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}
