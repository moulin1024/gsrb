#include "kernels.h"

// Original kernel function with strided memory access
__global__ void gsrb_subloop(
    const int        kinit,
    const int        kfinal,
    const int* __restrict__ redblack_indices,
    const int* __restrict__ i,
    const int* __restrict__ j,
    const double*    __restrict__ val,
    const double*    __restrict__ dinv,
    double*          __restrict__ x,
    const double*    __restrict__ b)
{
    const int kk0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int dkk = blockDim.x * gridDim.x;
    for (auto kk = kk0 + kinit - 1; kk < kfinal; kk += dkk) {
        const int ll            = redblack_indices[kk] - 1;
        double    inner_product = 0.0;
        for (auto mm = i[ll] - 1; mm < i[ll + 1] - 1; ++mm) {
            inner_product += val[mm] * x[j[mm] - 1];
        }
        x[ll] += (b[ll] - inner_product) * dinv[ll];
    }
}


// Optimized kernel function with contiguous data access
__global__ void gsrb_subloop_contiguous(
    const int                     kinit,
    const int                     kfinal,
    const int*       __restrict__ i,
    const int*       __restrict__ j,
    const double*    __restrict__ val,
    const double*    __restrict__ dinv,
    double*          __restrict__ x,
    const double*    __restrict__ b)
{
    const int kk0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int dkk = gridDim.x * blockDim.x;
    
    for (int kk = kk0 + kinit - 1; kk < kfinal; kk += dkk) {
        const int ll = kk + 1; // Continuous indexing
        double inner_product = 0.0;
        const int row_start = i[ll] - 1;
        const int row_end = i[ll + 1] - 1;

        #pragma unroll 5
        for (int mm = row_start; mm < row_end; ++mm) {
            inner_product += val[mm] * x[j[mm] - 1];
        }

        // Update x[ll]
        x[ll] += (b[ll] - inner_product) * dinv[ll];
    }
}
