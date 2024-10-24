#ifndef KERNELS_H
#define KERNELS_H
#include "switch_gpu_backend.h"
// Kernel function declarations

__global__ void gsrb_subloop(
    const int                     kinit,
    const int                     kfinal,
    const int*       __restrict__ redblack_indices,
    const int*       __restrict__ i,
    const int*       __restrict__ j,
    const double*    __restrict__ val,
    const double*    __restrict__ dinv,
    double*          __restrict__ x,
    const double*    __restrict__ b);

__global__ void gsrb_subloop_contiguous(
    const int                     kinit,
    const int                     kfinal,
    const int*       __restrict__ i,
    const int*       __restrict__ j,
    const double*    __restrict__ val,
    const double*    __restrict__ dinv,
    double*          __restrict__ x,
    const double*    __restrict__ b);

#endif // KERNELS_H
