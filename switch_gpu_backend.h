#ifndef SWITCH_GPU_BACKEND_H
#define SWITCH_GPU_BACKEND_H

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// Redefine CUDA types to HIP types
#ifdef USE_HIP
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t

// Redefine CUDA functions to HIP functions

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy

// Redefine CUDA kernel launch syntax
// #define <<<gridSize, blockSize>>> hipLaunchKernelGGL


// Redefine CUDA device functions
#define __device__ __device__
#define __global__ __global__
#define __host__ __host__

// Redefine CUDA synchronization functions
#define cudaDeviceSynchronize hipDeviceSynchronize

// Redefine CUDA math functions (if needed)

#define __syncthreads() __syncthreads()
#endif

// Error checking macro
#ifdef USE_HIP
#define GPU_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << hipGetErrorString(error) << std::endl; \
            throw std::runtime_error("HIP error"); \
        } \
    } while(0)
#else
#define GPU_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error("CUDA error"); \
        } \
    } while(0)
#endif

// Add more redefinitions as needed for your specific use case

#endif //SWITCH_GPU_BACKEND
