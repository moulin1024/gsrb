# Gauss-Seidel Red-Black Algorithm Benchmark

This repository contains an implementation and benchmark of the Gauss-Seidel Red-Black algorithm for solving systems of linear equations. The project aims to compare the performance of this algorithm across different hardware platforms and programming models.

## Purpose

The main objectives of this project are:

1. Implement the Gauss-Seidel Red-Black algorithm for CPU, CUDA, and HIP platforms.
2. Provide a benchmarking framework to measure and compare the performance of these implementations.
3. Analyze the scalability and efficiency of the algorithm on different hardware architectures.

## Features

- CPU implementation of the Gauss-Seidel and Gauss-Seidel Red-Black algorithms
- CUDA implementation for NVIDIA GPUs
- HIP implementation for AMD GPUs
- Benchmarking tools to measure performance across different platforms
- Flexible CMake-based build system for easy compilation on various systems

## Building the Project

This project uses CMake as its build system. To build the project, follow these steps:

1. Clone the repository
2. Create a build directory: `mkdir build && cd build`
3. Run CMake with your desired GPU backend:
   - For CUDA: `cmake -DGPU_BACKEND=CUDA ..`
   - For HIP: `cmake -DGPU_BACKEND=HIP ..`
   - For CPU-only: `cmake -DGPU_BACKEND=NONE ..`
4. Build the project: `make`

## Usage

After building the project, you can run the benchmark using:
```
./gauss_seidel
```


The program will run the Gauss-Seidel and Gauss-Seidel Red-Black algorithms on various problem sizes and report the performance metrics.
