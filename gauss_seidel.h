#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H

#include "csr_matrix.h"
#include <vector>

std::vector<double> gauss_seidel_red_black(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance);
double compute_residual(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b);

#if defined(USE_CUDA)
std::vector<double> gauss_seidel_red_black_cuda(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance);
double compute_residual_cuda(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b);
#elif defined(USE_HIP)
std::vector<double> gauss_seidel_red_black_hip(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance);
double compute_residual_hip(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b);
#endif

#endif // GAUSS_SEIDEL_H
