#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H

#include "csr_matrix.h"
#include <vector>

std::vector<double> gauss_seidel_red_black_cpu(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance);
double compute_residual_cpu(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b);

std::vector<double> gauss_seidel_red_black_gpu(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance);
double compute_residual_gpu(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b);


#endif // GAUSS_SEIDEL_H
