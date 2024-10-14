#include "gauss_seidel.h"
#include <cmath>
#include <iostream>

std::vector<double> gauss_seidel_red_black(const CSRMatrix& A, const std::vector<double>& b, int max_iterations, double tolerance) {
    int n = A.rows;
    std::vector<double> x(n, 0.0); // Initial guess
    std::vector<double> x_new(n, 0.0);

    for (int iter = 0; iter < max_iterations; ++iter) {
        double max_diff = 0.0;

        // Red sweep
        for (int i = 0; i < n; i += 2) {
            double sigma = 0.0;
            double a_ii = 0.0;

            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                int col = A.col_ind[j];
                double val = A.values[j];
                if (col == i) {
                    a_ii = val;
                } else {
                    sigma += val * x[col];
                }
            }

            if (a_ii == 0.0) {
                std::cerr << "Zero diagonal element detected at row " << i << ". Cannot proceed." << std::endl;
                exit(EXIT_FAILURE);
            }

            x_new[i] = (b[i] - sigma) / a_ii;
            double diff = std::abs(x_new[i] - x[i]);
            if (diff > max_diff) max_diff = diff;
        }

        // Update x with red values
        for (int i = 0; i < n; i += 2) {
            x[i] = x_new[i];
        }

        // Black sweep
        for (int i = 1; i < n; i += 2) {
            double sigma = 0.0;
            double a_ii = 0.0;

            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                int col = A.col_ind[j];
                double val = A.values[j];
                if (col == i) {
                    a_ii = val;
                } else {
                    sigma += val * x[col];
                }
            }

            if (a_ii == 0.0) {
                std::cerr << "Zero diagonal element detected at row " << i << ". Cannot proceed." << std::endl;
                exit(EXIT_FAILURE);
            }

            x_new[i] = (b[i] - sigma) / a_ii;
            double diff = std::abs(x_new[i] - x[i]);
            if (diff > max_diff) max_diff = diff;
        }

        // Update x with black values
        for (int i = 1; i < n; i += 2) {
            x[i] = x_new[i];
        }

        // Check for convergence
        if (max_diff < tolerance) {
            std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            return x;
        }
    }

    std::cout << "Reached maximum iterations without full convergence." << std::endl;
    return x;
}

double compute_residual(const CSRMatrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> residual(b.size(), 0.0);
    
    for (int i = 0; i < A.rows; ++i) {
        double row_sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += A.values[j] * x[A.col_ind[j]];
        }
        residual[i] = b[i] - row_sum;
    }

    double norm = 0.0;
    for (double r : residual) {
        norm += r * r;
    }
    return std::sqrt(norm);
}