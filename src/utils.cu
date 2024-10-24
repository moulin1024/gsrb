#include "utils.h"
#include <omp.h>

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

void permute_back(const double* x_reordered, double* x_original, const int* new_to_old, int n_points) {
    for (int idx = 0; idx < n_points; ++idx) {
        int old_idx = new_to_old[idx];
        x_original[old_idx] = x_reordered[idx];
    }
}
