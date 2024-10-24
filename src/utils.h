#ifndef UTILS_H
#define UTILS_H

#include <cmath>

// Function declarations

double compute_residual(const int* i, const int* j, const double* val, const double* x, const double* b, int n_points);

void permute_back(const double* x_reordered, double* x_original, const int* new_to_old, int n_points);

#endif // UTILS_H
