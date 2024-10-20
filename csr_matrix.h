#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include <vector>

struct CSRMatrix {
    std::vector<double> values;
    std::vector<int> col_ind;
    std::vector<int> row_ptr;
    int rows;
    int cols;
    std::vector<double> diagonal;
    std::vector<double> diagonal_inv;
};

// Function to generate a sparse diagonally dominant matrix in CSR format
CSRMatrix generate_sparse_diagonal_matrix(int size, double density, double min_value, double max_value);

// Function to print the CSR matrix in dense format
void print_csr_matrix(const CSRMatrix& matrix);

// Function to generate a random vector
std::vector<double> generate_random_vector(int size, double min_value, double max_value);

// Function to print a vector
void print_vector(const std::vector<double>& vector);

#endif // CSR_MATRIX_H
