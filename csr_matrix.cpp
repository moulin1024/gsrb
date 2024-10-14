#include "csr_matrix.h"
#include <random>
#include <iostream>
#include <iomanip>

// Function to generate a sparse diagonally dominant matrix in CSR format
CSRMatrix generate_sparse_diagonal_matrix(int size, double density, double min_value, double max_value) {
    CSRMatrix matrix;
    matrix.rows = size;
    matrix.cols = size;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> value_dis(min_value, max_value);

    matrix.row_ptr.push_back(0);

    for (int i = 0; i < size; ++i) {
        double row_sum = 0.0;
        std::vector<std::pair<int, double>> row_elements;

        // Add off-diagonal elements based on density
        for (int j = 0; j < size; ++j) {
            if (i != j && dis(gen) < density) {
                double value = value_dis(gen);
                row_elements.push_back({j, value});
                row_sum += std::abs(value);
            }
        }

        // Ensure diagonal dominance
        double diag = row_sum + std::abs(value_dis(gen)) + 1.0;
        if (dis(gen) < 0.5) diag = -diag;  // Randomly make some diagonal elements negative

        // Add diagonal element
        matrix.values.push_back(diag);
        matrix.col_ind.push_back(i);

        // Add off-diagonal elements
        for (const auto& elem : row_elements) {
            matrix.values.push_back(elem.second);
            matrix.col_ind.push_back(elem.first);
        }

        matrix.row_ptr.push_back(matrix.values.size());
    }

    return matrix;
}

// Function to print the CSR matrix in dense format
void print_csr_matrix(const CSRMatrix& matrix) {
    if (matrix.rows > 20 || matrix.cols > 20) {
        std::cout << "Matrix is too large to print (size > 20x20)." << std::endl;
        return;
    }

    std::vector<std::vector<double>> dense_matrix(matrix.rows, std::vector<double>(matrix.cols, 0.0));

    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; ++j) {
            int col = matrix.col_ind[j];
            dense_matrix[i][col] = matrix.values[j];
        }
    }

    std::cout << "Matrix:" << std::endl;
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed << dense_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to generate a random vector
std::vector<double> generate_random_vector(int size, double min_value, double max_value) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> value_dis(min_value, max_value);

    std::vector<double> vector(size);
    for (int i = 0; i < size; ++i) {
        vector[i] = value_dis(gen);
    }

    return vector;
}

// Function to print a vector
void print_vector(const std::vector<double>& vector) {
    std::cout << "Vector:" << std::endl;
    for (const auto& value : vector) {
        std::cout << std::setw(8) << std::setprecision(4) << std::fixed << value << " ";
    }
    std::cout << std::endl;
}
