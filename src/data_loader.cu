#include "data_loader.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

void load_binary_data(const char* filename, void* data, size_t size) {
    FILE* file = fopen(filename, "rb");
    if (file == nullptr) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    size_t read_size = fread(data, 1, size, file);
    if (read_size != size) {
        std::cerr << "Error reading file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    fclose(file);
}
