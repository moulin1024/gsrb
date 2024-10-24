#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <cstddef> // For size_t

// Function to load binary data from files
void load_binary_data(const char* filename, void* data, size_t size);

#endif // DATA_LOADER_H
