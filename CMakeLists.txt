# CMakeLists.txt

cmake_minimum_required(VERSION 3.12)
project(GaussSeidel)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to choose between CUDA, HIP, or CPU-only
set(GPU_BACKEND "NONE" CACHE STRING "Choose the GPU backend: CUDA, HIP, or NONE")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS CUDA HIP NONE)

# Source files (headers are now in the same directory as source files)
set(SRC_FILES
    src/main.cu
    src/kernels.cu
    src/data_loader.cu
    src/utils.cu
)

if(GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)

    # Set CUDA architectures (adjust according to your GPU)
    set(CMAKE_CUDA_ARCHITECTURES 80)

    # Set CUDA compilation flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

    # Add executable
    add_executable(gauss_seidel ${SRC_FILES})

    # Set CUDA specific properties
    set_target_properties(gauss_seidel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )

    # Link against CUDA libraries
    target_link_libraries(gauss_seidel ${CUDA_LIBRARIES})

    # Print CUDA architectures for debugging
    message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

elseif(GPU_BACKEND STREQUAL "HIP")
    enable_language(HIP)
    find_package(HIP REQUIRED)
    add_definitions(-DUSE_HIP)

    # Set HIP as the C++ compiler
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

    # Tell CMake to treat .cu files as C++ source files
    set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CXX)

    # Set HIP compilation flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

    # Add executable
    add_executable(gauss_seidel ${SRC_FILES})

    # Link against HIP libraries
    target_link_libraries(gauss_seidel ${HIP_LIBRARIES})

else()
    # CPU-only version

    # Tell CMake that .cu files are C++ files
    set_source_files_properties(${SRC_FILES} PROPERTIES LANGUAGE CXX)

    # Add executable
    add_executable(gauss_seidel ${SRC_FILES})

    # Add OpenMP flags
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gauss_seidel PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Print the selected GPU backend
message(STATUS "Selected GPU backend: ${GPU_BACKEND}")
