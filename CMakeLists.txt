cmake_minimum_required(VERSION 3.10)
project(GaussSeidel LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Option to choose between CUDA, HIP, or CPU-only
set(GPU_BACKEND "NONE" CACHE STRING "Choose the GPU backend: CUDA, HIP, or NONE")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS CUDA HIP NONE)

if(GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    
    # # Detect CUDA compute capability
    # include(FindCUDA/select_compute_arch)
    # CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    # string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    # string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    # string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES 80)

    # Add CUDA-specific source files
    set(GPU_SOURCES
        gauss_seidel_red_black.cu
        benchmark.cu
    )

elseif(GPU_BACKEND STREQUAL "HIP")
    find_package(hip REQUIRED)
    add_definitions(-DUSE_HIP)

    # Set HIP as the C++ compiler
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

    # Add HIP-specific source files
    set(GPU_SOURCES
        gauss_seidel_red_black.cu
        benchmark.cu
    )
    # Tell CMake to treat .cu files as C++ source files
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE CXX)

    # Add HIP-specific compiler flags
    set(HIP_COMPILE_FLAGS "-fPIC -D__HIP_PLATFORM_HCC__ -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_COMPILE_FLAGS}")

    # Alternatively, you can use target_compile_options (recommended)
    # This will be applied after the add_executable command
    # target_compile_options(gauss_seidel PRIVATE ${HIP_COMPILE_FLAGS})

else()
    # CPU-only version
    set(GPU_SOURCES "")
endif()

# Add executable
add_executable(gauss_seidel
    csr_matrix.cpp
    gauss_seidel_red_black.cu
    ${GPU_SOURCES}
)

if(GPU_BACKEND STREQUAL "CUDA")
    # Set CUDA specific properties
    set_target_properties(gauss_seidel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    # Link against CUDA libraries
    target_link_libraries(gauss_seidel ${CUDA_LIBRARIES})

    # Print CUDA architectures for debugging
    message(STATUS "CUDA_ARCH_LIST = ${CUDA_ARCH_LIST}")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

elseif(GPU_BACKEND STREQUAL "HIP")
    # Link against HIP libraries
    target_link_libraries(gauss_seidel hip::device)

    # If you prefer to use target_compile_options, uncomment the following line
    # target_compile_options(gauss_seidel PRIVATE ${HIP_COMPILE_FLAGS})
endif()

# Print the selected GPU backend
message(STATUS "Selected GPU backend: ${GPU_BACKEND}")
