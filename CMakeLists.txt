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
    
    SET(CMAKE_CUDA_ARCHITECTURES 80)

elseif(GPU_BACKEND STREQUAL "HIP")
    find_package(hip REQUIRED)
    add_definitions(-DUSE_HIP)

    # Set HIP as the C++ compiler
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

    # Tell CMake to treat .cu files as C++ source files
    set_source_files_properties(main.cu PROPERTIES LANGUAGE CXX)

    # Add HIP-specific compiler flags
    set(HIP_COMPILE_FLAGS "-fPIC -D__HIP_PLATFORM_HCC__ -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_COMPILE_FLAGS}")

endif()

# Add executable
add_executable(gauss_seidel main.cu)

if(GPU_BACKEND STREQUAL "CUDA")
    # Set CUDA specific properties
    set_target_properties(gauss_seidel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    # Link against CUDA libraries
    target_link_libraries(gauss_seidel ${CUDA_LIBRARIES})

    # Print CUDA architectures for debugging
    message(STATUS "CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")

elseif(GPU_BACKEND STREQUAL "HIP")
    # Link against HIP libraries
    target_link_libraries(gauss_seidel hip::device)
endif()

# Print the selected GPU backend
message(STATUS "Selected GPU backend: ${GPU_BACKEND}")
