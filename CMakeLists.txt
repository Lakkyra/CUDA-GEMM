cmake_minimum_required(VERSION 3.20)

project(cuda_gemm_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA GPU Architectures:
# 80: Ampere (A100)
# 86: Ampere (RTX 30-series / Laptop GPU)
# 89: Ada Lovelace (RTX 40-series, L4, L40)
# 90: Hopper (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

message(STATUS "Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Compiler Optimization Flags
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /fp:fast /W3>
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    )
endif()

# CUDA Compiler Flags
# -Xptxas -v: outputs detailed register allocation and shared memory usage
set(CUDA_FLAGS_COMMON "-O3" "--use_fast_math" "-Xptxas=-v" "--expt-relaxed-constexpr")
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS_COMMON}>
)

# Find CUDA cuBLAS package
find_package(CUDAToolkit REQUIRED)

# Project Header Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Source Files
set(KERNEL_SOURCES
    src/kernels/00_naive.cu
    src/kernels/01_coalesced.cu
    src/kernels/02_smem_tiled.cu
    src/kernels/03_bank_resolved.cu
    src/kernels/04_reg_tiled.cu
    src/kernels/05_vectorized_pipelined.cu
)

# Main Executable
add_executable(gemm_benchmark
    src/main.cu
    ${KERNEL_SOURCES}
)

target_link_libraries(gemm_benchmark
    PRIVATE
    CUDA::cublas
    CUDA::cudart
)

set_target_properties(gemm_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Installation targets
install(TARGETS gemm_benchmark DESTINATION bin)
