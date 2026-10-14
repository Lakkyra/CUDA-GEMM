#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <algorithm>

// Macro for robust CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t status = (call);                                          \
        if (status != cudaSuccess) {                                          \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(status) << std::endl;   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Macro for cuBLAS error checking
#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__    \
                      << " -> Status code: " << status << std::endl;          \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// High-precision GPU timer using CUDA events
struct GpuTimer {
    cudaEvent_t start_event, stop_event;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
    }

    // Returns elapsed time in milliseconds
    float elapsed_ms() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// Compute GFLOPS given M, N, K and elapsed time in milliseconds
inline double compute_gflops(int M, int N, int K, float time_ms) {
    if (time_ms <= 0.0f) return 0.0;
    // Standard GEMM performs 2*M*N*K floating-point operations (multiply + accumulate)
    double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return (ops * 1e-9) / (static_cast<double>(time_ms) * 1e-3);
}

// Compute theoretical DRAM bandwidth used by ideal GEMM in GB/s
inline double compute_bandwidth_gbps(int M, int N, int K, float time_ms) {
    if (time_ms <= 0.0f) return 0.0;
    // Read A (M*K*4 bytes) + Read B (K*N*4 bytes) + Write C (M*N*4 bytes)
    double bytes = (static_cast<double>(M) * K + static_cast<double>(K) * N + static_cast<double>(M) * N) * sizeof(float);
    return (bytes * 1e-9) / (static_cast<double>(time_ms) * 1e-3);
}

// Initialize matrix with uniform random floats in [min_val, max_val]
inline void init_matrix_random(float* mat, int rows, int cols, float min_val = -1.0f, float max_val = 1.0f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        mat[i] = dis(gen);
    }
}

// Initialize matrix with a constant value
inline void init_matrix_const(float* mat, int rows, int cols, float val) {
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        mat[i] = val;
    }
}

// Host CPU reference GEMM: C_ref = alpha * (A * B) + beta * C
// A is M x K, B is K x N, C is M x N in row-major layout
inline void cpu_gemm(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[m * K + k]) * static_cast<double>(B[k * N + n]);
            }
            C[m * N + n] = static_cast<float>(alpha * sum + beta * static_cast<double>(C[m * N + n]));
        }
    }
}

// Verification result summary
struct VerificationResult {
    bool passed;
    float max_abs_diff;
    float max_rel_diff;
    float l2_error;
    int mismatch_count;
};

// Verify matrix correctness against CPU reference with PRD tolerance (1e-5)
inline VerificationResult verify_gemm(
    int M, int N,
    const float* C_calc,
    const float* C_ref,
    float tolerance = 1e-5f,
    float rel_tolerance = 1e-4f
) {
    VerificationResult res = {true, 0.0f, 0.0f, 0.0f, 0};
    double sum_sq_diff = 0.0;
    int total = M * N;

    for (int i = 0; i < total; ++i) {
        float calc = C_calc[i];
        float ref = C_ref[i];
        float diff = std::abs(calc - ref);
        float rel_diff = diff / (std::abs(ref) + 1e-8f);

        if (diff > res.max_abs_diff) res.max_abs_diff = diff;
        if (rel_diff > res.max_rel_diff) res.max_rel_diff = rel_diff;
        sum_sq_diff += diff * diff;

        // Check against tolerances (FP32 accumulation order differs slightly on GPU vs CPU)
        if (diff > tolerance && rel_diff > rel_tolerance) {
            res.mismatch_count++;
            if (res.passed) {
                res.passed = false;
            }
        }
    }

    res.l2_error = static_cast<float>(std::sqrt(sum_sq_diff / total));
    return res;
}

// Print GPU Device Specs
inline void print_device_info() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "=========================================================\n";
    std::cout << " GPU Device: " << prop.name << "\n";
    std::cout << " Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << " Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << " Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
    std::cout << " Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << " Warp Size: " << prop.warpSize << "\n";
    std::cout << " Multiprocessors (SMs): " << prop.multiProcessorCount << "\n";
    std::cout << " Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << " Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz\n";
    std::cout << "=========================================================\n";
}
