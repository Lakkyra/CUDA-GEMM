#include "gemm_kernels.cuh"
#include "utils.cuh"

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstring>
#include <sstream>

// cuBLAS GEMM wrapper for Row-Major FP32: C = alpha * (A * B) + beta * C
// A is [M, K], B is [K, N], C is [M, N]
void run_gemm_cublas(
    cublasHandle_t handle,
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C
) {
    // Note on cuBLAS row-major handling:
    // cuBLAS interprets matrices in column-major order.
    // By leveraging the transpose identity: C^T = (alpha * A * B + beta * C)^T = alpha * B^T * A^T + beta * C^T
    // Since row-major C[M, N] has identical memory layout to column-major C^T[N, M],
    // passing B as the first operand and A as the second operand computes row-major GEMM with zero copy overhead.
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,   // B is [K, N] in row-major -> [N, K] in col-major with ldb = N
        d_A, K,   // A is [M, K] in row-major -> [K, M] in col-major with lda = K
        &beta,
        d_C, N    // C is [M, N] in row-major -> [N, M] in col-major with ldc = N
    ));
}

struct BenchmarkRecord {
    int id;
    std::string tag;
    std::string name;
    float time_ms;
    double gflops;
    double speedup_vs_naive;
    double pct_cublas;
    bool verified;
    float max_diff;
};

void run_benchmark_suite(
    int M, int N, int K,
    int warmup_runs,
    int bench_runs,
    int target_kernel,
    bool verify,
    bool output_json
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    // Host memory allocation
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);
    std::vector<float> h_C_out(M * N, 0.0f);

    // Initialize inputs with deterministic random values
    init_matrix_random(h_A.data(), M, K, -1.0f, 1.0f, 1234);
    init_matrix_random(h_B.data(), K, N, -1.0f, 1.0f, 5678);

    // Device memory allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // If verification requested, compute reference solution
    if (verify) {
        if (!output_json) {
            std::cout << ">> Computing reference GEMM for verification (size: " 
                      << M << "x" << N << "x" << K << ")... " << std::flush;
        }
        if (M <= 2048 && N <= 2048 && K <= 2048) {
            // Compute host CPU reference
            cpu_gemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());
        } else {
            // For large sizes, use cuBLAS on GPU as high-precision reference
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
            run_gemm_cublas(cublas_handle, M, N, K, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
        }
        if (!output_json) {
            std::cout << "Done.\n\n";
        }
    }

    const auto& catalog = get_kernel_catalog();
    std::vector<BenchmarkRecord> records;
    GpuTimer timer;

    float naive_time_ms = 1.0f;
    float cublas_time_ms = 1.0f;

    // First measure cuBLAS as reference
    {
        // Warmup
        for (int i = 0; i < warmup_runs; ++i) {
            run_gemm_cublas(cublas_handle, M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark
        timer.start();
        for (int i = 0; i < bench_runs; ++i) {
            run_gemm_cublas(cublas_handle, M, N, K, alpha, d_A, d_B, beta, d_C);
        }
        timer.stop();
        cublas_time_ms = timer.elapsed_ms() / bench_runs;
    }

    // Benchmark each kernel
    for (const auto& meta : catalog) {
        if (target_kernel != -1 && meta.id != target_kernel) {
            continue;
        }

        auto run_kernel = [&](cudaStream_t stream = 0) {
            switch (meta.id) {
                case 0: run_gemm_naive(M, N, K, alpha, d_A, d_B, beta, d_C, stream); break;
                case 1: run_gemm_coalesced(M, N, K, alpha, d_A, d_B, beta, d_C, stream); break;
                case 2: run_gemm_smem_tiled(M, N, K, alpha, d_A, d_B, beta, d_C, stream); break;
                case 3: run_gemm_bank_resolved(M, N, K, alpha, d_A, d_B, beta, d_C, stream); break;
                case 4: run_gemm_reg_tiled(M, N, K, alpha, d_A, d_B, beta, d_C, stream); break;
                case 5: run_gemm_vectorized_pipelined(M, N, K, alpha, d_A, d_B, beta, d_C, stream); break;
                case 6: run_gemm_cublas(cublas_handle, M, N, K, alpha, d_A, d_B, beta, d_C); break;
                default: break;
            }
        };

        // Verification pass
        bool pass = true;
        float max_diff = 0.0f;
        if (verify) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));
            run_kernel();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_out.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
            auto vres = verify_gemm(M, N, h_C_out.data(), h_C_ref.data());
            pass = vres.passed;
            max_diff = vres.max_abs_diff;
        }

        // Timing runs
        for (int i = 0; i < warmup_runs; ++i) {
            run_kernel();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        for (int i = 0; i < bench_runs; ++i) {
            run_kernel();
        }
        timer.stop();
        float avg_time_ms = timer.elapsed_ms() / bench_runs;

        if (meta.id == 0) {
            naive_time_ms = avg_time_ms;
        }

        double gflops = compute_gflops(M, N, K, avg_time_ms);
        double speedup = static_cast<double>(naive_time_ms) / avg_time_ms;
        double pct_cublas = (cublas_time_ms / avg_time_ms) * 100.0;

        records.push_back({
            meta.id, meta.tag, meta.name,
            avg_time_ms, gflops, speedup, pct_cublas,
            pass, max_diff
        });
    }

    // Output formatted results
    if (output_json) {
        std::cout << "{\n";
        std::cout << "  \"matrix\": {\"M\": " << M << ", \"N\": " << N << ", \"K\": " << K << "},\n";
        std::cout << "  \"kernels\": [\n";
        for (size_t i = 0; i < records.size(); ++i) {
            const auto& r = records[i];
            std::cout << "    {\n";
            std::cout << "      \"id\": " << r.id << ",\n";
            std::cout << "      \"tag\": \"" << r.tag << "\",\n";
            std::cout << "      \"name\": \"" << r.name << "\",\n";
            std::cout << "      \"time_ms\": " << r.time_ms << ",\n";
            std::cout << "      \"gflops\": " << r.gflops << ",\n";
            std::cout << "      \"speedup_vs_naive\": " << r.speedup_vs_naive << ",\n";
            std::cout << "      \"pct_cublas\": " << r.pct_cublas << ",\n";
            std::cout << "      \"verified\": " << (r.verified ? "true" : "false") << ",\n";
            std::cout << "      \"max_diff\": " << r.max_diff << "\n";
            std::cout << "    }" << (i + 1 < records.size() ? "," : "") << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
    } else {
        std::cout << "=========================================================================================================\n";
        std::cout << " GEMM Benchmark Results: M=" << M << ", N=" << N << ", K=" << K 
                  << " (Warmup: " << warmup_runs << ", Benchmark Runs: " << bench_runs << ")\n";
        std::cout << "=========================================================================================================\n";
        std::cout << std::left
                  << std::setw(32) << "Kernel Name"
                  << std::setw(12) << "Time (ms)"
                  << std::setw(14) << "GFLOPS"
                  << std::setw(12) << "Speedup"
                  << std::setw(16) << "% cuBLAS"
                  << std::setw(12) << "Verified"
                  << std::setw(14) << "Max |Diff|"
                  << "\n";
        std::cout << "---------------------------------------------------------------------------------------------------------\n";

        for (const auto& r : records) {
            std::cout << std::left
                      << std::setw(32) << r.name
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms
                      << std::setw(14) << std::fixed << std::setprecision(1) << r.gflops
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.speedup_vs_naive
                      << std::setw(16) << std::fixed << std::setprecision(1) << r.pct_cublas
                      << std::setw(12) << (r.verified ? "PASSED" : "FAILED")
                      << std::setw(14) << std::scientific << std::setprecision(2) << r.max_diff
                      << "\n";
        }
        std::cout << "=========================================================================================================\n";
    }

    // Cleanup resources
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    int M = 2048;
    int N = 2048;
    int K = 2048;
    int warmup_runs = 3;
    int bench_runs = 10;
    int target_kernel = -1; // -1 means all kernels
    bool verify = true;
    bool output_json = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            M = N = K = std::stoi(argv[++i]);
        } else if (arg == "--dim" && i + 3 < argc) {
            M = std::stoi(argv[++i]);
            N = std::stoi(argv[++i]);
            K = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            bench_runs = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::stoi(argv[++i]);
        } else if (arg == "--kernel" && i + 1 < argc) {
            std::string k_str = argv[++i];
            if (k_str == "all") target_kernel = -1;
            else if (k_str == "cublas") target_kernel = 6;
            else target_kernel = std::stoi(k_str);
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--no-verify") {
            verify = false;
        } else if (arg == "--json") {
            output_json = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --size <N>             Square matrix dimension M=N=K (default: 2048)\n"
                      << "  --dim <M> <N> <K>      Arbitrary rectangular dimensions M N K\n"
                      << "  --kernel <all|0..5|cublas> Target kernel ID or 'all' (default: all)\n"
                      << "  --iterations <N>       Number of benchmark timing iterations (default: 10)\n"
                      << "  --warmup <N>           Number of warmup iterations (default: 3)\n"
                      << "  --verify               Verify numerical accuracy against reference (default: on)\n"
                      << "  --no-verify            Skip numerical verification\n"
                      << "  --json                 Output machine-readable JSON format\n";
            return 0;
        }
    }

    if (!output_json) {
        print_device_info();
    }

    run_benchmark_suite(M, N, K, warmup_runs, bench_runs, target_kernel, verify, output_json);
    return 0;
}
