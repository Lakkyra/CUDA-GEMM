#include "gemm_kernels.cuh"
#include "utils.cuh"

// Block tile dimensions (macro-tile)
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Thread tile dimensions (micro-tile computed in registers per thread)
constexpr int TM = 8;
constexpr int TN = 8;

// Total threads per block: (128 / 8) * (128 / 8) = 16 * 16 = 256 threads
constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN);

/**
 * Kernel 4: Register Tiling / 2D Thread Micro-Kernel
 *
 * Characteristics & Architectural Rationale:
 * - Shared Memory Bandwidth Bottleneck in Tiled Kernels:
 *     In Kernel 2 and Kernel 3, each thread loads from shared memory for every MAC operation.
 *     At peak compute, an SM issues two 32-bit shared memory reads per thread per FMA.
 *     Shared memory load ports quickly become the primary performance bottleneck!
 * - Register Caching (Micro-Kernel):
 *     Instead of computing 1 element per thread, each thread computes an 8x8 sub-tile (TM x TN = 64 elements)
 *     of matrix C held entirely in fast on-chip registers (accum[8][8]).
 * - Outer Product in Registers:
 *     For each step along BK (BK = 8):
 *       - Load 8 floats of matrix A from shared memory into registers: regA[8].
 *       - Load 8 floats of matrix B from shared memory into registers: regB[8].
 *       - Compute the 8x8 outer product: 64 FMAs (128 FLOPs) using only 16 shared memory reads!
 *     -> 8x reduction in shared memory read transactions relative to compute operations!
 * - Instruction-Level Parallelism (ILP):
 *     Unrolling the 8x8 accumulation loop exposes 64 independent FMAs to the warp scheduler,
 *     hiding pipeline latency and maximizing SM execution pipeline utilization.
 */
__global__ void gemm_reg_tiled_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Shared memory for macro-tile staging
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    // Thread index within the 2D thread block
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int tid = ty * blockDim.x + tx; // 0..255 linear thread ID

    // Top-left origin of this thread block's C sub-matrix
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Accumulator registers for 8x8 sub-tile of C
    float accum[TM][TN] = {0.0f};

    // Staging registers for outer-product calculation
    float regA[TM];
    float regB[TN];

    // Cooperative loading configuration:
    // s_A has BM * BK = 128 * 8 = 1024 floats.
    // 256 threads -> each thread loads 1024 / 256 = 4 floats.
    // Threads load A as 32 rows x 32 cols of thread mappings or row/col strides
    const int a_load_row = tid / (BK);          // 0 .. 31
    const int a_load_col = tid % (BK);          // 0 .. 7
    constexpr int a_row_stride = BLOCK_THREADS / BK; // 256 / 8 = 32

    // s_B has BK * BN = 8 * 128 = 1024 floats.
    // Each thread loads 4 floats of B.
    const int b_load_row = tid / (BN);          // 0 .. 1
    const int b_load_col = tid % (BN);          // 0 .. 127
    constexpr int b_row_stride = BLOCK_THREADS / BN; // 256 / 128 = 2

    const int num_k_tiles = (K + BK - 1) / BK;

    // Loop over all K tiles
    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_offset = t * BK;

        // 1. Cooperatively load matrix A from global memory into shared memory
        #pragma unroll
        for (int load_offset = 0; load_offset < BM; load_offset += a_row_stride) {
            int g_row = block_row + a_load_row + load_offset;
            int g_col = k_offset + a_load_col;
            if (g_row < M && g_col < K) {
                s_A[a_load_row + load_offset][a_load_col] = A[g_row * K + g_col];
            } else {
                s_A[a_load_row + load_offset][a_load_col] = 0.0f;
            }
        }

        // 2. Cooperatively load matrix B from global memory into shared memory
        #pragma unroll
        for (int load_offset = 0; load_offset < BK; load_offset += b_row_stride) {
            int g_row = k_offset + b_load_row + load_offset;
            int g_col = block_col + b_load_col;
            if (g_row < K && g_col < N) {
                s_B[b_load_row + load_offset][b_load_col] = B[g_row * N + g_col];
            } else {
                s_B[b_load_row + load_offset][b_load_col] = 0.0f;
            }
        }

        __syncthreads();

        // 3. Compute outer products: outer loop steps through BK
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Read 8 elements of A from shared memory into registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = s_A[ty * TM + i][dotIdx];
            }

            // Read 8 elements of B from shared memory into registers
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = s_B[dotIdx][tx * TN + j];
            }

            // Perform 64 FMA operations purely in registers
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // 4. Write results back to matrix C in global memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_col = block_col + tx * TN + j;
            if (g_row < M && g_col < N) {
                int idx = g_row * N + g_col;
                if (beta != 0.0f) {
                    C[idx] = alpha * accum[i][j] + beta * C[idx];
                } else {
                    C[idx] = alpha * accum[i][j];
                }
            }
        }
    }
}

void run_gemm_reg_tiled(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream
) {
    // 16x16 threads = 256 threads per block
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_reg_tiled_kernel<<<gridDim, blockDim, 0, stream>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
    CUDA_CHECK(cudaGetLastError());
}
