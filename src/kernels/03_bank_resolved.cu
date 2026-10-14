#include "gemm_kernels.cuh"
#include "utils.cuh"

// Tile dimensions for Kernel 3: 32x32 tiles
constexpr int TILE_SIZE = 32;
// Shared Memory Padding (+1 float) to shift bank indices and resolve bank conflicts
constexpr int SMEM_PAD = 1;
constexpr int TILE_STRIDE = TILE_SIZE + SMEM_PAD; // 33 words

/**
 * Kernel 3: Shared Memory Bank Conflict Elimination
 *
 * Characteristics & Hardware Architecture Details:
 * - Shared Memory Banking:
 *     NVIDIA GPUs (Ampere / Ada / Hopper) organize on-chip shared memory into 32 banks.
 *     Each bank has a width of 4 bytes (32-bit word) per clock cycle.
 *     Bank Index = (byte_address / 4) % 32.
 * - Bank Conflict Serialization:
 *     If N threads within the same warp access distinct 32-bit addresses mapped to the SAME bank,
 *     the hardware serializes the accesses into N separate single-bank operations.
 *     This causes an N-way bank conflict, multiplying shared memory instruction latency by up to 32x!
 * - Padding Resolution:
 *     By padding the row stride from 32 to 33 (TILE_SIZE + 1 = 33 floats):
 *     Row 0: Elements 0..31  -> Banks (0..31) % 32
 *     Row 1: Elements 33..64 -> Banks (33..64) % 32 = (1..32) % 32
 *     Row k: Elements k*33   -> Banks (k*33) % 32 = k % 32
 *     Any column-wise or transposed traversal (stride = 33) visits bank (k + col) % 32.
 *     Since gcd(33, 32) = 1, every 32 consecutive accesses map to 32 mutually distinct banks,
 *     guaranteeing ZERO bank conflicts and maximal shared memory throughput!
 */
__global__ void gemm_bank_resolved_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Shared memory allocations with padding: 32 rows x 33 columns
    __shared__ float s_A[TILE_SIZE][TILE_STRIDE];
    __shared__ float s_B[TILE_SIZE][TILE_STRIDE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * TILE_SIZE + tx;
    const int row = blockIdx.y * TILE_SIZE + ty;

    float accum = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_k_col = t * TILE_SIZE + tx;
        int tiled_k_row = t * TILE_SIZE + ty;

        // Stage Matrix A into padded shared memory with boundary checks
        if (row < M && tiled_k_col < K) {
            s_A[ty][tx] = A[row * K + tiled_k_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Stage Matrix B into padded shared memory with boundary checks
        if (tiled_k_row < K && col < N) {
            s_B[ty][tx] = B[tiled_k_row * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Bank-conflict-free accumulation across padded shared memory rows and columns
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            accum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    // Write final accumulated result to global memory C
    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * accum + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * accum;
        }
    }
}

void run_gemm_bank_resolved(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream
) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    gemm_bank_resolved_kernel<<<gridDim, blockDim, 0, stream>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
    CUDA_CHECK(cudaGetLastError());
}
