#include "gemm_kernels.cuh"
#include "utils.cuh"

// Tile dimension for 2D Shared Memory Tiling: 32x32 = 1024 threads per block
constexpr int TILE_SIZE = 32;

/**
 * Kernel 2: 2D Shared Memory Tiling
 *
 * Characteristics & Optimization Focus:
 * - Solves the Memory Bandwidth Bottleneck:
 *     Global GPU memory (DRAM) has high latency (~400-800 cycles).
 *     On-chip Shared Memory (__shared__) has very low latency (~20-30 cycles) and high bandwidth.
 * - Tiling Concept:
 *     Partition matrices A, B, and C into sub-matrices (tiles) of size TILE_SIZE x TILE_SIZE.
 *     Instead of streaming A and B directly from global memory for every MAC, a thread block
 *     cooperatively stages a tile of A and a tile of B into shared memory.
 * - Global Memory Access Amortization:
 *     Each element loaded from DRAM into shared memory is reused TILE_SIZE (32) times
 *     by all threads in the block. This reduces DRAM traffic by a factor of 32 (96.8% reduction!).
 * - Synchronization:
 *     __syncthreads() ensures all block threads have finished loading DRAM -> shared memory
 *     before compute begins, and ensures all compute finishes before overwriting the tile.
 */
__global__ void gemm_smem_tiled_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Shared memory allocations for staged tiles
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Thread coordinates within the block and global grid
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int col = blockIdx.x * TILE_SIZE + tx;
    const int row = blockIdx.y * TILE_SIZE + ty;

    float accum = 0.0f;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Traverse across the K dimension in chunks of TILE_SIZE
    for (int t = 0; t < num_tiles; ++t) {
        int tiled_k_col = t * TILE_SIZE + tx;
        int tiled_k_row = t * TILE_SIZE + ty;

        // Stage Matrix A tile into Shared Memory with boundary guards
        if (row < M && tiled_k_col < K) {
            s_A[ty][tx] = A[row * K + tiled_k_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Stage Matrix B tile into Shared Memory with boundary guards
        if (tiled_k_row < K && col < N) {
            s_B[ty][tx] = B[tiled_k_row * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // Wait for all threads in the block to finish staging shared memory
        __syncthreads();

        // Accumulate partial matrix product from fast on-chip shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            accum += s_A[ty][k] * s_B[k][tx];
        }

        // Ensure all threads finished computing before the next iteration overwrites shared memory
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

void run_gemm_smem_tiled(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream
) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    gemm_smem_tiled_kernel<<<gridDim, blockDim, 0, stream>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
    CUDA_CHECK(cudaGetLastError());
}
