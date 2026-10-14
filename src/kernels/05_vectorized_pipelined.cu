#include "gemm_kernels.cuh"
#include "utils.cuh"

// Block tile dimensions (macro-tile)
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

// Thread tile dimensions (micro-tile per thread)
constexpr int TM = 8;
constexpr int TN = 8;

// Total threads per block: 16 * 16 = 256
constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN);

// Vectorization width: 128 bits = 4 x 32-bit float
constexpr int VECTOR_WIDTH = 4;

/**
 * Kernel 5: Vectorized Memory Access & Double Buffering (Software Pipelining)
 *
 * Characteristics & Architectural Rationale:
 * - 128-bit Vectorized Memory Operations (LDG.E.128):
 *     NVIDIA GPUs execute 128-bit vector loads (float4) in a single instruction.
 *     Vectorizing memory accesses cuts instruction count by 4x, lowers instruction cache pressure,
 *     and achieves maximum global memory bus transaction density.
 * - Double Buffering (Software Pipelining):
 *     In Kernel 4, the compute pipeline stalls at __syncthreads() waiting for global DRAM transfers.
 *     Double Buffering allocates two ping-pong shared memory buffers (s_A[2], s_B[2]).
 *     The kernel pipelines execution:
 *       1. Asynchronously prefetch global memory tile (k + 1) into registers.
 *       2. Simultaneously execute 8x8 register-level FMAs on tile (k) from shared memory.
 *       3. Barrier sync, commit prefetched registers into shared memory buffer, and repeat.
 *     -> Global memory latency is completely hidden behind arithmetic execution cycles!
 * - Peak Throughput:
 *     Approaches the hardware roofline ceiling and rivals vendor cuBLAS performance.
 */
__global__ void gemm_vectorized_pipelined_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Ping-Pong Double Buffers in Shared Memory: 2 buffers of shape [BM][BK] and [BK][BN]
    // Pad the shared memory stride to eliminate bank conflicts
    __shared__ float s_A[2][BK][BM + 4];
    __shared__ float s_B[2][BK][BN + 4];

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int tid = ty * blockDim.x + tx; // 0..255

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Register accumulators for 8x8 thread tile
    float accum[TM][TN] = {0.0f};

    // Staging registers for computation from shared memory
    float regA[TM];
    float regB[TN];

    // Staging registers for prefetching next global memory tile
    float4 ldg_reg_a[BM / (BLOCK_THREADS / (BK / VECTOR_WIDTH))]; // (128 / (256 / 2)) = 1 float4 per thread
    float4 ldg_reg_b[BK / (BLOCK_THREADS / (BN / VECTOR_WIDTH))]; // (8 / (256 / 32)) = 1 float4 per thread

    // Loading thread mappings:
    // A tile: BM x BK = 128 x 8 floats = 256 float4s.
    // Exactly 256 threads in block -> each thread loads 1 float4 of A!
    const int a_load_row = tid / (BK / VECTOR_WIDTH); // tid / 2 -> 0..127
    const int a_load_col = (tid % (BK / VECTOR_WIDTH)) * VECTOR_WIDTH; // (tid % 2) * 4 -> 0 or 4

    // B tile: BK x BN = 8 x 128 floats = 256 float4s.
    // Exactly 256 threads in block -> each thread loads 1 float4 of B!
    const int b_load_row = tid / (BN / VECTOR_WIDTH); // tid / 32 -> 0..7
    const int b_load_col = (tid % (BN / VECTOR_WIDTH)) * VECTOR_WIDTH; // (tid % 32) * 4 -> 0, 4, 8, .. 124

    const int num_k_tiles = (K + BK - 1) / BK;

    // -------------------------------------------------------------------------
    // PROLOGUE: Load tile 0 from global memory into shared memory buffer 0
    // -------------------------------------------------------------------------
    {
        // Load float4 from A
        int g_row_a = block_row + a_load_row;
        int g_col_a = a_load_col;
        float4 val_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g_row_a < M && (g_col_a + 3) < K) {
            val_a = *reinterpret_cast<const float4*>(&A[g_row_a * K + g_col_a]);
        } else {
            if (g_row_a < M && g_col_a + 0 < K) val_a.x = A[g_row_a * K + g_col_a + 0];
            if (g_row_a < M && g_col_a + 1 < K) val_a.y = A[g_row_a * K + g_col_a + 1];
            if (g_row_a < M && g_col_a + 2 < K) val_a.z = A[g_row_a * K + g_col_a + 2];
            if (g_row_a < M && g_col_a + 3 < K) val_a.w = A[g_row_a * K + g_col_a + 3];
        }
        // Write transposed into s_A[0][col][row] to ensure conflict-free vector reads
        s_A[0][a_load_col + 0][a_load_row] = val_a.x;
        s_A[0][a_load_col + 1][a_load_row] = val_a.y;
        s_A[0][a_load_col + 2][a_load_row] = val_a.z;
        s_A[0][a_load_col + 3][a_load_row] = val_a.w;

        // Load float4 from B
        int g_row_b = b_load_row;
        int g_col_b = block_col + b_load_col;
        float4 val_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g_row_b < K && (g_col_b + 3) < N) {
            val_b = *reinterpret_cast<const float4*>(&B[g_row_b * N + g_col_b]);
        } else {
            if (g_row_b < K && g_col_b + 0 < N) val_b.x = B[g_row_b * N + g_col_b + 0];
            if (g_row_b < K && g_col_b + 1 < N) val_b.y = B[g_row_b * N + g_col_b + 1];
            if (g_row_b < K && g_col_b + 2 < N) val_b.z = B[g_row_b * N + g_col_b + 2];
            if (g_row_b < K && g_col_b + 3 < N) val_b.w = B[g_row_b * N + g_col_b + 3];
        }
        s_B[0][b_load_row][b_load_col + 0] = val_b.x;
        s_B[0][b_load_row][b_load_col + 1] = val_b.y;
        s_B[0][b_load_row][b_load_col + 2] = val_b.z;
        s_B[0][b_load_row][b_load_col + 3] = val_b.w;
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // MAIN PIPELINED LOOP
    // -------------------------------------------------------------------------
    int write_buf = 1;
    int read_buf = 0;

    for (int t = 1; t < num_k_tiles; ++t) {
        // 1. Asynchronously prefetch tile (t) from Global DRAM into Local Registers
        int k_offset = t * BK;

        int g_row_a = block_row + a_load_row;
        int g_col_a = k_offset + a_load_col;
        float4 prefetch_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g_row_a < M && (g_col_a + 3) < K) {
            prefetch_a = *reinterpret_cast<const float4*>(&A[g_row_a * K + g_col_a]);
        } else {
            if (g_row_a < M && g_col_a + 0 < K) prefetch_a.x = A[g_row_a * K + g_col_a + 0];
            if (g_row_a < M && g_col_a + 1 < K) prefetch_a.y = A[g_row_a * K + g_col_a + 1];
            if (g_row_a < M && g_col_a + 2 < K) prefetch_a.z = A[g_row_a * K + g_col_a + 2];
            if (g_row_a < M && g_col_a + 3 < K) prefetch_a.w = A[g_row_a * K + g_col_a + 3];
        }
        ldg_reg_a[0] = prefetch_a;

        int g_row_b = k_offset + b_load_row;
        int g_col_b = block_col + b_load_col;
        float4 prefetch_b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (g_row_b < K && (g_col_b + 3) < N) {
            prefetch_b = *reinterpret_cast<const float4*>(&B[g_row_b * N + g_col_b]);
        } else {
            if (g_row_b < K && g_col_b + 0 < N) prefetch_b.x = B[g_row_b * N + g_col_b + 0];
            if (g_row_b < K && g_col_b + 1 < N) prefetch_b.y = B[g_row_b * N + g_col_b + 1];
            if (g_row_b < K && g_col_b + 2 < N) prefetch_b.z = B[g_row_b * N + g_col_b + 2];
            if (g_row_b < K && g_col_b + 3 < N) prefetch_b.w = B[g_row_b * N + g_col_b + 3];
        }
        ldg_reg_b[0] = prefetch_b;

        // 2. Compute tile (t - 1) from shared memory buffer [read_buf]
        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = s_A[read_buf][dotIdx][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = s_B[read_buf][dotIdx][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        // Wait until all threads have finished reading from shared memory buffer [read_buf]
        __syncthreads();

        // 3. Write prefetched register values into shared memory buffer [write_buf]
        s_A[write_buf][a_load_col + 0][a_load_row] = ldg_reg_a[0].x;
        s_A[write_buf][a_load_col + 1][a_load_row] = ldg_reg_a[0].y;
        s_A[write_buf][a_load_col + 2][a_load_row] = ldg_reg_a[0].z;
        s_A[write_buf][a_load_col + 3][a_load_row] = ldg_reg_a[0].w;

        s_B[write_buf][b_load_row][b_load_col + 0] = ldg_reg_b[0].x;
        s_B[write_buf][b_load_row][b_load_col + 1] = ldg_reg_b[0].y;
        s_B[write_buf][b_load_row][b_load_col + 2] = ldg_reg_b[0].z;
        s_B[write_buf][b_load_row][b_load_col + 3] = ldg_reg_b[0].w;

        // Ensure shared memory write is visible to all threads for the next iteration
        __syncthreads();

        // Swap ping-pong buffer indices
        read_buf ^= 1;
        write_buf ^= 1;
    }

    // -------------------------------------------------------------------------
    // EPILOGUE: Compute the final buffered tile
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            regA[i] = s_A[read_buf][dotIdx][ty * TM + i];
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            regB[j] = s_B[read_buf][dotIdx][tx * TN + j];
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                accum[i][j] += regA[i] * regB[j];
            }
        }
    }

    // -------------------------------------------------------------------------
    // WRITE BACK TO GLOBAL MEMORY MATRIX C
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += VECTOR_WIDTH) {
            int g_col = block_col + tx * TN + j;
            if (g_row < M && (g_col + 3) < N) {
                float4* c_vec = reinterpret_cast<float4*>(&C[g_row * N + g_col]);
                float4 old_c = (beta != 0.0f) ? *c_vec : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 out_c;
                out_c.x = alpha * accum[i][j + 0] + beta * old_c.x;
                out_c.y = alpha * accum[i][j + 1] + beta * old_c.y;
                out_c.z = alpha * accum[i][j + 2] + beta * old_c.z;
                out_c.w = alpha * accum[i][j + 3] + beta * old_c.w;
                *c_vec = out_c;
            } else {
                // Scalar fallback for boundary edges
                for (int v = 0; v < VECTOR_WIDTH; ++v) {
                    if (g_row < M && (g_col + v) < N) {
                        int idx = g_row * N + g_col + v;
                        C[idx] = alpha * accum[i][j + v] + (beta != 0.0f ? beta * C[idx] : 0.0f);
                    }
                }
            }
        }
    }
}

void run_gemm_vectorized_pipelined(
    int M, int N, int K,
    float alpha, const float* d_A, const float* d_B,
    float beta, float* d_C,
    cudaStream_t stream
) {
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    gemm_vectorized_pipelined_kernel<<<gridDim, blockDim, 0, stream>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
    CUDA_CHECK(cudaGetLastError());
}
