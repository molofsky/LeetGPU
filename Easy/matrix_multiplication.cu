#include <cuda_runtime.h>
#define TILE 16

/*
 * Naive matmul: one thread per output cell C[row][col].
 *
 * Each thread reads an entire row of A and an entire column of B from global memory,
 * computing the dot product scalar-by-scalar.
 *
 * Launch: 2D grid of 2D blocks. Block is (TILE x TILE) = 16x16 = 256 threads.
 *         Grid covers the (M x K) output matrix: (ceil(K/TILE), ceil(M/TILE)) blocks.
 *
 * Memory access:
 *   - A: each thread reads N elements. Threads in the same warp share `row` and vary `col`,
 *        so they read DIFFERENT rows of A for different columns of C. Row-stride access
 *        across threads in a warp means uncoalesced loads of A (bad).
 *   - B: threads in a warp share `i` (inner loop index) and vary `col`, so they read
 *        consecutive elements of B[i, col..col+31]. Coalesced (good).
 *   - C: one write per thread to C[row, col]. Coalesced (good).
 *
 * Arithmetic intensity (flops per byte loaded):
 *   - Per output cell: 2N flops (N multiplies + N adds), 2N loads (N from A, N from B), 4 bytes per load.
 *   - Intensity = 2N flops / 8N bytes = 0.25 flops/byte. Very low — memory-bound.
 *
 * Time:  O(M * N * K)     — each of M*K output cells does N multiply-adds.
 * Space: O(M*N + N*K + M*K) — three matrices in global memory; negligible per-thread state.
 *
 * Bottleneck: redundant global memory loads. Each element of A is loaded K times across
 *   threads (once per output column); each element of B is loaded M times across threads
 *   (once per output row). Tiling fixes this.
 */
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    if (row < M && col < K) {
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}


/*
 * Tiled matmul: one block per output tile, shared memory staging.
 *
 * Each block of 16x16 threads cooperatively computes a 16x16 tile of the output.
 * The block walks along the shared dimension N in steps of TILE, loading one tile of A
 * and one tile of B into shared memory, computing their partial product, and accumulating.
 *
 * Launch: same as naive — 2D grid, 2D blocks of (TILE x TILE) = 256 threads.
 *         Grid: (ceil(K/TILE), ceil(M/TILE)).
 *
 * Key insight — data reuse:
 *   - Naive:  each element of A is loaded K times, each of B is loaded M times.
 *   - Tiled:  each element of A is loaded K/TILE times (once per output tile column),
 *             each of B is loaded M/TILE times (once per output tile row).
 *   - With TILE=16, global memory traffic drops by 16x. The 16x fewer global loads
 *     are replaced by shared-memory loads, which are ~100x faster than DRAM.
 *
 * Per-tile structure:
 *   1. Cooperative load: each thread loads one element of As and one of Bs from global.
 *      Bounds-checked; out-of-bounds positions pad with 0 (identity for multiply-add).
 *   2. __syncthreads(): wait for all threads in the block to finish the load before computing.
 *   3. Inner multiply-accumulate: each thread computes a TILE-length dot product using
 *      only shared memory. With #pragma unroll, this becomes 16 straight-line FMAs.
 *   4. __syncthreads(): wait before overwriting shared memory with the next tile.
 *
 * Why #pragma unroll helps:
 *   - The inner loop has a compile-time-known bound (TILE = 16).
 *   - Unrolling eliminates loop overhead (compare, increment, branch) = ~3 instructions per iteration saved.
 *   - The flat instruction stream lets the compiler interleave the 16 FMAs and pipeline
 *     shared-memory loads, hiding shared-memory latency behind compute.
 *   - Typically 10-20% faster than the rolled version.
 *
 * Why shared memory doesn't hit bank conflicts here:
 *   - As[ty][i]: all threads in a warp share `ty`, varying `tx`. They read DIFFERENT values
 *     of i in sequence — but within one iteration of i, all threads hit the same bank (broadcast, free).
 *   - Bs[i][tx]: all threads in a warp share `i`, varying `tx`. They read consecutive
 *     elements of row i → 32 different banks → conflict-free.
 *   - If the layout were transposed (e.g., using Bs[tx][i] instead), we'd get 32-way bank
 *     conflicts and need the classic +1 padding trick.
 *
 * Potential further optimizations (not implemented here):
 *   - Thread coarsening: each thread computes a 2x2 or 4x4 sub-tile of outputs, reducing
 *     shared-memory traffic and increasing arithmetic intensity. ~2x gain.
 *   - Double buffering: load the next tile while computing on the current one to hide
 *     global memory latency. ~10-20% gain.
 *   - Vectorized (float4) loads into shared memory to reduce load instructions in the fill phase.
 *   - Tensor cores (WMMA): if inputs can be FP16/BF16, mma.sync-based matmul is 8-16x faster,
 *     as each instruction computes a 16x16x16 matmul in hardware.
 *
 * Time:  O(M * N * K)           — same work as naive; just reorganized.
 * Space: O(M*N + N*K + M*K) global + O(TILE^2) per-block shared = O(2*16*16*4) = 2 KB per block.
 */
__global__ void matrix_multiplication_tiled(const float* A, const float* B, float* C,
                                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const uint col = blockIdx.x * TILE + threadIdx.x;
    const uint row = blockIdx.y * TILE + threadIdx.y;

    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) C[row * K + col] = sum;
}


/*
 * Launch configurations for both kernels.
 *
 * Both use the same 2D launch shape: block is (TILE x TILE) = 256 threads, grid covers
 * the (M x K) output matrix. The launch math is identical — tiling doesn't change the
 * grid/block geometry, only what happens inside each block.
 *
 * Note on block size:
 *   - For the naive kernel, the block size is flexible (any 2D shape with <= 1024 threads works).
 *   - For the tiled kernel, block size is FIXED at TILE x TILE because each thread owns exactly
 *     one element of the shared tile. Changing block size would break the shared-memory layout.
 */
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE, TILE);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid((K + TILE - 1) / TILE,   // ceil(K / TILE) blocks along x (columns of C)
                       (M + TILE - 1) / TILE);  // ceil(M / TILE) blocks along y (rows of C)

    // ---- Launch the NAIVE kernel (baseline, memory-bound, ~10-20% of peak) ----
    // matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    // ---- Launch the TILED kernel (shared-memory staging, ~40-60% of peak FP32) ----
    matrix_multiplication_tiled<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    cudaDeviceSynchronize();
}