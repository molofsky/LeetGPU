#include <cuda_runtime.h>

#define TILE_DIM 16
#define ROW_DIM 32
#define BLOCK_ROWS 8

// Naive: each thread does one element. Read coalesced, write strided (slow).
__global__ void matrix_transpose_naive(const float* input, float* output, int rows, int cols) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (yIndex < rows && xIndex < cols) {
        output[xIndex * rows + yIndex] = input[yIndex * cols + xIndex];
    }
}

// Shared-memory tiled (16x16): each thread does one element via shared memory.
// Both reads and writes are coalesced. +1 padding kills bank conflicts.
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float sdata[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    if (yIndex < rows && xIndex < cols) {
        sdata[threadIdx.y][threadIdx.x] = input[yIndex * cols + xIndex];
    }
    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    if (yIndex < cols && xIndex < rows) {
        output[yIndex * rows + xIndex] = sdata[threadIdx.x][threadIdx.y];
    }
}

// Blocked variant (32x32 tile, 32x8 thread block). Each thread handles
// 4 rows of the tile via the j loop. Better register/ILP characteristics.
__global__ void matrix_transpose_blocked(const float* input, float* output, int rows, int cols) {
    __shared__ float sdata[ROW_DIM][ROW_DIM + 1];

    int xIndex = blockIdx.x * ROW_DIM + threadIdx.x;
    int yIndex = blockIdx.y * ROW_DIM + threadIdx.y;

    // Load tile (coalesced read from input)
    for (int j = 0; j < ROW_DIM; j += BLOCK_ROWS) {
        if ((yIndex + j) < rows && xIndex < cols) {
            sdata[threadIdx.y + j][threadIdx.x] = input[(yIndex + j) * cols + xIndex];
        }
    }
    __syncthreads();

    // Swap block indices for transposed write; threadIdx.x stays on fast axis
    xIndex = blockIdx.y * ROW_DIM + threadIdx.x;
    yIndex = blockIdx.x * ROW_DIM + threadIdx.y;

    // Write transposed tile (coalesced write to output)
    for (int j = 0; j < ROW_DIM; j += BLOCK_ROWS) {
        if ((yIndex + j) < cols && xIndex < rows) {
            output[(yIndex + j) * rows + xIndex] = sdata[threadIdx.x][threadIdx.y + j];
        }
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // Option A: simple 16x16 tiled (each thread does one element)
    // dim3 threadsPerBlock(TILE_DIM, TILE_DIM);    // (16, 16) = 256 threads
    // dim3 blocksPerGrid((cols + TILE_DIM - 1) / TILE_DIM,
    //                    (rows + TILE_DIM - 1) / TILE_DIM);
    // matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);

    // Option B: blocked 32x8 (each thread does 4 elements, generally faster)
    dim3 threadsPerBlock(ROW_DIM, BLOCK_ROWS);   // (32, 8) = 256 threads
    dim3 blocksPerGrid((cols + ROW_DIM - 1) / ROW_DIM,
                       (rows + ROW_DIM - 1) / ROW_DIM);
    matrix_transpose_blocked<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);

    cudaDeviceSynchronize();
}