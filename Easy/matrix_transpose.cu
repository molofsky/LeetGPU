// #include <cuda_runtime.h>
// #define ROW_DIM 32
// #define BLOCK_ROWS 8


// __global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
//     __shared__ float sdata[ROW_DIM][ROW_DIM + 1];
//     uint xIndex = blockIdx.x * ROW_DIM + threadIdx.x;
//     uint yIndex = blockIdx.y * ROW_DIM + threadIdx.y;

//     for (int j = 0; j < ROW_DIM; j += BLOCK_ROWS) {
//         if (xIndex < cols && (yIndex + j) < rows) {
//             sdata[threadIdx.y + j][threadIdx.x] = input[(yIndex + j) * cols + xIndex];
//         }
//     }
//     __syncthreads();

//     yIndex = blockIdx.x * ROW_DIM + threadIdx.x;
//     xIndex = blockIdx.y * ROW_DIM + threadIdx.y;

//     for (int j = 0; j < ROW_DIM; j += BLOCK_ROWS) {
//         output[(yIndex + j) * cols + xIndex] = sdata[threadIdx.y + j][threadIdx.x];
//     }
// }

// extern "C" void solve(const float* input, float* output, int rows, int cols) {
//     dim3 threadsPerBlock(ROW_DIM, BLOCK_ROWS);
//     dim3 blocksPerGrid((cols + threadsPerBlock.y - 1) / threadsPerBlock.y, (rows + threadsPerBlock.x - 1) / threadsPerBlock.x);
//     matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
//     cudaDeviceSynchronize();
// }

