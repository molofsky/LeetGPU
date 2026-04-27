// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #define TILE 16

// __global__ void kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {

//     const uint col = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint row = blockIdx.y * blockDim.y + threadIdx.y;

//     float sum = 0.0f;
//     for (int i = 0; i < K; ++i) {
//         sum += (row < M && col < N) ? __half2float(A[row * N + i]) * __half2float(B[i * K + col]) : 0.0f;
//     }   
//     sum *= alpha;

//     if (row < M && col < N) 
//         C[row * K + col] = __float2half(sum + beta * __half2float(C[row * K + col]));
// }

// __global__ void kernel_tiled(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
//     __shared__ float As[TILE][TILE];
//     __shared__ float Bs[TILE][TILE];

//     int row = blockIdx.y * TILE + threadIdx.y;
//     int col = blockIdx.x * TILE + threadIdx.x;
//     float sum = 0.0f;

//     for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
//         int aCol = t * TILE + threadIdx.x;
//         As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? __half2float(A[row * K + aCol]) : 0.0f;

//         int bRow = t * TILE + threadIdx.y;
//         Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? __half2float(B[bRow * N + col]) : 0.0f;
//         __syncthreads();

//         for (int i = 0; i < TILE; ++i) {
//             sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
//         }

//         __syncthreads();
//     }

//     if (row < M && col < N) {
//         float c_old = __half2float(C[row * N + col]);
//         C[row * N + col] = __float2half(alpha * sum + beta * c_old);
//     }
// }

// // A, B, and C are device pointers
// extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha,
//                       float beta) {

//     // int threadsPerBlock = 256;
//     // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (K + threadsPerBlock.y - 1) / threadsPerBlock.y);
//     kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
//     cudaDeviceSynchronize();
//                       }
