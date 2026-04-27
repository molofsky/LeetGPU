#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define WARP_SIZE 32


// __global__ void dot_product_kernel(const float* A, const float* B, float* result, int N) {
//     const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int vecN = N / 4;
//     if (idx < vecN) {
//         float4 a = reinterpret_cast<const float4*>(A)[idx];
//         float4 b = reinterpret_cast<const float4*>(B)[idx];
//         float val = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
//         atomicAdd(result, val);
//     }

//     if (idx == 0) {
//         float sum = 0.0f;
//         for (int i = vecN * 4; i < N; ++i) {
//             sum += A[i] * B[i];
//         }
//         atomicAdd(result, sum);
//     }
// }


__device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}


__global__ void dot_product_kernel(const float* A, const float* B, float* result, int N) {
    __shared__ float s_mem[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    float val = (idx < N) ? A[idx] * B[idx] : 0.0f;

    val = warp_reduce_sum(val);

    if (lane == 0) s_mem[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? s_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);

        if (lane == 0) {
            atomicAdd(result, val);
        }
    }   
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 256;
    // int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
    // if (blocksPerGrid == 0) blocksPerGrid = 1;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dot_product_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
