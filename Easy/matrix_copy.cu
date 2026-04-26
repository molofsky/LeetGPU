#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    // const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < N * N) {
    //     B[idx] = A[idx];
    // }

    // for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N * N; idx += blockDim.x * gridDim.x) {
    //     B[idx] = A[idx];
    // }

    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int vecN = N * N/ 4;

    for (int i = idx; i < vecN; idx += stride) {
        float4 a = reinterpret_cast<const float4*>(A)[i];
        reinterpret_cast<float4*>(B)[i] = a;
    }


    for (int i = idx + vecN * 4; i < N * N; i += stride) {
        B[i] = A[i];
    }

}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    // int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = 1024;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
