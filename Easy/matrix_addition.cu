#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    // const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < N * N) {
    //     C[idx] = A[idx] + B[idx];
    // }

    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = (N * N) / 4;
    if (idx < n) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];
        float4 c;

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        reinterpret_cast<float4*>(C)[idx] = c;
    }

    if (idx == 0) {
        for (int i = n * 4; i < N * N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
