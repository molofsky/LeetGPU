#include <cuda_runtime.h>

// Scalar: one thread per element, one 32-bit load/store per thread
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Vectorized: one thread per 4 elements, one 128-bit load/store per thread
__global__ void vector_add_f4(const float* A, const float* B, float* C, int N) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecN = N / 4;
    if (idx < vecN) {
        const float4 a = reinterpret_cast<const float4*>(A)[idx];
        const float4 b = reinterpret_cast<const float4*>(B)[idx];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(C)[idx] = c;
    }
    // Tail: one thread handles leftover 0-3 scalar elements
    if (idx == 0) {
        for (int i = vecN * 4; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;

    // For the scalar kernel:
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    // For the vectorized kernel:
    int vecN = N / 4;
    int blocksPerGrid = (vecN + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_f4<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    cudaDeviceSynchronize();
}