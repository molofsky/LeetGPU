#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a_val = A[idx];
        float b_val = B[idx];
        output[idx * 2]  = a_val;
        output[idx * 2 + 1] = b_val;
    }
}

__global__ void interleave_kernel_vectorized(const float* A, const float* B, float* output, int N) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecN = N / 4;
    if (idx < vecN) {
        float4 a = reinterpret_cast<const float4*>(A)[idx];
        float4 b = reinterpret_cast<const float4*>(B)[idx];

        float4 out = reinterpret_cast<float4*>(output) + 2 * idx;

        out[0] = make_float4(a.x, b.x, a.y, b.y);
        out[1] = make_float4(a.z, b.z, a.w, b.z);
    }

    if (idx == 0) {
        for (int i = vecN * 4; i < N; ++i) {
            output[idx * 2] = A[idx];
            output[idx * 2 + 1] = B[idx];
        }
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
