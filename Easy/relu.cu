#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        val = fmaxf(val, 0.0f);
        output[idx] = val;
    }
}

__global__ void relu_vectorized_kernel(const float* input, float* output, int N)  {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecN = N / 4;
    if (idx < vecN) {
        float4 val = reinterpret_cast<const float4*>(input)[idx];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        reinterpret_cast<float4*>(output)[idx] = val;
    }

    if (idx == 0) {
        for (int i = vecN * 4; i < N; ++i) {
            float val = input[i];
            val = fmaxf(val, 0.0f);
            output[i] = val;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    // int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0) blocksPerGrid = 1;
    relu_kernel_vectorize<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}