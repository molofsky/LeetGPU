#include <cuda_runtime.h>

__global__ void rms_norm_kernel(const float* input, float gamma, float beta, float* output, int N, float eps) {
    int tid = threadIdx.x;
    extern __shared__ float smem[];

    float sq_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)  {
        sq_sum += input[i] * input[i];
    }
    smem[tid] = sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(smem[0] / N + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = gamma * (input[i] * inv_rms) + beta;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    int threadsPerBlock = min(1024, (N + 31) / 32 * 32);
    int blocksPerGrid = 1;
    size_t sharedMem = threadsPerBlock * sizeof(float);

    rms_norm_kernel<<<blocksPerGrid, threadsPerBlock, sharedMem>>>(input, gamma, beta, output, N, eps);
    cudaDeviceSynchronize();
}
