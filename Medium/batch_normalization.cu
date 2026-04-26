#include <cuda_runtime.h>

__global__ void batch_normalization_kernel(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= C) return;

    float mean = 0.0f;
    for (int i = 0; i < N; ++i) {
        mean += input[i * C + j];
    }
    mean /= N;

    float var = 0.0f;
    for (int i= 0; i < N; ++i) {
        float diff = input[i * C + j] - mean;
        var += diff * diff;
    }
    var /= N;

    float inv_std = rsqrtf(var + eps);
    for (int i= 0; i < N; ++i) {
        float x_hat = (input[i * C + j] - mean)  * inv_std;
        output[i * C + j] = gamma[j]* x_hat + beta[j];
    }
}

__global__ void batch_normalization_optimized(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps) {
    int j = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += input[i * C + j];
    }
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / N;

    float var_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff =input[i * C + j] - mean;
        var_sum += diff * diff;
    }
    smem[tid] = var_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    float inv_std = rsqrtf(smem[0] / N + eps);

    for (int i = tid; i < N; i += blockDim.x) {
        float x_hat = (input[i * C + j] - mean) * inv_std;
        output[i * C + j] = gamma[j] * x_hat + beta[j];
    }
}

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {
    int threadsPerBlock = 256;
    // int blocksPerGrid = (C + threadsPerBlock - 1) / threadsPerBlock;
    // batch_normalization_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, C, eps);

    size_t sharedMem = threadsPerBlock * sizeof(float);
    batch_normalization_kernel<<<C, threadsPerBlock, sharedMem>>>(input, gamma, beta, output, N, C, eps);
    cudaDeviceSynchronize();
}
