#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    __shared__ float smax;
    __shared__ float ssum;
    __shared__ float smaxArr[256];
    __shared__ float ssumArr[256];

    int tid = threadIdx.x;

    // Pass 1: find max 
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        local_max = fmaxf(local_max, input[j]);
    }

    smaxArr[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smaxArr[tid] = fmaxf(smaxArr[tid], smaxArr[tid + stride]);
        __syncthreads();
    }

    if (tid == 0) smax = smaxArr[0];
    __syncthreads();

    // Pass 2: find sum
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        local_sum += expf(input[j] - smax);
    }
    ssumArr[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) ssumArr[tid] += ssumArr[tid + stride];
        __syncthreads();
    }

    if (tid == 0) ssum = ssumArr[0];
    __syncthreads();

    // Pass 3: write output
    for (int j = tid; j < N; j += blockDim.x) {
        output[j] = expf(input[j] - smax) / ssum;
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
