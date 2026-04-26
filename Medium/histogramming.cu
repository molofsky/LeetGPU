#include <cuda_runtime.h>

__global__ void histogramming_kernel(const int* input, int* histogram, int N, int num_bins) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&histogram[input[idx]], 1);
    }
}


__global__ void histogramming_optimized(const int* input, int* histogram, int N, int num_bins) {
    extern __shared__ int local[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = tid; j < num_bins; j += blockDim.x) {
        local[j] = 0;
    }
    __syncthreads();

    if (i < N) {
        atomicAdd(&local[input[i]], 1);
    }
    __syncthreads();

    for (int j = tid; j < num_bins; j += blockDim.x) {
        atomicAdd(&histogram[j], local[j]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    histogramming_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, histogram, N, num_bins);
    // int blocksPerGrid = min((N + threadsPerBlock - 1) / threadsPerBlock, 1024);
    // size_t sharedMemBytes = num_bins * sizeof(int);
    // histogramming_optimized<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
