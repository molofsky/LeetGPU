#include <cuda_runtime.h>
#define BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__global__ void block_reduce_sum(const float* input, float* output, int N) {
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    float val = (idx < N) ? input[idx] : 0.0f;
    val = warp_reduce_sum(val);

    if (lane == 0) {
      smem[warp_id] = val;
    }
    __syncthreads();


    if (warp_id == 0) {
        float val = (lane < BLOCK_SIZE / WARP_SIZE ? smem[lane] : 0.0f);
        val = warp_reduce_sum(val);

        if (lane == 0) {
            atomicAdd(output, val);
        }
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim;
    block_reduce_sum<<<gridDim, blockDim>>>(input, output, N);
    cudaDeviceSynchronize();
}
