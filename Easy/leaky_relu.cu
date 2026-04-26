#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    // const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx < N) {
    //     float val = input[idx];
    //     val = val > 0 ? val : 0.01 * val;
    //     output[idx] = val;
    // }
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecN = N / 4;
    float alpha = 0.01;

    if (idx < vecN) {
        float4 val = reinterpret_cast<const float4*>(input)[idx];
        val.x = (val.x > 0) ? val.x : alpha * val.x;
        val.y = (val.y > 0) ? val.y : alpha * val.y;
        val.z = (val.z > 0) ? val.z : alpha * val.z;
        val.w = (val.w > 0) ? val.w : alpha * val.w;
        reinterpret_cast<float4*>(output)[idx] = val;
    }

    if (idx == 0) {
        for (int i = vecN * 4; i < N; ++i) {
            float val = input[i];
            val = val > 0 ? val : alpha * val;
            output[i] =  val;
        }
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0) blocksPerGrid = 1;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
