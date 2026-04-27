#include <cuda_runtime.h>
#include <stdio.h>


__global__ void euclidean_distance_kernel(const float2* p1, const float2* p2, float* distance, int N) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float dx = p1[idx].x - p2[idx].x;  
        float dy = p1[idx].y - p2[idx].y;
        distance[idx] = sqrtf(dx * dx + dy * dy);
    }
}


int main() {
    int N = 10;

    float2* p1;
    float2* p2;
    float*  distance;

    cudaMalloc(&p1,       N * sizeof(float2));
    cudaMalloc(&p2,       N * sizeof(float2));
    cudaMalloc(&distance, N * sizeof(float));

    float2* h_p1 = new float2[N];
    float2* h_p2 = new float2[N];
    float*  h_dist = new float[N];

    for (int i = 0; i < N; i++) {
        h_p1[i] = {(float)i, (float)i};
        h_p2[i] = {0.0f, 0.0f};
    }

    cudaMemcpy(p1, h_p1, N * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(p2, h_p2, N * sizeof(float2), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    euclidean_distance_kernel<<<blocksPerGrid, threadsPerBlock>>>(p1, p2, distance, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_dist, distance, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("dist[%d] = %f\n", i, h_dist[i]);

    cudaFree(p1);
    cudaFree(p2);
    cudaFree(distance);
    delete[] h_p1;
    delete[] h_p2;
    delete[] h_dist;

    return 0;
}