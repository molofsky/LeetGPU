#include <cuda_runtime.h>
#include <float.h>
#define BLOCK_SIZE 256

__global__ void topk_gating_kernel(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ char smem_row[];
    float* s_logits = reinterpret_cast<float*>(smem_row);
    float* s_vals = s_logits + E;
    int* s_idxs = reinterpret_cast<int*>(s_vals + BLOCK_SIZE);
    float* s_topvals = reinterpret_cast<float*>(s_idxs + BLOCK_SIZE);

    const float* row_logits = logits + row * E;

    for (int i = tid; i < E; i += blockDim.x) {
        s_logits[i] = row_logits[i];
    }
    __syncthreads();


    for (int it = 0; it < k; ++it) {
        float best_val = -FLT_MAX;
        int best_idx = -1;
        for (int i = tid; i < E; i += blockDim.x) {
            float v = s_logits[i];
            if (v > best_val || (v == best_val && i < best_idx)) {
                best_val = v;
                best_idx = i;
            }
        }

        s_vals[tid] = best_val;
        s_idxs[tid] = best_idx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float v1 = s_vals[tid], v2 = s_vals[tid + stride];
                int i1 = s_idxs[tid], i2 = s_idxs[tid + stride];
                if (v2 > v1 || (v2 == v1 && i2 >= 0 && (i1 <0 || i2 < i1))) {
                    s_vals[tid] = v2;
                    s_idxs[tid] = i2;
                }
            }

            __syncthreads();
        }

        if (tid == 0) {
            int chosen = s_idxs[0];
            float chosen_val = s_vals[0];
            topk_indices[row * k + it] = chosen;
            s_topvals[it] = chosen_val;
            s_logits[chosen] = -FLT_MAX;
        }
        __syncthreads();
    }

    float max_val = s_topvals[0];

    if (tid < k) {
        s_vals[tid] = __expf(s_topvals[tid] - max_val);
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i= 0; i < k; ++i) sum += s_vals[i];
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < k; ++i) {
            topk_weights[row * k + i] = s_vals[i] * inv_sum;
        }
    }
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {
    int block_size = BLOCK_SIZE;
    size_t smem_bytes = E * sizeof(float) + BLOCK_SIZE * sizeof(float) + BLOCK_SIZE * sizeof(int) + k * sizeof(float);

    topk_gating_kernel<<<M, block_size, smem_bytes>>>(logits, topk_weights, topk_indices, M, E, k);
}
