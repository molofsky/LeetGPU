// #include <cuda_runtime.h>
// #include <float.h>
// #include <math.h>
// #define BLOCK_SIZE 256

// __global__ void causal_attention_kernel(const float* Q, const float* K, const float* V, float* output, int M, int d) {
//     int row = blockIdx.x;
//     int tid = threadIdx.x;

//     extern __shared__ char smem_raw[];
//     float* s_q = reinterpret_cast<float*>(smem_raw);
//     float* s_scores = s_q + d;
//     float* s_red = s_scores + (row + 1);

//     const float* q_row = Q + row * d;
//     const double inv_sqrt_d = 1.0 / sqrt((double)d);


//     for (int f = tid; f < d; f += blockDim.x) {
//         s_q[f] = q_row[f];
//     }
//     __syncthreads();

//     int num_keys = row + 1;
//     for (int j = tid; j < num_keys; j += blockDim.x) {
//         const float* k_row = K + j * d;
//         double dot = 0.0;
//         for (int f = 0; f < d; ++f) {
//             dot += (double)s_q[f] * (double)k_row[f];
//         }
//         s_scores[j] = (float)(dot * inv_sqrt_d);
//     }
//     __syncthreads();

//     float local_max = -FLT_MAX;
//     for (int j = tid; j < num_keys; j += blockDim.x) {
//         float v = s_scores[j];
//         if (v > local_max) local_max = v;
//     }
//     s_red[tid] = local_max;
//     __syncthreads();

//     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if (tid < stride) {
//             float a = s_red[tid], b = s_red[tid + stride];
//             s_red[tid] =(b > a) ? b : a;
//         }
//         __syncthreads();
//     }
//     float max_score = s_red[0];
//     __syncthreads();

//     double local_sum = 0.0;
//     for (int j = tid; j < num_keys; j += blockDim.x) {
//         float e = expf(s_scores[j] - max_score);
//         s_scores[j] = e;
//         local_sum += (double)e;
//     }
//     s_red[tid] = (float)local_sum;
//     __syncthreads();

//     for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if (tid < stride) {
//             s_red[tid] += s_red[tid + stride];
//         }
//         __syncthreads();
//     }
//     float sum_exp = s_red[0];
//     float inv_sum = 1.0f / sum_exp;
//     __syncthreads();

//     for (int j = tid; j < num_keys; j += blockDim.x) {
//         s_scores[j] *= inv_sum;
//     }
//     __syncthreads();

//     for (int f = tid; f < d; f += blockDim.x) {
//         double acc = 0.0;
//         for (int j = 0; j < num_keys; ++j) {
//             acc += (double)s_scores[j] * (double)V[j * d + f];
//         }
//         output[row * d + f] = (float)acc;
//     }
// }
// // Q, K, V, output are device pointers
// extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {
//     int block_size = BLOCK_SIZE;

//     size_t smem_bytes = d * sizeof(float) + M * sizeof(float) + BLOCK_SIZE * sizeof(float);

//     causal_attention_kernel<<<M, block_size, smem_bytes>>>(Q, K, V, output, M, d);
// }
