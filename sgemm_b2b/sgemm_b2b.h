#include <cuda_runtime.h>

void sgemm_b2b(float* __restrict__ a, float* __restrict__ b1, float* __restrict__ b2, float* __restrict__ mid, float* __restrict__ c, int M, int K1, int K2, int N, cudaStream_t stream);
