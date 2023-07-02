#include <cuda_runtime.h>

void sgemm_cuda(float* __restrict__ dInput, float* __restrict__ dWeight, float* __restrict__ dInputTrans, float* __restrict__ dWeightTrans, float* __restrict__ dOutput, int M, int N, int K, cudaStream_t stream);
