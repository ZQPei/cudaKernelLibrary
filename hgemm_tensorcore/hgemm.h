#include <cuda_runtime.h>

void hgemm_cuda(half* __restrict__ dInput, half* __restrict__ dWeight, half* __restrict__ dInputTrans, half* __restrict__ dWeightTrans, half* __restrict__ dOutput, int M, int N, int K, cudaStream_t stream);
