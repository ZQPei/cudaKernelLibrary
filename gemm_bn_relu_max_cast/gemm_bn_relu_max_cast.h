#include <cuda_runtime.h>

void fused_nn_dense_cast(float* __restrict__ p0, half* __restrict__ T_pad, half* __restrict__ p1, float* __restrict__ T_cast, cudaStream_t stream);
void fused_nn_dense_bn_relu_max_cast(float* __restrict__ p0, half* __restrict__ T_pad, half* __restrict__ p1, half* __restrict__ p_gamma, half* __restrict__ p_beta, float* __restrict__ T_cast, cudaStream_t stream);
