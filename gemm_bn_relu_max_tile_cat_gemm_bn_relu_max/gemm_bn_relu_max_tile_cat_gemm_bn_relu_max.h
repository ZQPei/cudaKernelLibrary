#include <cuda.h>
#include <cuda_fp16.h>

void fused_reshape_cast_pad(float* inp, half* outPad, cudaStream_t stream);
void fused_gemm1(half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream);
void fused_gemm_bn_relu_max_tile(half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream);
void fused_gemm2(half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream);
void fused_gemm_bn_relu_max(half* inp, half* w, half* g, half* b, float* outGemm, cudaStream_t stream);
