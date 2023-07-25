#include <cuda.h>
#include <cuda_fp16.h>

void fused_reshape_cast_pad(int MD, float* inp, half* outPad, cudaStream_t stream);
void fused_gemm1(int MD, half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream);
void fused_gemm_bn_relu_max_tile(int MD, half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream);
void fused_gemm2(int MD, half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream);
void fused_gemm_bn_relu_max(int MD, half* inp, half* w, half* g, half* b, float* outGemm, cudaStream_t stream);
