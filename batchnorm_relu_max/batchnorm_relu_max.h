#include <cuda_runtime.h>

namespace BRMImplType {
  int constexpr NAIVE = 0;
}

template<typename inT, typename outT>
void fused_batchnorm_relu_max(
    outT* __restrict__ dOutput,  // [N, C]
    inT* __restrict__ dInput,  // [N, D, C]
    inT* __restrict__ dGamma,  // [C]
    inT* __restrict__ dBeta,  // [C]
    int N, int D, int C,
    int impl,
    cudaStream_t stream
);
