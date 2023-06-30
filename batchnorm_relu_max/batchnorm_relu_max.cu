#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include "batchnorm_relu_max.h"

int constexpr _N =  18000;
int constexpr _D =  32;
int constexpr _C =  64;
int constexpr warpSz = 32;

#ifdef AUTO_TUNE
int constexpr blockSz = BLOCK_SIZE;
int constexpr loopNr = LOOP_NUMBER;
int constexpr grpBits = GROUP_BITS;
#else
int constexpr blockSz = 128;
int constexpr loopNr = 1;
int constexpr grpBits = 0;
#endif

__host__ __device__ __inline__ unsigned generateMask(int st, int ed) {
    return static_cast<unsigned>((1ULL<<ed)-1 - ((1ULL<<st)-1));
}

template<typename To, typename From>
__device__ __host__ __inline__ To convertTo(From);
template<>
__device__ __host__ __inline__ __half convertTo<half>(half h) { return h;}
template<>
__device__ __host__ __inline__ __half convertTo<half>(float f) { return __float2half(f);}
template<>
__device__ __host__ __inline__ float convertTo<float>(float f) { return f;}
template<>
__device__ __host__ __inline__ float convertTo<float>(__half h) { return __half2float(h);}

///////////////////////////////////////////////////////////////
template<typename inT, typename outT, int N, int D, int C, int impl>
__global__ void fused_batchnorm_relu_max_kernel(
    outT* __restrict__ dOutput,  // [N, C]
    inT* __restrict__ dInput,  // [N, D, C]
    inT* __restrict__ dGamma,  // [C]
    inT* __restrict__ dBeta  // [C]
) {
  // float constexpr epsilon = 0.001;
  int constexpr groupSz = 1 << grpBits;
  int constexpr nPerBlkOnce = blockSz / groupSz;
  int const laneId = threadIdx.x & (warpSz - 1);
  auto const mask = generateMask(laneId/groupSz*groupSz, (laneId/groupSz+1)*groupSz);
  static_assert(N == _N, "N == _N Check failed");
  static_assert(D == _D, "D == _D Check failed");
  static_assert(C == _C, "C == _C Check failed");
  static_assert(D % 32 == 0, "D should be divisble by warpSz");
  static_assert(grpBits >= 0 && grpBits <= 5, "groupSz should be in [0, 32]");

  #pragma unroll
  for (int _ = 0, groupId = blockIdx.x * nPerBlkOnce + threadIdx.x / groupSz; _ < loopNr; ++_, groupId += gridDim.x * nPerBlkOnce) {
    if (groupId >= N * C) return;

    // if (blockIdx.x == 0 && threadIdx.x == 0) printf("%d %d\n", D / groupSz, grpBits);

    int tId = threadIdx.x & (groupSz - 1);
    int n = groupId >> 6;
    int c = groupId & (64 - 1);
    float vmax = 0.f;

    float gamma = convertTo<float>(dGamma[c]);
    float beta = convertTo<float>(dBeta[c]);
    inT* din = dInput + n * D * C;
    outT* dout = dOutput + n * C;

    // // read value
    // int constexpr dLocalSz = D / groupSz;
    // float dLocal[dLocalSz];
    // #pragma unroll
    // for (int d = 0, offset = tId; d < D / groupSz; ++d, offset += groupSz) {
    //   dLocal[d] = convertTo<float>(din[offset * C + c]);
    // }

    // // max
    // #pragma unroll
    // for (int d = 0; d < D / groupSz; ++d) {
    //   vmax = fmaxf(vmax, dLocal[d]);
    // }

    // read value && max
    #pragma unroll
    for (int d = 0, offset = tId; d < D / groupSz; ++d, offset += groupSz) {
      vmax = fmaxf(vmax, convertTo<float>(din[offset * C + c]) * gamma + beta);
    }

    // shuffle
    #pragma unroll
    for (int _ = 0, delta = 1; _ < grpBits; ++_, delta <<= 1) {
      vmax = fmaxf(vmax, __shfl_xor_sync(mask, vmax, delta));
    }

    if (tId == 0) dout[c] = convertTo<outT>(vmax);
  }
}

///////////////////////////////////////////////////////////////
template<typename inT, typename outT, int N, int D, int C, int impl>
void _launch_fused_batchnorm_relu_max_kernel(
    outT* __restrict__ dOutput,  // [N, C]
    inT* __restrict__ dInput,  // [N, D, C]
    inT* __restrict__ dGamma,  // [C]
    inT* __restrict__ dBeta,  // [C]
    cudaStream_t stream
) {
  int constexpr groupSz = 1 << grpBits;
  int constexpr nPerBlk = blockSz / groupSz * loopNr;
  int gridSz = (N * C + nPerBlk - 1) / nPerBlk;
  // std::cout << gridSz << " " << nPerBlk << " " << blockSz << " " << groupSz << " " << loopNr << std::endl;
  fused_batchnorm_relu_max_kernel<inT, outT, N, D, C, impl><<<gridSz, blockSz, 0, stream>>>(dOutput, dInput, dGamma, dBeta);
}


///////////////////////////////////////////////////////////////
// template<>
// void fused_batchnorm_relu_max(
//     float* __restrict__ dOutput,  // [N, C]
//     float* __restrict__ dInput,  // [N, D, C]
//     float* __restrict__ dGamma,  // [C]
//     float* __restrict__ dBeta,  // [C]
//     int N, int D, int C,
//     int impl,
//     cudaStream_t stream
// ) {
//   // #define IF_STAT(n, d, c, it) if (N == n && D == d && C == c && impl == it) { _launch_fused_batchnorm_relu_max_kernel<n, d, c, it>(dOutput, dInput, dGamma, dBeta, stream); }
//   // #define ELIF_STAT(n, d, c, it) else if (N == n && D == d && C == c && impl == it) { _launch_fused_batchnorm_relu_max_kernel<n, d, c, it>(dOutput, dInput, dGamma, dBeta, stream); }
//   // #define ELSE_STAT else { std::cout << "Not Implemented Error" << std::endl; __builtin_trap(); }
//   // IF_STAT(18000, 32, 64, BRMImplType::NAIVE)
//   // ELSE_STAT

//   _launch_fused_batchnorm_relu_max_kernel<float, float, 18000, 32, 64, BRMImplType::NAIVE>(dOutput, dInput, dGamma, dBeta, stream);

//   // cudaStreamSynchronize(stream);
// }

template<>
void fused_batchnorm_relu_max(
    float* __restrict__ dOutput,  // [N, C]
    __half* __restrict__ dInput,  // [N, D, C]
    __half* __restrict__ dGamma,  // [C]
    __half* __restrict__ dBeta,  // [C]
    int N, int D, int C,
    int impl,
    cudaStream_t stream
) {
  // #define IF_STAT(n, d, c, it) if (N == n && D == d && C == c && impl == it) { _launch_fused_batchnorm_relu_max_kernel<n, d, c, it>(dOutput, dInput, dGamma, dBeta, stream); }
  // #define ELIF_STAT(n, d, c, it) else if (N == n && D == d && C == c && impl == it) { _launch_fused_batchnorm_relu_max_kernel<n, d, c, it>(dOutput, dInput, dGamma, dBeta, stream); }
  // #define ELSE_STAT else { std::cout << "Not Implemented Error" << std::endl; __builtin_trap(); }
  // IF_STAT(18000, 32, 64, BRMImplType::NAIVE)
  // ELSE_STAT

  // std::cout << "inp: half, out: float" << std::endl;
  _launch_fused_batchnorm_relu_max_kernel<__half, float, 18000, 32, 64, BRMImplType::NAIVE>(dOutput, dInput, dGamma, dBeta, stream);

  // cudaStreamSynchronize(stream);
}

// template<>
// void fused_batchnorm_relu_max(
//     __half* __restrict__ dOutput,  // [N, C]
//     __half* __restrict__ dInput,  // [N, D, C]
//     __half* __restrict__ dGamma,  // [C]
//     __half* __restrict__ dBeta,  // [C]
//     int N, int D, int C,
//     int impl,
//     cudaStream_t stream
// ) {
//   // #define IF_STAT(n, d, c, it) if (N == n && D == d && C == c && impl == it) { _launch_fused_batchnorm_relu_max_kernel<n, d, c, it>(dOutput, dInput, dGamma, dBeta, stream); }
//   // #define ELIF_STAT(n, d, c, it) else if (N == n && D == d && C == c && impl == it) { _launch_fused_batchnorm_relu_max_kernel<n, d, c, it>(dOutput, dInput, dGamma, dBeta, stream); }
//   // #define ELSE_STAT else { std::cout << "Not Implemented Error" << std::endl; __builtin_trap(); }
//   // IF_STAT(18000, 32, 64, BRMImplType::NAIVE)
//   // ELSE_STAT

//   // std::cout << "inp: half, out: float" << std::endl;
//   _launch_fused_batchnorm_relu_max_kernel<__half, __half, 18000, 32, 64, BRMImplType::NAIVE>(dOutput, dInput, dGamma, dBeta, stream);

//   // cudaStreamSynchronize(stream);
// }
