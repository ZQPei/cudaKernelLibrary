#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>

#define cuAssert(expr)                                                                       \
  do {                                                                                       \
    auto err = (expr);                                                                       \
    if (err != cudaSuccess) {                                                                \
      std::cerr << __FILE__ << ' ' << __LINE__ << ' ' << #expr << " failed with error code " \
                << err << ": " << cudaGetErrorString(err) << '\n';                           \
      std::cerr.flush();                                                                     \
      __builtin_trap();                                                                      \
    }                                                                                        \
  } while (0)

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
#define tz threadIdx.z
#define ty threadIdx.y
#define tx threadIdx.x

__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

// __device__ half max(half a, half b)
// {
//   return __float2half(max(__half2float(a), __half2float(b)));
// }
// __device__ half min(half a, half b)
// {
//   return __hlt(__half(a), __half(b)) ? a : b;
// }

/////////////////////////////////////////////////////////////////////////////////////////
// grid=(250, 1, 1),  block=(1024, 1, 1)
extern "C" __global__ void __launch_bounds__(1024) fused_reshape_cast_pad_kernel(int outer_loop_num, half* __restrict__ T_pad, float* __restrict__ p0) {
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < outer_loop_num; ++ax0_ax1_fused_outer_outer) {
    #pragma unroll
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 2; ++ax0_ax1_fused_inner_s) {
      T_pad[((((ax0_ax1_fused_outer_outer * 512000) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_inner_s)] = (((((int)threadIdx.x) & 7) < 5) ? ((half)p0[(((((ax0_ax1_fused_outer_outer * 320000) + (((int)blockIdx.x) * 1280)) + ((((int)threadIdx.x) >> 3) * 10)) + ((((int)threadIdx.x) & 7) * 2)) + ax0_ax1_fused_inner_s)]) : __float2half_rn(0.000000000000000000000000000000000000000000000e+00f));
    }
  }
}

void _launch_fused_reshape_cast_pad_kernel(int MD, float* __restrict__ inp, half* __restrict__ outPad, cudaStream_t stream) {
  int const outer_loop_num = MD / 1000;
  fused_reshape_cast_pad_kernel<<<dim3(250, 1, 1), dim3(1024, 1, 1), 0, stream>>>(outer_loop_num, outPad, inp);
}

/////////////////////////////////////////////////////////////////////////////////////////
// grid=(10000,1,1),  block=(32,4,2)
extern "C" __global__ void __launch_bounds__(256) fused_gemm1_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[1];
  __shared__ half placeholder_shared[8704];
  __shared__ half placeholder_d_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000000000000000000000000000000000000000000e+00f);
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 2; ++ax0_ax1_fused_outer_outer_outer_outer) {
    *(uint1*)(placeholder_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 4352) + (((int)threadIdx.z) * 2176)) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder + (((((((int)blockIdx.x) * 1024) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  }
  *(uint1*)(placeholder_d_shared + ((((((int)threadIdx.z) * 2176) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder_1 + (((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[(((int)threadIdx.y) * 2176)])), 136);
  nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[0], (&(placeholder_d_shared[(((int)threadIdx.z) * 2176)])), 136);
  nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[((((int)threadIdx.y) * 640) + (((int)threadIdx.z) * 16))])), T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    *(uint1*)(T_dense + (((((((int)blockIdx.x) * 2048) + (i_inner_j_inner_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2))) = *(uint1*)(placeholder_d_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 640) + (((int)threadIdx.z) * 320)) + (((int)threadIdx.y) * 80)) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)));
  }
}

void _launch_fused_gemm1_kernel(int MD, half* __restrict__ inp, half* __restrict__ w, half* __restrict__ g, half* __restrict__ b, half* __restrict__ outGemm, cudaStream_t stream) {
  fused_gemm1_kernel<<<dim3(MD/2,1,1), dim3(32,4,2), 0, stream>>>(inp, w, g, b, outGemm);
}

/////////////////////////////////////////////////////////////////////////////////////////
// grid=(10000,1,1),  block=(32,4,2)
extern "C" __global__ void __launch_bounds__(256) fused_gemm_bn_relu_max_tile_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[1];
  __shared__ half placeholder_shared[8704];
  __shared__ half placeholder_d_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000000000000000000000000000000000000000000e+00f);
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 2; ++ax0_ax1_fused_outer_outer_outer_outer) {
    *(uint1*)(placeholder_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 4352) + (((int)threadIdx.z) * 2176)) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder + (((((((int)blockIdx.x) * 1024) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  }
  *(uint1*)(placeholder_d_shared + ((((((int)threadIdx.z) * 2176) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder_1 + (((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[(((int)threadIdx.y) * 2176)])), 136);
  nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[0], (&(placeholder_d_shared[(((int)threadIdx.z) * 2176)])), 136);
  nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[((((int)threadIdx.y) * 640) + (((int)threadIdx.z) * 16))])), T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
  __syncthreads();

  // BN RELU MAX REDUCE
  int constexpr loopNum = 4;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    half2 r_val = *(half2*)(placeholder_d_shared + tz * 32 * 40 + tx * 40 + loopid * 8 + ty * 2);
    half2 r_gamma = *(half2*)(g + loopid * 8 + ty * 2);
    half2 r_beta = *(half2*)(b + loopid * 8 + ty * 2);
    half2 r_vmax;

    // bn relu
    // r_vmax.x = max(__half(0x0000), __float2half(__half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x)));
    // r_vmax.y = max(__half(0x0000), __float2half(__half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y)));
    r_vmax.x = max(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = max(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));
    r_val = r_vmax;

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = max(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = max(r_vmax.y, r_vmax_shfl.y);
    }

    // stg, tile && concat
    int offset1 = bx * 2 * 32 * 64 + tz * 32 * 64 + tx * 64;
    int offset2 = loopid * 4 * 2 + ty * 2;
    *(half2*)(T_dense + offset1 + offset2) = r_val;
    *(half2*)(T_dense + offset1 + 32 + offset2) = r_vmax;
  }

  // __syncthreads();
  // for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
  //   *(uint1*)(T_dense + (((((((int)blockIdx.x) * 2048) + (i_inner_j_inner_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2))) = *(uint1*)(placeholder_d_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 640) + (((int)threadIdx.z) * 320)) + (((int)threadIdx.y) * 80)) + ((((int)threadIdx.x) >> 4) * 40)) + ((((int)threadIdx.x) & 15) * 2)));
  // }
}

// grid=(10000,1,1),  block=(32,4,2)
extern "C" __global__ void __launch_bounds__(256) fused_gemm_bn_relu_max_tile_v2_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[1];
  __shared__ half placeholder_shared[8704];
  __shared__ half placeholder_d_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000000000000000000000000000000000000000000e+00f);
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 2; ++ax0_ax1_fused_outer_outer_outer_outer) {
    *(uint1*)(placeholder_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 4352) + (((int)threadIdx.z) * 2176)) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder + (((((((int)blockIdx.x) * 1024) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  }
  *(uint1*)(placeholder_d_shared + ((((((int)threadIdx.z) * 2176) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder_1 + (((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[(((int)threadIdx.y) * 2176)])), 136);
  nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[0], (&(placeholder_d_shared[(((int)threadIdx.z) * 2176)])), 136);
  nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[((((int)threadIdx.y) * 640) + (((int)threadIdx.z) * 16))])), T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
  __syncthreads();

  // BN RELU MAX REDUCE
  int constexpr loopNum = 4;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    half2 r_val = *(half2*)(placeholder_d_shared + tz * 32 * 40 + tx * 40 + loopid * 8 + ty * 2);
    half2 r_gamma = *(half2*)(g + loopid * 8 + ty * 2);
    half2 r_beta = *(half2*)(b + loopid * 8 + ty * 2);
    half2 r_vmax;

    // bn relu
    // r_vmax.x = max(__half(0x0000), __float2half(__half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x)));
    // r_vmax.y = max(__half(0x0000), __float2half(__half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y)));
    r_vmax.x = max(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = max(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));
    r_val = r_vmax;

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = max(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = max(r_vmax.y, r_vmax_shfl.y);
    }

    // sts, tile
    int offset1 = tz * 32 * 66 + tx * 66;
    int offset2 = loopid * 4 * 2 + ty * 2;
    *(half2*)(placeholder_shared + offset1 + offset2) = r_val;
    *(half2*)(placeholder_shared + offset1 + 32 + offset2) = r_vmax;
  }

  __syncthreads();
  // stg, concate
  int constexpr gloopNum = 8;
  #pragma unroll
  for (int gloopid = 0; gloopid < gloopNum; ++gloopid) {
    *(uint1*)(T_dense + bx * 64 * 64 + tz * 32 * 64 + gloopid * 4 * 64 + ty * 64 + tx *2) = *(uint1*)(placeholder_shared + tz * 32 * 66 + gloopid * 4 * 66 + ty * 66 + tx *2);
  }
}

// grid=(10000,1,1),  block=(32,4,2)
extern "C" __global__ void __launch_bounds__(256) fused_gemm_bn_relu_max_tile_v3_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[1];
  __shared__ half placeholder_shared[8704];
  __shared__ half placeholder_d_shared[4352];
  // __shared__ half s_buffer[32*33*2];
  // __shared__ half s_g[32];
  // __shared__ half s_b[32];
  // if (ty == 0 && tz == 0) {
  //   *(half*)(s_g + tx) = *(half*)(g + tx);
  //   *(half*)(s_b + tx) = *(half*)(b + tx);
  // }
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000000000000000000000000000000000000000000e+00f);
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 2; ++ax0_ax1_fused_outer_outer_outer_outer) {
    *(uint1*)(placeholder_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 4352) + (((int)threadIdx.z) * 2176)) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder + (((((((int)blockIdx.x) * 1024) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  }
  *(uint1*)(placeholder_d_shared + ((((((int)threadIdx.z) * 2176) + (((int)threadIdx.y) * 544)) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 2))) = *(uint1*)(placeholder_1 + (((((int)threadIdx.z) * 256) + (((int)threadIdx.y) * 64)) + (((int)threadIdx.x) * 2)));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[(((int)threadIdx.y) * 2176)])), 136);
  nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[0], (&(placeholder_d_shared[(((int)threadIdx.z) * 2176)])), 136);
  nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[((((int)threadIdx.y) * 640) + (((int)threadIdx.z) * 16))])), T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
  __syncthreads();


  // #pragma unroll
  // for (int loopid = 0; loopid < 4; ++loopid) {
  //   *(half2*)(s_buffer + tz * 16 * 66 + ty * 4 * 66 + loopid * 66 + tx * 2) = *(half2*)(placeholder_d_shared + tz * 16 * 80 + ty * 4 * 80 + loopid * 80 + (tx >> 4) * 40 + (tx & 15) * 2);
  // }
  // __syncthreads();

  // BN RELU MAX REDUCE
  int constexpr loopNum = 4;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    // half2 r_val = *(half2*)(s_buffer + tz * 16 * 66 + (tx >> 1) * 66 + (tx & 1) * 16 * 2 + ((ty << 2) + loopid) * 2);
    half2 r_val = *(half2*)(placeholder_d_shared + tz * 32 * 40 + tx * 40 + loopid * 8 + ty * 2);
    half2 r_gamma = *(half2*)(g + loopid * 8 + ty * 2);
    half2 r_beta = *(half2*)(b + loopid * 8 + ty * 2);
    half2 r_vmax;

    // bn relu
    // r_vmax.x = max(__half(0x0000), __float2half(__half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x)));
    // r_vmax.y = max(__half(0x0000), __float2half(__half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y)));
    r_vmax.x = max(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = max(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));
    r_val = r_vmax;

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = max(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = max(r_vmax.y, r_vmax_shfl.y);
    }

    // sts, tile
    int offset1 = tz * 32 * 66 + tx * 66;
    int offset2 = loopid * 4 * 2 + ty * 2;
    *(half2*)(placeholder_shared + offset1 + offset2) = r_val;
    *(half2*)(placeholder_shared + offset1 + 32 + offset2) = r_vmax;
  }

  __syncthreads();
  // stg, concate
  int constexpr gloopNum = 8;
  #pragma unroll
  for (int gloopid = 0; gloopid < gloopNum; ++gloopid) {
    *(uint1*)(T_dense + bx * 64 * 64 + tz * 32 * 64 + gloopid * 4 * 64 + ty * 64 + tx *2) = *(uint1*)(placeholder_shared + tz * 32 * 66 + gloopid * 4 * 66 + ty * 66 + tx *2);
  }
}

void _launch_fused_gemm_bn_relu_max_tile_kernel(int MD, half* __restrict__ inp, half* __restrict__ w, half* __restrict__ g, half* __restrict__ b, half* __restrict__ outGemm, cudaStream_t stream) {
  // fused_gemm_bn_relu_max_tile_kernel<<<dim3(MD/2,1,1), dim3(32,4,2), 0, stream>>>(inp, w, g, b, outGemm);
  // fused_gemm_bn_relu_max_tile_v2_kernel<<<dim3(MD/2,1,1), dim3(32,4,2), 0, stream>>>(inp, w, g, b, outGemm);
  fused_gemm_bn_relu_max_tile_v3_kernel<<<dim3(MD/2,1,1), dim3(32,4,2), 0, stream>>>(inp, w, g, b, outGemm);
}

/////////////////////////////////////////////////////////////////////////////////////////
// grid=(10000,1,1),  block=(32,2,4)
extern "C" __global__ void __launch_bounds__(256) fused_gemm2_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half placeholder_shared[2560];
  __shared__ half placeholder_d_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[2];
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000000000000000000000000000000000000000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    *(uint4*)(placeholder_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder + ((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(placeholder_d_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder_1 + (((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[((((int)threadIdx.y) * 1280) + (k_outer_inner * 16))])), 40);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax0_outer], (&(placeholder_d_shared[(((((int)threadIdx.z) * 640) + (ax0_outer * 320)) + (k_outer_inner * 16))])), 40);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[(((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 16)) + (ax1_outer_inner * 8))])), T_dense_wmma_accumulator[ax1_outer_inner], 72, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    *(uint4*)(T_dense + (((((((int)blockIdx.x) * 4096) + (i_inner_j_inner_fused_outer_outer_outer_outer * 2048)) + (((int)threadIdx.z) * 512)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(placeholder_d_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 2304) + (((int)threadIdx.z) * 576)) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)));
  }
}

void _launch_fused_gemm2_kernel(int MD, half* __restrict__ inp, half* __restrict__ w, half* __restrict__ g, half* __restrict__ b, half* __restrict__ outGemm, cudaStream_t stream) {
  fused_gemm2_kernel<<<dim3(MD/2,1,1), dim3(32,2,4), 0, stream>>>(inp, w, g, b, outGemm);
}

/////////////////////////////////////////////////////////////////////////////////////////
// grid=(10000,1,1),  block=(32,2,4)
extern "C" __global__ void __launch_bounds__(256) fused_gemm_bn_relu_max_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, float* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half placeholder_shared[2560];
  __shared__ half placeholder_d_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[2];
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000000000000000000000000000000000000000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    *(uint4*)(placeholder_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder + ((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(placeholder_d_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder_1 + (((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[((((int)threadIdx.y) * 1280) + (k_outer_inner * 16))])), 40);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax0_outer], (&(placeholder_d_shared[(((((int)threadIdx.z) * 640) + (ax0_outer * 320)) + (k_outer_inner * 16))])), 40);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[(((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 16)) + (ax1_outer_inner * 8))])), T_dense_wmma_accumulator[ax1_outer_inner], 72, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  
  // BN RELU MAX REDUCE
  int constexpr loopNum = 8;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    half2 r_val = *(half2*)(placeholder_d_shared + ty * 32 * 72 + tx * 72 + tz * 16 + loopid * 2);
    half2 r_gamma = *(half2*)(g + tz * 16 + loopid * 2);
    half2 r_beta = *(half2*)(b + tz * 16 + loopid * 2);
    half2 r_vmax;
    float2 r_vmaxf;

    // bn relu
    // r_vmax.x = max(__half(0x0000), __float2half(__half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x)));
    // r_vmax.y = max(__half(0x0000), __float2half(__half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y)));
    r_vmax.x = max(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = max(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = max(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = max(r_vmax.y, r_vmax_shfl.y);
    }
    r_vmaxf.x = __half2float(r_vmax.x);
    r_vmaxf.y = __half2float(r_vmax.y);

    // // sts
    // if (tx == 0) *(float2*)(placeholder_shared + ty * 64 + tz * 16 + loopid * 2) = r_vmaxf;

    // stg
    if (tx == 0) *(float2*)(T_dense + bx * 128 + ty * 64 + tz * 16 + loopid * 2) = r_vmaxf;
  }
}

// grid=(10000,1,1),  block=(32,2,4)
extern "C" __global__ void __launch_bounds__(256) fused_gemm_bn_relu_max_v2_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, float* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half placeholder_shared[2560];
  __shared__ half placeholder_d_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[2];
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000000000000000000000000000000000000000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    *(uint4*)(placeholder_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder + ((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(placeholder_d_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder_1 + (((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[((((int)threadIdx.y) * 1280) + (k_outer_inner * 16))])), 40);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax0_outer], (&(placeholder_d_shared[(((((int)threadIdx.z) * 640) + (ax0_outer * 320)) + (k_outer_inner * 16))])), 40);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[(((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 16)) + (ax1_outer_inner * 8))])), T_dense_wmma_accumulator[ax1_outer_inner], 72, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  
  // BN RELU MAX REDUCE
  int constexpr loopNum = 8;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    half2 r_val = *(half2*)(placeholder_d_shared + ty * 32 * 72 + tx * 72 + tz * 16 + loopid * 2);
    half2 r_gamma = *(half2*)(g + tz * 16 + loopid * 2);
    half2 r_beta = *(half2*)(b + tz * 16 + loopid * 2);
    half2 r_vmax;
    float2 r_vmaxf;

    // bn relu
    // r_vmax.x = max(__half(0x0000), __float2half(__half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x)));
    // r_vmax.y = max(__half(0x0000), __float2half(__half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y)));
    r_vmax.x = max(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = max(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = max(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = max(r_vmax.y, r_vmax_shfl.y);
    }
    r_vmaxf.x = __half2float(r_vmax.x);
    r_vmaxf.y = __half2float(r_vmax.y);

    // sts
    if (tx == 0) *(float2*)((float*)(placeholder_shared) + ty * 64 + tz * 16 + loopid * 2) = r_vmaxf;
  }
  __syncwarp();
  // stg
  if (tx < 16) *(float*)(T_dense + bx * 128 + ty * 64 + tz * 16 + tx) = *(float*)((float*)(placeholder_shared) + ty * 64 + tz * 16 + tx);
}

// grid=(10000,1,1),  block=(32,2,4)
extern "C" __global__ void __launch_bounds__(256) fused_gemm_bn_relu_max_v3_kernel(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ g, half* __restrict__ b, float* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half placeholder_shared[2560];
  __shared__ half placeholder_d_shared[4608];
  // __shared__ half s_buffer[2*32*33*2];
  // __shared__ half s_g[64];
  // __shared__ half s_b[64];
  // if (ty == 0 && tz == 0) {
  //   *(half2*)(s_g + tx * 2) = *(half2*)(g + tx * 2);
  //   *(half2*)(s_b + tx * 2) = *(half2*)(b + tx * 2);
  // }
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[2];
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000000000000000000000000000000000000000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    *(uint4*)(placeholder_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder + ((((((((int)blockIdx.x) * 4096) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(placeholder_d_shared + ((((((int)threadIdx.z) * 640) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4*)(placeholder_1 + (((((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 2) * 64)) + (k_outer_outer * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], (&(placeholder_shared[((((int)threadIdx.y) * 1280) + (k_outer_inner * 16))])), 40);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax0_outer], (&(placeholder_d_shared[(((((int)threadIdx.z) * 640) + (ax0_outer * 320)) + (k_outer_inner * 16))])), 40);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    nvcuda::wmma::store_matrix_sync((&(placeholder_d_shared[(((((int)threadIdx.y) * 2304) + (((int)threadIdx.z) * 16)) + (ax1_outer_inner * 8))])), T_dense_wmma_accumulator[ax1_outer_inner], 72, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();

  // copy shared memory to avoid bank conflict
  // #pragma unroll
  // for (int loopid = 0; loopid < 8; ++loopid) {
  //   *(half2*)(s_buffer + ty * 32 * 66 + tz * 8 * 66 + loopid * 66 + tx * 2) = *(half2*)(placeholder_d_shared + ty * 32 * 72 + tz * 8 * 72 + loopid * 72 + tx * 2);
  // }
  // __syncthreads();
  
  // BN RELU MAX REDUCE
  int constexpr loopNum = 8;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    // half2 r_val = *(half2*)(s_buffer + ty * 32 * 66 + tx * 66 + tz * 16 + loopid * 2);
    half2 r_val = *(half2*)(placeholder_d_shared + ty * 32 * 72 + tx * 72 + tz * 16 + loopid * 2);
    half2 r_gamma = *(half2*)(g + tz * 16 + loopid * 2);
    half2 r_beta = *(half2*)(b + tz * 16 + loopid * 2);
    half2 r_vmax;
    float2 r_vmaxf;

    // bn relu
    // r_vmax.x = max(__half(0x0000), __float2half(__half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x)));
    // r_vmax.y = max(__half(0x0000), __float2half(__half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y)));
    r_vmax.x = max(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = max(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = max(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = max(r_vmax.y, r_vmax_shfl.y);
    }
    r_vmaxf.x = __half2float(r_vmax.x);
    r_vmaxf.y = __half2float(r_vmax.y);

    // sts
    if (tx == 0) *(float2*)((float*)(placeholder_shared) + ty * 64 + tz * 16 + loopid * 2) = r_vmaxf;

    // // stg
    // if (tx == 0) *(float2*)((float*)(T_dense) + bx *128 + ty * 64 + tz * 16 + loopid * 2) = r_vmaxf;
  }
  __syncwarp();
  // stg
  if (tx < 16) *(float*)(T_dense + bx * 128 + ty * 64 + tz * 16 + tx) = *(float*)((float*)(placeholder_shared) + ty * 64 + tz * 16 + tx);
}

void _launch_fused_gemm_bn_relu_max_kernel(int MD, half* __restrict__ inp, half* __restrict__ w, half* __restrict__ g, half* __restrict__ b, float* __restrict__ outGemm, cudaStream_t stream) {
  // fused_gemm_bn_relu_max_kernel<<<dim3(MD/2,1,1), dim3(32,2,4), 0, stream>>>(inp, w, g, b, outGemm);
  // fused_gemm_bn_relu_max_v2_kernel<<<dim3(MD/2,1,1), dim3(32,2,4), 0, stream>>>(inp, w, g, b, outGemm);
  fused_gemm_bn_relu_max_v3_kernel<<<dim3(MD/2,1,1), dim3(32,2,4), 0, stream>>>(inp, w, g, b, outGemm);
}

/////////////////////////////////////////////////////////////////////////////////////////
void fused_reshape_cast_pad(int MD, float* inp, half* outPad, cudaStream_t stream) {
  _launch_fused_reshape_cast_pad_kernel(MD, inp, outPad, stream);
  // cuAssert(cudaStreamSynchronize(stream));
  cuAssert(cudaGetLastError());
}

void fused_gemm1(int MD, half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream) {
  _launch_fused_gemm1_kernel(MD, inp, w, g, b, outGemm, stream);
  // cuAssert(cudaStreamSynchronize(stream));
  cuAssert(cudaGetLastError());
}

void fused_gemm_bn_relu_max_tile(int MD, half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream) {
  _launch_fused_gemm_bn_relu_max_tile_kernel(MD, inp, w, g, b, outGemm, stream);
  // cuAssert(cudaStreamSynchronize(stream));
  cuAssert(cudaGetLastError());
}

void fused_gemm2(int MD, half* inp, half* w, half* g, half* b, half* outGemm, cudaStream_t stream) {
  _launch_fused_gemm2_kernel(MD, inp, w, g, b, outGemm, stream);
  // cuAssert(cudaStreamSynchronize(stream));
  cuAssert(cudaGetLastError());
}

void fused_gemm_bn_relu_max(int MD, half* inp, half* w, half* g, half* b, float* outGemm, cudaStream_t stream) {
  _launch_fused_gemm_bn_relu_max_kernel(MD, inp, w, g, b, outGemm, stream);
  // cuAssert(cudaStreamSynchronize(stream));
  cuAssert(cudaGetLastError());
}
