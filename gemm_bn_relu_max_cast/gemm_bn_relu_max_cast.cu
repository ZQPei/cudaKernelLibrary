#include <cuda.h>
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

__device__ half hmaxh(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half hminh(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_cast_nn_pad_kernel0(half* __restrict__ T_pad, float* __restrict__ p0) {
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 18; ++ax0_ax1_fused_outer_outer) {
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 2; ++ax0_ax1_fused_inner_s) {
      T_pad[((((ax0_ax1_fused_outer_outer * 512000) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_inner_s)] = (((((int)threadIdx.x) & 7) < 5) ? ((half)p0[(((((ax0_ax1_fused_outer_outer * 320000) + (((int)blockIdx.x) * 1280)) + ((((int)threadIdx.x) >> 3) * 10)) + ((((int)threadIdx.x) & 7) * 2)) + ax0_ax1_fused_inner_s)]) : __float2half_rn(0.000000000000000000000000000000000000000000000e+00f));
    }
  }
}

extern "C" __global__ void __launch_bounds__(32) tvmgen_default_fused_nn_dense_cast_kernel0(half* __restrict__ p0, half* __restrict__ p1, float* __restrict__ T_cast) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[1];
  __shared__ half p0_shared[512];
  __shared__ half p1_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> p0_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> p1_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000000000000000000000000000000000000000000e+00f);
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 16; ++ax0_ax1_fused_outer_outer_outer_outer) {
    p0_shared[((ax0_ax1_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x))] = p0[(((((int)blockIdx.x) * 512) + (ax0_ax1_fused_outer_outer_outer_outer * 32)) + ((int)threadIdx.x))];
  }
  for (int ax0_ax1_fused_outer_outer_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_outer_outer_1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer_1) {
    p1_shared[((ax0_ax1_fused_outer_outer_outer_outer_1 * 32) + ((int)threadIdx.x))] = p1[(((((int)blockIdx.y) * 128) + (ax0_ax1_fused_outer_outer_outer_outer_1 * 32)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(p0_shared_wmma_matrix_a[0], (&(p0_shared[0])), 16);
  nvcuda::wmma::load_matrix_sync(p1_shared_wmma_matrix_b[0], (&(p1_shared[0])), 16);
  nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], p0_shared_wmma_matrix_a[0], p1_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(p0_shared[0])), T_dense_wmma_accumulator[0], 8, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int ax0_inner_ax1_inner_fused_outer_outer_outer_outer = 0; ax0_inner_ax1_inner_fused_outer_outer_outer_outer < 8; ++ax0_inner_ax1_inner_fused_outer_outer_outer_outer) {
    T_cast[(((((((int)blockIdx.x) * 2048) + (ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.x) & 7))] = ((float)p0_shared[((ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x))]);
  }
}

extern "C" __global__ void __launch_bounds__(32) tvmgen_default_fused_nn_dense_cast_bn_relu_max_kernel0(half* __restrict__ p0, half* __restrict__ p1, half* __restrict__ p_gamma, half* __restrict__ p_beta, float* __restrict__ T_cast) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[1];
  __shared__ half p0_shared[512];
  __shared__ half p1_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> p0_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> p1_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000000000000000000000000000000000000000000e+00f);
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 16; ++ax0_ax1_fused_outer_outer_outer_outer) {
    p0_shared[((ax0_ax1_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x))] = p0[(((((int)blockIdx.x) * 512) + (ax0_ax1_fused_outer_outer_outer_outer * 32)) + ((int)threadIdx.x))];
  }
  for (int ax0_ax1_fused_outer_outer_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_outer_outer_1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer_1) {
    p1_shared[((ax0_ax1_fused_outer_outer_outer_outer_1 * 32) + ((int)threadIdx.x))] = p1[(((((int)blockIdx.y) * 128) + (ax0_ax1_fused_outer_outer_outer_outer_1 * 32)) + ((int)threadIdx.x))];
  }
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(p0_shared_wmma_matrix_a[0], (&(p0_shared[0])), 16);
  nvcuda::wmma::load_matrix_sync(p1_shared_wmma_matrix_b[0], (&(p1_shared[0])), 16);
  nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], p0_shared_wmma_matrix_a[0], p1_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(p0_shared[0])), T_dense_wmma_accumulator[0], 8, nvcuda::wmma::mem_row_major);
  // store global
  // __syncthreads();
  // for (int ax0_inner_ax1_inner_fused_outer_outer_outer_outer = 0; ax0_inner_ax1_inner_fused_outer_outer_outer_outer < 8; ++ax0_inner_ax1_inner_fused_outer_outer_outer_outer) {
  //   T_cast[(((((((int)blockIdx.x) * 2048) + (ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.x) & 7))] = ((float)p0_shared[((ax0_inner_ax1_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x))]);
  // }

  __syncthreads();
  int constexpr loopNum = 8;  // 64 / 8 = 8
  int constexpr shflNum = 5;  // 32
  #pragma unroll
  for (int loopId = 0; loopId < loopNum; ++loopId) {
    float gamma = __half2float(p_gamma[blockIdx.y * 8 + loopId]);
    float beta = __half2float(p_beta[blockIdx.y * 8 + loopId]);
    float vmax = fmaxf(0.f, __half2float(p0_shared[threadIdx.x * 8 + loopId]) * gamma + beta);

    // max
    #pragma unroll
    for (int _ = 0, delta = 1; _ < shflNum; ++_, delta <<= 1) {
      vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, delta));
    }

    // // debug
    // if (blockIdx.x == 0 && blockIdx.y == 1 && loopId == 0) {
    //   printf("%d %f %f\n", threadIdx.x, vmax, __half2float(p0_shared[threadIdx.x * 8 + loopId]) * gamma + beta);
    // }

    // stg
    if (threadIdx.x == 0) T_cast[blockIdx.x * 64 + blockIdx.y * 8 + loopId] = vmax;
  }
}

// grid=(4500,1,1),  block=(32,2,2)
extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_dense_cast_bn_relu_max_v2_kernel0(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ p_gamma, half* __restrict__ p_beta, float* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[8];
  __shared__ half placeholder_shared[8192];
  __shared__ half placeholder_d_shared[1536];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[2];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000000000000000000000000000000000000000000e+00f);
    }
  }
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
    *(uint2*)(placeholder_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 768) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 3) * 4))) = *(uint2*)(placeholder + (((((((int)blockIdx.x) * 2048) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  }
  for (int ax0_ax1_fused_outer_outer_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_outer_outer_1 < 2; ++ax0_ax1_fused_outer_outer_outer_outer_1) {
    *(uint2*)(placeholder_d_shared + (((((ax0_ax1_fused_outer_outer_outer_outer_1 * 768) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 3) * 4))) = *(uint2*)(placeholder_1 + ((((ax0_ax1_fused_outer_outer_outer_outer_1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
    nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[ax0_outer], (&(placeholder_shared[((((int)threadIdx.y) * 1536) + (ax0_outer * 384))])), 24);
  }
  for (int ax0_outer_1 = 0; ax0_outer_1 < 2; ++ax0_outer_1) {
    nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax0_outer_1], (&(placeholder_d_shared[((((int)threadIdx.z) * 768) + (ax0_outer_1 * 384))])), 24);
  }
  for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
    for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
      nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], placeholder_shared_wmma_matrix_a[i_c_outer], placeholder_d_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
    }
  }
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 4; ++ax0_outer_inner) {
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
      nvcuda::wmma::store_matrix_sync((&(placeholder_shared[((((((int)threadIdx.y) * 4096) + (ax0_outer_inner * 1024)) + (((int)threadIdx.z) * 32)) + (ax1_outer_inner * 16))])), T_dense_wmma_accumulator[((ax0_outer_inner * 2) + ax1_outer_inner)], 64, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 16; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
  //   *(uint2*)(T_dense + (((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))) = *(uint2*)(placeholder_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  // }


  // // copy shared memory to avoid bank conflict

  // __syncthreads();

  // BN RELU MAX REDUCE
  int constexpr loopNum = 32;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    half2 r_val = *(half2*)(placeholder_shared + ((tz << 1) + ty) * 32 * 64 + tx * 64 + loopid * 2);
    half2 r_gamma = *(half2*)(p_gamma + loopid * 2);
    half2 r_beta = *(half2*)(p_beta + loopid * 2);
    half2 r_vmax;
    float2 r_vmaxf;

    // bn relu
    r_vmax.x = hmaxh(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = hmaxh(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = hmaxh(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = hmaxh(r_vmax.y, r_vmax_shfl.y);
    }
    r_vmaxf.x = __half2float(r_vmax.x);
    r_vmaxf.y = __half2float(r_vmax.y);

    // half2 r_val = *(half2*)(placeholder_shared + ((tz << 1) + ty) * 32 * 64 + tx * 64 + loopid * 2);
    // half2 r_gamma = *(half2*)(p_gamma + loopid * 2);
    // half2 r_beta = *(half2*)(p_beta + loopid * 2);
    // float2 r_vmaxf;

    // // bn relu
    // r_vmaxf.x = max(0.f, __half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x));
    // r_vmaxf.y = max(0.f, __half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y));

    // // max on warp level
    // #pragma unroll
    // for (int boff = 0; boff < shflNum; ++boff) {
    //   float r_vmaxf_shfl_x = __shfl_xor_sync(0xffffffff, r_vmaxf.x, 1<<boff);
    //   float r_vmaxf_shfl_y = __shfl_xor_sync(0xffffffff, r_vmaxf.y, 1<<boff);
    //   r_vmaxf.x = max(r_vmaxf.x, r_vmaxf_shfl_x);
    //   r_vmaxf.y = max(r_vmaxf.y, r_vmaxf_shfl_y);
    // }

    // sts
    if (tx == 0) *(float2*)((float*)(placeholder_d_shared) + ((tz << 1) + ty) * 64 + loopid * 2) = r_vmaxf;

    // if (tx == 0) *(float2*)(T_dense + bx * 4 * 64 + ((tz << 1) + ty) * 64 + loopid * 2) = r_vmaxf;
  }
  __syncwarp();

  // stg
  int constexpr gloopNum = 2;
  #pragma unroll
  for (int gloopid = 0; gloopid < gloopNum; ++gloopid) {
    *(float*)(T_dense + bx * 4 * 64 + ((tz << 1) + ty) * 64 + gloopid * 32 + tx) = *(float*)((float*)(placeholder_d_shared) + ((tz << 1) + ty) * 64 + gloopid * 32 + tx);
  }
}

// grid=(4500,1,1),  block=(32,2,2)
extern "C" __global__ void __launch_bounds__(128) tvmgen_default_fused_nn_dense_cast_bn_relu_max_v3_kernel0(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ p_gamma, half* __restrict__ p_beta, float* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[8];
  int const tyz = (tz << 1) + ty;
  __shared__ half placeholder_shared[8192];
  __shared__ half placeholder_d_shared[1536];
  __shared__ half s_buffer[4*32*33*2];
  __shared__ half s_gamma[64];
  __shared__ half s_beta[64];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[2];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000000000000000000000000000000000000000000e+00f);
    }
  }
  for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
    *(uint2*)(placeholder_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 768) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 3) * 4))) = *(uint2*)(placeholder + (((((((int)blockIdx.x) * 2048) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  }
  for (int ax0_ax1_fused_outer_outer_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_outer_outer_1 < 2; ++ax0_ax1_fused_outer_outer_outer_outer_1) {
    *(uint2*)(placeholder_d_shared + (((((ax0_ax1_fused_outer_outer_outer_outer_1 * 768) + (((int)threadIdx.z) * 384)) + (((int)threadIdx.y) * 192)) + ((((int)threadIdx.x) >> 2) * 24)) + ((((int)threadIdx.x) & 3) * 4))) = *(uint2*)(placeholder_1 + ((((ax0_ax1_fused_outer_outer_outer_outer_1 * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
    nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[ax0_outer], (&(placeholder_shared[((((int)threadIdx.y) * 1536) + (ax0_outer * 384))])), 24);
  }
  for (int ax0_outer_1 = 0; ax0_outer_1 < 2; ++ax0_outer_1) {
    nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax0_outer_1], (&(placeholder_d_shared[((((int)threadIdx.z) * 768) + (ax0_outer_1 * 384))])), 24);
  }
  for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
    for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
      nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], placeholder_shared_wmma_matrix_a[i_c_outer], placeholder_d_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
    }
  }
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 4; ++ax0_outer_inner) {
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
      nvcuda::wmma::store_matrix_sync((&(placeholder_shared[((((((int)threadIdx.y) * 4096) + (ax0_outer_inner * 1024)) + (((int)threadIdx.z) * 32)) + (ax1_outer_inner * 16))])), T_dense_wmma_accumulator[((ax0_outer_inner * 2) + ax1_outer_inner)], 64, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 16; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
  //   *(uint2*)(T_dense + (((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer_outer * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4))) = *(uint2*)(placeholder_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 512) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)));
  // }

  // copy shared memory to avoid bank conflict
  #pragma unroll
  for (int loopid = 0; loopid < 32; ++loopid) {
    *(float*)(s_buffer + tyz * 32 * 66 + loopid * 33 * 2 + tx * 2) = *(float*)(placeholder_shared + tyz * 32 * 64 + loopid * 32 * 2 + tx * 2);
  }
  if (tyz == 0) {
    *(half2*)(s_gamma + tx * 2) = *(half2*)(p_gamma + tx * 2);
    *(half2*)(s_beta + tx * 2) = *(half2*)(p_beta + tx * 2);
  }
  __syncthreads();

  // BN RELU MAX REDUCE
  int constexpr loopNum = 32;
  int constexpr shflNum = 5;

  #pragma unroll
  for (int loopid = 0; loopid < loopNum; ++loopid) {
    half2 r_val = *(half2*)(s_buffer + tyz * 32 * 66 + tx * 66 + loopid * 2);
    half2 r_gamma = *(half2*)(s_gamma + loopid * 2);
    half2 r_beta = *(half2*)(s_beta + loopid * 2);
    half2 r_vmax;
    float2 r_vmaxf;

    // bn relu
    r_vmax.x = hmaxh(__half(0x0000), __hadd(__hmul(r_val.x, r_gamma.x), r_beta.x));
    r_vmax.y = hmaxh(__half(0x0000), __hadd(__hmul(r_val.y, r_gamma.y), r_beta.y));

    // max on warp level
    #pragma unroll
    for (int boff = 0; boff < shflNum; ++boff) {
      half2 r_vmax_shfl = __shfl_xor_sync(0xffffffff, r_vmax, 1<<boff);
      r_vmax.x = hmaxh(r_vmax.x, r_vmax_shfl.x);
      r_vmax.y = hmaxh(r_vmax.y, r_vmax_shfl.y);
    }
    // #pragma unroll
    // for (int i = 16; i > 0; i >>= 1) {
    //   half2 r_vmax_shfl = __shfl_down_sync(0xffffffff, r_vmax, i);
    //   r_vmax.x = hmaxh(r_vmax.x, r_vmax_shfl.x);
    //   r_vmax.y = hmaxh(r_vmax.y, r_vmax_shfl.y);
    // }
    r_vmaxf.x = __half2float(r_vmax.x);
    r_vmaxf.y = __half2float(r_vmax.y);

    // half2 r_val = *(half2*)(placeholder_shared + tyz * 32 * 64 + tx * 64 + loopid * 2);
    // half2 r_gamma = *(half2*)(p_gamma + loopid * 2);
    // half2 r_beta = *(half2*)(p_beta + loopid * 2);
    // float2 r_vmaxf;

    // // bn relu
    // r_vmaxf.x = max(0.f, __half2float(r_val.x) * __half2float(r_gamma.x) + __half2float(r_beta.x));
    // r_vmaxf.y = max(0.f, __half2float(r_val.y) * __half2float(r_gamma.y) + __half2float(r_beta.y));

    // // max on warp level
    // #pragma unroll
    // for (int boff = 0; boff < shflNum; ++boff) {
    //   float r_vmaxf_shfl_x = __shfl_xor_sync(0xffffffff, r_vmaxf.x, 1<<boff);
    //   float r_vmaxf_shfl_y = __shfl_xor_sync(0xffffffff, r_vmaxf.y, 1<<boff);
    //   r_vmaxf.x = max(r_vmaxf.x, r_vmaxf_shfl_x);
    //   r_vmaxf.y = max(r_vmaxf.y, r_vmaxf_shfl_y);
    // }

    // sts
    // if (tx == 0) *(float2*)((float*)(placeholder_d_shared) + tyz * 64 + loopid * 2) = r_vmaxf;

    if (tx == 0) *(float2*)(T_dense + bx * 4 * 64 + tyz * 64 + loopid * 2) = r_vmaxf;
  }
  // __syncwarp();

  // // stg
  // int constexpr gloopNum = 2;
  // #pragma unroll
  // for (int gloopid = 0; gloopid < gloopNum; ++gloopid) {
  //   *(float*)(T_dense + bx * 4 * 64 + tyz * 64 + gloopid * 32 + tx) = *(float*)((float*)(placeholder_d_shared) + tyz * 64 + gloopid * 32 + tx);
  // }
}

void fused_nn_dense_cast(float* __restrict__ p0, half* __restrict__ T_pad, half* __restrict__ p1, float* __restrict__ T_cast, cudaStream_t stream) {
  tvmgen_default_fused_reshape_cast_nn_pad_kernel0<<<dim3(250, 1, 1), dim3(1024, 1, 1), 0, stream>>>(T_pad, p0);
  tvmgen_default_fused_nn_dense_cast_kernel0<<<dim3(18000, 8, 1), dim3(32, 1, 1), 0, stream>>>(T_pad, p1, T_cast);
  // cudaStreamSynchronize(stream);
  cuAssert(cudaGetLastError());
}

void fused_nn_dense_bn_relu_max_cast(float* __restrict__ p0, half* __restrict__ T_pad, half* __restrict__ p1, half* __restrict__ p_gamma, half* __restrict__ p_beta, float* __restrict__ T_cast, cudaStream_t stream) {
  tvmgen_default_fused_reshape_cast_nn_pad_kernel0<<<dim3(250, 1, 1), dim3(1024, 1, 1), 0, stream>>>(T_pad, p0);
  // tvmgen_default_fused_nn_dense_cast_bn_relu_max_kernel0<<<dim3(18000, 8, 1), dim3(32, 1, 1), 0, stream>>>(T_pad, p1, p_gamma, p_beta, T_cast);
  // tvmgen_default_fused_nn_dense_cast_bn_relu_max_v2_kernel0<<<dim3(4500, 1, 1), dim3(32, 2, 2), 0, stream>>>(T_pad, p1, p_gamma, p_beta, T_cast);
  tvmgen_default_fused_nn_dense_cast_bn_relu_max_v3_kernel0<<<dim3(4500, 1, 1), dim3(32, 2, 2), 0, stream>>>(T_pad, p1, p_gamma, p_beta, T_cast);
  // cudaStreamSynchronize(stream);
  cuAssert(cudaGetLastError());
}
