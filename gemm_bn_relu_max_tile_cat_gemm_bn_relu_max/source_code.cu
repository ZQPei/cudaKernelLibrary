#include <cuda.h>
#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(1024) tvmgen_default_fused_reshape_cast_nn_pad_kernel0(half* __restrict__ T_pad, float* __restrict__ p0) {
  for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 20; ++ax0_ax1_fused_outer_outer) {
    for (int ax0_ax1_fused_inner_s = 0; ax0_ax1_fused_inner_s < 2; ++ax0_ax1_fused_inner_s) {
      T_pad[((((ax0_ax1_fused_outer_outer * 512000) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.x) * 2)) + ax0_ax1_fused_inner_s)] = (((((int)threadIdx.x) & 7) < 5) ? ((half)p0[(((((ax0_ax1_fused_outer_outer * 320000) + (((int)blockIdx.x) * 1280)) + ((((int)threadIdx.x) >> 3) * 10)) + ((((int)threadIdx.x) & 7) * 2)) + ax0_ax1_fused_inner_s)]) : __float2half_rn(0.000000000000000000000000000000000000000000000e+00f));
    }
  }
}

// tvmgen_default_fused_reshape_cast_nn_pad_kernel0<<<(1024, 1, 1), (250, 1, 1)>>>

extern "C" __global__ void __launch_bounds__(32) tvmgen_default_fused_nn_dense_kernel0(half* __restrict__ p0, half* __restrict__ p1, float* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, float> T_dense_wmma_accumulator[1];
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
  nvcuda::wmma::store_matrix_sync((&(((float*)p0_shared)[0])), T_dense_wmma_accumulator[0], 8, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 8; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    T_dense[(((((((int)blockIdx.x) * 1024) + (i_inner_j_inner_fused_outer_outer_outer_outer * 128)) + ((((int)threadIdx.x) >> 3) * 32)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.x) & 7))] = ((float*)p0_shared)[((i_inner_j_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x))];
  }
}

// tvmgen_default_fused_nn_dense_kernel0<<<(20000, 4, 1), (32, 1, 1)>>>

extern "C" __global__ void __launch_bounds__(32) tvmgen_default_fused_nn_dense_1_kernel0(half* __restrict__ p0, half* __restrict__ p1, float* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, float> T_dense_wmma_accumulator[1];
  __shared__ half p0_shared[512];
  __shared__ half p1_shared[128];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> p0_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> p1_shared_wmma_matrix_b[1];
  nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000000000000000000000000000000000000000000e+00f);
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 16; ++ax0_ax1_fused_outer_outer_outer_outer) {
      p0_shared[((ax0_ax1_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x))] = p0[(((((((int)blockIdx.x) * 2048) + (ax0_ax1_fused_outer_outer_outer_outer * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer_1 = 0; ax0_ax1_fused_outer_outer_outer_outer_1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer_1) {
      p1_shared[((ax0_ax1_fused_outer_outer_outer_outer_1 * 32) + ((int)threadIdx.x))] = p1[(((((((int)blockIdx.y) * 512) + (ax0_ax1_fused_outer_outer_outer_outer_1 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (k_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    }
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(p0_shared_wmma_matrix_a[0], (&(p0_shared[0])), 16);
    nvcuda::wmma::load_matrix_sync(p1_shared_wmma_matrix_b[0], (&(p1_shared[0])), 16);
    nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], p0_shared_wmma_matrix_a[0], p1_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  }
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((float*)p0_shared)[0])), T_dense_wmma_accumulator[0], 8, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 8; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    T_dense[(((((((int)blockIdx.x) * 2048) + (i_inner_j_inner_fused_outer_outer_outer_outer * 256)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)blockIdx.y) * 8)) + (((int)threadIdx.x) & 7))] = ((float*)p0_shared)[((i_inner_j_inner_fused_outer_outer_outer_outer * 32) + ((int)threadIdx.x))];
  }
}

// tvmgen_default_fused_nn_dense_1_kernel0<<<(20000, 8, 1), (32, 1, 1)>>>
