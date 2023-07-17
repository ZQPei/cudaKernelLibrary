#include <cuda_fp16.h>
#include <mma.h>

// grid=(10000,1,1),  block=(32,4,2)
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ T_dense) {
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

// grid=(10000,1,1),  block=(32,2,4)
extern "C" __global__ void __launch_bounds__(256) default_function_kernel1(half* __restrict__ placeholder, half* __restrict__ placeholder_1, half* __restrict__ T_dense) {
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
