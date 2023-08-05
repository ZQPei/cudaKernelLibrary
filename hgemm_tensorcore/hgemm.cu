#include <cuda.h>
#include <cublas_v2.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

template<int X>
__host__ __device__ __inline__ int constexpr getBits() {
  switch (X) {
    #define CASE_X(bits) { case (1<<(bits)): return bits; break; }
    CASE_X(0); CASE_X(1); CASE_X(2); CASE_X(3); CASE_X(4); CASE_X(5); CASE_X(6); CASE_X(7); CASE_X(8); CASE_X(9);
    CASE_X(10); CASE_X(11); CASE_X(12); CASE_X(13); CASE_X(14); CASE_X(15); CASE_X(16); CASE_X(17); CASE_X(18); CASE_X(19);
    #undef CASE_X
    default: return 0;
  }
}

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
// #define OFFSET(row, col, stride) (((row) << (getBits<(stride)>())) + (col))
#define OFFSET(row, col, stride) (((row) * stride) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

////////////////////////////////////////////////////////////////////////
// 1. hgemm tensorcore
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
template<int M, int N, int K,int BM, int BN, int BK>
__global__ void hgemm_tensorcore_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  int constexpr blockSz = 256;

  int constexpr WM = 64;
  int constexpr WN = 64;
  int constexpr WK = 32;
  static_assert((WM<=BM) && (WN<=BN) && (WK<=BK), "shape error");

  // use memory padding here to avoid shared memory bank conflict
  // select `padding=8` because ldm of load_matrix_sync must be a multiple of 8 for __half element type or multiple of 4 for float element type
  int constexpr PAD_A = 8;
  int constexpr PAD_B = 8;
  __shared__ half s_a[BM][(BK+PAD_A)];
  __shared__ half s_b[BK][(BN+PAD_B)];

  int const tid = threadIdx.x;
  int const wid = tid >> 5;
  // int const laneid = tid & 31;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[WM/16][WK/16];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[WK/16][WN/16];
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[WM/16][WN/16];

  // initialize output to zero
  #pragma unroll
  for (int m = 0; m < WM/16; ++m) {
    #pragma unroll
    for (int n = 0; n < WN/16; ++n) {
      wmma::fill_fragment(frag_c[m][n], (half)0.0);
    }
  }

  // ldg, sts, mma
  #pragma unroll
  for (int bk = 0; bk < K/BK; ++bk) {
    // ldg sts
    *(float4*)&s_a[0 * 64 + (tid>>2)][(tid&3) * 8] = *(float4*)(A + by * BM * K + (0 * 64 + (tid>>2)) * K + bk * BK + (tid&3) * 8);
    *(float4*)&s_a[1 * 64 + (tid>>2)][(tid&3) * 8] = *(float4*)(A + by * BM * K + (1 * 64 + (tid>>2)) * K + bk * BK + (tid&3) * 8);
    *(float4*)&s_b[(tid>>3)][0 * 8 * 8 + (tid&7) * 8] = *(float4*)(B + bk * BK * N + (tid>>3) * N + bx * BN + (0 * 8 * 8 + (tid&7) * 8));
    *(float4*)&s_b[(tid>>3)][1 * 8 * 8 + (tid&7) * 8] = *(float4*)(B + bk * BK * N + (tid>>3) * N + bx * BN + (1 * 8 * 8 + (tid&7) * 8));
    *(float4*)&s_b[(tid>>3)][2 * 8 * 8 + (tid&7) * 8] = *(float4*)(B + bk * BK * N + (tid>>3) * N + bx * BN + (2 * 8 * 8 + (tid&7) * 8));
    *(float4*)&s_b[(tid>>3)][3 * 8 * 8 + (tid&7) * 8] = *(float4*)(B + bk * BK * N + (tid>>3) * N + bx * BN + (3 * 8 * 8 + (tid&7) * 8));

    __syncthreads();

    // load matrix
    #pragma unroll
    for (int m = 0; m < WM/16; ++m) {
      #pragma unroll
      for (int k = 0; k < WK/16; ++k) {
        wmma::load_matrix_sync(frag_a[m][k], &s_a[(wid>>2) * 64 + m * 16][k * 16], BK+PAD_A);
      }
    }
    #pragma unroll
    for (int k = 0; k < WM/16; ++k) {
      #pragma unroll
      for (int n = 0; n < WN/16; ++n) {
        wmma::load_matrix_sync(frag_b[k][n], &s_b[k * 16][(wid&3) * 64 + n * 16], BN+PAD_B);
      }
    }

    // mma
    #pragma unroll
    for (int m = 0; m < WM/16; ++m) {
      #pragma unroll
      for (int n = 0; n < WN/16; ++n) {
        #pragma unroll
        for (int k = 0; k < WK/16; ++k) {
          wmma::mma_sync(frag_c[m][n], frag_a[m][k], frag_b[k][n], frag_c[m][n]);
        }
      }
    }

    __syncthreads();
  }

  // stg
  #pragma unroll
  for (int m = 0; m < WM/16; ++m) {
    #pragma unroll
    for (int n = 0; n < WN/16; ++n) {
      wmma::store_matrix_sync(C + by * BM * N + (wid>>2) * WM * N + m * 16 * N + bx * BN + (wid&3) * WN + n * 16, frag_c[m][n], N, wmma::mem_row_major);
    }
  }
}

template<int M, int N, int K>
void _launch_hgemm_tensorcore_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C, cudaStream_t stream) {
  int constexpr BM = 128;
  int constexpr BN = 256;
  int constexpr BK = 32;
  static_assert((M&(BM-1))==0 && (N&(BN-1))==0 && (K&(BK-1))==0, "M, N, K shape mismatch");

  int constexpr blockSz = 256;
  dim3 const gridSz = {N/BN, M/BM};
  hgemm_tensorcore_kernel<M,N,K,BM,BN,BK><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}


////////////////////////////////////////////////////////////////////////
// benchmark: cublas
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
// cublas
template<int M, int N, int K>
void _launch_hgemm_cudnn_kernel(half* __restrict__ a, half* __restrict__ b, half* __restrict__ c, cudaStream_t stream) {
  static cublasHandle_t cublas_handle = nullptr;
  if (cublas_handle == nullptr) {
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
  }
  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  half alpha = 1.0;
  half beta = 0;
  cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
        &alpha, b, CUDA_R_16F, N, a, CUDA_R_16F, K, &beta, c, CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F, algo);
}


////////////////////////////////////////////////////////////////////////
void hgemm_cuda(half* __restrict__ A, half* __restrict__ B, half* __restrict__ AT, half* __restrict__ BT, half* __restrict__ C, int M, int N, int K, cudaStream_t stream) {
  #define IF_STAT if (false)
  #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_hgemm_tensorcore_kernel<(m), (n), (k)>(A, B, C, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_hgemm_cudnn_kernel<(m), (n), (k)>(A, B, C, stream)
  #define ELSE_STAT else { std::cout << "NOT_IMPLEMENTED" << std::endl; __builtin_trap(); }

  IF_STAT;
  ELIF_STAT(1024, 1024, 1024);
  ELSE_STAT;

  #undef IF_STAT
  #undef ELIF_STAT
  #undef ELSE_STAT

  // cudaStreamSynchronize(stream);
}
