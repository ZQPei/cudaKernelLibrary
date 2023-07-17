#include <cuda.h>


template<int X>
__host__ __device__ __inline__ getBits() {
  switch(X) {
    #define CASE_X(x) { case (1<<x): return (x); break; }
    CASE_X(1); CASE_X(2); CASE_X(3); CASE_X(4); CASE_X(5); CASE_X(6); CASE_X(7); CASE_X(8); CASE_X(9); CASE_X(10);
    #undef CASE_X
  }
}

template<int M, int N, int K>
__global__ void sgemm_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  dim3 constexpr blockSz(16, 16, 1);
  dim3 constexpr gridSz((N+15)/16, (M+15)/16, 1);
  int constexpr BM = 128;  // tunable
  int constexpr BN = 128;  // tunable
  int constexpr BK = 8;
  int constexpr TM = 8;
  int constexpr TN = 8;

  __shared__ float s_a[BK][BM];
  __shared__ float s_b[BK][BM];

  float r_c[TM][TN] = {0.0f};

  #pragma unroll
  for (int bk = 0; bk < K/BK; ++bk) {
    load_gmem_to_s_a();
    load_gmem_to_s_b();

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
          r_c[m][n] += s_a[(k << getBits<TM>()) + i] * s_b[(k << getBits<TN>()) + j];
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      store_r_c_to_gmem();
    }
  }
}

template<int M, int N, int K>
void _launch_sgemm_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int constexpr BM = 128;  // tunable
  int constexpr BN = 128;  // tunable
  int constexpr BK = 8;
  int constexpr TM = 8;
  int constexpr TN = 8;
  dim3 constexpr blockSz(16, 16, 1);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM, 1);
  sgemm_kernel<M,N,K><<<gridSz, blockSz>>>(A, B, C);
}
