#include <cuda.h>
#include <cublas_v2.h>
#include <iostream>

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
#define OFFSET(row, col, stride) (((row) << (getBits<(stride)>())) + (col))
// #define OFFSET(row, col, stride) (((row) * stride) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])


////////////////////////////////////////////////////////////////////////
// 1. 
// a: row_major [M, K1]
// b1: row_major [K1, K2]
// b2: row_major [K2, N]
// mid: row_major [M, K2]
// c: row_major [M, N]

// #undef tx
// #undef ty
// #undef tz
// #undef bx
// #undef by
// #undef bz

template<int M, int K1, int K2, int N>
__global__ void __launch_bounds__(256) sgemm_b2b_kernel(float* __restrict__ a, float* __restrict__ b1, float* __restrict__ b2, float* __restrict__ mid, float* __restrict__ c) {
  int constexpr BM = 128;
  int constexpr SPLITK1 = 8;
  int constexpr BK2 = 128;
  int constexpr SPLITK2 = 8;
  int constexpr BN = 128;

  int constexpr bdimx = 16;
  int constexpr bdimy = 16;
  int constexpr TM = BM/bdimy;
  int constexpr TN = BN/bdimx;

  // const int bx = blockIdx.x;
  // const int by = blockIdx.y;
  // const int tx = threadIdx.x;
  // const int ty = threadIdx.y;

  __shared__ float s_a[2*SPLITK1*BM];
  __shared__ float s_b1[2*SPLITK1*BK2];
  // __shared__ float s_b2[2*SPLITK2*BN];
  #define s_mid s_a
  #define s_b2 s_b1
  float r_mid[TM*TN] = {0.0,};
  float r_c[TM*TN] = {0.0,};
  float r_a[TM];
  float r_b[TN];
  float4 r_load_a;
  float4 r_load_b;

  int const tid = (ty * bdimx) + tx;

  // gemm 1
  int s_buf_curr_id = 0;

  // first sts
  {
    int bk = 0;
    // ldg, sts
    r_load_a = *(float4*)(a + by * BM * K1 + (tid >> 1) * K1 + bk * SPLITK1 + (tid & 1) * 4);
    r_load_b = *(float4*)(b1 + bk * SPLITK1 * K2 + (tid >> 5) * K2 + (bx&(K2/BK2-1)) * BK2 + (tid & 31) * 4);
    #pragma unroll
    for (int loopid = 0; loopid < 4; ++loopid) {
      *(float*)(s_a + s_buf_curr_id * SPLITK1 * BM + (tid & 1) * 4 * BM + loopid * BM + (tid >> 1)) = *((float*)&r_load_a + loopid);
    }
    *(float4*)(s_b1 + s_buf_curr_id * SPLITK1 * BK2 + (tid >> 5) * BK2 + (tid & 31) * 4) = r_load_b;
    __syncthreads();
  }

  // main loop
  #pragma unroll
  for (int bk = 1; bk < (K1+SPLITK1-1)/SPLITK1; ++bk) {
    // ldg
    r_load_a = *(float4*)(a + by * BM * K1 + (tid >> 1) * K1 + bk * SPLITK1 + (tid & 1) * 4);
    r_load_b = *(float4*)(b1 + bk * SPLITK1 * K2 + (tid >> 5) * K2 + (bx&(K2/BK2-1)) * BK2 + (tid & 31) * 4);
    // #pragma unroll
    // for (int loopid = 0; loopid < 4; ++loopid) {
    //   *(float*)(s_a + (s_buf_curr_id ^ 1) * SPLITK1 * BM + (tid & 1) * 4 * BM + loopid * BM + (tid >> 1)) = *((float*)&r_load_a + loopid);
    // }
    // *(float4*)(s_b1 + (s_buf_curr_id ^ 1) * SPLITK1 * BK2 + (tid >> 5) * BK2 + (tid & 31) * 4) = r_load_b;

    // mma
    #pragma unroll
    for (int k = 0; k < SPLITK1; ++k) {
      *(float4*)(r_a + 0) = *(float4*)(s_a + s_buf_curr_id * SPLITK1 * BM + k * BM + 0 * 16 * 4 + ty * 4);
      *(float4*)(r_a + 4) = *(float4*)(s_a + s_buf_curr_id * SPLITK1 * BM + k * BM + 1 * 16 * 4 + ty * 4);
      *(float4*)(r_b + 0) = *(float4*)(s_b1 + s_buf_curr_id * SPLITK1 * BK2 + k * BK2 + 0 * 16 * 4 + tx * 4);
      *(float4*)(r_b + 4) = *(float4*)(s_b1 + s_buf_curr_id * SPLITK1 * BK2 + k * BK2 + 1 * 16 * 4 + tx * 4);
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
          *(float*)(r_mid + m * TN + n) += (*(float*)(r_a + m)) * (*(float*)(r_b + n));
        }
      }
    }

    // sts (do sts after mma to hide latency of ldg)
    #pragma unroll
    for (int loopid = 0; loopid < 4; ++loopid) {
      *(float*)(s_a + (s_buf_curr_id ^ 1) * SPLITK1 * BM + (tid & 1) * 4 * BM + loopid * BM + (tid >> 1)) = *((float*)&r_load_a + loopid);
    }
    *(float4*)(s_b1 + (s_buf_curr_id ^ 1) * SPLITK1 * BK2 + (tid >> 5) * BK2 + (tid & 31) * 4) = r_load_b;

    s_buf_curr_id ^= 1;
    __syncthreads();
  }

  // last mma
  #pragma unroll
  for (int k = 0; k < SPLITK1; ++k) {
    *(float4*)(r_a + 0) = *(float4*)(s_a + s_buf_curr_id * SPLITK1 * BM + k * BM + 0 * 16 * 4 + ty * 4);
    *(float4*)(r_a + 4) = *(float4*)(s_a + s_buf_curr_id * SPLITK1 * BM + k * BM + 1 * 16 * 4 + ty * 4);
    *(float4*)(r_b + 0) = *(float4*)(s_b1 + s_buf_curr_id * SPLITK1 * BK2 + k * BK2 + 0 * 16 * 4 + tx * 4);
    *(float4*)(r_b + 4) = *(float4*)(s_b1 + s_buf_curr_id * SPLITK1 * BK2 + k * BK2 + 1 * 16 * 4 + tx * 4);
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
      #pragma unroll
      for (int n = 0; n < TN; ++n) {
        *(float*)(r_mid + m * TN + n) += (*(float*)(r_a + m)) * (*(float*)(r_b + n));
      }
    }
  }

  // stg
  // #pragma unroll
  // for (int i = 0; i < 2; ++i) {
  //   #pragma unroll
  //   for (int m = 0; m < TM/2; ++m) {
  //     #pragma unroll
  //     for (int j = 0; j < 2; ++j) {
  //       *(float4*)(mid + by * BM * K2 + i * (BM>>1) * K2 + ty * (TM>>1) * K2 + m * K2 + (bx&(K2/BK2-1)) * BK2 + j * (BK2>>1) + tx * 4) = *(float4*)(r_mid + i * 32 + m * 8 + j * 4);
  //     }
  //   }
  // }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  // gemm 2
  s_buf_curr_id = 0;

  // first sts
  {
    int bk = 0;
    // ldg, sts
    r_load_b = *(float4*)(b2 + bk * SPLITK2 * N + (tid >> 5) * N + bx * BN + (tid & 31) * 4);

    if (tx / 2 == bk % 8) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int m = 0; m < TM/2; ++m) {
          #pragma unroll
          for (int n = 0; n < TN/2; ++n) {
            *(float*)(s_mid + s_buf_curr_id * SPLITK2 * BM + (tx & 1) * 4 * 128 + n * 128 + i * 64 + ty * 4 + m) = *(float*)(r_mid + i * 4 * 8 + m * 8 + (bk / 8) * 4 + n);
          }
        }
      }
    }
    *(float4*)(s_b2 + s_buf_curr_id * SPLITK2 * BN + (tid >> 5) * BN + (tid & 31) * 4) = r_load_b;
    __syncthreads();
  }

  // main loop
  #pragma unroll
  for (int bk = 1; bk < (K2+SPLITK2-1)/SPLITK2; ++bk) {
    // ldg
    r_load_b = *(float4*)(b2 + bk * SPLITK2 * N + (tid >> 5) * N + bx * BN + (tid & 31) * 4);
    // *(float4*)(s_b2 + (s_buf_curr_id ^ 1) * SPLITK2 * BN + (tid >> 5) * BN + (tid & 31) * 4) = r_load_b;

    // mma
    #pragma unroll
    for (int k = 0; k < SPLITK2; ++k) {
      *(float4*)(r_a + 0) = *(float4*)(s_mid + s_buf_curr_id * SPLITK2 * BM + k * BM + 0 * 16 * 4 + ty * 4);
      *(float4*)(r_a + 4) = *(float4*)(s_mid + s_buf_curr_id * SPLITK2 * BM + k * BM + 1 * 16 * 4 + ty * 4);
      *(float4*)(r_b + 0) = *(float4*)(s_b2 + s_buf_curr_id * SPLITK2 * BN + k * BN + 0 * 16 * 4 + tx * 4);
      *(float4*)(r_b + 4) = *(float4*)(s_b2 + s_buf_curr_id * SPLITK2 * BN + k * BN + 1 * 16 * 4 + tx * 4);
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
          *(float*)(r_c + m * TN + n) += (*(float*)(r_a + m)) * (*(float*)(r_b + n));
        }
      }
    }

    // sts (do sts after mma to hide latency of ldg)
    if (tx / 2 == bk % 8) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int m = 0; m < TM/2; ++m) {
          #pragma unroll
          for (int n = 0; n < TN / 2; ++n) {
            *(float*)(s_mid + (s_buf_curr_id ^ 1) * SPLITK2 * BM + (tx & 1) * 4 * 128 + n * 128 + i * 64 + ty * 4 + m) = *(float*)(r_mid + i * 4 * 8 + m * 8 + (bk / 8) * 4 + n);
          }
        }
      }
    }
    *(float4*)(s_b2 + (s_buf_curr_id ^ 1) * SPLITK2 * BN + (tid >> 5) * BN + (tid & 31) * 4) = r_load_b;

    s_buf_curr_id ^= 1;
    __syncthreads();
  }

  // last mma
  #pragma unroll
  for (int k = 0; k < SPLITK2; ++k) {
    *(float4*)(r_a + 0) = *(float4*)(s_mid + s_buf_curr_id * SPLITK2 * BM + k * BM + 0 * 16 * 4 + ty * 4);
    *(float4*)(r_a + 4) = *(float4*)(s_mid + s_buf_curr_id * SPLITK2 * BM + k * BM + 1 * 16 * 4 + ty * 4);
    *(float4*)(r_b + 0) = *(float4*)(s_b2 + s_buf_curr_id * SPLITK2 * BK2 + k * BK2 + 0 * 16 * 4 + tx * 4);
    *(float4*)(r_b + 4) = *(float4*)(s_b2 + s_buf_curr_id * SPLITK2 * BK2 + k * BK2 + 1 * 16 * 4 + tx * 4);
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
      #pragma unroll
      for (int n = 0; n < TN; ++n) {
        *(float*)(r_c + m * TN + n) += (*(float*)(r_a + m)) * (*(float*)(r_b + n));
      }
    }
  }

  // stg
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int m = 0; m < TM/2; ++m) {
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        *(float4*)(c + by * BM * N + i * (BM>>1) * N + ty * (TM>>1) * N + m * N + bx * BN + j * (BK2>>1) + tx * 4) = *(float4*)(r_c + i * 32 + m * 8 + j * 4);
      }
    }
  }
}

template<int M, int K1, int K2, int N>
void _launch_sgemm_b2b_kernel(float* __restrict__ a, float* __restrict__ b1, float* __restrict__ b2, float* __restrict__ mid, float* __restrict__ c, cudaStream_t stream) {
  static_assert((M&127) == 0 && (K1&7) == 0 && (K2&127) == 0 && (N&127) == 0, "M, K1, K2, N should be divisible by 32");
  int constexpr BM = 128;
  int constexpr SPLITK1 = 8;
  int constexpr BK2 = 128;
  int constexpr SPLITK2 = 8;
  int constexpr BN = 128;

  int constexpr bdimx = 16;
  int constexpr bdimy = 16;
  int constexpr TM = BM/bdimy;
  int constexpr TN = BN/bdimx;

  dim3 constexpr blockSz(bdimx, bdimy);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM);
  sgemm_b2b_kernel<M, K1, K2, N><<<gridSz, blockSz, 0, stream>>>(a, b1, b2, mid, c);
}


////////////////////////////////////////////////////////////////////////
void sgemm_b2b(float* __restrict__ a, float* __restrict__ b1, float* __restrict__ b2, float* __restrict__ mid, float* __restrict__ c, int M, int K1, int K2, int N, cudaStream_t stream) {
  #define IF_STAT if (false)
  #define ELIF_STAT(m, k1, k2, n) else if ((m) == M && (k1) == K1 && (k2) == K2 && (n) == N) _launch_sgemm_b2b_kernel<(m), (k1), (k2), (n)>(a, b1, b2, mid, c, stream)
  #define ELSE_STAT else { std::cerr << "NOT_IMPLEMENTED" << std::endl << std::flush; __builtin_trap(); }

  IF_STAT;
  ELIF_STAT(1024, 128, 128, 128);
  ELIF_STAT(1024, 1024, 128, 128);
  ELIF_STAT(1024, 1024, 128, 1024);
  ELIF_STAT(640000, 128, 128, 128);
  ELSE_STAT;

  #undef IF_STAT
  #undef ELIF_STAT
  #undef ELSE_STAT

  // cudaStreamSynchronize(stream);
}
