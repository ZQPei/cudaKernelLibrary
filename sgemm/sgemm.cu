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
  }
}

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z
#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z
#define OFFSET(row, col, stride) (((row) << (getBits<(stride)>())) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

////////////////////////////////////////////////////////////////////////
// 1. naive
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
template<int M, int N, int K>
__global__ void sgemm_naive_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= M * N) return;

  int constexpr nbits = getBits<N>();
  int constexpr kbits = getBits<K>();
  int m = (threadId >> nbits);
  int n = threadId & (N - 1);
  float vsum = 0.f;
  #pragma unroll
  for (int k = 0; k < K; ++k) {
    // vsum += A[OFFSET(m, k, K)] * B[OFFSET(n, k, N)];
    vsum += A[OFFSET(m, k, K)] * B[OFFSET(k, n, K)];
  }

  C[OFFSET(m, n, N)] = vsum;
}

template<int M, int N, int K>
void _launch_sgemm_naive_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&8) == 0, "M, N, K should be divisible by 32");
  int constexpr blockSz = 128;
  int constexpr gridSz = (M*N + blockSz - 1) / blockSz;
  int constexpr nbits = getBits<N>();
  int constexpr kbits = getBits<K>();
  // std::cout << gridSz << " " << blockSz << " " << nbits << " " << kbits << std::endl;
  sgemm_naive_kernel<M, N, K><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}

////////////////////////////////////////////////////////////////////////
// 2. naive vector
// A: row_major [M, K]
// B: col_major [N, K]
// C: row_major [M, N]
template<int M, int N, int K, typename vecT>
__global__ void sgemm_naive_vec_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadId >= M * N) return;

  int constexpr nbits = getBits<N>();
  int constexpr kbits = getBits<K>();
  int constexpr vecSz = sizeof(vecT) / sizeof(float);
  int m = (threadId >> nbits);
  int n = threadId & (N - 1);
  float vsum = 0.f;
  vecT a_val;
  vecT b_val;

  #pragma unroll
  for (int ki = 0; ki < K/vecSz; ++ki) {
    a_val = ((vecT*)A)[(m << getBits<K/vecSz>()) + ki];
    b_val = ((vecT*)B)[(n << getBits<K/vecSz>()) + ki];
    #pragma unroll
    for (int kj = 0; kj < vecSz; ++kj) {
      float a = ((float*)(&a_val))[kj];
      float b = ((float*)(&b_val))[kj];
      vsum += a * b;
    }
  }

  C[(m << nbits) + n] = vsum;
}

template<int M, int N, int K>
void _launch_sgemm_naive_vec_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&8) == 0, "M, N, K should be divisible by 32");
  int constexpr blockSz = 128;
  int constexpr gridSz = (M*N + blockSz - 1) / blockSz;
  sgemm_naive_vec_kernel<M, N, K, double4><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}

////////////////////////////////////////////////////////////////////////
// 3. 2d index
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
template<int M, int N, int K>
__global__ void sgemm_2dindex_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c) {
  dim3 constexpr blockSz(32, 32);
  dim3 constexpr gridSz((N + blockSz.x - 1) / blockSz.x, (M + blockSz.y - 1) / blockSz.y);  (void)gridSz;

  int n = blockIdx.x * blockSz.x + threadIdx.x;
  int m = blockIdx.y * blockSz.y + threadIdx.y;
  if (m < M && n < N) {
    float psum = 0.0;
    #pragma unroll
    for (int k = 0; k < K; k++) {
        psum += a[OFFSET(m, k, K)] * b[OFFSET(n, k, K)];
        // psum += a[OFFSET(k, m, M)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = psum;
  }
}

template<int M, int N, int K>
void _launch_sgemm_2dindex_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&8) == 0, "M, N, K should be divisible by 32");
  dim3 constexpr blockSz(32, 32);
  dim3 constexpr gridSz((N + blockSz.x - 1) / blockSz.x, (M + blockSz.y - 1) / blockSz.y);
  sgemm_2dindex_kernel<M, N, K><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}

////////////////////////////////////////////////////////////////////////
// 4. 2d index vector
// A: row_major [M, K]
// B: col_major [N, K]
// C: row_major [M, N]
template<int M, int N, int K, typename vecT>
__global__ void sgemm_2dindex_vec_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  dim3 constexpr blockSz = {32, 32, 1};
  dim3 constexpr gridSz = {(N + blockSz.x - 1) / blockSz.x, (M + blockSz.y - 1) / blockSz.y, 1};  (void)gridSz;

  int n = blockIdx.x * blockSz.x + threadIdx.x;
  int m = blockIdx.y * blockSz.y + threadIdx.y;
  // if (blockIdx.x == 0 && blockIdx.y == 0) printf("%d %d\n", m, n);
  // printf("%d %d %d %d %d\n", m, n, m >= M, N >= N, m >= M || N >= N);
  if (m >= M || n >= N) return;

  int constexpr nbits = getBits<N>();
  int constexpr kbits = getBits<K>();
  int constexpr vecSz = sizeof(vecT) / sizeof(float);

  float vsum = 0.f;
  vecT a_val;
  vecT b_val;

  #pragma unroll
  for (int ki = 0; ki < K/vecSz; ++ki) {
    a_val = ((vecT*)A)[(m << getBits<K/vecSz>()) + ki];
    b_val = ((vecT*)B)[(n << getBits<K/vecSz>()) + ki];
    #pragma unroll
    for (int kj = 0; kj < vecSz; ++kj) {
      float a = ((float*)(&a_val))[kj];
      float b = ((float*)(&b_val))[kj];
      vsum += a * b;
    }
  }

  C[(m << nbits) + n] = vsum;
}

template<int M, int N, int K>
void _launch_sgemm_2dindex_vec_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&8) == 0, "M, N, K should be divisible by 32");
  dim3 constexpr blockSz = {32, 32, 1};
  dim3 constexpr gridSz = {(N + blockSz.x - 1) / blockSz.x, (M + blockSz.y - 1) / blockSz.y, 1};
  sgemm_2dindex_vec_kernel<M, N, K, double2><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}

////////////////////////////////////////////////////////////////////////
// 5. block level tile
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
template<int M, int N, int K>
__global__ void sgemm_block_tile_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int constexpr BM = 128;  // tunable
  int constexpr BN = 128;  // tunable
  int constexpr BK = 8;
  int constexpr TM = 8;  // tunable
  int constexpr TN = 8;  // tunable

  dim3 constexpr blockSz(BN/TN, BM/TM);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM); (void)gridSz;

  __shared__ float s_a[BM*BK];
  __shared__ float s_b[BK*BN];
  float r_c[TM*TN] = {0.0,};

  int const tid = (ty * blockSz.x) + tx;

  #pragma unroll
  for (int bk = 0; bk < (K+BK-1)/BK; ++bk) {
    // ldg, sts
    *(float4*)(s_a + (tid >> 1) * BK + (tid & 1) * 4) = *(float4*)(A + by * BM * K + (tid >> 1) * K + bk * BK + (tid & 1) * 4);
    *(float4*)(s_b + (tid >> 5) * BN + (tid & 31) * 4) = *(float4*)(B + bk * BK * N + (tid >> 5) * N + bx * BN + (tid & 31) * 4);
    __syncthreads();

    // mma
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
      #pragma unroll
      for (int n = 0; n < TN; ++n) {
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
          *(float*)(r_c + m * TN + n) += (*(float*)(s_a + ty * TM * BK + m * BK + k)) * (*(float*)(s_b + k * BN + tx * TN + n));
        }
      }
    }
    __syncthreads();
  }

  // stg
  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    #pragma unroll
    for (int vec_n = 0; vec_n < TN/4; ++vec_n) {
      *(float4*)(C + by * BM * N + ty * TM * N + m * N + bx * BN + tx * TN + vec_n * 4) = *(float4*)(r_c + m * TN + vec_n * 4);
    }
  }
}

template<int M, int N, int K>
void _launch_sgemm_block_tile_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&8) == 0, "M, N, K should be divisible by 32");
  int constexpr BM = 128; (void)BM;
  int constexpr BN = 128; (void)BN;
  int constexpr BK = 8; (void)BK;
  int constexpr TM = 8; (void)TM;
  int constexpr TN = 8; (void)TN;

  dim3 constexpr blockSz(BN/TN, BM/TM);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM);
  sgemm_block_tile_kernel<M, N, K><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}


////////////////////////////////////////////////////////////////////////
// 6. block level tile, low bank conflict
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
template<int M, int N, int K>
__global__ void sgemm_block_tile_bank_conflict_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int constexpr BM = 128;  // tunable
  int constexpr BN = 128;  // tunable
  int constexpr BK = 8;
  int constexpr TM = 8;  // tunable
  int constexpr TN = 8;  // tunable

  dim3 constexpr blockSz(BN/TN, BM/TM);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM); (void)gridSz;

  __shared__ float s_a[BK*BM];
  __shared__ float s_b[BK*BN];
  float r_c[TM*TN] = {0.0,};
  float r_a[TM];
  float r_b[TN];
  float4 r_load_a;
  float4 r_load_b;

  int const tid = (ty * blockSz.x) + tx;

  #pragma unroll
  for (int bk = 0; bk < (K+BK-1)/BK; ++bk) {
    // ldg, sts
    r_load_a = *(float4*)(A + by * BM * K + (tid >> 1) * K + bk * BK + (tid & 1) * 4);
    r_load_b = *(float4*)(B + bk * BK * N + (tid >> 5) * N + bx * BN + (tid & 31) * 4);
    #pragma unroll
    for (int loopid = 0; loopid < 4; ++loopid) {
      *(float*)(s_a + (tid & 1) * 4 * BM + loopid * BM + (tid >> 1)) = *((float*)&r_load_a + loopid);
    }
    *(float4*)(s_b + (tid >> 5) * BN + (tid & 31) * 4) = r_load_b;
    __syncthreads();

    // mma
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      *(float4*)(r_a + 0) = *(float4*)(s_a + k * BM + 0 * 16 * 4 + ty * 4);
      *(float4*)(r_a + 4) = *(float4*)(s_a + k * BM + 1 * 16 * 4 + ty * 4);
      *(float4*)(r_b + 0) = *(float4*)(s_b + k * BN + 0 * 16 * 4 + tx * 4);
      *(float4*)(r_b + 4) = *(float4*)(s_b + k * BN + 1 * 16 * 4 + tx * 4);
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
          *(float*)(r_c + m * TN + n) += (*(float*)(r_a + m)) * (*(float*)(r_b + n));
        }
      }
    }
    __syncthreads();
  }

  // stg
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    #pragma unroll
    for (int m = 0; m < TM/2; ++m) {
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        *(float4*)(C + by * BM * N + i * (BM>>1) * N + ty * (TM>>1) * N + m * N + bx * BN + j * (BN>>1) + tx * 4) = *(float4*)(r_c + i * 32 + m * 8 + j * 4);
      }
    }
  }
}

template<int M, int N, int K>
void _launch_sgemm_block_tile_bank_conflict_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&8) == 0, "M, N, K should be divisible by 32");
  int constexpr BM = 128; (void)BM;
  int constexpr BN = 128; (void)BN;
  int constexpr BK = 8; (void)BK;
  int constexpr TM = 8; (void)TM;
  int constexpr TN = 8; (void)TN;

  dim3 constexpr blockSz(BN/TN, BM/TM);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM);
  sgemm_block_tile_bank_conflict_kernel<M, N, K><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}


////////////////////////////////////////////////////////////////////////
// 7. block level tile, low bank conflict, double buffer
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
template<int M, int N, int K>
__global__ void sgemm_double_buffer_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int constexpr BM = 128;  // tunable
  int constexpr BN = 128;  // tunable
  int constexpr BK = 8;
  int constexpr TM = 8;  // tunable
  int constexpr TN = 8;  // tunable

  dim3 constexpr blockSz(BN/TN, BM/TM);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM); (void)gridSz;

  __shared__ float s_a[2*BK*BM];
  __shared__ float s_b[2*BK*BN];
  float r_c[TM*TN] = {0.0,};
  float r_a[TM];
  float r_b[TN];
  float4 r_load_a;
  float4 r_load_b;

  int const tid = (ty * blockSz.x) + tx;

  int s_buf_curr_id = 0;

  {
    int bk = 0;
    // ldg, sts
    r_load_a = *(float4*)(A + by * BM * K + (tid >> 1) * K + bk * BK + (tid & 1) * 4);
    r_load_b = *(float4*)(B + bk * BK * N + (tid >> 5) * N + bx * BN + (tid & 31) * 4);
    #pragma unroll
    for (int loopid = 0; loopid < 4; ++loopid) {
      *(float*)(s_a + s_buf_curr_id * BK * BM + (tid & 1) * 4 * BM + loopid * BM + (tid >> 1)) = *((float*)&r_load_a + loopid);
    }
    *(float4*)(s_b + s_buf_curr_id * BK * BN + (tid >> 5) * BN + (tid & 31) * 4) = r_load_b;
    __syncthreads();
  }

  #pragma unroll
  for (int bk = 1; bk < (K+BK-1)/BK; ++bk) {
    // ldg
    r_load_a = *(float4*)(A + by * BM * K + (tid >> 1) * K + bk * BK + (tid & 1) * 4);
    r_load_b = *(float4*)(B + bk * BK * N + (tid >> 5) * N + bx * BN + (tid & 31) * 4);
    // #pragma unroll
    // for (int loopid = 0; loopid < 4; ++loopid) {
    //   *(float*)(s_a + (s_buf_curr_id ^ 1) * BK * BM + (tid & 1) * 4 * BM + loopid * BM + (tid >> 1)) = *((float*)&r_load_a + loopid);
    // }
    // *(float4*)(s_b + (s_buf_curr_id ^ 1) * BK * BN + (tid >> 5) * BN + (tid & 31) * 4) = r_load_b;

    // mma
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      *(float4*)(r_a + 0) = *(float4*)(s_a + s_buf_curr_id * BK * BM + k * BM + 0 * 16 * 4 + ty * 4);
      *(float4*)(r_a + 4) = *(float4*)(s_a + s_buf_curr_id * BK * BM + k * BM + 1 * 16 * 4 + ty * 4);
      *(float4*)(r_b + 0) = *(float4*)(s_b + s_buf_curr_id * BK * BN + k * BN + 0 * 16 * 4 + tx * 4);
      *(float4*)(r_b + 4) = *(float4*)(s_b + s_buf_curr_id * BK * BN + k * BN + 1 * 16 * 4 + tx * 4);
      #pragma unroll
      for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
          *(float*)(r_c + m * TN + n) += (*(float*)(r_a + m)) * (*(float*)(r_b + n));
        }
      }
    }

    // sts (do sts after mma to hide latency of ldg)
    #pragma unroll
    for (int loopid = 0; loopid < 4; ++loopid) {
      *(float*)(s_a + (s_buf_curr_id ^ 1) * BK * BM + (tid & 1) * 4 * BM + loopid * BM + (tid >> 1)) = *((float*)&r_load_a + loopid);
    }
    *(float4*)(s_b + (s_buf_curr_id ^ 1) * BK * BN + (tid >> 5) * BN + (tid & 31) * 4) = r_load_b;

    s_buf_curr_id ^= 1;
    __syncthreads();
  }

  // mma
  #pragma unroll
  for (int k = 0; k < BK; ++k) {
    *(float4*)(r_a + 0) = *(float4*)(s_a + s_buf_curr_id * BK * BM + k * BM + 0 * 16 * 4 + ty * 4);
    *(float4*)(r_a + 4) = *(float4*)(s_a + s_buf_curr_id * BK * BM + k * BM + 1 * 16 * 4 + ty * 4);
    *(float4*)(r_b + 0) = *(float4*)(s_b + s_buf_curr_id * BK * BN + k * BN + 0 * 16 * 4 + tx * 4);
    *(float4*)(r_b + 4) = *(float4*)(s_b + s_buf_curr_id * BK * BN + k * BN + 1 * 16 * 4 + tx * 4);
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
        *(float4*)(C + by * BM * N + i * (BM>>1) * N + ty * (TM>>1) * N + m * N + bx * BN + j * (BN>>1) + tx * 4) = *(float4*)(r_c + i * 32 + m * 8 + j * 4);
      }
    }
  }
}

template<int M, int N, int K>
void _launch_sgemm_double_buffer_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&8) == 0, "M, N, K should be divisible by 32");
  int constexpr BM = 128; (void)BM;
  int constexpr BN = 128; (void)BN;
  int constexpr BK = 8; (void)BK;
  int constexpr TM = 8; (void)TM;
  int constexpr TN = 8; (void)TN;

  dim3 constexpr blockSz(BN/TN, BM/TM);
  dim3 constexpr gridSz((N+BN-1)/BN, (M+BM-1)/BM);
  sgemm_double_buffer_kernel<M, N, K><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}


////////////////////////////////////////////////////////////////////////
// benchmark: cublas
// A: row_major [M, K]
// B: row_major [K, N]
// C: row_major [M, N]
// cublas
template<int M, int N, int K>
void _launch_sgemm_cudnn_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static cublasHandle_t cublas_handle = nullptr;
  if (cublas_handle == nullptr) {
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
  }
  float cublas_alpha = 1.0;
  float cublas_beta = 0;
  // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, A, K, B, N, &cublas_beta, C, M);
  cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, B, N, A, K, &cublas_beta, C, N);
}


////////////////////////////////////////////////////////////////////////
void sgemm_cuda(float* __restrict__ A, float* __restrict__ B, float* __restrict__ AT, float* __restrict__ BT, float* __restrict__ C, int M, int N, int K, cudaStream_t stream) {
  #define IF_STAT if (false)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_naive_kernel<(m), (n), (k)>(A, B, C, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_naive_vec_kernel<(m), (n), (k)>(A, BT, C, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_2dindex_kernel<(m), (n), (k)>(A, BT, C, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_2dindex_vec_kernel<(m), (n), (k)>(A, BT, C, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_block_tile_kernel<(m), (n), (k)>(A, B, C, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_block_tile_bank_conflict_kernel<(m), (n), (k)>(A, B, C, stream)
  #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_double_buffer_kernel<(m), (n), (k)>(A, B, C, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_cudnn_kernel<(m), (n), (k)>(A, B, C, stream)
  #define ELSE_STAT else { std::cout << "NOT_IMPLEMENTED" << std::endl; __builtin_trap(); }

  IF_STAT;
  ELIF_STAT(32, 32, 32);
  ELIF_STAT(128, 128, 128);
  ELIF_STAT(1024, 1024, 1024);
  ELIF_STAT(640000, 128, 32);
  ELIF_STAT(640000, 32, 16);
  ELSE_STAT;

  #undef IF_STAT
  #undef ELIF_STAT
  #undef ELSE_STAT

  cudaStreamSynchronize(stream);
}
