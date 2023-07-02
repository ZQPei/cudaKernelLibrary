#include <cuda.h>
#include <iostream>

template<int X>
__host__ __device__ __inline__ int constexpr getBits() {
  switch (X) {
    #define CASE_X(bits) { case (1<<(bits)): return bits; break; }
    CASE_X(0); CASE_X(1); CASE_X(2); CASE_X(3); CASE_X(4); CASE_X(5); CASE_X(6); CASE_X(7); CASE_X(8); CASE_X(9);
    CASE_X(10); CASE_X(11); CASE_X(12); CASE_X(13); CASE_X(14); CASE_X(15); CASE_X(16); CASE_X(17); CASE_X(18); CASE_X(19);
  }
}

#define OFFSET(row, col, stride) (((row) << (getBits<(stride)>())) + (col))

////////////////////////////////////////////////////////////////////////
// 1. naive
// A: row_major [M, K]
// B: col_major [N, K]
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
  static_assert((M&31) == 0 && (N&31) == 0 && (K&31) == 0, "M, N, K should be divisible by 32");
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
  static_assert((M&31) == 0 && (N&31) == 0 && (K&31) == 0, "M, N, K should be divisible by 32");
  int constexpr blockSz = 128;
  int constexpr gridSz = (M*N + blockSz - 1) / blockSz;
  sgemm_naive_vec_kernel<M, N, K, double4><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}

////////////////////////////////////////////////////////////////////////
// 3. 2d index
// A: row_major [M, K]
// B: col_major [N, K]
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
        // psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        psum += a[OFFSET(k, m, M)] * b[OFFSET(k, n, N)];
    }
    c[OFFSET(m, n, N)] = psum;
  }
}

template<int M, int N, int K>
void _launch_sgemm_2dindex_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, cudaStream_t stream) {
  static_assert((M&31) == 0 && (N&31) == 0 && (K&31) == 0, "M, N, K should be divisible by 32");
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
  static_assert((M&31) == 0 && (N&31) == 0 && (K&31) == 0, "M, N, K should be divisible by 32");
  dim3 constexpr blockSz = {32, 32, 1};
  dim3 constexpr gridSz = {(N + blockSz.x - 1) / blockSz.x, (M + blockSz.y - 1) / blockSz.y, 1};
  sgemm_2dindex_vec_kernel<M, N, K, double2><<<gridSz, blockSz, 0, stream>>>(A, B, C);
}


////////////////////////////////////////////////////////////////////////
void sgemm_cuda(float* __restrict__ dInput, float* __restrict__ dWeight, float* __restrict__ dInputTrans, float* __restrict__ dWeightTrans, float* __restrict__ dOutput, int M, int N, int K, cudaStream_t stream) {
  #define IF_STAT if (false)
  #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_naive_kernel<(m), (n), (k)>(dInput, dWeightTrans, dOutput, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_naive_vec_kernel<(m), (n), (k)>(dInput, dWeight, dOutput, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_2dindex_kernel<(m), (n), (k)>(dInput, dWeightTrans, dOutput, stream)
  // #define ELIF_STAT(m, n, k) else if ((m) == M && (n) == N && (k) == K) _launch_sgemm_2dindex_vec_kernel<(m), (n), (k)>(dInput, dWeight, dOutput, stream)
  #define ELSE_STAT else { std::cout << "NOT_IMPLEMENTED" << std::endl; __builtin_trap(); }

  IF_STAT;
  ELIF_STAT(32, 32, 32);
  ELIF_STAT(128, 128, 128);
  ELIF_STAT(1024, 1024, 1024);
  ELSE_STAT;

  #undef IF_STAT
  #undef ELIF_STAT
  #undef ELSE_STAT

  // cudaStreamSynchronize(stream);
}