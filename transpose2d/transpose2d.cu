#include <cuda.h>
#include <iostream>


////////////////////////////////////////////////////////////////////
template<int N, int C>
__global__ void transpose_naive_kernel(float* dOutput, float* dInput) {
  int constexpr blockSz = 128;

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int n = threadId / C;
  int c = threadId % C;

  // if (c >= C || n >= N) return;

  dOutput[c * N + n] = dInput[n * C + c];
}

template<int N, int C>
void _launch_transpose_naive_kernel(float* dOutput, float* dInput, cudaStream_t stream) {
  static_assert((N & 31) == 0 && (C & 31) == 0, "shape should be divisible by 32");

  int constexpr blockSz = 128;
  int constexpr gridSz = (N * C + blockSz - 1) / blockSz;
  transpose_naive_kernel<N, C><<<gridSz, blockSz, 0, stream>>>(dOutput, dInput);
}

////////////////////////////////////////////////////////////////////
template<int N, int C>
__global__ void transpose2d_index_kernel(float* dOutput, float* dInput) {
  int nx = blockIdx.x * blockDim.x + threadIdx.x;  // 0->63
  int ny = blockIdx.y * blockDim.y + threadIdx.y;  // 0->127

  int dimx = blockDim.x * gridDim.x;  // 64
  int dimy = blockDim.y * gridDim.y;  // 128

  // if (nx >= C || ny >= N) return;

  dOutput[nx * dimy + ny] = dInput[ny * dimx + nx];
}

template<int N, int C>
void _launch_transpose_2d_index_kernel(float* dOutput, float* dInput, cudaStream_t stream) {
  static_assert((N & 31) == 0 && (C & 31) == 0, "shape should be divisible by 32");
  // std::cout << N << " " << C << std::endl;

  dim3 constexpr blockSz = {32, 4, 1};
  dim3 constexpr gridSz = {(C + blockSz.x - 1)/blockSz.x, (N + blockSz.y - 1)/blockSz.y, 1};
  transpose2d_index_kernel<N, C><<<gridSz, blockSz, 0, stream>>>(dOutput, dInput);
}

////////////////////////////////////////////////////////////////////
template<int N, int C>
__global__ void transpose2d_shared_kernel(float* __restrict__ dOutput, float* __restrict__ dInput) {
  int ni = blockIdx.x * blockDim.x + threadIdx.x;
  int nj = blockIdx.y * blockDim.y + threadIdx.y;
  int dimi = blockDim.x * gridDim.x; (void)dimi;
  int dimj = blockDim.y * gridDim.y; (void)dimj;

  int ri = blockIdx.y * blockDim.x + threadIdx.x;
  int rj = blockIdx.x * blockDim.y + threadIdx.y;
  int rdimi = blockDim.x * gridDim.y; (void)rdimi;
  int rdimj = blockDim.y * gridDim.x; (void)rdimj;

  int input_idx = nj * dimi + ni;
  int output_idx = rj * rdimi + ri;

  __shared__ float smem[32][32];

  // if (ni >= C || nj >= N) return;

  smem[threadIdx.x][threadIdx.y] = dInput[input_idx];

  __syncthreads();

  dOutput[output_idx] = smem[threadIdx.y][threadIdx.x];
}

template<int N, int C>
void _launch_transpose_2d_shared_kernel(float* __restrict__ dOutput, float* __restrict__ dInput, cudaStream_t stream) {
  static_assert((N & 31) == 0 && (C & 31) == 0, "shape should be divisible by 32");

  dim3 blockSz = {32, 32, 1};
  dim3 gridSz = {(C + blockSz.x - 1)/blockSz.x, (N + blockSz.y - 1)/blockSz.y, 1};
  transpose2d_shared_kernel<N, C><<<gridSz, blockSz, 0, stream>>>(dOutput, dInput);
}

////////////////////////////////////////////////////////////////////
template<int N, int C>
__global__ void transpose2d_shared_no_bank_conflict_kernel(float* __restrict__ dOutput, float* __restrict__ dInput) {
  dim3 constexpr blockSz = {32, 32, 1};
  dim3 constexpr gridSz = {(C + blockSz.x - 1)/blockSz.x, (N + blockSz.y - 1)/blockSz.y, 1};
  __shared__ float smem[32][32 + 1];

  int ni = (blockIdx.x << 5) + threadIdx.x;
  int nj = (blockIdx.y << 5) + threadIdx.y;
  int constexpr dimi = blockSz.x * gridSz.x; (void)dimi;
  int constexpr dimj = blockSz.y * gridSz.y; (void)dimj;

  int ri = (blockIdx.y << 5) + threadIdx.x;
  int rj = (blockIdx.x << 5) + threadIdx.y;
  int constexpr rdimi = blockSz.x * gridSz.y; (void)rdimi;
  int constexpr rdimj = blockSz.y * gridSz.x; (void)rdimj;

  int input_idx = (nj << 10) + ni;
  int output_idx = (rj << 10) + ri;

  // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d\n", dimi, rdimi);

  // if (ni >= C || nj >= N) return;

  smem[threadIdx.y][threadIdx.x] = dInput[input_idx];

  __syncthreads();

  dOutput[output_idx] = smem[threadIdx.x][threadIdx.y];
}

template<int N, int C>
void _launch_transpose_2d_shared_no_bank_conflict_kernel(float* __restrict__ dOutput, float* __restrict__ dInput, cudaStream_t stream) {
  static_assert((N & 31) == 0 && (C & 31) == 0, "shape should be divisible by 32");

  dim3 constexpr blockSz = {32, 32, 1};
  dim3 constexpr gridSz = {(C + blockSz.x - 1)/blockSz.x, (N + blockSz.y - 1)/blockSz.y, 1};
  // std::cout << gridSz.x << " " << gridSz.y << std::endl;
  transpose2d_shared_no_bank_conflict_kernel<N, C><<<gridSz, blockSz, 0, stream>>>(dOutput, dInput);
}

////////////////////////////////////////////////////////////////////
template<int N, int C>
__global__ void transpose2d_shared_no_bank_conflict_ilp_kernel(float* __restrict__ dOutput, float* __restrict__ dInput) {
  dim3 constexpr loopSz = {1, 4, 1};
  dim3 constexpr blockSz = {32, 32, 1};
  dim3 constexpr gridSz = {(C + blockSz.x/loopSz.x - 1)/blockSz.x/loopSz.x, (N + (blockSz.y*loopSz.y) - 1)/(blockSz.y*loopSz.y), 1};
  __shared__ float smem1[32][32 + 1];
  __shared__ float smem2[32][32 + 1];
  __shared__ float smem3[32][32 + 1];
  __shared__ float smem4[32][32 + 1];

  int ni = (blockIdx.x << 5) + threadIdx.x;
  int nj = (blockIdx.y << 5) + threadIdx.y;
  int constexpr dimi = blockSz.x * gridSz.x; (void)dimi;
  int constexpr dimj = blockSz.y * gridSz.y; (void)dimj;

  int ri = (blockIdx.y << 5) + threadIdx.x;
  int rj = (blockIdx.x << 5) + threadIdx.y;
  int constexpr rdimi = blockSz.x * gridSz.y; (void)rdimi;
  int constexpr rdimj = blockSz.y * gridSz.x; (void)rdimj;

  int input_idx = (nj << 10) + ni;
  // int output_idx = (rj << 9) + ri;
  int output_idx = rj * rdimi * 4 + 0 * rdimi + ri;

  int input_idx2 = input_idx + 1 * dimi * dimj;
  int output_idx2 = rj * rdimi * 4 + 1 * rdimi + ri;

  int input_idx3 = input_idx + 2 * dimi * dimj;
  int output_idx3 = rj * rdimi * 4 + 2 * rdimi + ri;

  int input_idx4 = input_idx + 3 * dimi * dimj;
  int output_idx4 = rj * rdimi * 4 + 3 * rdimi + ri;

  // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d\n", dimi * dimj, rdimi * rdimj);

  // LGS && STS
  smem1[threadIdx.y][threadIdx.x] = dInput[input_idx];
  smem2[threadIdx.y][threadIdx.x] = dInput[input_idx2];
  smem3[threadIdx.y][threadIdx.x] = dInput[input_idx3];
  smem4[threadIdx.y][threadIdx.x] = dInput[input_idx4];

  __syncthreads();

  // LDS && STG
  dOutput[output_idx] = smem1[threadIdx.x][threadIdx.y];
  dOutput[output_idx2] = smem2[threadIdx.x][threadIdx.y];
  dOutput[output_idx3] = smem3[threadIdx.x][threadIdx.y];
  dOutput[output_idx4] = smem4[threadIdx.x][threadIdx.y];
}

template<int N, int C>
void _launch_transpose_2d_shared_no_bank_conflict_ilp_kernel(float* __restrict__ dOutput, float* __restrict__ dInput, cudaStream_t stream) {
  static_assert((N & 31) == 0 && (C & 31) == 0, "shape should be divisible by 32");

  dim3 constexpr loopSz = {1, 4, 1};
  dim3 constexpr blockSz = {32, 32, 1};
  dim3 constexpr gridSz = {(C + blockSz.x/loopSz.x - 1)/blockSz.x/loopSz.x, (N + (blockSz.y*loopSz.y) - 1)/(blockSz.y*loopSz.y), 1};
  // std::cout << gridSz.x << " " << gridSz.y << std::endl;
  transpose2d_shared_no_bank_conflict_ilp_kernel<N, C><<<gridSz, blockSz, 0, stream>>>(dOutput, dInput);
}

////////////////////////////////////////////////////////////////////
void transpose2d(float* __restrict__ dOutput, float* __restrict__ dInput, int N, int C, cudaStream_t stream) {
  #define IF_STAT if (false)
  // #define ELIF_STAT(n, c) else if ((n) == N && (c) == C) _launch_transpose_naive_kernel<(n), (c)>(dOutput, dInput, stream)
  // #define ELIF_STAT(n, c) else if ((n) == N && (c) == C) _launch_transpose_2d_index_kernel<(n), (c)>(dOutput, dInput, stream)
  // #define ELIF_STAT(n, c) else if ((n) == N && (c) == C) _launch_transpose_2d_shared_kernel<(n), (c)>(dOutput, dInput, stream)
  // #define ELIF_STAT(n, c) else if ((n) == N && (c) == C) _launch_transpose_2d_shared_no_bank_conflict_kernel<(n), (c)>(dOutput, dInput, stream)
  #define ELIF_STAT(n, c) else if ((n) == N && (c) == C) _launch_transpose_2d_shared_no_bank_conflict_ilp_kernel<(n), (c)>(dOutput, dInput, stream)
  #define ELSE_STAT else std::cout << "NOT_IMPLEMENTED" << std::endl

  IF_STAT;
  ELIF_STAT(32, 32);
  ELIF_STAT(128, 64);
  ELIF_STAT(128, 128);
  ELIF_STAT(1024, 1024);
  ELIF_STAT(1024, 6400);
  ELIF_STAT(10240, 10240);
  ELSE_STAT;

  // cudaStreamSynchronize(stream);
}
