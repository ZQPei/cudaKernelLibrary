#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <omp.h>
#include <unistd.h>

#include "utils.h"
#include "hgemm.h"


DEFINE_int32(M, 1024, "");
DEFINE_int32(N, 1024, "");
DEFINE_int32(K, 1024, "");
DEFINE_bool(doRefCheck, true, "");
DEFINE_bool(doProfile, true, "");
DEFINE_bool(verbose, false, "");
DEFINE_int32(runNum, 1000, "");

void hgemm_cpu(half* A, half* B, half* C, int M, int N, int K) {
  // A [M, K]
  // B [N, K]
  // C [M, N]
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c_sum = 0.f;
      for (int k = 0; k < K; ++k) {
        c_sum += (float)A[i * K + k] * (float)B[k * N + j];
      }
      C[i * N + j] = (half)c_sum;
    }
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int M = FLAGS_M;
  int N = FLAGS_N;
  int K = FLAGS_K;

  std::string _postfix = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K) + "_" + 
                         getTypeString<half>() + "_" + getTypeString<half>() + ".bin";

  // alloc data
  Tensor<half> A({M, K}, std::string("./data/A_") + _postfix);
  Tensor<half> B({K, N}, std::string("./data/B_") + _postfix);
  Tensor<half> AT({M, K}, std::string("./data/AT_") + _postfix);
  Tensor<half> BT({N, K}, std::string("./data/BT_") + _postfix);
  Tensor<half> CRef({M, N}, std::string("./data/C_ref_") + _postfix);
  Tensor<half> C({M, N});

  // init
  cudaStream_t stream = cudaStreamDefault;
  // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  // if (FLAGS_verbose) A.show();
  // if (FLAGS_verbose) B.show();
  cuAssert(cudaGetLastError());

  // run
  if (FLAGS_verbose) std::cout << "run ..." << std::endl;
  cuAssert(cudaGetLastError());
  auto test_func = [&]() {
    hgemm_cuda(A.d_ptr, B.d_ptr, AT.d_ptr, BT.d_ptr, C.d_ptr, M, N, K, stream);
    // hgemm_cpu(A.h_ptr, B.h_ptr, C.h_ptr, M, N, K); C.h2d();
    // cudaStreamSynchronize(stream);
    // sleep(0.01);
  };

  test_func();
  cudaStreamSynchronize(stream);
  cuAssert(cudaGetLastError());

  // compare output
  if (FLAGS_doRefCheck) {
    if (FLAGS_verbose) std::cout << "compare output with reference ..." << std::endl;
    cudaStreamSynchronize(stream);
    cuAssert(cudaGetLastError());
    C.d2h();
    if (FLAGS_verbose) CRef.show();
    if (FLAGS_verbose) C.show();
    compareValue(C.h_ptr, CRef.h_ptr, C.size, 1e-1, true);
  }

  // time profile
  if (FLAGS_doProfile) {
    if (FLAGS_verbose) std::cout << "test performance ..." << std::endl;
    cudaStreamSynchronize(stream);
    cuAssert(cudaGetLastError());
    float ms = testPerf(test_func, stream, FLAGS_runNum);
    std::cout << "PERF: " << ms << " ms" << std::endl;
    float flops = 2 * long(M) * N * K;
    float GFLOPS = float(flops) / 1024/1024/1024/(ms/1e3);
    std::cout << "GFLOPS: " << GFLOPS << std::endl;
  }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cudaStreamSynchronize(stream);
  cuAssert(cudaGetLastError());
  return 0;
}
