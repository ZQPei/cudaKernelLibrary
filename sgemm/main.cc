#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "utils.h"
#include "sgemm.h"


DEFINE_int32(M, 576000, "");
DEFINE_int32(N, 64, "");
DEFINE_int32(K, 10, "");
DEFINE_bool(doRefCheck, true, "");
DEFINE_bool(doProfile, true, "");
DEFINE_bool(verbose, false, "");
DEFINE_int32(runNum, 1000, "");

void sgemm_cpu(float* A, float* B, float* C, int M, int N, int K) {
  // A [M, K]
  // B [N, K]
  // C [M, N]
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c_sum = 0.f;
      for (int k = 0; k < K; ++k) {
        c_sum += A[i * K + k] * B[j * K + k];
      }
      C[i * N + j] = c_sum;
    }
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int M = FLAGS_M;
  int N = FLAGS_N;
  int K = FLAGS_K;

  std::string _postfix = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K) + "_" + 
                         getTypeString<float>() + "_" + getTypeString<float>() + ".bin";

  // alloc data
  Tensor<float> A({M, K}, std::string("./data/A_") + _postfix);
  Tensor<float> B({K, N}, std::string("./data/B_") + _postfix);
  Tensor<float> AT({M, K}, std::string("./data/AT_") + _postfix);
  Tensor<float> BT({N, K}, std::string("./data/BT_") + _postfix);
  Tensor<float> CRef({M, N}, std::string("./data/C_Ref") + _postfix);
  Tensor<float> C({M, N});

  // init
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  if (FLAGS_verbose) profilerShowData(A.d_ptr, M * K, std::string("./data/A_") + _postfix, true);
  if (FLAGS_verbose) profilerShowData(B.d_ptr, N * K, std::string("./data/B_") + _postfix, true);
  cuAssert(cudaGetLastError());

  // get reference
  if (FLAGS_verbose) std::cout << "get reference ..." << std::endl;

  // run kernel
  if (FLAGS_verbose) std::cout << "get comparison ..." << std::endl;
  cuAssert(cudaGetLastError());
  auto run_sgemm_cuda = [&]() {
    sgemm_cuda(A.d_ptr, B.d_ptr, AT.d_ptr, BT.d_ptr, C.d_ptr, M, N, K, stream);
    // cudaStreamSynchronize(stream);
  };

  run_sgemm_cuda();

  // compare output
  if (FLAGS_doRefCheck) {
    if (FLAGS_verbose) std::cout << "compare output between cpu and gpu ..." << std::endl;
    cuAssert(cudaGetLastError());
    sgemm_cpu(hInput, hWeight, hOutput, M, N, K);
    if (FLAGS_verbose) profilerShowData(hOutputRef, M * N, std::string("./data/C_ref_") + _postfix, false);
    if (FLAGS_verbose) profilerShowData(hOutput, M * N, std::string("./data/C_") + _postfix, false);
    cudaMemcpyAsync(hOutput, dOutput, sizeof(float) * M * N, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (FLAGS_verbose) profilerShowData(hOutput, M * N, std::string("./data/C_") + _postfix, false);
    if (FLAGS_verbose) profilerSaveData(hOutput, M * N, std::string("./data/C_") + _postfix, false);
    auto res = compareValue(hOutput, hOutputRef, M, N);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;
  }

  // // time profile
  // if (FLAGS_doProfile) {
  //   cudaStreamSynchronize(stream);
  //   PROFILER_TIMER_START(sgemm, stream);
  //   for (int _ = 0; _ < FLAGS_runNum; ++_) {
  //     sgemm_cuda(dInput, dWeight, dInputTrans, dWeightTrans, dOutput, M, N, K, stream);
  //   }
  //   cudaStreamSynchronize(stream);
  //   PROFILER_TIMER_STOP(sgemm, stream);
  //   float _elapsed_time_us = PROFILER_TIMER_RESULT(sgemm);
  //   if (FLAGS_verbose) std::cout << " us" << std::endl;
  //   if (FLAGS_verbose) {
  //     float flops = 2 * long(M) * N * K;
  //     float GFLOPS = float(flops) / 1024/1024/1024/(_elapsed_time_us/1e6);
  //     std::cout << "GFLOPS: " << GFLOPS << std::endl;
  //   }
  // }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cuAssert(cudaGetLastError());
  return 0;
}
