#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <omp.h>
#include <unistd.h>

#include "utils.h"
#include "sgemm_b2b.h"


DEFINE_int32(M, 1024, "");
DEFINE_int32(K1, 128, "");
DEFINE_int32(K2, 128, "");
DEFINE_int32(N, 128, "");
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
  int K1 = FLAGS_K1;
  int K2 = FLAGS_K2;
  int N = FLAGS_N;

  std::string _postfix = std::to_string(M) + "_" + std::to_string(K1) + "_" + std::to_string(K2) + "_" + std::to_string(N) + ".bin";

  // alloc data
  Tensor<float> inp({M, K1}, std::string("./data/inp_") + _postfix);
  Tensor<float> w1({K1, K2}, std::string("./data/w1_") + _postfix);
  Tensor<float> w2({K2, N}, std::string("./data/w2_") + _postfix);
  Tensor<float> midRef({M, K2}, std::string("./data/mid_ref_") + _postfix);
  Tensor<float> mid({M, K2});
  Tensor<float> outRef({M, N}, std::string("./data/out_ref_") + _postfix);
  Tensor<float> out({M, N});

  // init
  cudaStream_t stream = cudaStreamDefault;
  // cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  if (FLAGS_verbose) inp.show();
  if (FLAGS_verbose) w1.show();
  if (FLAGS_verbose) w2.show();
  cuAssert(cudaGetLastError());

  // run
  if (FLAGS_verbose) std::cout << "run ..." << std::endl;
  cuAssert(cudaGetLastError());
  auto test_func = [&]() {
    sgemm_b2b(inp.d_ptr, w1.d_ptr, w2.d_ptr, mid.d_ptr, out.d_ptr, M, K1, K2, N, stream);
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
    mid.d2h();
    out.d2h();
    if (FLAGS_verbose) midRef.show();
    if (FLAGS_verbose) mid.show();
    compareValue(mid.h_ptr, midRef.h_ptr, mid.size, 1e-2, true);
    if (FLAGS_verbose) outRef.show();
    if (FLAGS_verbose) out.show();
    if (FLAGS_verbose) out.save(std::string("./data/out_") + _postfix);
    compareValue(out.h_ptr, outRef.h_ptr, out.size, 1e-2, true);
  }

  // return 0;

  // time profile
  if (FLAGS_doProfile) {
    if (FLAGS_verbose) std::cout << "test performance ..." << std::endl;
    cudaStreamSynchronize(stream);
    cuAssert(cudaGetLastError());
    float ms = testPerf(test_func, stream, FLAGS_runNum);
    std::cout << "PERF: " << ms << " ms" << std::endl;
    float flops = 2 * long(M) * K1 * K2 + 2 * long(M) * K2 * N;
    float GFLOPS = float(flops) / 1024/1024/1024/(ms/1e3);
    std::cout << "GFLOPS: " << GFLOPS << std::endl;
  }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cudaStreamSynchronize(stream);
  cuAssert(cudaGetLastError());
  return 0;
}
