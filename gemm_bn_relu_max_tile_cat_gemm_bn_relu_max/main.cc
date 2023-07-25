#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "profiler.h"
#include "gemm_bn_relu_max_tile_cat_gemm_bn_relu_max.h"


DEFINE_int32(MD, 20000, "");
DEFINE_int32(MR, 32, "");
DEFINE_int32(K0, 10, "");
DEFINE_int32(K1, 16, "");
DEFINE_int32(N1, 32, "");
DEFINE_int32(K2, 64, "");
DEFINE_int32(N2, 64, "");
DEFINE_bool(doRefCheck, true, "");
DEFINE_bool(doProfile, true, "");
DEFINE_bool(verbose, false, "");
DEFINE_int32(runNum, 1000, "");


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int MD = FLAGS_MD;
  int MR = FLAGS_MR;
  int K0 = FLAGS_K0;
  int K1 = FLAGS_K1;
  int N1 = FLAGS_N1;
  int K2 = FLAGS_K2;
  int N2 = FLAGS_N2;
  
  cudaStream_t stream;
  std::string _postfix = std::to_string(MD) + "_" + std::to_string(MR) + "_" + std::to_string(K0) + "_" +
                         std::to_string(K1) + "_" + std::to_string(N1) + "_" +
                         std::to_string(K2) + "_" + std::to_string(N2) + ".bin";

  // alloc data
  if (FLAGS_verbose) std::cout << "alloc data ..." << std::endl;
  cudaData<float> inp(MD*MR*K0, std::string("data/inp_") + _postfix);
  cudaData<half> w1(N1*K1, std::string("data/w1_") + _postfix);
  cudaData<half> g1(N1, std::string("data/g1_") + _postfix);
  cudaData<half> b1(N1, std::string("data/b1_") + _postfix);
  cudaData<half> w2(N2*K2, std::string("data/w2_") + _postfix);
  cudaData<half> g2(N2, std::string("data/g2_") + _postfix);
  cudaData<half> b2(N2, std::string("data/b2_") + _postfix);
  cudaData<half> outPad(MD*MR*K1);
  cudaData<half> outGemm1(MD*MR*K2);
  cudaData<float> outGemm2(MD*1*N2);
  cudaData<half> outPadRef(MD*MR*K1, std::string("data/outPad_") + _postfix);
  cudaData<half> outGemm1Ref(MD*MR*K2, std::string("data/outGemm1_") + _postfix);
  cudaData<float> outGemm2Ref(MD*1*N2, std::string("data/outGemm2_") + _postfix);

  cudaData<half> out_gemm1(MD*MR*N1);
  cudaData<half> out_gemm1Ref(MD*MR*N1, std::string("data/out_gemm1_") + _postfix);
  cudaData<half> out_gemm2(MD*MR*N2);
  cudaData<half> out_gemm2Ref(MD*MR*N2, std::string("data/out_gemm2_") + _postfix);

  // init
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (FLAGS_verbose) {
    inp.show();
    w1.show();
    g1.show();
    b1.show();
    w2.show();
    g2.show();
    b2.show();
  }
  cuAssert(cudaGetLastError());

  // define fused_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max
  auto fused_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max = [&]() {
    fused_reshape_cast_pad(MD, inp.d_ptr, outPad.d_ptr, stream);
    // fused_gemm1(MD, outPad.d_ptr, w1.d_ptr, g1.d_ptr, b1.d_ptr, out_gemm1.d_ptr, stream);
    fused_gemm_bn_relu_max_tile(MD, outPad.d_ptr, w1.d_ptr, g1.d_ptr, b1.d_ptr, outGemm1.d_ptr, stream);
    // fused_gemm2(MD, outGemm1.d_ptr, w2.d_ptr, g2.d_ptr, b2.d_ptr, out_gemm2.d_ptr, stream);
    fused_gemm_bn_relu_max(MD, outGemm1.d_ptr, w2.d_ptr, g2.d_ptr, b2.d_ptr, outGemm2.d_ptr, stream);

    // cuAssert(cudaStreamSynchronize(stream));
    cuAssert(cudaGetLastError());
  };

  // cpu version as ref
  if (FLAGS_verbose) std::cout << "start on cpu as reference ..." << std::endl;

  // cuda version
  if (FLAGS_verbose) std::cout << "start on gpu as comparison ..." << std::endl;
  cuAssert(cudaGetLastError());
  fused_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max();
  cudaStreamSynchronize(stream);
  cuAssert(cudaGetLastError());

  outPad.d2h(stream);
  outGemm1.d2h(stream);
  outGemm2.d2h(stream);
  out_gemm1.d2h(stream);
  out_gemm2.d2h(stream);

  // compare output
  if (FLAGS_doRefCheck) {
    bool res;
    // check outPad
    outPad.show();
    outPadRef.show();
    res = compareValue(outPad.h_ptr, outPadRef.h_ptr, outPad.size);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;

    // check out_gemm1
    out_gemm1.show();
    out_gemm1Ref.show();
    res = compareValue(out_gemm1.h_ptr, out_gemm1Ref.h_ptr, out_gemm1.size);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;

    // // check outGemm1
    outGemm1.show();
    outGemm1Ref.show();
    res = compareValue(outGemm1.h_ptr, outGemm1Ref.h_ptr, outGemm1.size, 1);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;

    // check out_gemm2
    out_gemm2.show();
    out_gemm2Ref.show();
    res = compareValue(out_gemm2.h_ptr, out_gemm2Ref.h_ptr, out_gemm2.size);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;


    // check outGemm2
    outGemm2.show();
    outGemm2Ref.show();
    res = compareValue(outGemm2.h_ptr, outGemm2Ref.h_ptr, outGemm2.size, 1);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;
  }

  // time profile
  if (FLAGS_doProfile) {
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_START(sgemm, stream);
    for (int _ = 0; _ < FLAGS_runNum; ++_) {
      fused_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max();
    }
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_STOP(sgemm, stream);
    float _elapsed_time_us = PROFILER_TIMER_RESULT(sgemm);
    if (FLAGS_verbose) std::cout << " us" << std::endl;
    if (FLAGS_verbose) {
      // float flops = 2 * long(M) * N * K;
      // float GFLOPS = float(flops) / 1024/1024/1024/(_elapsed_time_us/1e6);
      // std::cout << "GFLOPS: " << GFLOPS << std::endl;
    }
  }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cuAssert(cudaGetLastError());
  return 0;
}
