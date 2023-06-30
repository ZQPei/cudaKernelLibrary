#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "profiler.h"
#include "batchnorm_relu_max.h"

// #define IN_TYPE float
#define IN_TYPE __half
#define OUT_TYPE float
// #define OUT_TYPE __half

template<typename T>
inline std::string getTypeString();
template<>
inline std::string getTypeString<__half>() { return "float16"; }
template<>
inline std::string getTypeString<float>() { return "float32"; }

DEFINE_int32(N, 18000, "");
DEFINE_int32(D, 32, "");
DEFINE_int32(C, 64, "");
DEFINE_bool(doRefCheck, true, "");
DEFINE_bool(doProfile, true, "");
DEFINE_bool(verbose, false, "");
DEFINE_int32(runNum, 1000, "");

template<typename inT, typename outT>
void init(outT **hOutput, outT **hOutputRef, outT **dOutput, inT **hInput, inT **dInput, inT **dGamma, inT **dBeta,
          int N, int D, int C, cudaStream_t stream) {
  cudaMallocHost(hOutput, sizeof(outT) * N * C);
  cudaMallocHost(hOutputRef, sizeof(outT) * N * C);
  cudaMalloc(dOutput, sizeof(outT) * N * C);
  cudaMallocHost(hInput, sizeof(inT) * N * D * C);
  cudaMalloc(dInput, sizeof(inT) * N * D * C);
  cudaMalloc(dGamma, sizeof(inT) * C);
  cudaMalloc(dBeta, sizeof(inT) * C);

  cuAssert(cudaGetLastError());
}

template<typename inT, typename outT>
void free(outT *hOutput, outT *hOutputRef, outT *dOutput, inT *hInput, inT *dInput, inT *dGamma, inT *dBeta, int N,
          int D, int C) {
  cudaFreeHost(hOutput);
  cudaFreeHost(hInput);
  cudaFreeHost(hOutputRef);
  cudaFree(dOutput);
  cudaFree(dInput);
  cudaFree(dGamma);
  cudaFree(dBeta);
}

template<typename T>
bool compareValue(T *hOutput, T *hOutputRef, int N, int C);

template<>
bool compareValue(float *hOutput, float *hOutputRef, int N, int C) {
    const float eps = 0.00001;
    bool result = true;

    #pragma omp parallel for reduction(&&:result)
    for (int i = 0; i < N * C; i++) {
        if (fabs(hOutput[i] - hOutputRef[i]) > eps) {
            result = false;
        }
    }
    return result;
}

template<>
bool compareValue(__half *hOutput, __half *hOutputRef, int N, int C) {
    const float eps = 0.001;
    bool result = true;

    #pragma omp parallel for reduction(&&:result)
    for (int i = 0; i < N * C; i++) {
        if (fabs(__half2float(hOutput[i]) - __half2float(hOutputRef[i])) > eps) {
            result = false;
        }
    }
    return result;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int N = FLAGS_N;
  int D = FLAGS_D;
  int C = FLAGS_C;
  OUT_TYPE *hOutput, *hOutputRef, *dOutput;
  IN_TYPE *hInput, *dInput, *dGamma, *dBeta;
  cudaStream_t stream;

  // init
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  init(&hOutput, &hOutputRef, &dOutput, &hInput, &dInput, &dGamma, &dBeta, N, D, C, stream);
  std::string _postfix = std::to_string(N) + "_" + std::to_string(D) + "_" + std::to_string(C) + "_" + getTypeString<IN_TYPE>() + "_" + getTypeString<OUT_TYPE>() + ".bin";
  profilerLoadData(hInput, N * D * C, std::string("./data/input_") + _postfix, false);
  profilerLoadData(dInput, N * D * C, std::string("./data/input_") + _postfix, true);
  profilerLoadData(dGamma, C, std::string("./data/gamma_") + _postfix, true);
  profilerLoadData(dBeta, C, std::string("./data/beta_") + _postfix, true);
  if (FLAGS_verbose) profilerShowData(hInput, N * D * C, std::string("./data/input_") + _postfix, false);
  cuAssert(cudaGetLastError());

  // cpu version as ref
  if (FLAGS_verbose) std::cout << "start batchnorm_relu_max on cpu as reference ..." << std::endl;
  profilerLoadData(hOutputRef, N * C, std::string("./data/output_ref_") + _postfix, false);

  // cuda version
  if (FLAGS_verbose) std::cout << "start batchnorm_relu_max on gpu as comparison ..." << std::endl;
  fused_batchnorm_relu_max<IN_TYPE, OUT_TYPE>(dOutput, dInput, dGamma, dBeta, N, D, C, BRMImplType::NAIVE, stream);
  cudaStreamSynchronize(stream);

  // compare output
  if (FLAGS_doRefCheck) {
    if (FLAGS_verbose) std::cout << "compare output between cpu and gpu ..." << std::endl;
    cuAssert(cudaGetLastError());
    cudaMemcpyAsync(hOutput, dOutput, sizeof(OUT_TYPE) * N * C, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (FLAGS_verbose) profilerShowData(hOutputRef, N * C, std::string("./data/output_ref_") + _postfix, false);
    if (FLAGS_verbose) profilerShowData(hOutput, N * C, std::string("./data/output_") + _postfix, false);
    if (FLAGS_verbose) profilerSaveData(hOutput, N * C, std::string("./data/output_") + _postfix, false);
    auto res = compareValue(hOutput, hOutputRef, N, C);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;
  }

  // time profile
  if (FLAGS_doProfile) {
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_START(batchnorm_relu_max, stream);
    for (int _ = 0; _ < FLAGS_runNum; ++_) {
      fused_batchnorm_relu_max<IN_TYPE, OUT_TYPE>(dOutput, dInput, dGamma, dBeta, N, D, C, BRMImplType::NAIVE, stream);
    }
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_STOP(batchnorm_relu_max, stream);
    if (FLAGS_verbose) std::cout << " us" << std::endl;
  }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cuAssert(cudaGetLastError());
  free(hOutput, hOutputRef, dOutput, hInput, dInput, dGamma, dBeta, N, D, C);
  return 0;
}
