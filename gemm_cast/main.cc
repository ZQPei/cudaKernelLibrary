#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "profiler.h"
#include "gemm_cast.h"

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

DEFINE_int32(M, 576000, "");
DEFINE_int32(N, 64, "");
DEFINE_int32(K, 10, "");
DEFINE_bool(doRefCheck, true, "");
DEFINE_bool(doProfile, true, "");
DEFINE_bool(verbose, false, "");
DEFINE_int32(runNum, 1000, "");

template<typename inT, typename outT>
void init(outT **hOutput, outT **hOutputRef, outT **dOutput, half **dOutputHalf, half **hOutputHalf, float **hInput, float **dInput, inT **hWeight, inT **dWeight, inT **dInputPad, inT **dWeightPad,
          int M, int N, int K, cudaStream_t stream) {
  cudaMallocHost(hOutput, sizeof(outT) * M * N);
  cudaMallocHost(hOutputRef, sizeof(outT) * M * N);
  cudaMalloc(dOutput, sizeof(outT) * M * N);
  cudaMalloc(dOutputHalf, sizeof(half) * M * N);
  cudaMallocHost(hOutputHalf, sizeof(half) * M * N);
  cudaMallocHost(hInput, sizeof(float) * M * K);
  cudaMallocHost(hWeight, sizeof(inT) * N * K);
  cudaMalloc(dInput, sizeof(float) * M * K);
  cudaMalloc(dWeight, sizeof(inT) * N * K);
  // cudaMalloc(dInputPad, sizeof(inT) * M * K * 2);
  cudaMalloc(dInputPad, sizeof(inT) * 18 * 512000);
  cudaMalloc(dWeightPad, sizeof(inT) * N * 16);
  // cudaMemset(*dInputPad, 0x00, sizeof(inT) * M * K * 2);

  cuAssert(cudaGetLastError());
}

template<typename inT, typename outT>
void free(outT *hOutput, outT *hOutputRef, outT *dOutput, half *dOutputHalf, half *hOutputHalf, float *hInput, float *dInput, inT *hWeight, inT *dWeight, inT *dInputPad, inT *dWeightPad, int M,
          int N, int K) {
  cudaFreeHost(hOutput);
  cudaFreeHost(hInput);
  cudaFreeHost(hOutputRef);
  cudaFree(dOutput);
  cudaFree(dInput);
  cudaFree(dWeight);
  cudaFree(dInputPad);
}

template<typename T>
bool compareValue(T *hOutput, T *hOutputRef, int M, int N);

template<>
bool compareValue(float *hOutput, float *hOutputRef, int M, int N) {
    const float eps = 0.00001;
    bool result = true;

    #pragma omp parallel for reduction(&&:result)
    for (int i = 0; i < M * N; i++) {
        if (fabs(hOutput[i] - hOutputRef[i]) > eps) {
            result = false;
        }
    }
    return result;
}

template<>
bool compareValue(__half *hOutput, __half *hOutputRef, int M, int N) {
    const float eps = 0.001;
    bool result = true;

    #pragma omp parallel for reduction(&&:result)
    for (int i = 0; i < M * N; i++) {
        if (fabs(__half2float(hOutput[i]) - __half2float(hOutputRef[i])) > eps) {
            result = false;
        }
    }
    return result;
}

void gemm_cast_cpu(float* A, half* B, float* C, int M, int N, int K) {
  // A [M, K]
  // B [N, K]
  // C [M, N]
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c_sum = 0.f;
      for (int k = 0; k < K; ++k) {
        c_sum += A[i * K + k] * __half2float(B[j * K + k]);
        // c_sum += A[i * K + k] * __half2float(B[k * N + j]);
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
  OUT_TYPE *hOutput, *hOutputRef, *dOutput;
  IN_TYPE *hWeight, *dWeight, *dInputPad, *dWeightPad;
  float *hInput, *dInput;
  half *dOutputHalf, *hOutputHalf;
  cudaStream_t stream;

  // init
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  init(&hOutput, &hOutputRef, &dOutput, &dOutputHalf, &hOutputHalf, &hInput, &dInput, &hWeight, &dWeight, &dInputPad, &dWeightPad, M, N, K, stream);
  std::string _postfix = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K) + "_" + getTypeString<float>() + "_" + getTypeString<OUT_TYPE>() + ".bin";
  profilerLoadData(dInput, M * K, std::string("./data/A_") + _postfix, true);
  profilerLoadData(dWeight, N * K, std::string("./data/B_") + _postfix, true);
  profilerLoadData(dWeightPad, N * 16, std::string("./data/B_pad_") + _postfix, true);
  profilerLoadData(hInput, M * K, std::string("./data/A_") + _postfix, false);
  profilerLoadData(hWeight, N * K, std::string("./data/B_") + _postfix, false);
  // cudaMemcpy(hInput, dInput, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
  // cudaMemcpy(hWeight, dWeight, sizeof(half) * N * K, cudaMemcpyDeviceToHost);
  if (FLAGS_verbose) profilerShowData(dInput, M * K, std::string("./data/A_") + _postfix, true);
  if (FLAGS_verbose) profilerShowData(dWeight, N * K, std::string("./data/B_") + _postfix, true);
  cuAssert(cudaGetLastError());

  // cpu version as ref
  if (FLAGS_verbose) std::cout << "start gemm_cast on cpu as reference ..." << std::endl;
  profilerLoadData(hOutputRef, M * N, std::string("./data/C_ref_") + _postfix, false);

  // cuda version
  if (FLAGS_verbose) std::cout << "start gemm_cast on gpu as comparison ..." << std::endl;
  cuAssert(cudaGetLastError());
  fused_nn_dense_cast(dInput, dInputPad, dWeightPad, dOutput, stream);
  cudaStreamSynchronize(stream);

  if (FLAGS_verbose) profilerShowData(dInputPad, 18000 * 32 * 16, std::string("./data/A_") + _postfix, true);

  // compare output
  if (FLAGS_doRefCheck) {
    if (FLAGS_verbose) std::cout << "compare output between cpu and gpu ..." << std::endl;
    cuAssert(cudaGetLastError());
    gemm_cast_cpu(hInput, hWeight, hOutput, M, N, K);
    if (FLAGS_verbose) profilerShowData(hOutputRef, M * N, std::string("./data/C_ref_") + _postfix, false);
    if (FLAGS_verbose) profilerShowData(hOutput, M * N, std::string("./data/C_") + _postfix, false);
    cudaMemcpyAsync(hOutput, dOutput, sizeof(OUT_TYPE) * M * N, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (FLAGS_verbose) profilerShowData(hOutput, M * N, std::string("./data/C_") + _postfix, false);
    if (FLAGS_verbose) profilerSaveData(hOutput, M * N, std::string("./data/C_") + _postfix, false);
    auto res = compareValue(hOutput, hOutputRef, M, N);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;
  }

  // time profile
  if (FLAGS_doProfile) {
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_START(gemm_cast, stream);
    for (int _ = 0; _ < FLAGS_runNum; ++_) {
      fused_nn_dense_cast(dInput, dInputPad, dWeightPad, dOutput, stream);
    }
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_STOP(gemm_cast, stream);
    if (FLAGS_verbose) std::cout << " us" << std::endl;
  }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cuAssert(cudaGetLastError());
  free(hOutput, hOutputRef, dOutput, dOutputHalf, hOutputHalf, hInput, dInput, hWeight, dWeight, dInputPad, dWeightPad, M, N, K);
  return 0;
}
