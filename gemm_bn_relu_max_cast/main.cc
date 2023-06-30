#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_fp16.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "profiler.h"
#include "gemm_bn_relu_max_cast.h"

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
DEFINE_int32(MM, 18000, "");
DEFINE_int32(N, 64, "");
DEFINE_int32(K, 10, "");
DEFINE_bool(doRefCheck, true, "");
DEFINE_bool(doProfile, true, "");
DEFINE_bool(verbose, false, "");
DEFINE_int32(runNum, 1000, "");

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

void gemm_bn_relu_max_cast_cpu(float* A, half* B, half* gamma, half* beta, float* C, int M, int MM, int N, int K) {
  int D = M / MM;
  for (int ii = 0; ii < MM; ++ii) {
    for (int j = 0; j < N; ++j) {
      float vmax = 0.f;
      for (int d = 0; d < D; ++d) {
        int i = ii * D + d;
        float _gamma = __half2float(gamma[j]);
        float _beta = __half2float(beta[j]);
        float c_sum = 0.f;
        for (int k = 0; k < K; ++k) {
          c_sum += A[i * K + k] * __half2float(B[j * K + k]);
        }
        vmax = std::max(vmax, c_sum * _gamma + _beta);
      }
      C[ii * N + j] = vmax;
    }
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int M = FLAGS_M;
  int MM = FLAGS_MM;
  int N = FLAGS_N;
  int K = FLAGS_K;
  OUT_TYPE *hOutput, *hOutputRef, *dOutput;
  IN_TYPE *hWeight, *dWeight, *dInputPad, *dWeightPad;
  float *hInput, *dInput;
  half *dOutputHalf, *hOutputHalf;
  half *dGamma, *dBeta, *hGamma, *hBeta;
  cudaStream_t stream;

  // init
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaMallocHost(&hOutput, sizeof(OUT_TYPE) * MM * N);
  cudaMallocHost(&hOutputRef, sizeof(OUT_TYPE) * MM * N);
  cudaMalloc(&dOutput, sizeof(OUT_TYPE) * MM * N);
  cudaMalloc(&dOutputHalf, sizeof(half) * MM * N);
  cudaMallocHost(&hOutputHalf, sizeof(half) * MM * N);
  cudaMallocHost(&hInput, sizeof(float) * M * K);
  cudaMallocHost(&hWeight, sizeof(IN_TYPE) * N * K);
  cudaMalloc(&dInput, sizeof(float) * M * K);
  cudaMalloc(&dWeight, sizeof(IN_TYPE) * N * K);
  // cudaMalloc(&dInputPad, sizeof(IN_TYPE) * M * K * 2);
  cudaMalloc(&dInputPad, sizeof(IN_TYPE) * 18 * 512000);
  cudaMalloc(&dWeightPad, sizeof(IN_TYPE) * N * 16);
  // cudaMemset(&*dInputPad, 0x00, sizeof(IN_TYPE) * M * K * 2);
  cudaMallocHost(&hGamma, sizeof(IN_TYPE) * N);
  cudaMallocHost(&hBeta, sizeof(IN_TYPE) * N);
  cudaMalloc(&dGamma, sizeof(IN_TYPE) * N);
  cudaMalloc(&dBeta, sizeof(IN_TYPE) * N);
  cuAssert(cudaGetLastError());
  std::string _postfix = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(K) + "_" + getTypeString<float>() + "_" + getTypeString<OUT_TYPE>() + ".bin";
  profilerLoadData(dInput, M * K, std::string("./data/A_") + _postfix, true);
  profilerLoadData(dWeight, N * K, std::string("./data/B_") + _postfix, true);
  profilerLoadData(dWeightPad, N * 16, std::string("./data/B_pad_") + _postfix, true);
  profilerLoadData(hInput, M * K, std::string("./data/A_") + _postfix, false);
  profilerLoadData(hWeight, N * K, std::string("./data/B_") + _postfix, false);
  profilerLoadData(hGamma, N, std::string("./data/gamma_") + _postfix, false);
  profilerLoadData(hBeta, N, std::string("./data/beta_") + _postfix, false);
  profilerLoadData(dGamma, N, std::string("./data/gamma_") + _postfix, true);
  profilerLoadData(dBeta, N, std::string("./data/beta_") + _postfix, true);
  // cudaMemcpy(hInput, dInput, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
  // cudaMemcpy(hWeight, dWeight, sizeof(half) * N * K, cudaMemcpyDeviceToHost);
  if (FLAGS_verbose) profilerShowData(dInput, M * K, std::string("./data/A_") + _postfix, true);
  if (FLAGS_verbose) profilerShowData(dWeight, N * K, std::string("./data/B_") + _postfix, true);
  cuAssert(cudaGetLastError());

  // cpu version as ref
  if (FLAGS_verbose) std::cout << "start gemm_cast on cpu as reference ..." << std::endl;
  profilerLoadData(hOutputRef, MM * N, std::string("./data/C_ref_") + _postfix, false);

  // cuda version
  if (FLAGS_verbose) std::cout << "start gemm_cast on gpu as comparison ..." << std::endl;
  cuAssert(cudaGetLastError());
  // fused_nn_dense_cast(dInput, dInputPad, dWeightPad, dOutput, stream);
  fused_nn_dense_bn_relu_max_cast(dInput, dInputPad, dWeightPad, dGamma, dBeta, dOutput, stream);
  cudaStreamSynchronize(stream);

  if (FLAGS_verbose) profilerShowData(dInputPad, 18000 * 32 * 16, std::string("./data/A_") + _postfix, true);

  // compare output
  if (FLAGS_doRefCheck) {
    if (FLAGS_verbose) std::cout << "compare output between cpu and gpu ..." << std::endl;
    cuAssert(cudaGetLastError());
    gemm_bn_relu_max_cast_cpu(hInput, hWeight, hGamma, hBeta, hOutput, M, MM, N, K);
    if (FLAGS_verbose) profilerShowData(hOutputRef, MM * N, std::string("./data/C_ref_") + _postfix, false);
    if (FLAGS_verbose) profilerShowData(hOutput, MM * N, std::string("./data/C_") + _postfix, false);
    cudaMemcpyAsync(hOutput, dOutput, sizeof(OUT_TYPE) * MM * N, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (FLAGS_verbose) profilerShowData(hOutput, MM * N, std::string("./data/C_") + _postfix, false);
    if (FLAGS_verbose) profilerSaveData(hOutput, MM * N, std::string("./data/C_") + _postfix, false);
    auto res = compareValue(hOutput, hOutputRef, MM, N);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;
  }

  // time profile
  if (FLAGS_doProfile) {
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_START(gemm_cast, stream);
    for (int _ = 0; _ < FLAGS_runNum; ++_) {
      // fused_nn_dense_cast(dInput, dInputPad, dWeightPad, dOutput, stream);
      fused_nn_dense_bn_relu_max_cast(dInput, dInputPad, dWeightPad, dGamma, dBeta, dOutput, stream);
    }
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_STOP(gemm_cast, stream);
    if (FLAGS_verbose) std::cout << " us" << std::endl;
  }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cuAssert(cudaGetLastError());
  cudaFreeHost(hOutput);
  cudaFreeHost(hInput);
  cudaFreeHost(hOutputRef);
  cudaFree(dOutput);
  cudaFree(dInput);
  cudaFree(dWeight);
  cudaFree(dInputPad);
  cudaFreeHost(hGamma);
  cudaFreeHost(hBeta);
  cudaFree(dGamma);
  cudaFree(dBeta);
  return 0;
}
