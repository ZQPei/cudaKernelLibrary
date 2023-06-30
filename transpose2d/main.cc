#include <iostream>
#include <random>
#include <cuda.h>
#include <gflags/gflags.h>
#include <omp.h>

#include "profiler.h"
#include "transpose2d.h"


DEFINE_int32(N, 128, "");
DEFINE_int32(C, 128, "");
DEFINE_bool(random, false, "");
DEFINE_bool(doRefCheck, true, "");
DEFINE_bool(doProfile, true, "");
DEFINE_bool(verbose, false, "");
DEFINE_int32(runNum, 1000, "");

void init(float **hOutput, float **hInput, float **hOutputRef, float **dOutput, float **dOutputRef, float **dInput,
          int N, int C, cudaStream_t stream) {
  cudaMallocHost(hOutput, sizeof(float) * N * C);
  cudaMallocHost(hInput, sizeof(float) * N * C);
  cudaMallocHost(hOutputRef, sizeof(float) * N * C);
  cudaMalloc(dOutput, sizeof(float) * N * C);
  cudaMalloc(dOutputRef, sizeof(float) * N * C);
  cudaMalloc(dInput, sizeof(float) * N * C);

  if (FLAGS_random) {
    // fill with random value
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    float* data = *hInput;
    #pragma omp parallel for
    for (int i = 0; i < N * C; ++i) {
      // std::cout << i << std::endl;
      data[i] = dist(gen);
    }
  }

  cuAssert(cudaGetLastError());
}

void free(float *hOutput, float *hInput, float *hOutputRef, float *dOutput, float *dOutputRef, float *dInput, int N,
          int C) {
  cudaFreeHost(hInput);
  cudaFreeHost(hOutput);
  cudaFreeHost(hOutputRef);
  cudaFree(dInput);
  cudaFree(dOutput);
}

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

void transpose2d_cpu(float* hOutput, float* hInput, int N, int C) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < C; ++j) {
      hOutput[j * N + i] = hInput[i * C + j];
    }
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int N = FLAGS_N;
  int C = FLAGS_C;
  float *hOutput, *hInput, *hOutputRef;
  float *dOutput, *dInput, *dOutputRef;
  cudaStream_t stream;

  // init
  if (FLAGS_verbose) std::cout << "init ..." << std::endl;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  init(&hOutput, &hInput, &hOutputRef, &dOutput, &dOutputRef, &dInput, N, C, stream);
  std::string _postfix = std::to_string(N) + "_" + std::to_string(C) + ".bin";
  if (!FLAGS_random) {
    profilerLoadData(hInput, N * C, std::string("./data/input_") + _postfix, false);
  }
  if (FLAGS_verbose) {
    profilerSaveData(hInput, N * C, std::string("./data/input_") + _postfix, false);
    profilerShowData(hInput, N * C, std::string("./data/input_") + _postfix, false);
  }
  cudaMemcpyAsync(dInput, hInput, sizeof(float) * N * C, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  cuAssert(cudaGetLastError());

  // cpu version as ref
  if (FLAGS_verbose) std::cout << "start transpose2d on cpu as reference ..." << std::endl;
  // transpose2d_cpu(hOutputRef, hInput, N, C);
  profilerLoadData(hOutputRef, N * C, std::string("./data/output_ref_") + _postfix, false);

  // cuda version
  if (FLAGS_verbose) std::cout << "start transpose2d on gpu as comparison ..." << std::endl;
  transpose2d(dOutput, dInput, N, C, stream);
  cudaStreamSynchronize(stream);

  // compare output
  if (FLAGS_doRefCheck) {
    if (FLAGS_verbose) std::cout << "compare output between cpu and gpu ..." << std::endl;
    cuAssert(cudaGetLastError());
    cudaMemcpyAsync(hOutput, dOutput, sizeof(float) * N * C, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (FLAGS_verbose) profilerShowData(hOutputRef, N * C, std::string("./data/output_ref_") + _postfix, false);
    if (FLAGS_verbose) profilerShowData(hOutput, N * C, std::string("./data/output_") + _postfix, false);
    if (FLAGS_verbose) profilerSaveData(hOutputRef, N * C, std::string("./data/output_ref_") + _postfix, false);
    if (FLAGS_verbose) profilerSaveData(hOutput, N * C, std::string("./data/output_") + _postfix, false);
    auto res = compareValue(hOutput, hOutputRef, N, C);
    if (res) std::cout << "check pass ..." << std::endl;
    else std::cout << "check failed ..." << std::endl;
  }

  // time profile
  if (FLAGS_doProfile) {
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_START(transpose2d, stream);
    for (int _ = 0; _ < FLAGS_runNum; ++_) {
      transpose2d(dOutput, dInput, N, C, stream);
      // cudaMemcpyAsync(dOutputRef, dOutput, N * C * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    PROFILER_TIMER_STOP(transpose2d, stream);
    if (FLAGS_verbose) std::cout << " us" << std::endl;
  }

  // free
  if (FLAGS_verbose) std::cout << "free ..." << std::endl;
  cuAssert(cudaGetLastError());
  free(hOutput, hInput, hOutputRef, dOutput, dOutputRef, dInput, N, C);
  return 0;
}
