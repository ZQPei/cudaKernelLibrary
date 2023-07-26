#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <sys/time.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>


bool time_profiler = true;
bool nvtx_profiler = false;

bool profilerParseEnv(const char* name);


static uint64_t GetCurrentMicrosecond() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1e6 + tv.tv_usec;
}

#define PROFILER_TIMER_START(_X, _stream)                                                    \
  uint64_t _X##_start, _X##_stop;                                                            \
  do {                                                                                       \
    if (time_profiler) {                                                                     \
      cudaStreamSynchronize((cudaStream_t)(_stream));                                        \
      _X##_start = GetCurrentMicrosecond();                                                  \
    }                                                                                        \
  } while (0)

#define PROFILER_TIMER_STOP(_X, _stream)                                                     \
  do {                                                                                       \
    if (time_profiler) {                                                                     \
      cudaStreamSynchronize((cudaStream_t)(_stream));                                        \
      _X##_stop = GetCurrentMicrosecond();                                                   \
      float _X##_elasped_time_ms = (_X##_stop - _X##_start) / 1000.f;                        \
      std::cout << _X##_elasped_time_ms;                                                     \
    }                                                                                        \
  } while (0)

static int profiler_cnt = 0;

#define PROFILER_CNT_THRESH 0

#define PROFILER_NVTX_PUSH(_X)                                                               \
  do {                                                                                       \
    profiler_cnt++;                                                                          \
    if (nvtx_profiler && profiler_cnt > PROFILER_CNT_THRESH) {                               \
      nvtxRangePushA(_X);                                                                    \
    }                                                                                        \
  } while (0)

#define PROFILER_NVTX_POP()                                                                  \
  do {                                                                                       \
    if (nvtx_profiler && profiler_cnt > PROFILER_CNT_THRESH) {                               \
      nvtxRangePop();                                                                        \
    }                                                                                        \
  } while (0)

#define PROFILER_SLEEP(USec)                                                                 \
  do {                                                                                       \
    int t1 = USec / 1000000, t2 = USec % 1000000;                                            \
    if (t1) sleep(t1);                                                                       \
    if (t2) usleep(t2);                                                                      \
  } while (0)


#define cuAssert(expr)                                                                       \
  do {                                                                                       \
    auto err = (expr);                                                                       \
    if (err != cudaSuccess) {                                                                \
      std::cerr << __FILE__ << ' ' << __LINE__ << ' ' << #expr << " failed with error code " \
                << err << ": " << cudaGetErrorString(err) << '\n';                           \
      std::cerr.flush();                                                                     \
      __builtin_trap();                                                                      \
    }                                                                                        \
  } while (0)

template <typename T>
void profilerLoadData(T *data, const size_t count, std::string data_path, bool is_gpu = true) {
  T *cpu_data = reinterpret_cast<T *>(new T[count]);

  FILE *fp = fopen(data_path.c_str(), "r");
  auto  n  = fread(cpu_data, sizeof(T), count, fp);
  assert(n == count);

  if (is_gpu) {
    cuAssert(cudaMemcpy(reinterpret_cast<void *>(data), reinterpret_cast<void *>(cpu_data),
                        sizeof(T) * count, cudaMemcpyHostToDevice));
  } else {
    memcpy(reinterpret_cast<void *>(data), reinterpret_cast<void *>(cpu_data),
           sizeof(T) * count);
  }

  fclose(fp);
  delete[] cpu_data;
}

template <typename T>
void profilerShowData(T *data, const size_t count, std::string data_name = "",
              bool is_gpu = true) {
  T *cpu_data = nullptr;
  if (is_gpu) {
    cpu_data = reinterpret_cast<T *>(new T[count]);
    // for (int i = 0; i < count; ++i) cpu_data[i] = (T)rand();
    cuAssert(cudaMemcpy(reinterpret_cast<void *>(cpu_data), reinterpret_cast<void *>(data),
                        sizeof(T) * count, cudaMemcpyDeviceToHost));
  } else {
    cpu_data = reinterpret_cast<T *>(data);
  }

  float sum = 0.f;
  std::cout << data_name << std::endl;
  for (size_t i = 0; i < count; ++i) {
    if (i < 100) {
      if (std::is_same<T, float>::value)
        printf("%f ", static_cast<float>(cpu_data[i]));
      else if (std::is_same<T, double>::value)
        printf("%f ", static_cast<float>(cpu_data[i]));
      else if (std::is_same<T, __half>::value)
        printf("%f ", __half2float(cpu_data[i]));
      else
        printf("%d ", static_cast<int>(cpu_data[i]));
    }

    if (std::is_same<T, __half>::value)
      sum += __half2float(cpu_data[i]);
    else
      sum += static_cast<float>(cpu_data[i]);
  }
  printf("Sum: %f\n\n", sum);
  // cout << "Sum: " << sum << endl << endl;

  if (is_gpu) delete[] cpu_data;
}

template <typename T>
void profilerSaveData(T *data, const size_t count, std::string save_name, bool is_gpu = true) {
  T *cpu_data = nullptr;
  if (is_gpu) {
    cpu_data = reinterpret_cast<T *>(new T[count]);
    // for (int i = 0; i < count; ++i) cpu_data[i] = (T)rand();
    printf("%p %p %lu %lu\n", cpu_data, data, sizeof(T), count);
    cuAssert(cudaMemcpy(reinterpret_cast<void *>(cpu_data), reinterpret_cast<void *>(data),
                        sizeof(T) * count, cudaMemcpyDeviceToHost));
  } else {
    cpu_data = reinterpret_cast<T *>(data);
  }
  FILE *fp = fopen(save_name.c_str(), "wb");
  auto  n  = fwrite(cpu_data, sizeof(T), count, fp);
  assert(n == count);

  fclose(fp);
  if (is_gpu) delete[] cpu_data;
}
