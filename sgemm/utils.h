#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <sys/time.h>
#include <omp.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>


bool time_profiler = true;
bool nvtx_profiler = false;

std::unordered_set<void*> _device_ptr_set;
std::unordered_set<void*> _pinned_ptr_set;

bool profilerParseEnv(const char* name) {
  char* env = getenv(name);
  bool ret = false;
  if (env)
    ret = bool(atoi(env));
  std::cout << name << ": " << env << ", " << ret << std::endl << std::flush;
  return ret;
}

static uint64_t GetCurrentMicrosecond() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1e6 + tv.tv_usec;
}

#define PROFILER_TIMER_START(_X, _stream)                                                    \
  uint64_t _X##_start, _X##_stop, _X##_elasped_time_ms;                                      \
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
      _X##_elasped_time_ms = (_X##_stop - _X##_start) / 1000.f;                              \
      std::cout << _X##_elasped_time_ms;                                                     \
    }                                                                                        \
  } while (0)

#define PROFILER_TIMER_RESULT(_X) _X##_elasped_time_ms


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

template<typename T>
bool compareValue(T *hOutput, T *hOutputRef, size_t size, float const eps = 1e-3);

template<>
bool compareValue(float *hOutput, float *hOutputRef, size_t size, float const eps) {
    // const float eps = 0.001;
    bool result = true;

    #pragma omp parallel for reduction(&&:result)
    for (size_t i = 0; i < size; i++) {
        if (fabs(hOutput[i] - hOutputRef[i]) > eps) {
            result = false;
        }
    }
    return result;
}

template<>
bool compareValue(__half *hOutput, __half *hOutputRef, size_t size, float const eps) {
    // const float eps = 0.001;
    bool result = true;

    #pragma omp parallel for reduction(&&:result)
    for (size_t i = 0; i < size; i++) {
        if (fabs(__half2float(hOutput[i]) - __half2float(hOutputRef[i])) > eps) {
            result = false;
        }
    }
    return result;
}

template<typename T>
inline std::string getTypeString();
template<>
inline std::string getTypeString<half>() { return "float16"; }
template<>
inline std::string getTypeString<float>() { return "float32"; }
template<>
inline std::string getTypeString<double>() { return "float64"; }
template<>
inline std::string getTypeString<int>() { return "int32"; }

#define CUDA_MALLOC_PTR(_T, _ptrname, _size, _is_gpu) \
  _T *_ptrname; \
  do { \
    cuAssert(cudaGetLastError()); \
    if (_is_gpu) { \
      cudaMalloc(&_ptrname, sizeof(_T) * (_size)); \
      _device_ptr_set.insert(reinterpret_cast<void*>(_ptrname)); \
    } else { \
      cudaMallocHost(&_ptrname, sizeof(_T) * (_size)); \
      _pinned_ptr_set.insert(reinterpret_cast<void*>(_ptrname)); \
    } \
    cuAssert(cudaGetLastError()); \
  } while (0)

#define CUDA_FREE_PTR() \
  do { \
    cuAssert(cudaGetLastError()); \
    for (auto& _ptr: _pinned_ptr_set) { \
      cudaFreeHost(_ptr); \
    } \
    _pinned_ptr_set.clear(); \
    for (auto& _ptr: _device_ptr_set) { \
      cudaFree(_ptr); \
    } \
    _device_ptr_set.clear(); \
    cuAssert(cudaGetLastError()); \
  } while (0)

template<typename T>
class Tensor {
public:
  Tensor(size_t _size, std::string _data_path = ""):
      size(_size) {
    cudaMallocHost(&h_ptr, _size * sizeof(T));
    cudaMalloc(&d_ptr, _size * sizeof(T));
    cuAssert(cudaGetLastError());
    if (_data_path.size()) load(_data_path);
  }

  Tensor(std::vector<size_t> _shape, std::string _data_path = ""):
      shape(_shape) {
    size = 1; for (auto _s: shape) size *= _s;
    cudaMallocHost(&h_ptr, size * sizeof(T));
    cudaMalloc(&d_ptr, size * sizeof(T));
    cuAssert(cudaGetLastError());
    if (_data_path.size()) load(_data_path);
  }

  // Tensor(std::vector<int> _shape, std::string _data_path = "") {
  //   shape.clear();
  //   size = 1;
  //   for (auto _s: _shape) {
  //     std::cout << _s << std::endl;
  //     size *= _s;
  //     shape.push_back(_s);
  //   }
  //   cudaMallocHost(&h_ptr, size * sizeof(T));
  //   cudaMalloc(&d_ptr, size * sizeof(T));
  //   cuAssert(cudaGetLastError());
  //   if (_data_path.size()) load(_data_path);
  // }

  ~Tensor() {
    cudaFreeHost(h_ptr);
    cudaFree(d_ptr);
  }

  void load(std::string& _data_path) {
    data_path = _data_path;
    std::cout << "load data from " << data_path << std::endl;
    profilerLoadData(h_ptr, size, data_path, false);
    h2d();
    cuAssert(cudaGetLastError());
  }

  void show() {
    std::cout << "show data of size " << size << std::endl;
    profilerShowData(h_ptr, size, data_path, false);
  }

  void fill_random() {
    // fill with random value
    std::mt19937 gen(0);
    T _min, _max;

    if (std::is_same<T, float>::value) {
      _min = 0.0f, _max = 1.0f;
    } else if (std::is_same<T, half>::value) {
      _min = (half)0.0, _max = (half)1.0;
    } else if (std::is_same<T, int>::value) {
      _min = 0, _max = 99;
    }
    std::uniform_real_distribution<T> dist(_min, _max);
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
      h_ptr[i] = dist(gen);
    }
    h2d();
  }

  void fill_value(T val) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
      h_ptr[i] = (T)val;
    }
    h2d();
  }

  void fill_ones() {
    fill_value((T)1.0);
  }

  void fill_zero() {
    fill_value((T)0.0);
  }

  void h2d(cudaStream_t stream = cudaStreamDefault) {
    cudaMemcpyAsync(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    cuAssert(cudaGetLastError());
  }

  void d2h(cudaStream_t stream = cudaStreamDefault) {
    cudaMemcpyAsync(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cuAssert(cudaGetLastError());
  }

  T* h_ptr = nullptr;
  T* d_ptr =nullptr;
  std::vector<size_t> shape;
  size_t size;
  std::string data_path = "";
};
