#include "profiler.h"
#include <cstdlib>
#include <gflags/gflags.h>


bool profilerParseEnv(const char* name) {
  char* env = getenv(name);
  bool ret = false;
  if (env)
    ret = bool(atoi(env));
  // std::cout << name << ": " << env << ", " << ret << std::endl;
  return ret;
}

bool time_profiler = true;  // profilerParseEnv("TIME_PROFILER");
bool nvtx_profiler = false;  // profilerParseEnv("NVTX_PROFILER");
