#include <cuda_runtime.h>

void transpose2d(float* dOutput, float* dInput, int N, int C, cudaStream_t stream);
