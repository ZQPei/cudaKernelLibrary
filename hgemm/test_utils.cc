#include <iostream>
#include <gflags/gflags.h>
#include <cuda_fp16.h>

#include "utils.h"


int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Tensor<float> tf(100);
  std::vector<int> shape = {10, 20};
  Tensor<half> th({10, 20});

  tf.fill_random();
  th.fill_ones();
  th.fill_zero();

  tf.show();
  th.show();
}


