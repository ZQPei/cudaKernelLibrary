#!/usr/bin/env python
import os
import opentuner
from opentuner import Result
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner.search.manipulator import PowerOfTwoParameter
from opentuner import MeasurementInterface

import platform
arch = platform.machine()

if arch == "x86_64":
  nvccArch = "-gencode=arch=compute_86,code=sm_86"
  # nvccArch = "-gencode=arch=compute_75,code=sm_75"
elif arch == "aarch64":
  # nvccArch = "-gencode=arch=compute_72,code=sm_72"
  nvccArch = "-gencode=arch=compute_87,code=sm_87"
else:
  raise NotImplementedError(f"machine {arch} not supported")

class SoftmaxTuner(MeasurementInterface):

  def manipulator(self):
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(PowerOfTwoParameter('BLOCK_SIZE', 32, 1024))
    manipulator.add_parameter(IntegerParameter('LOOP_NUMBER', 1, 64))
    manipulator.add_parameter(IntegerParameter('GROUP_BITS', 0, 5))

    return manipulator

  def run(self, desired_result, input, limit):
    cfg = desired_result.configuration.data

    # compile
    blockSz = str(cfg["BLOCK_SIZE"])
    loopNr  = str(cfg["LOOP_NUMBER"])
    grpbits = str(cfg["GROUP_BITS"])

    cmd_compile = 'nvcc --use_fast_math --expt-relaxed-constexpr -std=c++14 -Xcompiler "-fopenmp" -O3 ' + nvccArch + \
      ' batchnorm_relu_max.cu -DAUTO_TUNE=1 ' \
      " -DBLOCK_SIZE=" + blockSz + \
      " -DLOOP_NUMBER=" + loopNr + \
      " -DGROUP_BITS=" + grpbits + \
      f" -c -o ./build/batchnorm_relu_max.cu.{arch}.o"
    print("    " + cmd_compile)
    compile_result = self.call_program(cmd_compile)
    if compile_result['returncode'] != 0:
      return Result(time=1e9)

    # link
    cmd_link = 'nvcc -std=c++14 -Xcompiler "-fopenmp" -O3 ' + nvccArch + \
      f" ./build/batchnorm_relu_max.cu.{arch}.o ./build/main.cc.{arch}.o ./build/profiler.cc.{arch}.o" + \
      f" -o ./build/run.{arch}.out" + \
      " -lgflags "
    print("    " + cmd_link)
    compile_result = self.call_program(cmd_link)
    if compile_result['returncode'] != 0:
      return Result(time=1e9)

    run_cmd = f'./build/run.{arch}.out -doRefCheck=false -doProfile=true -verbose=false'

    print("    " + run_cmd)
    run_result = self.call_program(run_cmd)
    if run_result['returncode'] != 0:
      return Result(time=1e9)

    cost = float(run_result['stdout'])
    if cost < 1: cost = 1e9
    print("    cost is ", cost)
    return Result(time=cost)

  def save_final_config(self, configuration):
    """called at the end of tuning"""
    print(configuration.data) 
    '''
    print("Optimal block size written to mmm_final_config.json: ", configuration.data)
    self.manipulator().save_to_file(configuration.data,
                                    'mmm_final_config.json',
                                    format='json')
                                    '''

if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  SoftmaxTuner.main(argparser.parse_args())
