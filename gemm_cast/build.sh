#!/bin/bash

set -ex

platform=$(uname -p)
if [[ $platform == "x86_64" ]]; then
    sm="86"
    # sm="75"
elif [[ $platform == "aarch64" ]]; then
    sm="87"
fi
cmd="nvcc --use_fast_math --expt-relaxed-constexpr -std=c++14 -Xcompiler -fopenmp -O3 -lgflags -gencode arch=compute_${sm},code=sm_${sm}"
# cmd="nvcc --use_fast_math --expt-relaxed-constexpr -std=c++14 -Xcompiler -fopenmp -O3 -lgflags -lineinfo -gencode arch=compute_${sm},code=sm_${sm}"
# cmd="nvcc --use_fast_math --expt-relaxed-constexpr -std=c++14 -Xcompiler -fopenmp -O3 -lgflags -maxrregcount 32 -lineinfo -gencode arch=compute_${sm},code=sm_${sm}"
target_dir="./build"
target_file="run.${platform}.sm_${sm}.out"

if [[ $# == 1 && -f $1 ]]; then
    ${cmd} $1 -c -o ${target_dir}/$1.${platform}.o
elif [[ $# == 0 ]]; then
    ${cmd} gemm_cast.cu profiler.cc main.cc -o ${target_dir}/${target_file}
    # nvprof \
    # nsys profile --trace=cuda,nvtx --backtrace=none --sample=none -o "/home/administrator/workspace/tmp/`date +%F-%T`.qdrep" \
    run_cmd="${target_dir}/${target_file} -M=576000 -N=64 -K=10 -verbose=true -doProfile=true -doRefCheck=true -runNum=1000"
    echo $run_cmd
    $run_cmd
fi
