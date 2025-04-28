#!/bin/bash

set -e

# CUDA
export CXX=g++
export NVCC=nvcc

# # ROCM
# export CXX=hipcc
# export NVCC=hipcc

SRC_PATH=$(pwd)

cd ./version4_sumup_far_part_no_jit_only
make clean
make -j8
./build/sum_up.out 2>&1 | tee ./test_sum_up_far_part.log
cat ./test_sum_up_far_part.log | grep "[[Result]]" | tee ./test_sumup_far_part.log.csv

cd $SRC_PATH

cd ./version5_sumup_far_part_with_jit_or_aot
make clean
make -j8
./build/sum_up.out 2>&1 | tee ./test_sum_up_far_part.log
cat ./test_sum_up_far_part.log | grep "[[Result]]" | tee ./test_sumup_far_part.log.csv

cd $SRC_PATH

echo "================================\n"

echo "Done"
