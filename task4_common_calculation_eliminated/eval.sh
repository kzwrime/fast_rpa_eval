#!/bin/bash

set -e

# CUDA
export CXX=g++
export NVCC=nvcc

# # ROCM
# export CXX=hipcc
# export NVCC=hipcc

SRC_PATH=$(pwd)

cd ./version6_rho_with_bias_calculate
./test_rho.sh 2>&1 | tee ./test_rho.log

cd $SRC_PATH

cd ./version7_rho_with_bias_directly
./test_rho.sh 2>&1 | tee ./test_rho.log

cd $SRC_PATH

echo "================================\n"

cat ./version6_rho_with_bias_calculate/test_rho.log | grep "[[Result]]" | tee ./version6_rho_with_bias_calculate/test_rho.log.csv
echo "--------------------------------"
cat ./version7_rho_with_bias_directly/test_rho.log | grep "[[Result]]" | tee ./version7_rho_with_bias_directly/test_rho.log.csv

echo "Done"
