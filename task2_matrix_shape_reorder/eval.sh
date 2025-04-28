#!/bin/bash

set -e

# CUDA
export CXX=g++
export NVCC=nvcc

# # ROCM
# export CXX=hipcc
# export NVCC=hipcc

SRC_PATH=$(pwd)

cd ./version1_gradient_rho_only
./test_rho.sh 2>&1 | tee ./test_rho.log

cd $SRC_PATH

cd ./version2_gradient_rho_only
./test_rho.sh 2>&1 | tee ./test_rho.log

cd $SRC_PATH

echo "================================\n"

cat ./version1_gradient_rho_only/test_rho.log | grep "[[Result]]" | tee ./version1_gradient_rho_only/test_rho.log.csv
echo "--------------------------------"
cat ./version2_gradient_rho_only/test_rho.log | grep "[[Result]]" | tee ./version2_gradient_rho_only/test_rho.log.csv

echo "Done"
