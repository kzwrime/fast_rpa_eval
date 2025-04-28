#!/bin/bash

set -e

# CUDA
export CXX=g++
export NVCC=nvcc

# # ROCM
# export CXX=hipcc
# export NVCC=hipcc

SRC_PATH=$(pwd)

cd ./base
./test_rho.sh 2>&1 | tee ./test_rho.log

cd $SRC_PATH

cd ./opt
./test_rho.sh 2>&1 | tee ./test_rho.log

cd $SRC_PATH

cat ./base/test_rho.log | grep "[[Result]]" | tee ./base/test_rho.log.csv
cat ./opt/test_rho.log | grep "[[Result]]" | tee ./opt/test_rho.log.csv

echo "Done"
