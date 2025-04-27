#! /bin/bash

set -e

# Check that FR_PROJECT_HOME is set
if [[ -z "${FR_PROJECT_HOME}" ]]; then
  echo "Error: FR_PROJECT_HOME is not set. Please export FR_PROJECT_HOME to the project root directory."
  exit 1
fi

# Check that N_JOBS is set
if [[ -z "${N_JOBS}" ]]; then
  echo "Error: N_JOBS is not set. Please export N_JOBS to specify the number of parallel jobs."
  exit 1
fi

cd ${FR_PROJECT_HOME}/third_party

if [[ ! -d eigen ]]; then
  git clone https://gitlab.com/libeigen/eigen.git
fi
cd eigen
# git checkout 3.4
git checkout 68f4e58cfacc686583d16cff90361f0b43bc2c1b
if [[ ! -d build ]]; then
  mkdir build
fi
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${FR_PROJECT_HOME}/third_party/local
make install -j${N_JOBS}

echo "Eigen 3.4 installed to ${FR_PROJECT_HOME}/third_party/local"
