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
  wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip
  unzip eigen-3.3.9.zip
  mv eigen-3.3.9 eigen
fi
cd eigen
if [[ ! -d build ]]; then
  mkdir build
fi
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${FR_PROJECT_HOME}/third_party/local
make install -j${N_JOBS}

echo "Eigen 3.3.9 installed to ${FR_PROJECT_HOME}/third_party/local"
