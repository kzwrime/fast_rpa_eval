```bash

cd ${FR_PROJECT_HOME}/third_party
wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.9.0.tar.gz
tar -xzf magma-2.9.0.tar.gz
cd magma-2.9.0
mkdir build 
cd build

# Please adjust the compilation parameters according to your hardware. Refer to MAGMA's documentation for details:

# https://icl.utk.edu/projectsfiles/magma/doxygen/

# After running cmake, make sure to verify that the printed paths for LAPACK, CUDA/HIP are correct.

# NVIDIA A100
cmake .. -DMAGMA_ENABLE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${FR_PROJECT_HOME}/third_party/local -DGPU_TARGET='sm_80' -DBLA_VENDOR=Intel10_64lp_seq

# AMDGPU gfx906
export GPU_TARGET=gfx906
export ROCM=$ROCM_PATH
export ROCM_HOME=$ROCM_PATH
cmake .. -DMAGMA_ENABLE_HIP=ON -DCMAKE_INSTALL_PREFIX=${FR_PROJECT_HOME}/third_party/local -DGPU_TARGET='gfx906' -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc

make -j${N_JOBS}
make install

```