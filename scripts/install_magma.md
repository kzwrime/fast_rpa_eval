```bash

cd ${FR_PROJECT_HOME}/third_party
wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.9.0.tar.gz
tar -xzf magma-2.9.0.tar.gz
cd magma-2.9.0
mkdir build 
cd build

# ================================================
# Please adjust the compilation parameters according to your hardware. Refer to MAGMA's documentation for details:

# https://icl.utk.edu/projectsfiles/magma/doxygen/

# After running cmake, make sure to verify that the printed paths for LAPACK, CUDA/HIP are correct.

# NVIDIA A100
cmake .. -DMAGMA_ENABLE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${FR_PROJECT_HOME}/third_party/local -DGPU_TARGET='sm_80' -DBLA_VENDOR=Intel10_64lp_seq

# AMDGPU gfx906
export GPU_TARGET=gfx906
export ROCM=$ROCM_PATH
export ROCM_HOME=$ROCM_PATH
export BACKEND=hip
cmake .. -DMAGMA_ENABLE_HIP=ON -DCMAKE_INSTALL_PREFIX=${FR_PROJECT_HOME}/third_party/local -DGPU_TARGET='gfx906' -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc

# Alternative Method
# If cmake or make failed on your AMDGPU machine, you can use Makefile
# export ... # as above
# cd ${FR_PROJECT_HOME}/third_party/magma-2.9.0
# cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
# sed -i 's/-lmkl_gnu_thread/-lmkl_sequential/g' make.inc
# make lib -j${N_JOBS}
# make install prefix=${FR_PROJECT_HOME}/third_party/local -j${N_JOBS}

# ================================================

make -j${N_JOBS}
make install -j${N_JOBS}

```