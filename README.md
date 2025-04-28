# FAST RPA Evaluation

This is an experimental repository for SC25 AE/AD.

# Before Clone

**Important !!** This repository adopts git lfs.

```bash
git lfs install
git clone https://github.com/kzwrime/fast_rpa_eval
```

Or

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/kzwrime/fast_rpa_eval
cd fast_rpa_eval
git lfs pull
```

Or download ZIP directly.

# Evaluation Setup

Description of the hardware and software environment:

The experiments described in the paper were tested on the **Hygon Z100 GPU** in the **ORISE supercomputer**, using DTK suite (base on ROCM).

However, this project supports both **NVIDIA CUDA** and **AMD ROCM HIP**, so testing can be conducted on either NVIDIA or AMD GPUs, which can reflect the effect of the experiment. But the data may be obviously different with the difference of hardware environment.

## Recommended Compiler Versions

This experiment has been tested in the following compiler environments:

- GNU Compiler: 9.4.0
- CUDA: 12.4
- or ROCM: 5.7
- or DTK: 24.04

# Recommended Hardware Environment

- NVIDIA A100 (Tested)
- or AMDGPU MI50/MI60
- or Hygon Z100 GPU (Tested)

## Third-Party Library Dependencies

- Manually required:
  - Intel MKL (verified version: oneapi 2024.1)
- Installed via the methods provided below:
  - Eigen (verified version: eigen-3.4.1)
    - git commit 68f4e58cfacc686583d16cff90361f0b43bc2c1b
  - MAGMA (verified version: magma-2.9.0)

## Environment Variable Setup

```bash
export FR_PROJECT_HOME=$(pwd)
export N_JOBS=16

# Load MKL environment variables, e.g.,  
source /opt/intel/oneapi/mkl/latest/env/vars.sh
```

## Installing Third-Party Libraries

- eigen
    - run script: `./scripts/install_eigen.sh`
- magma
    - follow guide: `./scripts/install_magma.md`

```bash
# Install eigen by directly running the script  
./scripts/install_eigen.sh

# Install magma. Due to differences in software/hardware environments,  
# please refer to scripts/install_magma.md for manual execution.  
```

# Evaluation

Modify env.sh, set relevant environment variables correctly, then set proper compiler.

```bash
source env.sh

# For CUDA
export CXX=g++
export NVCC=nvcc

# For ROCM
export CXX=hiccc
export NVCC=hipcc
```

## task1_atom_tiling

```bash
cd task1_atom_tiling/version3_rho_with_bias
./test_rho.sh 2>&1 | tee test_rho.log
cat test_rho.log | grep "[[Result]]" | tee test_rho.log.csv
```

## task2_matrix_shape_reorder

```bash
cd task2_matrix_shape_reorder
./eval.sh
```

result: 

- task2_matrix_shape_reorder/version1_gradient_rho_only/test_rho.log.csv
- task2_matrix_shape_reorder/version2_gradient_rho_only/test_rho.log.csv

## task3_jit

```bash
cd task3_jit
./eval.sh
```

result: 

- task3_jit/version4_sumup_far_part_no_jit_only/test_sumup_far_part.log.csv
- task3_jit/version5_sumup_far_part_with_jit_or_aot/test_sumup_far_part.log.csv

## task4_common_calculation_eliminated

```bash
cd task4_common_calculation_eliminated
./eval.sh
```

result: 
- task4_common_calculation_eliminated/version6_rho_with_bias_calculate/test_rho.sh
- task4_common_calculation_eliminated/version7_rho_with_bias_directly/test_rho.log.csv


