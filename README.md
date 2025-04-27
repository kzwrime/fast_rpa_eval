# FAST RPA Evaluation

This is an experimental repository for SC25 AE/AD.

# Before Clone

This repository adopts git lfs.

```bash
git lfs install
git clone https://github.com/kzwrime/fast_rpa_eval
```

Or download ZIP directly.

# Evaluation Setup

Description of the hardware and software environment:

The experiments described in the paper were tested on the **Hygon Z100 GPU** in the **ORISE supercomputer**. A similar environment would be the AMDGPU MI50/60.

However, this project supports both **NVIDIA CUDA** and **AMD ROCM HIP**, so testing can be conducted on either NVIDIA or AMD GPUs, which can reflect the effect of the experiment. But the data may be obviously different with the difference of hardware environment.

## Recommended Compiler Versions

This experiment has been tested in the following compiler environments:

- GNU Compiler: 9.4.0
- CUDA: 12.4
- ROCM: 5.7

# Recommended Hardware Environment

- NVIDIA A100
- AMDGPU MI50/MI60

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




