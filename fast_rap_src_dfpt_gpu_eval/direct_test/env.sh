
source /opt/intel/oneapi/mkl/latest/env/vars.sh
source /opt/intel/oneapi/mpi/latest/env/vars.sh

# Add magma lib path
export LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
