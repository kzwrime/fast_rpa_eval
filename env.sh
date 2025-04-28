
# load mkl env, adjust the path according to your installation
source /opt/intel/oneapi/mkl/latest/env/vars.sh

# Add magma lib path
export LD_LIBRARY_PATH=$(pwd)/third_party/local/lib:$LD_LIBRARY_PATH
