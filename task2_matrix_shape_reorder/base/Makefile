# 编译器设置
CXX := $(CXX)
NVCC := $(NVCC)

CXXFLAGS := -std=c++17 -O3 -g -DDIRECT_TEST=1 -DATOM_TILE_SIZE=4 -lmagma -lmkl_gf_lp64 -lmkl_sequential -lmkl_core

# 根据 NVCC 是否包含 nvcc，选择启用 CUDA 或 HIP 宏
ifeq ($(findstring nvcc,$(NVCC)),nvcc)
  CXXFLAGS += -D_DFPT_ENABLE_CUDA_
  NVCCFLAGS := $(CXXFLAGS) --expt-relaxed-constexpr --compiler-options -fPIC
else
  CXXFLAGS += -D_DFPT_ENABLE_HIP_
  NVCCFLAGS := $(CXXFLAGS) -fPIC
endif


# 包含路径
INCLUDE_DIRS := ../../third_party/local/include ../../third_party/local/include/eigen3
CXXFLAGS += $(addprefix -I,$(INCLUDE_DIRS))
NVCCFLAGS += $(addprefix -I,$(INCLUDE_DIRS))

SRC_DIRS := .
BUILD_DIR := ./build

# 源文件和头文件
COMMON_SRCS := 
SUM_UP_SRCS := 
EVAL_RHO_CPP_SRCS := ${SRC_DIRS}/evaluate_first_order_rho.cpp
EVAL_RHO_CU_SRCS := ${SRC_DIRS}/evaluate_first_order_rho_run.cu
CU_SRCS := ${SRC_DIRS}/device.cu ${SRC_DIRS}/device_data.cu

# 目标文件路径设置
COMMON_OBJS := $(patsubst ${SRC_DIRS}/%.cpp,${BUILD_DIR}/%.cpp.o,$(COMMON_SRCS))
SUM_UP_OBJS := $(patsubst ${SRC_DIRS}/%.cpp,${BUILD_DIR}/%.cpp.o,$(SUM_UP_SRCS))
EVAL_RHO_OBJS := $(patsubst ${SRC_DIRS}/%.cpp,${BUILD_DIR}/%.cpp.o,$(EVAL_RHO_CPP_SRCS)) \
                 $(patsubst ${SRC_DIRS}/%.cu,${BUILD_DIR}/%.cuo.o,$(EVAL_RHO_CU_SRCS))
CU_OBJS := $(patsubst ${SRC_DIRS}/%.cu,${BUILD_DIR}/%.cuo.o,$(CU_SRCS))

HEADERS := $(wildcard ./*.h) $(wildcard ./*.hpp) $(wildcard ${SRC_DIRS}/*.h) $(wildcard ${SRC_DIRS}/*.hpp)

# 最终目标
TARGETS := ${BUILD_DIR}/evaluate_first_order_rho_run.out

# 默认目标
all: prepare_build $(TARGETS)

# 准备构建目录
prepare_build:
	@mkdir -p $(BUILD_DIR)

# 链接目标
${BUILD_DIR}/sum_up.out: $(SUM_UP_OBJS) $(CU_OBJS) $(COMMON_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

${BUILD_DIR}/evaluate_first_order_rho_run.out: $(EVAL_RHO_OBJS) $(CU_OBJS) $(COMMON_OBJS)
	echo "EVAL_RHO_OBJS = $(EVAL_RHO_OBJS)"
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# 编译规则
${BUILD_DIR}/%.cpp.o: ${SRC_DIRS}/%.cpp $(HEADERS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA编译规则
${BUILD_DIR}/%.cuo.o: ${SRC_DIRS}/%.cu $(HEADERS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 清理
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean prepare_build