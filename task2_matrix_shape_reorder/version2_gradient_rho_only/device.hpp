#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
  #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__clang__)
    #define HOST_DEVICE __host__ __device__
  #else
    #define HOST_DEVICE
  #endif
#else
  #define HOST_DEVICE
#endif

#define PRAGMA(X) _Pragma(#X)

#define XDEF_UNROLL      _Pragma("unroll")
#define XDEF_NO_UNROLL   _Pragma("unroll 1")
#define XDEF_UNROLL_N(n) PRAGMA(unroll n)

// 统一错误检查宏
#if defined(__CUDACC__)
  #define DEV_CHECK(call)                                                                                              \
    {                                                                                                                  \
      const cudaError_t error = call;                                                                                  \
      if (error != cudaSuccess) {                                                                                      \
        printf("CUDA Error: %s:%d, ", __FILE__, __LINE__);                                                             \
        printf("code:%d, reason: %s\n", static_cast<int>(error), cudaGetErrorString(error));                           \
        exit(1);                                                                                                       \
      }                                                                                                                \
    }
#elif defined(__HIPCC__)
  #define DEV_CHECK(call)                                                                                              \
    {                                                                                                                  \
      const hipError_t error = call;                                                                                   \
      if (error != hipSuccess) {                                                                                       \
        printf("HIP Error: %s:%d, ", __FILE__, __LINE__);                                                              \
        printf("code:%d, reason: %s\n", static_cast<int>(error), hipGetErrorString(error));                            \
        exit(1);                                                                                                       \
      }                                                                                                                \
    }
#else
  #define DEV_CHECK(call) call
#endif

#if defined(__CUDACC__) || defined(_DFPT_ENABLE_CUDA_)
  #define DEV_SET_DEVICE       cudaSetDevice
  #define DEV_GET_DEVICE_COUNT cudaGetDeviceCount

  #define DEV_MALLOC                cudaMalloc
  #define DEV_MEMCPY_H2D            cudaMemcpyAsync
  #define DEV_MEMCPY_D2H            cudaMemcpyAsync
  #define DEV_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
  #define DEV_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
  #define DEV_FREE                  cudaFree

  #define DEV_EVENT_T            cudaEvent_t
  #define DEV_EVENT_CREATE       cudaEventCreate
  #define DEV_EVENT_RECORD       cudaEventRecord
  #define DEV_EVENT_SYNCHRONIZE  cudaEventSynchronize
  #define DEV_EVENT_ELAPSED_TIME cudaEventElapsedTime
  #define DEV_EVENT_DESTROY      cudaEventDestroy

  #define DEV_STREAM_T           cudaStream_t
  #define DEV_STREAM_CREATE      cudaStreamCreate
  #define DEV_STREAM_DESTROY     cudaStreamDestroy
  #define DEV_STREAM_SYNCHRONIZE cudaStreamSynchronize

#elif defined(__HIPCC__) || defined(_DFPT_ENABLE_HIP_)
  #define DEV_SET_DEVICE       hipSetDevice
  #define DEV_GET_DEVICE_COUNT hipGetDeviceCount

  #define DEV_MALLOC                hipMalloc
  #define DEV_MEMCPY_H2D            hipMemcpyAsync
  #define DEV_MEMCPY_D2H            hipMemcpyAsync
  #define DEV_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
  #define DEV_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
  #define DEV_FREE                  hipFree

  #define DEV_EVENT_T            hipEvent_t
  #define DEV_EVENT_CREATE       hipEventCreate
  #define DEV_EVENT_RECORD       hipEventRecord
  #define DEV_EVENT_SYNCHRONIZE  hipEventSynchronize
  #define DEV_EVENT_ELAPSED_TIME hipEventElapsedTime
  #define DEV_EVENT_DESTROY      hipEventDestroy

  #define DEV_STREAM_T           hipStream_t
  #define DEV_STREAM_CREATE      hipStreamCreate
  #define DEV_STREAM_DESTROY     hipStreamDestroy
  #define DEV_STREAM_SYNCHRONIZE hipStreamSynchronize
#else
  #define DEV_STREAM_T void *
#endif

#define TM_DEV_INIT(var)                                                                                               \
  RemoveConst_t<decltype(&var(0))> var##_dev = nullptr;                                                                \
  DEV_CHECK(DEV_MALLOC((void **)&var##_dev, var.size() * sizeof(decltype(var(0)))));

#define TM_DEV_H2D(var, stream)                                                                                        \
  DEV_CHECK(DEV_MEMCPY_H2D(                                                                                            \
      var##_dev, (const void *)var.data(), var.size() * sizeof(decltype(var(0))), DEV_MEMCPY_HOST_TO_DEVICE, stream));

#define TM_DEV_D2H(var, stream)                                                                                        \
  DEV_CHECK(DEV_MEMCPY_D2H(                                                                                            \
      (void *)var.data(), var##_dev, var.size() * sizeof(decltype(var(0))), DEV_MEMCPY_DEVICE_TO_HOST, stream));

#define CDIV(x, align)     (((x) + (align) - 1) / (align))
#define ALIGN_UP(x, align) (((x) + (align) - 1) / (align) * (align))

#define TESTING_CHECK(err)                                                                                             \
  do {                                                                                                                 \
    magma_int_t err_ = (err);                                                                                          \
    if (err_ != 0) {                                                                                                   \
      fprintf(                                                                                                         \
          stderr,                                                                                                      \
          "Error: %s\nfailed at %s:%d: error %lld: %s\n",                                                              \
          #err,                                                                                                        \
          __FILE__,                                                                                                    \
          __LINE__,                                                                                                    \
          (long long)err_,                                                                                             \
          magma_strerror(err_));                                                                                       \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)

#include "magma_v2.h"

struct DeviceInfo {
  int device_id;
  DEV_STREAM_T stream;
  magma_queue_t magma_queue;
};
extern DeviceInfo devInfo;

#if defined(__CUDACC__) || defined(__HIPCC__)

template <bool enable = true>
class EventHelper {
private:
  DEV_STREAM_T stream;
  DEV_EVENT_T cu_start, cu_stop;

public:
  EventHelper(DEV_STREAM_T &stream) : stream(stream) {
    DEV_CHECK(DEV_EVENT_CREATE(&cu_start));
    DEV_CHECK(DEV_EVENT_CREATE(&cu_stop));
  }
  ~EventHelper() {
    DEV_CHECK(DEV_EVENT_DESTROY(cu_start));
    DEV_CHECK(DEV_EVENT_DESTROY(cu_stop));
  }
  inline void record_start() {
    DEV_CHECK(DEV_EVENT_RECORD(cu_start, stream));
  }
  inline void record_stop() {
    DEV_CHECK(DEV_EVENT_RECORD(cu_stop, stream));
  }
  inline void synchronize() {
    DEV_CHECK(DEV_EVENT_SYNCHRONIZE(cu_stop));
  }
  inline float elapsed_time(const std::string &info) {
    DEV_CHECK(DEV_EVENT_RECORD(cu_stop, stream));
    DEV_CHECK(DEV_EVENT_SYNCHRONIZE(cu_stop));
    float milliseconds = 0;
    DEV_CHECK(DEV_EVENT_ELAPSED_TIME(&milliseconds, cu_start, cu_stop));
    printf("%s: %f ms\n", info.c_str(), milliseconds);
    return milliseconds;
  }
};

template <>
class EventHelper<false> {
private:
  DEV_STREAM_T stream;
  DEV_EVENT_T cu_start, cu_stop;

public:
  EventHelper(DEV_STREAM_T &stream) : stream(stream) {
  }
  ~EventHelper() {
  }
  inline void record_start() {
  }
  inline void record_stop() {
  }
  inline void synchronize() {
  }
  inline float elapsed_time(const std::string &info) {
    return 0;
  }
};

#endif

void init_dfpt_device_info();
void free_dfpt_device_info();
