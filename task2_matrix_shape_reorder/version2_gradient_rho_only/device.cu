#include "common.hpp"
#include "device.hpp"
#include "setting.h"

DeviceInfo devInfo;

void init_dfpt_device_info() {
  int device_id = 0;
  printf("Initializing device info..., using device %d\n", device_id);

  int num_devices;
  DEV_CHECK(DEV_GET_DEVICE_COUNT(&num_devices));
  if (device_id >= num_devices) {
    printf("Invalid device ID %d, num_devices = %d\n", device_id, num_devices);
    exit(1);
  }

  DEV_CHECK(DEV_SET_DEVICE(device_id));

  DEV_CHECK(DEV_STREAM_CREATE(&devInfo.stream));
  TESTING_CHECK(magma_init());
  magma_print_environment();

  magma_device_t magmadev;
  magma_getdevice(&magmadev);
  printf("MAGMA is using device: %d\n", magmadev);

#if defined(__CUDACC__)
  magma_queue_create_from_cuda(device_id, devInfo.stream, NULL, NULL, &devInfo.magma_queue);
#elif defined(__HIPCC__)
  magma_queue_create_from_hip(device_id, devInfo.stream, NULL, NULL, &devInfo.magma_queue);
#endif
}

void free_dfpt_device_info() {
  DEV_CHECK(DEV_STREAM_DESTROY(devInfo.stream));
  TESTING_CHECK(magma_finalize());
}
