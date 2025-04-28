
#include "common.hpp"
#include "setting.h"

static int save_dfpt_data_test_stop_iter = 0;
extern "C" void save_dfpt_data_test_stop_() {
#ifdef SAVE_DFPT_DATA_TEST
  save_dfpt_data_test_stop_iter++;
  if (save_dfpt_data_test_stop_iter >= 1) {
    printf("%s:%d: save_dfpt_data_test_stop_iter: %d\n", __FILE__, __LINE__, save_dfpt_data_test_stop_iter);
    exit(0);
  }
#endif
}
