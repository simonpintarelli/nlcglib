#pragma once

#include <unistd.h>

namespace nlcglib {

#if defined(__NLCGLIB__CUDA)
#define GPU_PREFIX(fname) cuda##fname
#elif defined(__NLCGLIB__ROCM)
#define GPU_PREFIX(fname) hip##fname
#endif

#if defined(__NLCGLIB_CUDA) || defined(__NLCGLIB_ROCM)
#define CALL_DEVICE_API(func__, args__)                      \
  {                                                          \
    acc_error_t error;                                       \
    error = GPU_PREFIX(func__) args__;                       \
    if (error != GPU_PREFIX(Success)) {                      \
      char nm[1024];                                         \
      gethostname(nm, 1024);                                 \
      std::printf("hostname: %s\n", nm);                     \
      std::printf("Error in %s at line %i of file %s: %s\n", \
                  #func__,                                   \
                  __LINE__,                                  \
                  __FILE__,                                  \
                  GPU_PREFIX(GetErrorString)(error));        \
      stack_backtrace();                                     \
    }                                                        \
  }
#else
#define CALL_DEVICE_API(func__, args__)
#endif

namespace acc {

template <class T, class U = T>
void
copy(T* target, const U* src, size_t n);

struct acc_meminfo
{
  size_t free;
  size_t total;
};

acc_meminfo
get_mem_info();

}  // namespace acc
}  // namespace nlcglib
