#include "gpu/acc.hpp"
#if defined(__NLCGLIB__CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(__NLCGLIB__ROCM)
#include <hip/hip_runtime_api.h>
#endif

namespace nlcglib {
namespace acc {

acc_meminfo
get_mem_info()
{
  acc_meminfo info{};
  CALL_DEVICE_API(MemGetInfo, (&info.free, &info.total));
  return info;
}

}  // namespace acc

}  // namespace nlcglib
