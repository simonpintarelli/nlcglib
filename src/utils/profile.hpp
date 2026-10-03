#pragma once


#include <string>

#if defined(__NLCGLIB__CUDA) && defined(__NLCGLIB__TX)
#include <nvtx3/nvToolsExt.h>
#endif

#if defined(__NLCGLIB__ROCM) && defined(__NLCGLIB__TX)
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

#define PROFILER_CONCAT_IMPL(x, y) x##y
#define PROFILER_CONCAT(x, y) PROFILER_CONCAT_IMPL(x, y)

namespace nlcglib {
namespace txprofiler {
enum class _vendor
{
  cuda,
  rocm
};

template <enum _vendor>
class TimerVendor
{
};

#if defined(__NLCGLIB__CUDA)
template <>
class TimerVendor<_vendor::cuda>
{
public:
#if defined(__NLCGLIB__TX)
  TimerVendor(const std::string& str) { nvtxRangePush(str.c_str()); }
  ~TimerVendor() { nvtxRangePop(); }
#else
  TimerVendor(const std::string&) {}
#endif /* __NLCGLIB__TX */
};

using Timer = TimerVendor<_vendor::cuda>;
#endif /* __NLCGLIB__CUDA */

#if defined(__NLCGLIB__ROCM)
template <>
class TimerVendor<_vendor::rocm>
{
public:
#if defined(__NLCGLIB__TX)
  TimerVendor(const std::string& str) { roctxRangePush(str.c_str()); }
  ~TimerVendor() { roctxRangePop(); }
#else
  TimerVendor(const std::string&) {}
#endif /* __NLCGLIB__TX */
};
using Timer = TimerVendor<_vendor::rocm>;
#endif /* __NLCGLIB__ROCM */


}  // namespace txprofiler


#if defined(__NLCGLIB__CUDA) || defined(__NLCGLIB__ROCM)
#define PROFILE(identifier) \
  txprofiler::Timer PROFILER_CONCAT(GeneratedScopedTmer, __COUNTER__)(identifier);
#else
#define PROFILE(identifier) ;
#endif


}  // namespace nlcglib
