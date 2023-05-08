#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace nlcglib {
namespace env {
/// Check if environment variable NLCG_DISABLE_NEWTON_EFERMI is set (using a singleton).
bool
get_skip_newton_efermi()
{
  static std::atomic<int> skip_newton{-1};
  if (skip_newton.load(std::memory_order_relaxed) == -1) {
    char* disable_efermi = std::getenv("NLCGLIB_DISABLE_NEWTON_EFERMI");
    if (disable_efermi == nullptr) {
      // the variable is not set
      skip_newton.store(0, std::memory_order_relaxed);
    } else {
      // the environment variable NLCGLIB_DISABLE_NEWTON_EFERMI is defined (holds any value except
      // 0)
      if (std::strcmp("0", disable_efermi) == 0) {
        skip_newton.store(0, std::memory_order_relaxed);
      } else {
        skip_newton.store(1, std::memory_order_relaxed);
      }
    }
  }
  return skip_newton.load(std::memory_order_relaxed) == 1;
}

}  // namespace env
}  // namespace nlcglib
