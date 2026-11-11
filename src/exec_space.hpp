#pragma once

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include "execution_backend.h"

namespace nlcglib {

/// Keep memory-space specialization inside implementation files.
template <class Function>
auto
dispatch_backend(execution_backend backend, Function&& function)
{
  switch (backend) {
    case execution_backend::host:
      return function.template operator()<Kokkos::HostSpace>();
    case execution_backend::device:
#ifdef __NLCGLIB__CUDA
      return function.template operator()<Kokkos::CudaSpace>();
#elif defined(__NLCGLIB__ROCM)
      return function.template operator()<Kokkos::HIPSpace>();
#else
      throw std::runtime_error("recompile nlcglib with CUDA or ROCM.");
#endif
    default:
      throw std::runtime_error("invalid execution backend");
  }
}

template <class SPACE>
struct exec
{
};

#ifdef __NLCGLIB__CUDA
template <>
struct exec<Kokkos::CudaSpace>
{
  using type = Kokkos::Cuda;
};
#endif

#ifdef __NLCGLIB__ROCM
template <>
struct exec<Kokkos::HIPSpace>
{
  using type = Kokkos::HIP;
};
#endif

template <>
struct exec<Kokkos::HostSpace>
{
#ifdef __USE_OPENMP
  using type = Kokkos::OpenMP;
#else
  using type = Kokkos::Serial;
#endif
};

template <class SPACE>
using exec_t = typename exec<SPACE>::type;


}  // namespace nlcglib
