#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_HIP_Space.hpp>

namespace nlcglib {

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
struct exec<Kokkos::Experimental::HIPSpace>
{
  using type = Kokkos::Experimental::HIP;
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


}  // nlcglib
