#pragma once

#include <Kokkos_Core.hpp>

namespace nlcglib {

template <class SPACE>
struct exec
{
};

template <>
struct exec<Kokkos::CudaSpace>
{
  using type = Kokkos::Cuda;
};

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
