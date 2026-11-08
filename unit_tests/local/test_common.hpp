#pragma once

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <type_traits>
#include "la/dvector.hpp"

namespace nlcglib::test {

#if defined(__NLCGLIB__ROCM)
using device_space_t = Kokkos::HIPSpace;
#elif defined(__NLCGLIB__CUDA)
using device_space_t = Kokkos::CudaSpace;
#endif

/// memory spaces typed tests are instantiated for: host always, device if enabled
#if defined(__NLCGLIB__ROCM) || defined(__NLCGLIB__CUDA)
using KokkosMemTypes = ::testing::Types<Kokkos::HostSpace, device_space_t>;
#else
using KokkosMemTypes = ::testing::Types<Kokkos::HostSpace>;
#endif

class MemSpaceNames
{
public:
  template <class T>
  static std::string GetName(int)
  {
    if (std::is_same<T, Kokkos::HostSpace>::value) return "Host";
    return "Device";
  }
};

using complex_double = Kokkos::complex<double>;

template <class memspace>
using cmatrix_t = KokkosDVector<complex_double**, SlabLayoutV, Kokkos::LayoutLeft, memspace>;

template <class memspace>
cmatrix_t<memspace>
make_matrix(int m, int n)
{
  return cmatrix_t<memspace>(Map<>(Communicator(), SlabLayoutV({{0, 0, m, n}})));
}

/// fill with uniform random numbers in [0,1) + i [0,1). Initialized on the host and copied,
/// so this works for any memory space (no device lambdas involved).
template <class MAT>
void
fill_random(MAT& mat, unsigned seed)
{
  std::uniform_real_distribution<double> unif01(0, 1);
  std::mt19937 gen(seed);
  auto host = Kokkos::create_mirror(mat.array());
  for (size_t j = 0; j < host.extent(1); ++j) {
    for (size_t i = 0; i < host.extent(0); ++i) {
      double re = unif01(gen);
      double im = unif01(gen);
      host(i, j) = complex_double(re, im);
    }
  }
  Kokkos::deep_copy(mat.array(), host);
}

/// host *copy* of the underlying Kokkos view (a snapshot, never aliases host memory)
template <class MAT>
auto
to_host(const MAT& mat)
{
  auto host = Kokkos::create_mirror(Kokkos::HostSpace(), mat.array());
  Kokkos::deep_copy(host, mat.array());
  return host;
}

/// max_ij |A_ij - delta_ij| for a host-accessible 2d view
template <class VIEW>
double
max_dev_from_identity(const VIEW& A)
{
  double err = 0;
  for (size_t i = 0; i < A.extent(0); ++i) {
    for (size_t j = 0; j < A.extent(1); ++j) {
      double ref = (i == j) ? 1.0 : 0.0;
      err = std::max(err, Kokkos::abs(A(i, j) - complex_double(ref)));
    }
  }
  return err;
}

}  // namespace nlcglib::test
