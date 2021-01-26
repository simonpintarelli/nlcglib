#include "overlap.hpp"
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "la/mvector.hpp"

using namespace nlcglib;

typedef std::complex<double> complex_double;

using typeX = KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace>;

template <typename mT>
const Matrix make_const(const mT &m)
{
  return make_const(m);
}

int main(int argc, char *argv[])
{
  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})));

  std::vector<complex_double> yptr(200 * 20);
  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutStride, Kokkos::MemoryUnmanaged> Y(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})),
      buffer_protocol<complex_double, 2>({1, 200}, {200, 20}, yptr.data(), memory_type::host));

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> H(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 20, 20}})));

  auto kokkos_array = X.array();

  mvector<decltype(X)> mvec;

  mvec[std::make_pair(0, 0)] = X;
  mvec[std::make_pair(0, 1)] = X;

  auto bX = make_buffer(mvec);
  auto bbX = make_const(mvec);

  return 0;
}
