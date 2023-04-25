#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include "hip/hip_space.hpp"
#include "la/dvector.hpp"
#include "la/lapack.hpp"

#ifdef __NLCGLIB__ROCM
using device_space_t = Kokkos::Experimental::HIPSpace;
#elif defined __NLCGLIB__CUDA
using device_space_t = Kokkos::CudaSpace;
#endif

using namespace nlcglib;

typedef Kokkos::complex<double> complex_double;

template<class T>
struct print_type {};

void
run()
{
  int n = 4000;
  int m = 400;

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, n, m}})));

  auto Y = create_mirror_view_and_copy(device_space_t(), X);

  if (Y.array().data() == X.array().data())
  {
    std::cout << "create_mirror_view_and_copy(SPACE, KokkosDVectors) works"
              << "\n";
  }
  else
  {
    std::cout << "create_mirror_view_and_copy(SPACE, KokkosDVectors) broken"
              << "\n";
  }
}


void
run_copy_to_host()
{
  int n = 4000;
  int m = 400;

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, n, m}})));

  auto Y = create_mirror_view_and_copy(Kokkos::HostSpace(), X);
}


void
test2()
{
  using matrix_t = Kokkos::View<double **, Kokkos::LayoutLeft ,device_space_t>;

  int n = 10;
  matrix_t A("foo", n, n);

  matrix_t B("bar", 10, 10);

  Kokkos::deep_copy(B, A);

  std::cout << "A.ptr = " << A.data() << "\n";
  std::cout << "B.ptr = " << B.data() << "\n";
  if ( A.data() != B.data()) {
    std::cout << "A and B are _distinct_ arrays!" << "\n";
  }

  auto C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
}

int
main(int argc, char *argv[])
{
  Kokkos::initialize();
  Communicator::init(argc, argv);

  run();
  test2();
  run_copy_to_host();

  Kokkos::finalize();
  Communicator::finalize();
  return 0;
}
