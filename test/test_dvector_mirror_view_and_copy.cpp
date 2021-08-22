#include <Kokkos_Core.hpp>
#include "la/dvector.hpp"
#include "la/lapack.hpp"

#include <mpi.h>
#include <iostream>

#include "cudaProfiler.h"

using namespace nlcglib;

typedef Kokkos::complex<double> complex_double;

template<class T>
struct print_type {};

void
run()
{
  int n = 4000;
  int m = 400;

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::CudaSpace> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, n, m}})));

  auto Y = create_mirror_view_and_copy(Kokkos::CudaSpace(), X);

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

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::CudaSpace> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, n, m}})));

  auto Y = create_mirror_view_and_copy(Kokkos::HostSpace(), X);
}


void
test2()
{
  using matrix_t = Kokkos::View<double **, Kokkos::LayoutLeft ,Kokkos::CudaSpace>;

  int n = 10;
  matrix_t A("foo", n, n);

  // Kokkos::parallel_for(const ExecPolicy &policy, const FunctorType &functor)
  // for (int i = 0; i < n; ++i) {
  //   for (int j = 0; j < n; ++j) {
  //     if (i > j)
  //       A(i,j) = i+j;
  //     else
  //       A(i, j) = i + j + 100;
  //   }
  // }

  matrix_t B("bar", 10, 10);

  Kokkos::deep_copy(B, A);

  std::cout << "A.ptr = " << A.data() << "\n";
  std::cout << "B.ptr = " << B.data() << "\n";
  if ( A.data() != B.data()) {
    std::cout << "A and B are _distinct_ arrays!" << "\n";
  }

  auto C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  // print_type<decltype(C)>::info;
}

int
main(int argc, char *argv[])
{
  Kokkos::initialize();
  Communicator::init(argc, argv);
  cuProfilerStart();

  run();
  test2();
  run_copy_to_host();

  cuProfilerStop();
  Kokkos::finalize();
  Communicator::finalize();
  return 0;
}
