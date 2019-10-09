#include "la/dvector.hpp"
#include "la/lapack.hpp"

#include <mpi.h>
#include <iostream>
#include <complex>

using namespace nlcglib;

// template<class T>
// class X
// {
// };

typedef std::complex<double> complex_double;


void
run_stacked_vector()
{
  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})));

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> H(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 20, 20}})));

  auto kokkos_array = X.array();

  auto X2 = std::make_tuple(X, X);
  // eigh(X);
  inner(H, X2, X2);
}


int main(int argc, char *argv[])
{
  Kokkos::initialize();
  MPI_Init(&argc, &argv);

  unsigned threads_count = 4;
  if (Kokkos::hwloc::available()) {
    threads_count = Kokkos::hwloc::get_available_numa_count() *
                    Kokkos::hwloc::get_available_cores_per_numa() *
                    Kokkos::hwloc::get_available_threads_per_core();
  }
  std::cout << "thread_count: " << threads_count;


  run();

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
