#include "la/dvector.hpp"
#include "la/lapack.hpp"

#include <mpi.h>
#include <iostream>

using namespace nlcglib;

// template<class T>
// class X
// {
// };

typedef std::complex<double> complex_double;

void run() {
  KokkosDVector<complex_double**, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})));

  KokkosDVector<complex_double**, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> H(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 20, 20}})));

  auto kokkos_array = X.array();

  // eigh(X);
  inner(H, X, X);
}


void
run_unmanaged()
{
  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})));

  std::vector<complex_double> yptr(200*20);
  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutStride, Kokkos::MemoryUnmanaged> Y(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})),
      buffer_protocol<complex_double, 2>({1, 200}, {200, 20}, yptr.data(), memory_type::host)
  );

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> H(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 20, 20}})));

  auto kokkos_array = X.array();

  inner(H, X, X);

  auto S = H.copy();

  Kokkos::View<double*, Kokkos::HostSpace> eigvals("eigvals", 20);
  eigh(H, eigvals, S);
}

#ifdef __CUDA
void
run_unmanaged_cuda()
{
  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::CudaSpace> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})));

  complex_double yptr[200 * 20];
  KokkosDVector<complex_double **,
                    SlabLayoutV,
                    Kokkos::LayoutStride,
                    Kokkos::CudaSpace,
                    Kokkos::MemoryUnmanaged>
      Y(Map<>(Communicator(), SlabLayoutV({{0, 0, 200, 20}})),
        buffer_protocol<complex_double, 2>({1, 200}, {200, 20}, yptr, memory_type::host));

  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::CudaSpace> H(
      Map<>(Communicator(), SlabLayoutV({{0, 0, 20, 20}})));

  auto kokkos_array = X.array();

  inner(H, X, X);

  auto S = H.copy();

  Kokkos::View<double*, Kokkos::CudaSpace> eigvals("eigvals", 20);

  eigh(S, eigvals, H);

}
#endif


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
  // inner(H, X2, X2);
}


int main(int argc, char *argv[])
{
  Kokkos::initialize();
  Communicator::init(argc, argv);

  unsigned threads_count = 4;
  if (Kokkos::hwloc::available()) {
    threads_count = Kokkos::hwloc::get_available_numa_count() *
                    Kokkos::hwloc::get_available_cores_per_numa() *
                    Kokkos::hwloc::get_available_threads_per_core();
  }
  std::cout << "thread_count: " << threads_count;

  run();

  #ifdef __CUDA
  run_unmanaged_cuda();
  #endif

  Communicator::finalize();
  Kokkos::finalize();
  return 0;
}
