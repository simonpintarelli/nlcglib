#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include "traits.hpp"

#ifdef __NLCGLIB_CUDA
void run()
{
  int n = 10;
  double arr[10];

  Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    c(arr, n);
  Kokkos::View<double*, Kokkos::HostSpace>
    a("A",  n);
  Kokkos::View<double*, Kokkos::CudaSpace>
    a_device("A", n);
  Kokkos::View<double*, Kokkos::HostSpace>
    b("b", n);
  for (int i = 0; i < n; ++i) {
    a(i) = i;
  }

  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0, n), KOKKOS_LAMBDA(int i) {
      a_device[i] = i;
    });

  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a_device);
  std::cout << "a.data: " << a.data() << "\n";
  std::cout << "b.data: " << b.data() << "\n";

  std::cout << "b" << "\n";
  for (int i = 0; i < n; ++i) {
    std::cout << b(i) << ", ";
  }
  std::cout << "\n";

  std::cout << "c (should be same as b)" << "\n";
  for (int i = 0; i < n; ++i) {
    std::cout << c(i) << ", ";
  }
  std::cout << "\n";
}


void
run_2d()
{
  int n = 3;
  double arr[n*n];

  Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged> c(arr, n, n);
  Kokkos::View<double **, Kokkos::HostSpace> a("A", n, n);
  Kokkos::View<double **, Kokkos::CudaSpace> a_device("A", n, n);
  Kokkos::View<double **, Kokkos::HostSpace> b("b", n, n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a(i, j) = i;
    }
  }

  // Kokkos::parallel_for(
  //     Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::Cuda>({{0, 0}}, {{n, n}}),
  //     KOKKOS_LAMBDA(int i, int j) { a_device(i, j) = i; });

  Kokkos::deep_copy(b, a);
  Kokkos::deep_copy(c, a_device);
  std::cout << "a.data: " << a.data() << "\n";
  std::cout << "b.data: " << b.data() << "\n";

  std::cout << "b"
            << "\n";
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << b(i, j) << ", ";
    }
  }
  std::cout << "\n";

  std::cout << "c (should be same as b)"
            << "\n";
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << c(i, j) << ", ";
    }
  }
  std::cout << "\n";
}
#endif

int
main(int argc, char *argv[])
{
  Kokkos::initialize();
#ifdef __NLCGLIB_CUDA
  run();
  run_2d();
#endif
  Kokkos::finalize();
  return 0;
}
