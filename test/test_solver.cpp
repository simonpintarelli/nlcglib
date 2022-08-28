#include "la/dvector.hpp"
#include "la/lapack.hpp"

#include <mpi.h>
#include <iostream>
#include <random>

using namespace nlcglib;

typedef std::complex<double> complex_double;

std::uniform_real_distribution<double> unif01(0, 1);
std::mt19937 gen(0);


template<class SPACE=Kokkos::HostSpace>
void
run_unmanaged()
{
  int ncols = 20;
  int nrows = 200;
  KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, SPACE> X(
      Map<>(Communicator(), SlabLayoutV({{0, 0, nrows, ncols}})));

  auto arr = X.array();
  auto host_view = Kokkos::create_mirror(arr);
  for (auto i = 0ul; i < host_view.size(); ++i) {
    *(host_view.data() + i) = unif01(gen);
  }

  Kokkos::deep_copy(arr, host_view);

  using matrix_t = KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, SPACE>;
  matrix_t H(
      Map<>(Communicator(), SlabLayoutV({{0, 0, ncols, ncols}})));

  inner(H, X, X);
  auto S = H.copy();

  solve_sym(H /* will be overwritten by Cholesky factorization */,
                S /* will be overwritten by solution */);

  std::cout <<  "extent " << S.array().extent(0) << "\n";
  std::cout <<  "extent " << S.array().extent(1) << "\n";

  auto S_host = Kokkos::create_mirror(S.array());
  Kokkos::deep_copy(S_host, S.array());
  // print result
  for (int i = 0; i < ncols; ++i) {
    for (int j = 0; j < ncols; ++j) {
      std::cout << S_host(i,j) << "  ";
    }
    std::cout << "\n";
  }
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

  std::cout << "run on HOST" << "\n";
  run_unmanaged<Kokkos::HostSpace>();

#ifdef __NLCGLIB__CUDA
  std::cout << "run on DEVICE"
            << "\n";
  run_unmanaged<Kokkos::CudaSpace>();
#endif

  // std::cout << "run non GPU" << "\n";
  // run_unmanaged<Kokkos::CudaSpace>();

  Communicator::finalize();
  Kokkos::finalize();
  return 0;
}
