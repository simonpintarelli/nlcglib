#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "utils/timer.hpp"

#include <mpi.h>
#include <iostream>
#include <random>

using namespace nlcglib;

typedef std::complex<double> complex_double;

std::uniform_real_distribution<double> unif01(0, 1);
std::mt19937 gen(0);


template <enum Kokkos::Iterate iterate_t = Kokkos::Iterate::Default,
          class M0,
          class M1,
          class T2,
          class... KOKKOS2>
M0&
scale(M0& dst, const M1& src, const Kokkos::View<T2*, KOKKOS2...>& x, double alpha, double beta = 0)
{
  auto mDST = dst.array();
  auto mSRC = src.array();
  int m = mSRC.extent(0);
  int n = mSRC.extent(1);

  using vector_t = M0;
  using memspace = typename vector_t::storage_t::memory_space;
  if (Kokkos::SpaceAccessibility<Kokkos::Cuda, memspace>::accessible) {
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2, iterate_t>, Kokkos::Cuda> mdrange_policy;
    if (src.array().stride(0) == 1) {
      Kokkos::parallel_for(
          "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
            mDST(i, j) = mDST(i, j) * beta + alpha * x(j) * mSRC(i, j);
          });
    }
  } else if (Kokkos::SpaceAccessibility<Kokkos::Serial, memspace>::accessible) {
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2, iterate_t>, exec_t<memspace>> mdrange_policy;
    if (src.array().stride(0) == 1) {
      Kokkos::parallel_for(
          "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
            mDST(i, j) = mDST(i, j) * beta + alpha * x(j) * mSRC(i, j);
          });
    }
  } else {
    throw std::runtime_error("no suitable ExecutionSpace found.");
  }

  return dst;
}


template<class SPACE=Kokkos::HostSpace, enum Kokkos::Iterate iterate_t>
void
run()
{
  int ncols = 800;
  int nrows = 20000;
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

  auto Y = empty_like()(H);

  auto fn = Kokkos::View<complex_double*, SPACE>("fn", ncols);

  Timer timer;
  timer.start();
  scale<iterate_t>(Y, H, fn, 1.0);
  Kokkos::fence();
  double tlap = timer.stop();

  std::cout << "Timing: " << tlap << "\n";
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
  run<Kokkos::HostSpace, Kokkos::Iterate::Left>();

  std::cout << "run on DEVICE"
            << "\n";
  run<Kokkos::CudaSpace, Kokkos::Iterate::Right>();

  // std::cout << "run non GPU" << "\n";
  // run_unmanaged<Kokkos::CudaSpace>();

  Communicator::finalize();
  Kokkos::finalize();
  return 0;
}
