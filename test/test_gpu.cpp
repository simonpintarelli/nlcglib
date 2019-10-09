#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include <Kokkos_Core.hpp>

#include <mpi.h>
#include <iostream>
#include <random>

#include "cudaProfiler.h"

using namespace nlcglib;

std::uniform_real_distribution<double> unif01(0, 1);
std::mt19937 gen(0);


// template<class T>
// class X
// {
// };


typedef Kokkos::complex<double> complex_double;

void run() {
  int n = 4000;
  int m = 400;

  // typedef KokkosDVector<complex_double**,
  //                           SlabLayoutV,
  //                           Kokkos::LayoutLeft,
  //                           Kokkos::CudaSpace>
  //     vector_t;

  for (int rep = 0; rep < 1000; ++rep) {
    std::cout << "rep " << rep << "\n";
    KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::CudaSpace> X(
        Map<>(Communicator(), SlabLayoutV({{0, 0, n, m}})));

    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X.array());
    for (auto i = 0ul; i < host.size(); ++i) {
      *(host.data() + i) = unif01(gen);
    }

    Kokkos::deep_copy(X.array(), host);


    // eigh(X);
    auto H = inner_()(X, X);
    auto HH = empty_like()(H);
    // auto  Z = transform_alloc(X, H, 1.0);
    auto Y = loewdin(X);
  }
  // Initialize A matrix on device
  // typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;
  // Kokkos::parallel_for("init_H",
  //                      mdrange_policy({0, 0}, {m, m}),
  //                      MDFunctor<typename vector_t::storage_t>(M));

}

int main(int argc, char *argv[])
{
  Kokkos::initialize();
  Communicator::init(argc, argv);
  cuProfilerStart();
  run();

  cuProfilerStop();
  Kokkos::finalize();
  Communicator::finalize();
  return 0;
}
