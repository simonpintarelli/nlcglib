#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include <Kokkos_Core.hpp>
#include "la/utils.hpp"
#include "la/lapack.hpp"

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


struct functor
{
  template<class X_t>
  std::tuple<to_layout_left_t<X_t>, double> operator()(const X_t& x)
  // auto
  {
    return std::make_tuple(inner_()(x, x), 1.0);
  }
};



typedef Kokkos::complex<double> complex_double;


#ifdef __NLCGLIB__CUDA
auto run() {

  typedef KokkosDVector<complex_double**,
                            SlabLayoutV,
                            Kokkos::LayoutLeft,
                            Kokkos::CudaSpace>
      vector_t;

  mvector<vector_t> X;

  auto res = eval_threaded(tapply(functor(), X));
  auto resa = tapply_async(functor(), X);

  // here the lambda is accepted, as it does not contain any calls to cuda.
  auto evaled = eval_threaded(tapply([](auto x) { return std::get<0>(eval(x)); }, resa));
  return res;
}
#endif

int main(int argc, char *argv[])
{
  Kokkos::initialize();
  Communicator::init(argc, argv);

#ifdef __NLCGLIB__CUDA
  cuProfilerStart();
  run();
  cuProfilerStop();
#endif

  Kokkos::finalize();
  Communicator::finalize();
  return 0;
}
