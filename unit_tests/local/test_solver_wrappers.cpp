#include <gtest/gtest.h>
#include <Kokkos_HIP_Space.hpp>
#include <Kokkos_HostSpace.hpp>
#include <iomanip>
#include <iostream>
#include "gtest/gtest.h"
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include <random>

using namespace nlcglib;

// using complex_double = std::complex<double>;
using complex_double = Kokkos::complex<double>;

int nrows = 200;
int ncols = 20;

template<typename T>
class TestSymSolve : public ::testing::Test
{
public:
  void SetUp();
  TestSymSolve()
      : X(Map<>(Communicator(), SlabLayoutV({{0, 0, nrows, ncols}})))
      , H(Map<>(Communicator(), SlabLayoutV({{0, 0, ncols, ncols}})))
  {}
protected:
  using vector_t = KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, T>;
  //
  vector_t X;
  vector_t H;
};


template <typename T>
void TestSymSolve<T>::SetUp()
{
  std::uniform_real_distribution<double> unif01(0, 1);
  std::mt19937 gen(0);

  auto arr = X.array();
  auto host_view = Kokkos::create_mirror(arr);
  for (int i = 0; i < host_view.size(); ++i) {
    *(host_view.data() + i) = unif01(gen);
  }
  Kokkos::deep_copy(arr, host_view);

  using matrix_t = KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, T>;
  matrix_t H(Map<>(Communicator(), SlabLayoutV({{0, 0, ncols, ncols}})));
}

// https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#typed-tests
#ifdef __NLCGLIB__CUDA
using KokkosMemTypes = ::testing::Types<Kokkos::CudaSpace, Kokkos::HostSpace>;
#endif

#ifdef __NLCGLIB__ROCM
using KokkosMemTypes = ::testing::Types<Kokkos::Experimental::HIPSpace, Kokkos::HostSpace>;
#endif

TYPED_TEST_SUITE_P(TestSymSolve);
TYPED_TEST_SUITE(TestSymSolve, KokkosMemTypes);

TYPED_TEST(TestSymSolve, PotrfPotrs)
{
  inner(this->H, this->X, this->X);
  auto S = this->H.copy();

  solve_sym(this->H /* will be overwritten by Cholesky factorization */,
            S /* will be overwritten by solution */);

  std::cout << "extent " << S.array().extent(0) << "\n";
  std::cout << "extent " << S.array().extent(1) << "\n";

  auto S_host = Kokkos::create_mirror(S.array());
  Kokkos::deep_copy(S_host, S.array());
  // check result
  for (int i = 0; i < ncols; ++i) {
    for (int j = 0; j < ncols; ++j) {
      if ( i == j ) {
        EXPECT_NEAR(S_host(i, j).real(), 1, 1e-8);
        EXPECT_NEAR(S_host(i, j).imag(), 0, 1e-8);
      }
      else {
        EXPECT_NEAR(S_host(i, j).imag(), 0, 1e-8);
        EXPECT_NEAR(S_host(i, j).real(), 0, 1e-8);
      }
    }
  }
}
