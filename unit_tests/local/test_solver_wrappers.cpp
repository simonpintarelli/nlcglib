#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include "gtest/gtest.h"
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "test_common.hpp"

using namespace nlcglib;

// using complex_double = std::complex<double>;
using complex_double = Kokkos::complex<double>;

int nrows = 200;
int ncols = 20;

template <typename T>
class TestSymSolve : public ::testing::Test
{
public:
  void SetUp();
  TestSymSolve()
      : X(Map<>(Communicator(), SlabLayoutV({{0, 0, nrows, ncols}})))
      , H(Map<>(Communicator(), SlabLayoutV({{0, 0, ncols, ncols}})))
  {
  }

protected:
  using vector_t = KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, T>;
  //
  vector_t X;
  vector_t H;
};


template <typename T>
void
TestSymSolve<T>::SetUp()
{
  std::uniform_real_distribution<double> unif01(0, 1);
  std::mt19937 gen(0);

  auto arr = X.array();
  auto host_view = Kokkos::create_mirror(arr);
  for (size_t i = 0; i < host_view.size(); ++i) {
    *(host_view.data() + i) = unif01(gen);
  }
  Kokkos::deep_copy(arr, host_view);

  using matrix_t = KokkosDVector<complex_double **, SlabLayoutV, Kokkos::LayoutLeft, T>;
  matrix_t H(Map<>(Communicator(), SlabLayoutV({{0, 0, ncols, ncols}})));
}

// host, and device if enabled
using nlcglib::test::KokkosMemTypes;
using nlcglib::test::MemSpaceNames;

TYPED_TEST_SUITE(TestSymSolve, KokkosMemTypes, MemSpaceNames);

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
      if (i == j) {
        EXPECT_NEAR(S_host(i, j).real(), 1, 1e-8);
        EXPECT_NEAR(S_host(i, j).imag(), 0, 1e-8);
      } else {
        EXPECT_NEAR(S_host(i, j).imag(), 0, 1e-8);
        EXPECT_NEAR(S_host(i, j).real(), 0, 1e-8);
      }
    }
  }
}

/// Loewdin orthogonalization: Y = X (X^H X)^{-1/2} has orthonormal columns
template <class memspace>
class Loewdin : public ::testing::Test
{
};

TYPED_TEST_SUITE(Loewdin, KokkosMemTypes, MemSpaceNames);

TYPED_TEST(Loewdin, Orthonormal)
{
  using namespace nlcglib::test;
  auto X = make_matrix<TypeParam>(nrows, ncols);
  fill_random(X, 7);

  auto Y = loewdin(X);
  auto YhY = inner_()(Y, Y);
  EXPECT_LT(max_dev_from_identity(to_host(YhY)), 1e-10) << "Y^H Y != I";

  // overlap variant with S = identity must agree
  auto Y2 = loewdin(X, X);
  auto Y_host = to_host(Y);
  auto Y2_host = to_host(Y2);
  double err = 0;
  for (size_t i = 0; i < Y_host.extent(0); ++i)
    for (size_t j = 0; j < Y_host.extent(1); ++j)
      err = std::max(err, Kokkos::abs(Y_host(i, j) - Y2_host(i, j)));
  EXPECT_LT(err, 1e-12) << "loewdin(X, X) != loewdin(X)";
}
