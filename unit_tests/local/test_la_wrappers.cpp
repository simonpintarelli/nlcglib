#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <iomanip>
#include <iostream>
#include "constants.hpp"
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "la/magma.hpp"
#include "test_common.hpp"

using namespace nlcglib;

class CPUKokkosVectors : public ::testing::Test
{
public:
  CPUKokkosVectors()
      : a_(Map<>(Communicator(), SlabLayoutV({{0, 0, 30, 5}})))
      , b_(Map<>(Communicator(), SlabLayoutV({{0, 0, 30, 5}})))
      , cRef_(Map<>(Communicator(), SlabLayoutV({{0, 0, 5, 5}})))
      , u_(Map<>(Communicator(), SlabLayoutV({{0, 0, 5, 5}})))
  {
  }

protected:
  void SetUp() override
  {
    int m = a_.map().nrows();
    int n = a_.map().ncols();
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        a_.array()(i, j) = j * m + i;
        b_.array()(i, j) = j * m + i;
      }
    }

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i == j)
          u_.array()(i, j) = 1;
        else
          u_.array()(i, j) = 0;
      }
    }

    double c_arr[25] = {8555,   21605,  34655,  47705,  60755,  21605,  61655, 101705, 141755,
                        181805, 34655,  101705, 168755, 235805, 302855, 47705, 141755, 235805,
                        329855, 423905, 60755,  181805, 302855, 423905, 544955};

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        cRef_.array()(i, j) = c_arr[i * n + j];
      }
    }
  }

  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> vector_t;

  vector_t a_;
  vector_t b_;
  vector_t cRef_;
  vector_t u_;
};

TEST_F(CPUKokkosVectors, InnerProductCPU)
{
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> vector_t;
  int n = 5;
  vector_t c(Map<>(Communicator(), SlabLayoutV({{0, 0, n, n}})));
  inner(c, a_, b_);
  // check results
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      auto x = c.array()(i, j);
      auto xRef = cRef_.array()(i, j);
      EXPECT_DOUBLE_EQ(x, xRef);
      if (std::abs(xRef - x) > 1e-10) {
        std::cout << i << " " << j << " "
                  << "\nfail: got " << x << " expected " << xRef << "\n";
      }
    }
  }
}

TEST_F(CPUKokkosVectors, TransformCPU)
{
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> vector_t;
  int n = a_.map().ncols();
  int m = a_.map().nrows();
  vector_t c(Map<>(Communicator(), SlabLayoutV({{0, 0, m, n}})));

  transform(c, 0., 1., a_, u_);
  // host vector type

  // compare c=a_*u_ with a_, since u_ is the identity.
  for (int i = 0; i < c.map().nrows(); ++i)
    for (int j = 0; j < c.map().ncols(); ++j) {
      auto x = c.array()(i, j);
      auto xRef = a_.array()(i, j);
      EXPECT_DOUBLE_EQ(x, xRef);
      if (std::abs(xRef - x) > 1e-10) {
        std::cout << i << " " << j << " "
                  << "\nfail: got " << x << " expected " << xRef << "\n";
      }
    }
  std::cout << "\n";
}

#if defined(__NLCGLIB__ROCM) || defined(__NLCGLIB__CUDA)

using nlcglib::test::device_space_t;

class GPUKokkosVectors : public ::testing::Test
{
public:
  GPUKokkosVectors()
      : a_(Map<>(Communicator(), SlabLayoutV({{0, 0, 30, 5}})))
      , b_(Map<>(Communicator(), SlabLayoutV({{0, 0, 30, 5}})))
      , cRef_(Map<>(Communicator(), SlabLayoutV({{0, 0, 5, 5}})))
      , u_(Map<>(Communicator(), SlabLayoutV({{0, 0, 5, 5}})))
  {
  }

  void SetUp() override
  {
    int m = a_.map().nrows();
    int n = a_.map().ncols();

    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;
    auto mA = a_.array();
    auto mB = b_.array();
    Kokkos::parallel_for(
        "init", mdrange_policy({0, 0}, {m, n}), KOKKOS_LAMBDA(int i, int j) {
          mA(i, j) = j * m + i;
          mB(i, j) = j * m + i;
        });

    double c_arr[25] = {8555,   21605,  34655,  47705,  60755,  21605,  61655, 101705, 141755,
                        181805, 34655,  101705, 168755, 235805, 302855, 47705, 141755, 235805,
                        329855, 423905, 60755,  181805, 302855, 423905, 544955};

    auto M = cRef_.array();
    auto U = u_.array();
    Kokkos::parallel_for(
        "init", mdrange_policy({0, 0}, {5, 5}), KOKKOS_LAMBDA(int i, int j) {
          M(i, j) = c_arr[i * n + j];
          if (i == j)
            U(i, j) = 1;
          else
            U(i, j) = 0;
        });
  }

protected:
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t> vector_t;

  vector_t a_;
  vector_t b_;
  vector_t cRef_;
  vector_t u_;
};


TEST_F(GPUKokkosVectors, InnerProductGPU)
{
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t> vector_t;
  int n = 5;
  vector_t c(Map<>(Communicator(), SlabLayoutV({{0, 0, n, n}})));
  inner(c, a_, b_);
  // host vector type
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> h_vector_t;

  h_vector_t h_c(c.map());
  h_vector_t h_cRef(c.map());

  // copy from device to host
  deep_copy(h_cRef, cRef_);
  deep_copy(h_c, c);

  // check results
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_DOUBLE_EQ(h_c.array()(i, j), h_cRef.array()(i, j));
    }
  }
}


TEST_F(GPUKokkosVectors, TransformGPU)
{
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t> vector_t;
  int n = a_.map().ncols();
  int m = a_.map().nrows();
  vector_t c(Map<>(Communicator(), SlabLayoutV({{0, 0, m, n}})));
  transform(c, 0., 1., a_, u_);
  // host vector type
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace> h_vector_t;

  h_vector_t h_c(c.map());
  h_vector_t h_cRef(c.map());
  deep_copy(h_cRef, a_);
  deep_copy(h_c, c);

  // compare c=a_*u_ with a_, since u_ is the identity.
  for (int i = 0; i < c.map().nrows(); ++i)
    for (int j = 0; j < c.map().ncols(); ++j) {
      EXPECT_DOUBLE_EQ(h_c.array()(i, j), h_cRef.array()(i, j));
    }
}


#endif

/* ------------------------------------------------------------------------------------------
 * Typed tests, instantiated for host and (if enabled) device memory.
 * ------------------------------------------------------------------------------------------ */

using namespace nlcglib::test;

template <class memspace>
class LinAlg : public ::testing::Test
{
protected:
  using matrix_t = cmatrix_t<memspace>;
};

TYPED_TEST_SUITE(LinAlg, KokkosMemTypes, MemSpaceNames);

/// Hermitian eigenvalue problem for the tridiagonal matrix tridiag(-2, 1, -2), whose
/// eigenvalues are 1 - 4 cos(k pi / (n+1)), k = 1..n (ascending).
TYPED_TEST(LinAlg, EigHermitian)
{
  const int n = 20;
  auto A = make_matrix<TypeParam>(n, n);
  {
    auto A_host = Kokkos::create_mirror(A.array());
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        A_host(i, j) = (i == j) ? 1.0 : ((i - j == 1 || j - i == 1) ? -2.0 : 0.0);
      }
    }
    Kokkos::deep_copy(A.array(), A_host);
  }

  auto V = make_matrix<TypeParam>(n, n);
  Kokkos::View<double *, TypeParam> w("w", n);
  eigh(V, w, A);

  auto w_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
  for (int k = 1; k <= n; ++k) {
    double ref = 1 - 4 * std::cos(k * constants::pi / (n + 1));
    EXPECT_NEAR(w_host(k - 1), ref, 1e-12) << "k = " << k;
  }

  // eigenvectors (sign/phase independent): A V = V diag(w) and V^H V = I
  auto AV = transform_alloc(A, V);
  auto Vw = make_matrix<TypeParam>(n, n);
  scale(Vw, V, w, 1.0, 0.0);
  auto AV_host = to_host(AV);
  auto Vw_host = to_host(Vw);
  double res = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) res = std::max(res, Kokkos::abs(AV_host(i, j) - Vw_host(i, j)));
  EXPECT_LT(res, 1e-12) << "residual |A V - V diag(w)|";

  auto VhV = inner_()(V, V);
  EXPECT_LT(max_dev_from_identity(to_host(VhV)), 1e-12) << "V^H V != I";
}

/// diag(make_diag(x)) == x, off-diagonal entries are zero
TYPED_TEST(LinAlg, DiagRoundtrip)
{
  const int n = 7;
  Kokkos::View<complex_double *, TypeParam> x("x", n);
  auto x_host = Kokkos::create_mirror(x);
  for (int i = 0; i < n; ++i) x_host(i) = complex_double(i + 1, -0.5 * i);
  Kokkos::deep_copy(x, x_host);

  auto D = make_diag()(x);
  auto D_host = to_host(D);
  auto d_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diag(D));
  ASSERT_EQ(d_host.extent(0), size_t(n));
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(d_host(i), x_host(i));
    for (int j = 0; j < n; ++j)
      if (i != j) EXPECT_EQ(D_host(i, j), complex_double(0));
  }
}

/// dst <- beta * dst + alpha * src * diag(x)
TYPED_TEST(LinAlg, ScaleColumns)
{
  const int m = 30, n = 5;
  auto src = make_matrix<TypeParam>(m, n);
  fill_random(src, 1);
  Kokkos::View<double *, TypeParam> x("x", n);
  auto x_host = Kokkos::create_mirror(x);
  for (int j = 0; j < n; ++j) x_host(j) = j + 1;
  Kokkos::deep_copy(x, x_host);

  auto dst = make_matrix<TypeParam>(m, n);
  scale(dst, src, x, 2.0);  // beta = 0 branch
  auto src_host = to_host(src);
  auto dst0_host = to_host(dst);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_NEAR(Kokkos::abs(dst0_host(i, j) - 2.0 * x_host(j) * src_host(i, j)), 0, 1e-14);

  scale(dst, src, x, 1.5, 0.5);  // beta != 0 branch
  auto dst1_host = to_host(dst);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_NEAR(
          Kokkos::abs(dst1_host(i, j) - (0.5 * dst0_host(i, j) + 1.5 * x_host(j) * src_host(i, j))),
          0,
          1e-14);
}

/// C = A B^H for non-square A, B
TYPED_TEST(LinAlg, OuterProduct)
{
  const int m = 7, n = 5, k = 3;
  auto A = make_matrix<TypeParam>(m, k);
  auto B = make_matrix<TypeParam>(n, k);
  fill_random(A, 2);
  fill_random(B, 3);
  auto C = make_matrix<TypeParam>(m, n);
  outer(C, A, B);

  auto A_host = to_host(A);
  auto B_host = to_host(B);
  auto C_host = to_host(C);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      complex_double ref = 0;
      for (int l = 0; l < k; ++l) ref += A_host(i, l) * Kokkos::conj(B_host(j, l));
      EXPECT_NEAR(Kokkos::abs(C_host(i, j) - ref), 0, 1e-13) << "i = " << i << ", j = " << j;
    }
  }
}

/// C <- beta * C + alpha * A
TYPED_TEST(LinAlg, Add)
{
  const int m = 11, n = 4;
  auto A = make_matrix<TypeParam>(m, n);
  auto C = make_matrix<TypeParam>(m, n);
  fill_random(A, 4);
  fill_random(C, 5);
  auto C0_host = to_host(C);
  auto A_host = to_host(A);

  complex_double alpha(2.0, 1.0), beta(0.5, 0);
  add(C, A, alpha, beta);
  auto C_host = to_host(C);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_NEAR(
          Kokkos::abs(C_host(i, j) - (beta * C0_host(i, j) + alpha * A_host(i, j))), 0, 1e-14);
}

namespace {

// Copying a whole input container to look up one key is never necessary.
class NoncopyableValues : public mvector<int>
{
public:
  explicit NoncopyableValues(const Communicator &comm)
      : mvector<int>(comm)
  {
  }
  NoncopyableValues(const NoncopyableValues &) = delete;
};

}  // namespace

TEST(MVectorApply, DeferredSnapshotAndCaching)
{
  const auto key = std::make_pair(1, 0);
  NoncopyableValues input(Communicator{MPI_COMM_WORLD});
  input[key] = 3;
  int lazy_calls = 0, cached_calls = 0;
  auto lazy = tapply(
      [&](int value) {
        ++lazy_calls;
        return 2 * value;
      },
      input);
  auto cached = tapply_async(
      [&](int value) {
        ++cached_calls;
        return 2 * value;
      },
      input);
  EXPECT_EQ(lazy_calls, 0);
  EXPECT_EQ(cached_calls, 0);
  EXPECT_TRUE(lazy.commk() == input.commk());
  EXPECT_TRUE(cached.commk() == input.commk());

  input[key] = 9;
  EXPECT_EQ(eval(lazy.at(key)), 6);
  EXPECT_EQ(eval(lazy.at(key)), 6);
  EXPECT_EQ(lazy_calls, 2);
  EXPECT_EQ(eval(cached.at(key)), 6);
  EXPECT_EQ(eval(cached.at(key)), 6);
  EXPECT_EQ(cached_calls, 1);

  mvector<std::function<int()>> source;
  source[key] = [counter = 0]() mutable { return ++counter; };
  auto immediate = tapply_eager([](int value) { return value; }, source);
  EXPECT_EQ(immediate.at(key), 1);
  EXPECT_EQ(eval(source.at(key)), 1);
}

TEST(MVectorApply, EagerValuesAndEmptyInput)
{
  const auto first = std::make_pair(1, 0);
  const auto second = std::make_pair(3, 1);
  NoncopyableValues input(Communicator{MPI_COMM_WORLD});
  input[second] = 9;
  input[first] = 4;
  mvector<double> weights;
  weights[first] = 0.5;
  weights[second] = 0.25;
  std::vector<int> visited;
  auto values = tapply_eager(
      [&](int value, double weight) {
        visited.push_back(value);
        return std::make_tuple(value, value * weight);
      },
      input,
      weights);
  EXPECT_EQ(visited, (std::vector<int>{4, 9}));
  EXPECT_EQ(values.at(first), std::make_tuple(4, 2.0));
  EXPECT_EQ(values.at(second), std::make_tuple(9, 2.25));
  EXPECT_TRUE(values.commk() == Communicator{MPI_COMM_SELF});

  NoncopyableValues empty(Communicator{MPI_COMM_WORLD});
  auto result = tapply_eager(
      [](int) -> int { throw std::runtime_error("empty input must not be evaluated"); }, empty);
  EXPECT_EQ(result.size(), 0);
  EXPECT_TRUE(result.commk() == Communicator{MPI_COMM_SELF});
}

TEST(MVectorApply, EagerInPlaceKeepsMatrixStorage)
{
  const auto key = std::make_pair(2, 0);
  auto matrix = nlcglib::test::make_matrix<Kokkos::HostSpace>(2, 2);
  Kokkos::deep_copy(matrix.array(), nlcglib::test::complex_double{0});
  mvector<decltype(matrix)> blocks;
  blocks[key] = matrix;
  int calls = 0;
  tapply_eager(
      [&](auto block) {
        ++calls;
        block.array()(0, 1) = nlcglib::test::complex_double{2.0, -3.0};
      },
      blocks);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(matrix.array()(0, 1), nlcglib::test::complex_double(2.0, -3.0));
}

int
main(int argc, char *argv[])
{
  int result = 0;
  // make sure initializers are called
  ::testing::InitGoogleTest(&argc, argv);
  Communicator::init(argc, argv);

  Kokkos::initialize();

#ifdef __NLCGLIB__MAGMA
  nlcg_init_magma();
#endif

  result = RUN_ALL_TESTS();

  Kokkos::finalize();

#ifdef __NLCGLIB__MAGMA
  nlcg_finalize_magma();
#endif

  Communicator::finalize();

  return result;
}
