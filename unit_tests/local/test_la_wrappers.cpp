#include <gtest/gtest.h>
#include <Kokkos_HIP_Space.hpp>
#include <iostream>
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "la/magma.hpp"
#include <iomanip>

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

  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace>
      vector_t;

  vector_t a_;
  vector_t b_;
  vector_t cRef_;
  vector_t u_;
};

TEST_F(CPUKokkosVectors, InnerProductCPU)
{
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace>
      vector_t;
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
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace>
      vector_t;
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
  // compare c=a_*u_ with a_, since u_ is the identity.
  // for (int i = 0; i < c.map().nrows(); ++i)
  //   for (int j = 0; j < c.map().ncols(); ++j) {
  //     // std::cout << h_cRef.array()(i, j) << " ";
  //     std::cout << h_c.array()(i, j) << " ";
  //   }
  std::cout << "\n";
}

#if defined(__NLCGLIB__ROCM) || defined(__NLCGLIB__CUDA)

#ifdef __NLCGLIB__ROCM
using device_space_t = Kokkos::Experimental::HIPSpace;
#else
using device_space_t = Kokkos::CudaSpace;
#endif

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
              if (i == j) U(i, j) = 1;
              else U(i, j) = 0;
        });
  }

protected:
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t>
      vector_t;

  vector_t a_;
  vector_t b_;
  vector_t cRef_;
  vector_t u_;
};


TEST_F(GPUKokkosVectors, InnerProductGPU)
{
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t>
      vector_t;
  int n = 5;
  vector_t c(Map<>(Communicator(), SlabLayoutV({{0, 0, n, n}})));
  inner(c, a_, b_);
  // host vector type
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace>
      h_vector_t;

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
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t>
      vector_t;
  int n = a_.map().ncols();
  int m = a_.map().nrows();
  vector_t c(Map<>(Communicator(), SlabLayoutV({{0, 0, m, n}})));
  transform(c, 0., 1., a_, u_);
  // host vector type
  typedef KokkosDVector<double **, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace>
      h_vector_t;

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


TEST(EigenValues, EigHermitian)
{
  // Poisson matrix: n =5, ones on diagonal, -2 on first off-diagonals
  typedef std::complex<double> numeric_t;
  typedef KokkosDVector<numeric_t **, SlabLayoutV, Kokkos::LayoutLeft, device_space_t>
      vector_t;

  const std::vector<numeric_t> _Varr = {
      2.88675135e-01,  5.00000000e-01,  5.77350269e-01,  5.00000000e-01, 2.88675135e-01,
      -5.00000000e-01, -5.00000000e-01, 2.26646689e-16,  5.00000000e-01, 5.00000000e-01,
      5.77350269e-01,  -1.82365965e-16, -5.77350269e-01, 1.24197014e-16, 5.77350269e-01,
      5.00000000e-01,  -5.00000000e-01, 1.14137498e-17,  5.00000000e-01, -5.00000000e-01,
      -2.88675135e-01, 5.00000000e-01,  -5.77350269e-01, 5.00000000e-01, -2.88675135e-01};
  const std::vector<double> eigs = {-2.46410162, -1., 1., 3., 4.46410162};
  int n = eigs.size();

  vector_t Vref(Map<>(Communicator(), SlabLayoutV({{0, 0, n, n}})));
  auto Vref_host = Kokkos::create_mirror(Vref.array());

  std::memcpy(Vref_host.data(), _Varr.data(), n * n);
  Kokkos::deep_copy(Vref.array(), Vref_host);

  vector_t A(Map<>(Communicator(), SlabLayoutV({{0, 0, n, n}})));
  typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;
  auto A_array = A.array();
  Kokkos::parallel_for(
      "init", mdrange_policy({0, 0}, {5, 5}), KOKKOS_LAMBDA(int i, int j) {
        A_array(i, i) = 1;
        if (std::abs(i-j) == 1) A_array(i, j) = -2;
      });

  //
  auto A_host = Kokkos::create_mirror(A_array);
  Kokkos::deep_copy(A_host, A_array);

  vector_t V(Map<>(Communicator(), SlabLayoutV({{0, 0, n, n}})));
  Kokkos::View<double *, device_space_t> w("w", n);
  eigh(V, w, A);

  // copy to host
  auto Vh = Kokkos::create_mirror(V.array());
  Kokkos::deep_copy(Vh, V.array());

  std::cout << std::setprecision(4);

  // copy eigenvalues to host
  auto wh = Kokkos::create_mirror(w);
  Kokkos::deep_copy(wh, w);

  std::cout << "eigenvalues" << "\n";
  for (int i = 0; i < wh.extent(0); ++i) {
    // double err = eigs[i] - wh(i);
    EXPECT_NEAR(wh(i), eigs[i], 1e-8);
  }
}
#endif

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
