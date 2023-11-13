#pragma once

// #include <rocblas.h>
// #include <rocsolver.h>

#include "la/cblas.hpp"
#include "rocm.hpp"
#include "rocblas.hpp"
#include "rocsolver.hpp"
#include "la/dvector.hpp"
#include "la/utils.hpp"
#include "la/lapack_cpu.hpp"

#ifdef __NLCGLIB__MAGMA
#include "magma.hpp"
#endif

namespace nlcglib {

#ifdef __NLCGLIB__ROCM
/// Hermitian eigenvalue problem CUDA
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space,
                              Kokkos::Experimental::HIPSpace>::value,
                 void>
eigh(KokkosDVector<T, LAYOUT, KOKKOS...>& U,
     Kokkos::View<double*, Kokkos::Experimental::HIPSpace>& w,
     const KokkosDVector<T, LAYOUT, KOKKOS...>& S)
{
  if (U.map().is_local() && S.map().is_local()) {

    deep_copy(U, S);

    // int n = U.map().nrows();
    // int lda = U.array().stride(1);

    auto w_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
    auto S_host = create_mirror_view_and_copy(Kokkos::HostSpace(), S);

    auto U_host = empty_like()(S_host);

    // eigh(U_host, w_host, S_host);

    {
      int lda = U_host.array().stride(1);
      int n = U_host.map().ncols();
      Kokkos::deep_copy(U_host.array(), S_host.array());
      LAPACKE_zheevd(
          LAPACK_COL_MAJOR,                                           /* matrix layout */
          'V',                                                        /* jobz */
          'U',                                                        /* uplot */
          n,                                                          /* matrix size */
          reinterpret_cast<lapack_complex_double*>(U_host.array().data()), /* Complex double */
          lda,                                                        /* lda */
          w_host.data()                                                    /* eigenvalues */
      );
    }
    // cblas::zheevd<typename T>
    // performance of Hermitian eigensolver in rocm is bad! use magma instead.
    // zheevd_magma(n, U.array().data(), lda, w_host.data());
    Kokkos::deep_copy(w, w_host);
    deep_copy(U, U_host);
    // rocm::heevd(rocblas_evect::rocblas_evect_original, rocblas_fill::rocblas_fill_upper, n, U.array().data(), lda, w.data());
  } else {
    throw std::runtime_error("distributed eigh not implemented");
  }
}

/// stores result in RHS, after the call A will contain the cholesky factorization of a
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space, Kokkos::Experimental::HIPSpace>::value>
cholesky(KokkosDVector<T, LAYOUT, KOKKOS...>& A)
{
  if (A.map().is_local()) {
    // first call potrf
    int n = A.map().nrows();
    int lda = A.array().stride(1);
    auto ptr_A = A.array().data();
    // auto uplo = rocblas_fill::rocblas_fill_upper;
    // int info_potrf;
    // rocm::potrf(uplo, n, ptr_A, lda, info_potrf);

    zpotrf_magma(n, ptr_A, lda);
  } else {
    throw std::runtime_error("distributed cholesky not implemented");
  }
}


/// stores result in RHS, after the call A will contain the cholesky factorization of a
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space, Kokkos::Experimental::HIPSpace>::value>
solve_sym(KokkosDVector<T, LAYOUT, KOKKOS...>& A,
          KokkosDVector<T, LAYOUT, KOKKOS...>& RHS)
{
  auto A_host = create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto RHS_host = create_mirror_view_and_copy(Kokkos::HostSpace(), RHS);

  solve_sym(A_host, RHS_host);
  deep_copy(A, A_host);
  deep_copy(RHS, RHS_host);

  // if (A.map().is_local() && RHS.map().is_local()) {
  //   // first call potrf
  //   int n = A.map().nrows();
  //   int lda = A.array().stride(1);
  //   int ldb = RHS.array().stride(1);
  //   auto ptr_B = RHS.array().data();
  //   auto ptr_A = A.array().data();

  //   // auto uplo = rocblas_fill::rocblas_fill_upper;
  //   // int info_potrf;
  //   // rocm::potrf(uplo, n, ptr_A, lda, info_potrf);
  //   // int nrhs = RHS.array().extent(1);

  //   // rocm::potrs(uplo, n, nrhs, ptr_A, lda, ptr_B, ldb);

  //   zpotrf_magma(n, ptr_A, lda);
  //   int nrhs = RHS.array().extent(1);
  //   zpotrs_magma(n, nrhs, ptr_A, lda, ptr_B, ldb);
  // } else {
  //   throw std::runtime_error("distributed solve_sym not implemented");
  // }
}

/// Inner product c = a^H * b, on GPU
template <class M0, class M1, class M2>
std::enable_if_t<std::is_same<typename M0::storage_t::memory_space, Kokkos::Experimental::HIPSpace>::value, void>
inner(M0& c,
      const M1& a,
      const M2& b,
      const typename M0::numeric_t& alpha = typename M0::numeric_t{1.0},
      const typename M0::numeric_t& beta = typename M0::numeric_t{0.0})
{
  typedef typename M1::storage_t::value_type numeric_t;

  static_assert(std::is_same<typename M0::storage_t::memory_space,
                             typename M1::storage_t::memory_space>::value,
                "c,a not on same memory");
  static_assert(std::is_same<typename M1::storage_t::memory_space,
                             typename M2::storage_t::memory_space>::value,
                "a,b not on same memory");
  // static_assert(std::is_same<LAYOUT1, LAYOUT2>::value, "matrix layout do not match");
  if (c.map().is_local()) {
    if (a.array().stride(0) != 1 || b.array().stride(0) != 1 || c.array().stride(0) != 1) {
      throw std::runtime_error("expecting column major layout");
    }

    int m = a.map().ncols();
    int k = a.map().nrows();
    int n = b.map().ncols();
    numeric_t* A_ptr = a.array().data();
    numeric_t* B_ptr = b.array().data();
    numeric_t* C_ptr = c.array().data();

    int lda = a.array().stride(1);
    int ldb = b.array().stride(1);
    int ldc = c.array().stride(1);

    auto H = rocblas_operation::rocblas_operation_conjugate_transpose;
    auto N = rocblas_operation::rocblas_operation_none;
    rocm::gemm(H, N, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);
    allreduce(c, a.map().comm());
  } else {
    throw std::runtime_error("distributed inner product not implemented.");
  }
}

/// Inner product c = a^H * b, on GPU
template <class M0,
          class M1,
          class M2>
std::enable_if_t<
    std::is_same<typename M0::storage_t::memory_space,
                 Kokkos::Experimental::HIPSpace>::value,
    void>
outer(M0& c,
      const M1& a,
      const M2& b,
      const typename M0::numeric_t& alpha = typename M0::numeric_t{1.0},
      const typename M0::numeric_t& beta = typename M0::numeric_t{0.0})
{
  typedef M0 vector0_t;
  typedef M1 vector1_t;
  typedef M2 vector2_t;
  typedef typename vector1_t::storage_t::value_type numeric_t;

  static_assert(std::is_same<typename vector0_t::storage_t::memory_space,
                             typename vector1_t::storage_t::memory_space>::value,
                "c,a not on same memory");
  static_assert(std::is_same<typename vector1_t::storage_t::memory_space,
                             typename vector2_t::storage_t::memory_space>::value,
                "a,b not on same memory");
  // static_assert(std::is_same<LAYOUT1, LAYOUT2>::value, "matrix layout do not match");
  if (a.map().is_local() && b.map().is_local() && c.map().is_local()) {
    if (a.array().stride(0) != 1 || b.array().stride(0) != 1 || c.array().stride(0) != 1) {
      throw std::runtime_error("expecting column major layout");
    }

    int m = a.map().ncols();
    int k = a.map().nrows();
    int n = b.map().ncols();
    numeric_t* A_ptr = a.array().data();
    numeric_t* B_ptr = b.array().data();
    numeric_t* C_ptr = c.array().data();

    int lda = a.array().stride(1);
    int ldb = b.array().stride(1);
    int ldc = c.array().stride(1);
    auto H = rocblas_operation::rocblas_operation_conjugate_transpose;
    auto N = rocblas_operation::rocblas_operation_none;

    rocm::gemm(N, H, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);

  } else {
    throw std::runtime_error("distributed outer product not implemented.");
  }
}

/// C <- beta * C + alpha * A @ B
template <class M0, class M1, class M2>
std::enable_if_t<std::is_same<typename M0::storage_t::memory_space, Kokkos::Experimental::HIPSpace>::value, void>
transform(M0& C,
          typename M0::numeric_t beta,
          typename M0::numeric_t alpha,
          const M1& A,
          const M2& B)
{
  typedef M0 vector0_t;
  typedef M1 vector1_t;
  typedef M2 vector2_t;
  typedef typename vector1_t::storage_t::value_type numeric_t;

  static_assert(std::is_same<typename vector0_t::storage_t::memory_space,
                             typename vector1_t::storage_t::memory_space>::value,
                "c,a not on same memory");
  static_assert(std::is_same<typename vector1_t::storage_t::memory_space,
                             typename vector2_t::storage_t::memory_space>::value,
                "a,b not on same memory");
  if (B.map().is_local()) {
    /* single rank */
    int m = A.map().nrows();
    int n = B.map().ncols();
    int k = A.map().ncols();
    numeric_t* A_ptr = A.array().data();
    numeric_t* B_ptr = B.array().data();
    numeric_t* C_ptr = C.array().data();

    if (A.array().stride(0) != 1 || B.array().stride(0) != 1 || C.array().stride(0) != 1) {
      throw std::runtime_error("expecting column major layout");
    }
    // assume there are no strides
    int lda = A.array().stride(1);
    int ldb = B.array().stride(1);
    int ldc = C.array().stride(1);

    auto N = rocblas_operation::rocblas_operation_none;
    rocm::gemm(N, N, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);
  } else {
    throw std::runtime_error("distributed transform not implemented.");
  }
}


/// add C <- alpha * A + beta * C
template <class M0, class M1>
std::enable_if_t<std::is_same<typename M0::storage_t::memory_space, Kokkos::Experimental::HIPSpace>::value, void>
add(M0& C,
    const M1& A,
    typename M0::numeric_t alpha,
    typename M0::numeric_t beta = typename M0::numeric_t{1.0})
{
  typedef M0 vector0_t;
  typedef M1 vector1_t;
  typedef typename vector1_t::storage_t::value_type numeric_t;

  static_assert(std::is_same<typename vector0_t::storage_t::memory_space,
                             typename vector1_t::storage_t::memory_space>::value,
                "c,a not on same memory");

  /* single rank */
  int m = A.map().nrows();
  int n = C.map().ncols();
  numeric_t* A_ptr = A.array().data();
  numeric_t* C_ptr = C.array().data();

  if (A.array().stride(0) != 1 || C.array().stride(0) != 1) {
    throw std::runtime_error("expecting column major layout");
  }
  // assume there are no strides
  int lda = A.array().stride(1);
  int ldc = C.array().stride(1);

  // using geam = rocm::geam<numeric_t>;
  auto N = rocblas_operation::rocblas_operation_none;
  // rocm::geam(N, N, m, n, alpha, A_ptr, lda, beta, B_ptr, ldb, C, ldc);
  rocm::geam(N, N, m, n, alpha, A_ptr, lda, beta, C_ptr, ldc, C_ptr, ldc);
}

#endif

}  // nlcglib
