#pragma once

#ifdef __USE_MKL
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

#include <type_traits>
#include "la/dvector.hpp"
#include "la/cblas.hpp"

#ifdef __USE_MKL
#define CPX MKL_Complex16
#else
#define CPX _Complex double
#endif


namespace nlcglib {

/// Hermitian eigenvalue problem on CPU
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<
  std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space,
               Kokkos::HostSpace>::value, void>
eigh(KokkosDVector<T, LAYOUT, KOKKOS...>& U,
     Kokkos::View<double*, Kokkos::HostSpace>& w,
     const KokkosDVector<T, LAYOUT, KOKKOS...>& S)
{
  static_assert(std::is_same<decltype(S.array().layout()), Kokkos::LayoutLeft>::value,
                "must be col-major layout");
  int lda = S.array().stride(1);

  // check number of MPI ranks in communicator
  if (S.map().is_local()) {
    int n = S.map().ncols();
    Kokkos::deep_copy(U.array(), S.array());
    LAPACKE_zheevd(LAPACK_COL_MAJOR,                                     /* matrix layout */
                   'V',                                                  /* jobz */
                   'U',                                                  /* uplot */
                   n,                                                    /* matrix size */
                   reinterpret_cast<CPX*>(U.array().data()), /* Complex double */
                   lda,                                                    /* lda */
                   w.data()                                              /* eigenvalues */
    lapack_int info = LAPACKE_zheevd(LAPACK_COL_MAJOR,                                     /* matrix layout */
                                     'V',                                                  /* jobz */
                                     'U',                                                  /* uplot */
                                     n,                                                    /* matrix size */
                                     reinterpret_cast<CPX*>(U.array().data()), /* Complex double */
                                     lda,                                                    /* lda */
                                     w.data()                                              /* eigenvalues */
    );
    if (info != 0)
      throw std::runtime_error("cblas zheevd failed");
  } else {
    throw std::runtime_error("not yet implemented");
  }
}


/// stores result in RHS, after the call A will contain the cholesky factorization of a
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space, Kokkos::HostSpace>::value>
solve_sym(KokkosDVector<T, LAYOUT, KOKKOS...>& A,
          KokkosDVector<T, LAYOUT, KOKKOS...>& RHS)
{
  if (A.map().is_local() && RHS.map().is_local()) {
    typedef KokkosDVector<T**, LAYOUT, KOKKOS...> vector_t;
    typedef typename vector_t::storage_t::value_type numeric_t;

    typedef cblas::potrf<numeric_t> potrf_t;
    if (A.array().stride(0) != 1 || RHS.array().stride(0) != 1) {
      throw std::runtime_error("expecting column major layout");
    }

    int n = A.map().nrows();
    int lda = A.array().stride(1);
    int ldb = RHS.array().stride(1);
    auto ptr_B = RHS.array().data();
    auto ptr_A = A.array().data();

    char uplo = 'U';
    auto order = CBLAS_ORDER::CblasColMajor;
    potrf_t::call(order, uplo, n, ptr_A, lda);
    int nrhs = RHS.array().extent(1);

    typedef cblas::potrs<numeric_t> potrs_t;
    potrs_t::call(order, uplo, n, nrhs, ptr_A, lda, ptr_B, ldb);
  } else {
    throw std::runtime_error("not implemented");
  }
}


///  Inner product: c = a^H * b, on CPU
template <class T0,
          class LAYOUT0,
          class... KOKKOS0,
          class T1,
          class LAYOUT1,
          class... KOKKOS1,
          class T2,
          class LAYOUT2,
          class... KOKKOS2>
std::enable_if_t<
    std::is_same<typename KokkosDVector<T0, LAYOUT0, KOKKOS0...>::storage_t::memory_space,
                 Kokkos::HostSpace>::value,
    void>
inner(KokkosDVector<T0**, LAYOUT0, KOKKOS0...>& C,
      const KokkosDVector<T1**, LAYOUT1, KOKKOS1...>& A,
      const KokkosDVector<T2**, LAYOUT2, KOKKOS2...>& B,
      const T0& alpha = T0{1.0},
      const T0& beta = T0{0.0})
{
  typedef KokkosDVector<T0**, LAYOUT0, KOKKOS0...> vector0_t;
  typedef KokkosDVector<T1**, LAYOUT1, KOKKOS1...> vector1_t;
  typedef KokkosDVector<T2**, LAYOUT2, KOKKOS2...> vector2_t;
  typedef typename vector1_t::storage_t::value_type numeric_t;

  static_assert(std::is_same<typename vector0_t::storage_t::memory_space,
                             typename vector1_t::storage_t::memory_space>::value,
                "c,a not on same memory");
  static_assert(std::is_same<typename vector1_t::storage_t::memory_space,
                             typename vector2_t::storage_t::memory_space>::value,
                "a,b not on same memory");
  static_assert(std::is_same<LAYOUT1, LAYOUT2>::value, "matrix layout do not match");

  // single rank
  if (A.map().is_local() && B.map().is_local() && C.map().is_local()) {
    int m = A.map().ncols();
    int k = A.map().nrows();
    int n = B.map().ncols();
    numeric_t* A_ptr = A.array().data();
    numeric_t* B_ptr = B.array().data();
    numeric_t* C_ptr = C.array().data();

    if (A.array().stride(0) != 1 || B.array().stride(0) != 1 || C.array().stride(0) != 1) {
      throw std::runtime_error("expecting column major layout");
    }
    int lda = A.array().stride(1);
    int ldb = B.array().stride(1);
    int ldc = C.array().stride(1);

    // single rank inner product
    cblas::gemm<numeric_t>::call(CblasColMajor,
                                 cblas::gemm<numeric_t>::H,
                                 CblasNoTrans,
                                 m,
                                 n,
                                 k,
                                 alpha,
                                 A_ptr,
                                 lda,
                                 B_ptr,
                                 ldb,
                                 beta,
                                 C_ptr,
                                 ldc);
  } else {
    throw std::runtime_error("not implemented.");
  }
}

///  Inner product: c = a^H * b, on CPU
template <class T0,
          class LAYOUT0,
          class... KOKKOS0,
          class T1,
          class LAYOUT1,
          class... KOKKOS1,
          class T2,
          class LAYOUT2,
          class... KOKKOS2>
std::enable_if_t<
    std::is_same<typename KokkosDVector<T0, LAYOUT0, KOKKOS0...>::storage_t::memory_space,
                  Kokkos::HostSpace>::value,
    void> outer(KokkosDVector<T0**, LAYOUT0, KOKKOS0...>& C,
                const KokkosDVector<T1**, LAYOUT1, KOKKOS1...>& A,
                const KokkosDVector<T2**, LAYOUT2, KOKKOS2...>& B,
                const T0& alpha = T0{1.0},
                const T0& beta = T0{0.0})
{
  typedef KokkosDVector<T0**, LAYOUT0, KOKKOS0...> vector0_t;
  typedef KokkosDVector<T1**, LAYOUT1, KOKKOS1...> vector1_t;
  typedef KokkosDVector<T2**, LAYOUT2, KOKKOS2...> vector2_t;
  typedef typename vector1_t::storage_t::value_type numeric_t;

  static_assert(std::is_same<typename vector0_t::storage_t::memory_space,
                             typename vector1_t::storage_t::memory_space>::value,
                "c,a not on same memory");
  static_assert(std::is_same<typename vector1_t::storage_t::memory_space,
                             typename vector2_t::storage_t::memory_space>::value,
                "a,b not on same memory");
  static_assert(std::is_same<LAYOUT1, LAYOUT2>::value, "matrix layout do not match");

  // single rank
  if (A.map().is_local() && B.map().is_local() && C.map().is_local()) {
    int m = A.map().ncols();
    int k = A.map().nrows();
    int n = B.map().ncols();
    numeric_t* A_ptr = A.array().data();
    numeric_t* B_ptr = B.array().data();
    numeric_t* C_ptr = C.array().data();

    if (A.array().stride(0) != 1 || B.array().stride(0) != 1 || C.array().stride(0) != 1) {
      throw std::runtime_error("expecting column major layout");
    }
    int lda = A.array().stride(1);
    int ldb = B.array().stride(1);
    int ldc = C.array().stride(1);

    // single rank inner product
    cblas::gemm<numeric_t>::call(CblasColMajor,
                                 CblasNoTrans,
                                 cblas::gemm<numeric_t>::H,
                                 m,
                                 n,
                                 k,
                                 alpha,
                                 A_ptr,
                                 lda,
                                 B_ptr,
                                 ldb,
                                 beta,
                                 C_ptr,
                                 ldc);
  } else {
    throw std::runtime_error("not implemented.");
  }
}

/// C <- beta * C + alpha * A @ B
template <class T0, class LAYOUT0, class... KOKKOS0,
          class T1, class LAYOUT1, class... KOKKOS1,
          class T2, class LAYOUT2, class... KOKKOS2>
std::enable_if_t<
    std::is_same<typename KokkosDVector<T0, LAYOUT0, KOKKOS0...>::storage_t::memory_space,
                 Kokkos::HostSpace>::value, void>
transform(KokkosDVector<T0**, LAYOUT0, KOKKOS0...>& C,
          T0 beta,
          T0 alpha,
          const KokkosDVector<T1**, LAYOUT1, KOKKOS1...>& A,
          const KokkosDVector<T2**, LAYOUT2, KOKKOS2...>& B)
{
  typedef KokkosDVector<T0**, LAYOUT0, KOKKOS0...> vector0_t;
  typedef KokkosDVector<T1**, LAYOUT1, KOKKOS1...> vector1_t;
  typedef KokkosDVector<T2**, LAYOUT2, KOKKOS2...> vector2_t;
  typedef typename vector1_t::storage_t::value_type numeric_t;

  static_assert(std::is_same<typename vector0_t::storage_t::memory_space,
                             typename vector1_t::storage_t::memory_space>::value,
                "c,a not on same memory");
  static_assert(std::is_same<typename vector1_t::storage_t::memory_space,
                             typename vector2_t::storage_t::memory_space>::value,
                "a,b not on same memory");
  static_assert(std::is_same<LAYOUT1, LAYOUT2>::value, "matrix layout do not match");

  if (A.map().is_local() && B.map().is_local() && C.map().is_local()) {
    /* single rank */
    int m = A.map().nrows();
    int n = B.map().ncols();
    int k = A.map().ncols();
    numeric_t* A_ptr = A.array().data();
    numeric_t* B_ptr = B.array().data();
    numeric_t* C_ptr = C.array().data();

    if(A.array().stride(0) != 1 || B.array().stride(0) != 1 || C.array().stride(0) != 1) {
      throw std::runtime_error("expecting column major layout");
    }
    int lda = A.array().stride(1);
    int ldb = B.array().stride(1);
    int ldc = C.array().stride(1);

    // single rank inner product
    cblas::gemm<numeric_t>::call(CblasColMajor,
                                 cblas::gemm<numeric_t>::N,
                                 cblas::gemm<numeric_t>::N,
                                 m,
                                 n,
                                 k,
                                 alpha,
                                 A_ptr,
                                 lda,
                                 B_ptr,
                                 ldb,
                                 beta,
                                 C_ptr,
                                 ldc);
  } else {
    throw std::runtime_error("not implemented.");
  }
}


/// add
template <class M0, class M1>
std::enable_if_t<std::is_same<typename M0::storage_t::memory_space, Kokkos::HostSpace>::value, void>
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

  if (A.map().is_local() && C.map().is_local()) {
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

    using geam = cblas::geam<numeric_t>;
    geam::call(
        CblasColMajor, geam::N, geam::N, m, n, alpha, A_ptr, lda, beta, C_ptr, ldc, C_ptr, ldc);
  } else {
    throw std::runtime_error("not implemented.");
  }
}



}  // namespace nlcglib
