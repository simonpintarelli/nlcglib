#pragma once
#include <type_traits>
#include "la/cuda.hpp"
#include "la/dvector.hpp"

namespace nlcglib {

/// Hermitian eigenvalue problem CUDA
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space,
                              Kokkos::CudaSpace>::value,
                 void>
eigh(KokkosDVector<T, LAYOUT, KOKKOS...>& U,
     Kokkos::View<double*, Kokkos::CudaSpace>& w,
     const KokkosDVector<T, LAYOUT, KOKKOS...>& S)
{
  if (U.map().is_local() && S.map().is_local()) {
    typedef KokkosDVector<T**, LAYOUT, KOKKOS...> vector_t;
    typedef typename vector_t::storage_t::value_type numeric_t;

    deep_copy(U, S);

    // assert status_create == CUSOLVER_STATUS_SUCCESS
    int n = U.map().nrows();
    int lda = U.array().stride(1);
    typedef cuda::zheevd<numeric_t> zheevd_t;
    int Info;
    zheevd_t::call(zheevd_t::VECTOR, zheevd_t::UPPER, n, U.array().data(), lda, w.data(), Info);
  } else {
    throw std::runtime_error("not implemented");
  }
}

/// stores result in RHS, after the call A will contain the cholesky factorization of a
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space, Kokkos::CudaSpace>::value>
cholesky(KokkosDVector<T, LAYOUT, KOKKOS...>& A)
{
  if (A.map().is_local()) {
    typedef KokkosDVector<T**, LAYOUT, KOKKOS...> vector_t;
    typedef typename vector_t::storage_t::value_type numeric_t;
    // first call potrf
    typedef cuda::potrf<numeric_t> potrf_t;
    int n = A.map().nrows();
    int lda = A.array().stride(1);
    auto ptr_A = A.array().data();
    auto uplo = potrf_t::UPPER;
    int info_potrf;
    potrf_t::call(uplo, n, ptr_A, lda, info_potrf);
  } else {
    throw std::runtime_error("not implemented");
  }
}


/// stores result in RHS, after the call A will contain the cholesky factorization of a
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<std::is_same<typename KokkosDVector<T, LAYOUT, KOKKOS...>::storage_t::memory_space, Kokkos::CudaSpace>::value>
solve_sym(KokkosDVector<T, LAYOUT, KOKKOS...>& A,
          KokkosDVector<T, LAYOUT, KOKKOS...>& RHS)
{
  if (A.map().is_local() && RHS.map().is_local()) {
    typedef KokkosDVector<T**, LAYOUT, KOKKOS...> vector_t;
    typedef typename vector_t::storage_t::value_type numeric_t;
    // first call potrf
    typedef cuda::potrf<numeric_t> potrf_t;
    int n = A.map().nrows();
    int lda = A.array().stride(1);
    int ldb = RHS.array().stride(1);
    auto ptr_B = RHS.array().data();
    auto ptr_A = A.array().data();

    auto uplo = potrf_t::UPPER;
    int info_potrf;
    potrf_t::call(uplo, n, ptr_A, lda, info_potrf);
    int nrhs = RHS.array().extent(1);

    typedef cuda::potrs<numeric_t> potrs_t;
    potrs_t::call(uplo, n, nrhs, ptr_A, lda, ptr_B, ldb);
  } else {
    throw std::runtime_error("not implemented");
  }
}

/// Inner product c = a^H * b, on GPU
template <class M0, class M1, class M2>
std::enable_if_t<std::is_same<typename M0::storage_t::memory_space, Kokkos::CudaSpace>::value, void>
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

    using gemm = cuda::gemm<numeric_t>;
    gemm::call(gemm::H, gemm::N, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);

  } else {
    throw std::runtime_error("not implemented.");
  }
}

/// Inner product c = a^H * b, on GPU
template <class M0,
          class M1,
          class M2>
std::enable_if_t<
    std::is_same<typename M0::storage_t::memory_space,
                 Kokkos::CudaSpace>::value,
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

    using gemm = cuda::gemm<numeric_t>;
    gemm::call(gemm::N, gemm::H, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);

  } else {
    throw std::runtime_error("not implemented.");
  }
}

/// C <- beta * C + alpha * A @ B
template <class M0, class M1, class M2>
std::enable_if_t<std::is_same<typename M0::storage_t::memory_space, Kokkos::CudaSpace>::value, void>
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
  // static_assert(std::is_same<LAYOUT1, LAYOUT2>::value, "matrix layout do not match");

  if (A.map().is_local() && B.map().is_local() && C.map().is_local()) {
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

    using gemm = cuda::gemm<numeric_t>;
    gemm::call(gemm::N, gemm::N, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc);
  } else {
    throw std::runtime_error("not implemented.");
  }
}


/// add
template <class M0, class M1>
std::enable_if_t<std::is_same<typename M0::storage_t::memory_space, Kokkos::CudaSpace>::value, void>
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

    using geam = cuda::geam<numeric_t>;
    geam::call(geam::N, geam::N, m, n, alpha, A_ptr, lda, beta, C_ptr, ldc, C_ptr, ldc);
  } else {
    throw std::runtime_error("not implemented.");
  }
}

}  // namespace nlcglib
