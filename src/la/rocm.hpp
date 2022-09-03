#pragma once

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <type_traits>
#include "la/lapack_rocm.hpp"
#include "rocblas.hpp"
#include "rocsolver.hpp"
#include <cstdlib>
#include <cstdio>
#include "backtrace.hpp"


#define CALL_ROCBLAS(func__, args__)                           \
  {                                                         \
    rocblas_status status = func__ args__;                  \
    if (status != rocblas_status::rocblas_status_success) { \
      char nm[1024];                                        \
      gethostname(nm, 1024);                                \
      printf("hostname: %s\n", nm);                         \
      printf("Error in %s at line %i of file %s: %s\n",     \
             #func__,                                       \
             __LINE__,                                      \
             __FILE__,                                      \
             rocblas_status_to_string(status));             \
      stack_backtrace();                                    \
    }                                                       \
  }

namespace nlcglib {
namespace rocm {



// use rocblas_status_to_string(rocblas_status status)

#ifdef __NLCGLIB__ROCM

template <class T>
inline void
potrf(rocblas_fill uplo, int n, T* A, int lda, int& Info)
{
  static_assert(std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, Kokkos::complex<double>>::value);
  auto handle = rocblasHandle::get();

  rocblas_double_complex* A_ptr = reinterpret_cast<rocblas_double_complex*>(A);

  rocblas_int* dev_info;
  CALL_ROCBLAS(rocsolver_zpotrf, (handle, uplo, n, A_ptr, lda, dev_info))
}


template <class T>
inline void
potrs(rocblas_fill uplo, int n, int nrhs, T* A, int lda, T* B, int ldb)
{
  static_assert(std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, Kokkos::complex<double>>::value);

  auto handle = rocblasHandle::get();

  rocblas_double_complex* A_ptr = reinterpret_cast<rocblas_double_complex*>(A);
  rocblas_double_complex* B_ptr = reinterpret_cast<rocblas_double_complex*>(B);

  CALL_ROCBLAS(rocsolver_zpotrs, (handle, uplo, n, nrhs, A_ptr, lda, B_ptr, ldb))

}

template <class T>
inline void
heevd(rocblas_evect mode, rocblas_fill uplo, int n, T* A, int lda, double* w)
{
  static_assert(std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, Kokkos::complex<double>>::value);

  auto handle = rocblasHandle::get();
  // rocsolver_zh
  if (mode != rocblas_evect::rocblas_evect_original) {
    throw std::runtime_error("unsupported mode in rocm::heevd");
  }

  rocblas_double_complex* A_ptr = reinterpret_cast<rocblas_double_complex*>(A);

  int* dev_info;

  double* E;

  // TODO: make a wrapper for hipMalloc
  hipError_t res = hipMalloc(&E, n);

  CALL_ROCBLAS(rocsolver_zheevd, (handle, mode, uplo, n, A_ptr, lda, w, E, dev_info))

  int info;
  hipMemcpyDtoH(&info, dev_info, sizeof(int));
  hipFree(E);
}

inline void
gemm(rocblas_operation transa,
     rocblas_operation transb,
     int m,
     int n,
     int k,
     Kokkos::complex<double> alpha,
     const Kokkos::complex<double>* A,
     int lda,
     const Kokkos::complex<double>* B,
     int ldb,
     Kokkos::complex<double> beta,
     Kokkos::complex<double>* C,
     int ldc)
{
  auto handle = rocblasHandle::get();

  CALL_ROCBLAS(rocblas_zgemm,
               (handle,
                transa,
                transb,
                m,
                n,
                k,
                reinterpret_cast<const rocblas_double_complex*>(&alpha),
                reinterpret_cast<const rocblas_double_complex*>(A),
                lda,
                reinterpret_cast<const rocblas_double_complex*>(B),
                ldb,
                reinterpret_cast<const rocblas_double_complex*>(&beta),
                reinterpret_cast<rocblas_double_complex*>(C),
                ldc))
}

template<class T>
inline void geam(rocblas_operation transa,
                 rocblas_operation transb,
                 int m,
                 int n,
                 T alpha,
                 T* A,
                 int lda,
                 T beta,
                 T* C,
                 int ldc)
{
  auto handle = rocblasHandle::get();
  throw std::runtime_error("NOT IMPLEMENTED!");

  // have to use axpy here ...
  // somehow this is missing..
  // rocblas_zgeam
}

#endif

}  // namespace rocm

}  // namespace nlcglib
