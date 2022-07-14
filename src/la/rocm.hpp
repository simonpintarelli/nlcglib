#pragma once

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <type_traits>
#include "la/lapack_rocm.hpp"
#include "rocblas.hpp"
#include "rocsolver.hpp"

namespace nlcglib {
namespace rocm {

// use rocblas_status_to_string(rocblas_status status)

#ifdef __NLCGLIB__ROCM

template <class T>
void
potrf(rocblas_fill uplo, int n, T* A, int lda, int& Info)
{
  static_assert(std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, Kokkos::complex<double>>::value);
  auto handle = rocblasHandle::get();

  rocblas_double_complex* A_ptr = reinterpret_cast<rocblas_double_complex*>(A);

  rocblas_int* dev_info;
  auto status = rocsolver_zpotrf(handle, uplo, n, A_ptr, lda, dev_info);

  if (status != rocblas_status::rocblas_status_success) {
    // exit
    throw std::runtime_error("rocsolver_zpotrf failed");
  }
}


template <class T>
void
potrs(rocblas_fill uplo, int n, int nrhs, T* A, int lda, T* B, int ldb)
{
  static_assert(std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, Kokkos::complex<double>>::value);

  auto handle = rocblasHandle::get();

  rocblas_double_complex* A_ptr = reinterpret_cast<rocblas_double_complex*>(A);
  rocblas_double_complex* B_ptr = reinterpret_cast<rocblas_double_complex*>(B);

  auto status = rocsolver_zpotrs(handle, uplo, n, nrhs, A_ptr, lda, B_ptr, ldb);

  if (status != rocblas_status::rocblas_status_success) {
    // exit
    throw std::runtime_error("rocsolver_zpotrs failed");
  }
}


template <class T>
void
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
  auto res = hipMalloc(&E, n);

  auto status = rocsolver_zheevd(handle, mode, uplo, n, A_ptr, lda, w, E, dev_info);

  hipDeviceSynchronize();

  int info;
  hipMemcpyDtoH(&info, dev_info, sizeof(int));

  if (status != rocblas_status::rocblas_status_success) {
    // exit
    throw std::runtime_error("rocsolver_zpotrs failed, info=" + std::to_string(info));
  }

  hipFree(E);
}

void
gemm(rocblas_operation transa,
     rocblas_operation transb,
     int m,
     int n,
     int k,
     Kokkos::complex<double> alpha,
     const Kokkos::complex<double>* A,
     int lda,
     Kokkos::complex<double> beta,
     const Kokkos::complex<double>* B,
     int ldb,
     Kokkos::complex<double>* C,
     int ldc)
{
  auto handle = rocblasHandle::get();
  auto info = rocblas_zgemm(handle,
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
                            ldc);

  if (info != rocblas_status::rocblas_status_success) {
    throw std::runtime_error("rocblas_zgemm failed\n");
  }
}

void geam()
{
  auto handle = rocblasHandle::get();

  // somehow this is missing..
  // rocblas_zgeam
}

#endif

}  // namespace rocm

}  // namespace nlcglib
