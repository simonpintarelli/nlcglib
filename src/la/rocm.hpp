#pragma once

#include <stdexcept>
#include "la/lapack_rocm.hpp"
#include "rocblas.hpp"
#include "rocsolver.hpp"


namespace nlcglib {
namespace rocm {

// use rocblas_status_to_string(rocblas_status status)

#ifdef __NLCGLIB__ROCM

struct rocsolver_base
{
  static const auto UPPER = rocblas_fill::rocblas_fill_upper;
  static const auto LOWER = rocblas_fill::rocblas_fill_lower;
  static const auto FULL = rocblas_fill::rocblas_fill_full;
  // static const auto VECTOR = rocblas
};

template <class T>
struct potrf
{};

template <>
struct potrf<std::complex<double>> : rocsolver_base
{
  inline static void call (rocblas_fill uplo, int n, std::complex<double>* A, int lda, int& Info);
};


void potrf<std::complex<double>>::call(rocblas_fill uplo, int n, std::complex<double>* A, int lda, int& Info)
{
  throw std::runtime_error("not implemented\n");

  auto handle = rocblasHandle::get();
  // rocsolver_zpotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, rocblas_double_complex *A, const rocblas_int lda, rocblas_int *info);

  rocblas_int* dev_info;
  rocsolver_zpotrf(handle, uplo, n, A, lda, dev_info);

  // rocblas_status::rocblas_status_success
}



#endif

}  // namespace rocm

}  // namespace nlcglib
