#pragma once

// #include <rocblas.h>
// #include <rocsolver.h>

#include "rocm.hpp"
#include "rocblas.hpp"
#include "rocsolver.hpp"

namespace nlcglib {
namespace rocm {

#ifdef __NLCGLIB__ROCM


void my_rocblas_potrfwrapper()
{
  // rocsolver_zpotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, rocblas_double_complex *A, const rocblas_int lda, rocblas_int *info);
}

#endif

}  // rocm
}  // nlcglib
