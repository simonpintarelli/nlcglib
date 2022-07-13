#pragma once

#include <cuda_runtime_api.h>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "backtrace.hpp"
#include "cublas.hpp"
#include "cusolver.hpp"

#define CALL_CUDA(func__, args__)                       \
  {                                                     \
    cudaError_t error = func__ args__;                  \
    if (error != cudaSuccess) {                         \
      char nm[1024];                                    \
      gethostname(nm, 1024);                            \
      printf("hostname: %s\n", nm);                     \
      printf("Error in %s at line %i of file %s: %s\n", \
             #func__,                                   \
             __LINE__,                                  \
             __FILE__,                                  \
             cudaGetErrorString(error));                \
      stack_backtrace();                                \
    }                                                   \
  }

#define CALL_CUSOLVER(func__, args__)                                                  \
  {                                                                                    \
    cusolverStatus_t status;                                                           \
    if ((status = func__ args__) != CUSOLVER_STATUS_SUCCESS) {                         \
      cusolver::error_message(status);                                                 \
      char nm[1024];                                                                   \
      gethostname(nm, 1024);                                                           \
      std::printf("hostname: %s\n", nm);                                               \
      std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
      stack_backtrace();                                                               \
    }                                                                                  \
  }


namespace nlcglib {
namespace cuda {
#ifdef __NLCGLIB__CUDA
struct cusolver_base
{
  static const cublasFillMode_t LOWER{CUBLAS_FILL_MODE_LOWER};
  static const cublasFillMode_t UPPER{CUBLAS_FILL_MODE_UPPER};
  static const cublasFillMode_t FULL{CUBLAS_FILL_MODE_FULL};
  static const cusolverEigMode_t VECTOR{CUSOLVER_EIG_MODE_VECTOR};
  static const cusolverEigMode_t NOVECTOR{CUSOLVER_EIG_MODE_NOVECTOR};
};

template <typename T>
struct potrf
{
};

template <>
struct potrf<std::complex<double>> : cusolver_base
{
  inline static cusolverStatus_t call(
      cublasFillMode_t uplo, int n, std::complex<double>* A, int lda, int& Info);
};

cusolverStatus_t
potrf<std::complex<double>>::call(
    cublasFillMode_t uplo, int n, std::complex<double>* A, int lda, int& Info)
{
  using numeric_t = std::complex<double>;
  cuDoubleComplex* cA = reinterpret_cast<cuDoubleComplex*>(A);
  int lwork = 0;
  cusolverDnHandle_t cusolver_handle = cusolver::cusolverDnHandle::get();
  cusolverStatus_t ret_buffer_size =
      cusolverDnZpotrf_bufferSize(cusolver_handle, uplo, n, cA, lda, &lwork);
  if (ret_buffer_size != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << ret_buffer_size << "\n";
    exit(1);
  }

  cuDoubleComplex* work_ptr;
  CALL_CUDA(cudaMalloc, (&work_ptr, lwork * sizeof(numeric_t)));
  int* dev_Info;
  CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
  cusolverStatus_t ret_cusolver =
      cusolverDnZpotrf(cusolver_handle, uplo, n, cA, lda, work_ptr, lwork, dev_Info);
  CALL_CUDA(cudaDeviceSynchronize, ());
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
  CALL_CUDA(cudaFree, (work_ptr));
  if (ret_cusolver != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << ret_cusolver << "\n"
              << "info: " << Info << "\n";
    exit(1);
  }
  return ret_cusolver;
}

template <>
struct potrf<Kokkos::complex<double>> : cusolver_base
{
  inline static cusolverStatus_t call(
      cublasFillMode_t uplo, int n, Kokkos::complex<double>* A, int lda, int& Info);
};

cusolverStatus_t
potrf<Kokkos::complex<double>>::call(
    cublasFillMode_t uplo, int n, Kokkos::complex<double>* A, int lda, int& Info)
{
  using numeric_t = Kokkos::complex<double>;
  cuDoubleComplex* cA = reinterpret_cast<cuDoubleComplex*>(A);
  int lwork = 0;
  cusolverDnHandle_t cusolver_handle = cusolver::cusolverDnHandle::get();
  cusolverStatus_t ret_buffer_size =
      cusolverDnZpotrf_bufferSize(cusolver_handle, uplo, n, cA, lda, &lwork);
  if (ret_buffer_size != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << ret_buffer_size << "\n";
    exit(1);
  }

  cuDoubleComplex* work_ptr;
  CALL_CUDA(cudaMalloc, (&work_ptr, lwork * sizeof(numeric_t)));
  int* dev_Info;
  CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
  cusolverStatus_t ret_cusolver =
      cusolverDnZpotrf(cusolver_handle, uplo, n, cA, lda, work_ptr, lwork, dev_Info);
  CALL_CUDA(cudaDeviceSynchronize, ());
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
  CALL_CUDA(cudaFree, (work_ptr));
  if (ret_cusolver != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << ret_cusolver << "\n"
              << "info: " << Info << "\n";
    exit(1);
  }
  return ret_cusolver;
}


template <typename T>
struct potrs
{
};

template <>
struct potrs<std::complex<double>> : cusolver_base
{
  inline static cusolverStatus_t call(cublasFillMode_t uplo,
                                      int n,
                                      int nrhs,
                                      const std::complex<double>* A,
                                      int lda,
                                      std::complex<double>* B,
                                      int ldb);
};

cusolverStatus_t
potrs<std::complex<double>>::call(cublasFillMode_t uplo,
                                  int n,
                                  int nrhs,
                                  const std::complex<double>* A,
                                  int lda,
                                  std::complex<double>* B,
                                  int ldb)
{
  const cuDoubleComplex* cA = reinterpret_cast<const cuDoubleComplex*>(A);
  cuDoubleComplex* cB = reinterpret_cast<cuDoubleComplex*>(B);
  cusolverDnHandle_t cusolver_handle = cusolver::cusolverDnHandle::get();

  int* dev_Info;
  int Info;
  CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
  cusolverStatus_t stat =
      cusolverDnZpotrs(cusolver_handle, uplo, n, nrhs, cA, lda, cB, ldb, dev_Info);
  CALL_CUDA(cudaDeviceSynchronize, ());
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));

  if (stat != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << stat << "\n"
              << "info: " << Info << "\n";
    exit(1);
  }

  return stat;
}

template <>
struct potrs<Kokkos::complex<double>> : cusolver_base
{
  inline static cusolverStatus_t call(cublasFillMode_t uplo,
                                      int n,
                                      int nrhs,
                                      const Kokkos::complex<double>* A,
                                      int lda,
                                      Kokkos::complex<double>* B,
                                      int ldb);
};

cusolverStatus_t
potrs<Kokkos::complex<double>>::call(cublasFillMode_t uplo,
                                     int n,
                                     int nrhs,
                                     const Kokkos::complex<double>* A,
                                     int lda,
                                     Kokkos::complex<double>* B,
                                     int ldb)
{
  const cuDoubleComplex* cA = reinterpret_cast<const cuDoubleComplex*>(A);
  cuDoubleComplex* cB = reinterpret_cast<cuDoubleComplex*>(B);
  cusolverDnHandle_t cusolver_handle = cusolver::cusolverDnHandle::get();

  int* dev_Info;
  int Info;
  CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
  cusolverStatus_t stat =
      cusolverDnZpotrs(cusolver_handle, uplo, n, nrhs, cA, lda, cB, ldb, dev_Info);
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
  if (stat != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << stat << "\n"
              << "info: " << Info << "\n";
    exit(1);
  }

  return stat;
}

template <typename T>
struct zheevd : cusolver_base
{
};

template <>
struct zheevd<std::complex<double>> : cusolver_base
{
  inline static cusolverStatus_t call(cusolverEigMode_t jobz,
                                      cublasFillMode_t uplo,
                                      int n,
                                      std::complex<double>* A,
                                      int lda,
                                      double* w,
                                      int& Info);
};

cusolverStatus_t
zheevd<std::complex<double>>::call(cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo,
                                   int n,
                                   std::complex<double>* A,
                                   int lda,
                                   double* w,
                                   int& Info)
{
  using numeric_t = std::complex<double>;
  cuDoubleComplex* cA = reinterpret_cast<cuDoubleComplex*>(A);
  int lwork = 0;
  cusolverDnHandle_t cusolver_handle = cusolver::cusolverDnHandle::get();
  cusolverStatus_t ret_buffer_size =
      cusolverDnZheevd_bufferSize(cusolver_handle, jobz, uplo, n, cA, lda, w, &lwork);
  if (ret_buffer_size != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << ret_buffer_size << "\n";
    exit(1);
  }

  cuDoubleComplex* work_ptr;
  CALL_CUDA(cudaMalloc, (&work_ptr, lwork * sizeof(numeric_t)));
  int* dev_Info;
  CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
  cusolverStatus_t ret_cusolver =
      cusolverDnZheevd(cusolver_handle, jobz, uplo, n, cA, lda, w, work_ptr, lwork, dev_Info);
  CALL_CUDA(cudaDeviceSynchronize, ());
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
  CALL_CUDA(cudaFree, (work_ptr));
  if (ret_cusolver != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << ret_cusolver << "\n"
              << "info: " << Info << "\n";

    exit(1);
  }
  return ret_cusolver;
}

template <>
struct zheevd<Kokkos::complex<double>> : cusolver_base
{
  inline static cusolverStatus_t call(cusolverEigMode_t jobz,
                                      cublasFillMode_t uplo,
                                      int n,
                                      Kokkos::complex<double>* A,
                                      int lda,
                                      double* w,
                                      int& Info);
};

cusolverStatus_t
zheevd<Kokkos::complex<double>>::call(cusolverEigMode_t jobz,
                                      cublasFillMode_t uplo,
                                      int n,
                                      Kokkos::complex<double>* A,
                                      int lda,
                                      double* w,
                                      int& Info)
{
  using numeric_t = Kokkos::complex<double>;
  cuDoubleComplex* cA = reinterpret_cast<cuDoubleComplex*>(A);
  int lwork = 0;
  cusolverDnHandle_t cusolver_handle = cusolver::cusolverDnHandle::get();
  cusolverStatus_t ret_buffer_size =
      cusolverDnZheevd_bufferSize(cusolver_handle, jobz, uplo, n, cA, lda, w, &lwork);
  if (ret_buffer_size != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Something went wrong\n"
              << "return value: " << ret_buffer_size << "\n";
    exit(1);
  }

  cuDoubleComplex* work_ptr;
  CALL_CUDA(cudaMalloc, (&work_ptr, lwork * sizeof(numeric_t)));
  int* dev_Info;
  CALL_CUDA(cudaMalloc, ((void**)&dev_Info, sizeof(int)));
  cusolverStatus_t ret_cusolver =
      cusolverDnZheevd(cusolver_handle, jobz, uplo, n, cA, lda, w, work_ptr, lwork, dev_Info);
  CALL_CUDA(cudaDeviceSynchronize, ());
  CALL_CUDA(cudaMemcpy, (&Info, dev_Info, sizeof(int), cudaMemcpyDeviceToHost));
  CALL_CUDA(cudaFree, (work_ptr));
  if (ret_cusolver != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Error in DnZheevd\n"
              << "return value: " << ret_cusolver << "\n"
              << "info: " << Info << "\n";

    exit(1);
  }
  return ret_cusolver;
}

template <class T>
struct gemm
{
};

template <>
struct gemm<std::complex<double>>
{
  static const cublasOperation_t H = cublasOperation_t::CUBLAS_OP_HERMITAN;
  static const cublasOperation_t N = cublasOperation_t::CUBLAS_OP_N;

  inline static void call(cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          std::complex<double> alpha,
                          const std::complex<double>* A,
                          int lda,
                          const std::complex<double>* B,
                          int ldb,
                          std::complex<double> beta,
                          std::complex<double>* C,
                          int ldc)
  {
    cublasZgemm_v2(cublas::cublasHandle::get(),
                   transa,
                   transb,
                   m,
                   n,
                   k,
                   reinterpret_cast<const cuDoubleComplex*>(&alpha),
                   reinterpret_cast<const cuDoubleComplex*>(A),
                   lda,
                   reinterpret_cast<const cuDoubleComplex*>(B),
                   ldb,
                   reinterpret_cast<const cuDoubleComplex*>(&beta),
                   reinterpret_cast<cuDoubleComplex*>(C),
                   ldc);
  }
};

template <>
struct gemm<Kokkos::complex<double>>
{
  static const cublasOperation_t H = cublasOperation_t::CUBLAS_OP_HERMITAN;
  static const cublasOperation_t N = cublasOperation_t::CUBLAS_OP_N;

  inline static void call(cublasOperation_t transa,
                          cublasOperation_t transb,
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
    cublasZgemm_v2(cublas::cublasHandle::get(),
                   transa,
                   transb,
                   m,
                   n,
                   k,
                   reinterpret_cast<const cuDoubleComplex*>(&alpha),
                   reinterpret_cast<const cuDoubleComplex*>(A),
                   lda,
                   reinterpret_cast<const cuDoubleComplex*>(B),
                   ldb,
                   reinterpret_cast<const cuDoubleComplex*>(&beta),
                   reinterpret_cast<cuDoubleComplex*>(C),
                   ldc);
  }
};

template <>
struct gemm<double>
{
  static const cublasOperation_t H = cublasOperation_t::CUBLAS_OP_T;
  static const cublasOperation_t N = cublasOperation_t::CUBLAS_OP_N;

  inline static void call(cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          int k,
                          double alpha,
                          const double* A,
                          int lda,
                          const double* B,
                          int ldb,
                          double beta,
                          double* C,
                          int ldc)
  {
    cublasDgemm_v2(cublas::cublasHandle::get(),
                   transa,
                   transb,
                   m,
                   n,
                   k,
                   &alpha,
                   A,
                   lda,
                   B,
                   ldb,
                   &beta,
                   C,
                   ldc);
  }
};


template <class T>
struct geam
{
};

template <>
struct geam<Kokkos::complex<double>>
{
  static const cublasOperation_t H = cublasOperation_t::CUBLAS_OP_HERMITAN;
  static const cublasOperation_t N = cublasOperation_t::CUBLAS_OP_N;

  inline static void call(cublasOperation_t transa,
                          cublasOperation_t transb,
                          int m,
                          int n,
                          Kokkos::complex<double> alpha,
                          const Kokkos::complex<double>* A,
                          int lda,
                          Kokkos::complex<double> beta,
                          const Kokkos::complex<double>* B,
                          int ldb,
                          Kokkos::complex<double>* C,
                          int ldc)
  {
    cublasZgeam(cublas::cublasHandle::get(),
                transa,
                transb,
                m,
                n,
                reinterpret_cast<const cuDoubleComplex*>(&alpha),
                reinterpret_cast<const cuDoubleComplex*>(A),
                lda,
                reinterpret_cast<const cuDoubleComplex*>(&beta),
                reinterpret_cast<const cuDoubleComplex*>(B),
                ldb,
                reinterpret_cast<cuDoubleComplex*>(C),
                ldc);
  }
};

#endif  //__NLCGLIB__CUDA
}  // namespace cuda
}  // namespace nlcglib
