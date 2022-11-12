#ifdef __NLCGLIB__MAGMA
#include <magma_auxiliary.h>
#include <magma_types.h>
#include <magma_v2.h>
#include <Kokkos_Core.hpp>
#include <complex>
#include <stdexcept>


/// Wrapper around `magma_zheevd_gpu`

template <class COMPLEX>
void
zheevd_magma(int n, COMPLEX* dA, int lda, double* w)
{
  // allocate workspace arrays according to magma doxygen
  const int magma_align = 32;
  magmaDoubleComplex* wA;  // workspace array
  int ldwa = magma_roundup(lda, magma_align);
  magma_zmalloc_pinned(&wA, n * ldwa);

  magmaDoubleComplex* work;
  int nb = magma_get_zhetrd_nb(n);
  int lwork = std::max(n + n * nb, 2 * n + n * n);
  magma_zmalloc(&work, n);

  double* rwork;
  int lrwork = 1 + 5 * n + 2 * n * n;
  magma_dmalloc(&rwork, lrwork);

  int* iwork;
  int liwork = 3 + 5 * n;
  magma_imalloc(&iwork, liwork);

  int info{0};

  // ref:
  // https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__heevd.html#ga841519bd19ba04630758ffda31bc1cbb
  magma_zheevd_gpu(magma_vec_t::MagmaVec,
                   magma_uplo_t::MagmaLower,
                   n,
                   reinterpret_cast<magmaDoubleComplex_ptr>(
                       dA),  // overwritten, if info=0, A contains the eigenvalues
                   lda,
                   w,                                               // eigenvalues
                   reinterpret_cast<magmaDoubleComplex_ptr>(wA),    // workspace
                   ldwa,                                            //
                   reinterpret_cast<magmaDoubleComplex_ptr>(work),  // work (complex)
                   lwork,
                   rwork,  // work real
                   lrwork,
                   iwork,  // work intenger
                   liwork,
                   &info);

  if (info != 0) {
    throw std::runtime_error("magma_zheevd_gpu failed, error=" + std::to_string(info));
  }

  magma_free_pinned(wA);
  magma_free(work);

  magma_free(rwork);
  magma_free(iwork);
}

template void
zheevd_magma(int, Kokkos::complex<double>*, int, double*);
template void
zheevd_magma(int, std::complex<double>*, int, double*);

template <class COMPLEX>
void
zpotrf_magma(int n, COMPLEX* dA, int lda)
{
  int info{-1};
  magma_zpotrf_gpu(
      magma_uplo_t::MagmaLower, n, reinterpret_cast<magmaDoubleComplex_ptr>(dA), lda, &info);

  if (info != 0) {
    throw std::runtime_error("magma_zpotrf_gpu failed with" + std::to_string(info));
  }
}

template void
zpotrf_magma(int, Kokkos::complex<double>*, int);
template void
zpotrf_magma(int, std::complex<double>*, int);


template <class COMPLEX>
void
zpotrs_magma(int n, int nrhs, COMPLEX* dA, int lda, COMPLEX* dB, int ldb)
{
  int info{-1};
  magma_zpotrs_gpu(magma_uplo_t::MagmaLower,
                   n,
                   nrhs,
                   reinterpret_cast<magmaDoubleComplex_ptr>(dA),
                   lda,
                   reinterpret_cast<magmaDoubleComplex_ptr>(dB),
                   ldb,
                   &info);

  if (info != 0) {
    throw std::runtime_error("magma_zpotrs_gpu failed with " + std::to_string(info));
  }
}

template void zpotrs_magma(int, int, Kokkos::complex<double>*, int, Kokkos::complex<double>*, int);
template void zpotrs_magma(int, int, std::complex<double>*, int, std::complex<double>*, int);

#endif /*__NLCGLIB__MAGMA*/
