#pragma once
#ifdef __NLCGLIB__MAGMA
#include <Kokkos_Core.hpp>

void nlcg_init_magma();
void nlcg_finalize_magma();

template<class COMPLEX>
void zheevd_magma(int n, COMPLEX* dA, int lda, double* w);

template <class COMPLEX>
void zpotrf_magma(int n, COMPLEX* dA, int lda);

template <class COMPLEX>
void zpotrs_magma(int n, int nrhs, COMPLEX* dA, int lda, COMPLEX* dB, int ldb);


#endif /*__NLCGLIB__MAGMA*/
