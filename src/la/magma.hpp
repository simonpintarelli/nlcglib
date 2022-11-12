#pragma once
#ifdef __NLCGLIB__MAGMA
#include <Kokkos_Core.hpp>

void zheevd_magma(int n, Kokkos::complex<double>* dA, int lda, double* w);

#endif /*__NLCGLIB__MAGMA*/
