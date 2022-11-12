#pragma once
#ifdef __NLCGLIB__MAGMA
#include <Kokkos_Core.hpp>

template<class COMPLEX>
void zheevd_magma(int n, COMPLEX* dA, int lda, double* w);

#endif /*__NLCGLIB__MAGMA*/
