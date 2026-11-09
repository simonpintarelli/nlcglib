#pragma once

#include "interface.hpp"
#include "la/mvector.hpp"
#include "la/types.h"

namespace nlcglib {

/// Host reductions used to enforce the electron-number constraint.
struct GradEtaHelper
{
  static double dFdmu(smearing_type smear,
                      const mvector<host_vector_t>& Hii,
                      const mvector<host_vector_t>& en,
                      const mvector<host_vector_t>& fn,
                      const mvector<double>& wk,
                      double mu,
                      double T,
                      double mo);

  static double dmu_deta(smearing_type smear,
                         const mvector<host_vector_t>& en,
                         const mvector<double>& wk,
                         double mu,
                         double T,
                         double mo);
};

/// Smearing dispatch stays outside the statically specialized Kokkos kernels.
template <class MemorySpace>
class GradEta
{
public:
  GradEta(double T, smearing_type smear);

  dense_matrix_t<MemorySpace> g_eta(const dense_matrix_t<MemorySpace>& Hij,
                                    double mu,
                                    double wk,
                                    const real_vector_t<MemorySpace>& ek,
                                    const real_vector_t<MemorySpace>& fn,
                                    double dmu_deta,
                                    double dFdmu,
                                    double mo);

private:
  double kT;
  smearing_type smear;
};

extern template class GradEta<Kokkos::HostSpace>;
#ifdef __NLCGLIB__CUDA
extern template class GradEta<Kokkos::CudaSpace>;
#elif defined(__NLCGLIB__ROCM)
extern template class GradEta<Kokkos::HIPSpace>;
#endif

}  // namespace nlcglib
