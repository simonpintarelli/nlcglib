#pragma once

#include <tuple>
#include "interface.hpp"
#include "la/mvector.hpp"
#include "la/types.h"

namespace nlcglib {

class FreeEnergy;
class Overlap;
class USPreconditioner;

struct slope_t
{
  double x, eta;
};

/// Search directions share one implementation per execution memory space.
template <class MemorySpace>
class descent_direction
{
public:
  descent_direction(double T, double kappa, smearing_type smear);

  std::tuple<slope_t, slope_t, mvector<host_matrix_t>, mvector<host_matrix_t>> conjugated(
      slope_t fr_old,
      const mvector<host_matrix_t>& X,
      const mvector<host_vector_t>& en,
      const mvector<host_vector_t>& fn,
      const mvector<host_matrix_t>& hx,
      const mvector<host_matrix_t>& zxp,
      const mvector<host_matrix_t>& zetap,
      const mvector<host_matrix_t>& ul,
      const mvector<double>& wk,
      double mu,
      const Overlap& S,
      const USPreconditioner& P,
      FreeEnergy& free_energy);

  std::tuple<slope_t, mvector<host_matrix_t>, mvector<host_matrix_t>> restarted(
      const mvector<host_matrix_t>& X,
      const mvector<host_vector_t>& en,
      const mvector<host_vector_t>& fn,
      const mvector<host_matrix_t>& hx,
      const mvector<double>& wk,
      double mu,
      const Overlap& S,
      const USPreconditioner& P,
      FreeEnergy& free_energy);

  std::tuple<slope_t, mvector<host_matrix_t>, mvector<host_matrix_t>> restarted_sd(
      const mvector<host_matrix_t>& X,
      const mvector<host_vector_t>& en,
      const mvector<host_vector_t>& fn,
      const mvector<host_matrix_t>& hx,
      const mvector<double>& wk,
      double mu,
      const Overlap& S,
      const Overlap& Sinv,
      const USPreconditioner& P,
      FreeEnergy& free_energy);

private:
  double T;
  double kappa;
  smearing_type smear;
};

extern template class descent_direction<Kokkos::HostSpace>;
#ifdef __NLCGLIB__CUDA
extern template class descent_direction<Kokkos::CudaSpace>;
#elif defined(__NLCGLIB__ROCM)
extern template class descent_direction<Kokkos::HIPSpace>;
#endif

}  // namespace nlcglib
