#pragma once

#include <tuple>
#include "execution_backend.h"
#include "interface.hpp"
#include "la/types.h"
#include "slope.h"

namespace nlcglib {

class FreeEnergy;
class Overlap;
class USPreconditioner;

/// Host-side search-direction control with backend-specific numerical kernels.
class descent_direction
{
public:
  descent_direction(execution_backend backend, double T, double kappa, smearing_type smear);

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
  execution_backend backend;
  double T;
  double kappa;
  smearing_type smear;
};

}  // namespace nlcglib
