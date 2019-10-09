#pragma once

#include "interface.hpp"

namespace nlcglib {

void
nlcg_mvp2(EnergyBase& energy_base,
          smearing_type smearing,
          double temp,
          double tol,
          double kappa,
          double tau,
          int maxiter,
          int restart);

void
nlcg_mvp2_cuda(EnergyBase& energy_base,
               smearing_type smearing,
               double temp,
               double tol,
               double kappa,
               double tau,
               int maxiter,
               int restart);


void
nlcg_check_gradient_host(EnergyBase& energy);

void
nlcg_check_gradient_cuda(EnergyBase& energy);


}  // namespace nlcglib
