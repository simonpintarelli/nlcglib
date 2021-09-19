#pragma once

#include "interface.hpp"

namespace nlcglib {

void initialize();
void finalize();

nlcg_info
nlcg_mvp2_cpu(EnergyBase& energy_base,
              smearing_type smearing,
              double temp,
              double tol,
              double kappa,
              double tau,
              int maxiter,
              int restart);
nlcg_info
nlcg_mvp2_device(EnergyBase& energy_base,
                 smearing_type smearing,
                 double temp,
                 double tol,
                 double kappa,
                 double tau,
                 int maxiter,
                 int restart);

nlcg_info
nlcg_mvp2_cpu_device(EnergyBase& energy_base,
                      smearing_type smearing,
                      double temp,
                      double tol,
                      double kappa,
                      double tau,
                      int maxiter,
                     int restart);
nlcg_info
nlcg_mvp2_device_cpu(EnergyBase& energy_base,
                     smearing_type smearing,
                     double temp,
                     double tol,
                     double kappa,
                     double tau,
                     int maxiter,
                     int restart);

nlcg_info
nlcg_us_device(EnergyBase& energy_base,
               UltrasoftPrecondBase& us_precond_base,
               OverlapBase& overlap_base,
               smearing_type smear,
               double T,
               double tol,
               double kappa,
               double tau,
               int maxiter,
               int restart);

nlcg_info
nlcg_us_cpu(EnergyBase& energy_base,
            UltrasoftPrecondBase& us_precond_base,
            OverlapBase& overlap_base,
            smearing_type smear,
            double T,
            double tol,
            double kappa,
            double tau,
            int maxiter,
            int restart);

nlcg_info
nlcg_us_device_cpu(EnergyBase& energy_base,
                   UltrasoftPrecondBase& us_precond_base,
                   OverlapBase& overlap_base,
                   smearing_type smear,
                   double T,
                   double tol,
                   double kappa,
                   double tau,
                   int maxiter,
                   int restart);

nlcg_info
nlcg_us_cpu_device(EnergyBase& energy_base,
                   UltrasoftPrecondBase& us_precond_base,
                   OverlapBase& overlap_base,
                   smearing_type smear,
                   double T,
                   double tol,
                   double kappa,
                   double tau,
                   int maxiter,
                   int restart);


// void nlcg_check_gradient_host(EnergyBase& energy);

// void
// nlcg_check_gradient_cuda(EnergyBase& energy);

// void nlcheck_overlap(EnergyBase&, OverlapBase&, OverlapBase&);


}  // namespace nlcglib
