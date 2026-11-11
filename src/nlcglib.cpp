#include <cmath>
#include <cstdio>
#include <iomanip>
#include <ios>
#include <map>
#include <nlcglib.hpp>
#include <tuple>
#include "interface.hpp"
#include "linesearch/linesearch.hpp"
#include "solver_state.h"
#include "utils/iteration_logger.h"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

enum class cg_state
{
  CG,   // conjugate gradient
  pSD,  // preconditioned restart
  SD,   // steepest descent
};

namespace nlcglib {

nlcg_info
nlcg_us(execution_backend backend,
        EnergyBase& energy_base,
        UltrasoftPrecondBase& us_precond_base,
        OverlapBase& overlap_base,
        InverseOverlapBase& inverse_overlap_base,
        smearing_type smearing_t,
        double T,
        int maxiter,
        double tol,
        double kappa,
        double tau,
        int restart)
{
  switch (smearing_t) {
    case smearing_type::FERMI_DIRAC:
    case smearing_type::GAUSSIAN_SPLINE:
    case smearing_type::GAUSS:
    case smearing_type::METHFESSEL_PAXTON:
    case smearing_type::COLD:
      break;
    default:
      throw std::runtime_error("invalid smearing type given");
  }
  PROFILE("nlcglib");
  nlcg_info info;

  Timer timer;
  solver_state solver(backend,
                      energy_base,
                      us_precond_base,
                      overlap_base,
                      inverse_overlap_base,
                      smearing_t,
                      T,
                      kappa);
  std::map<smearing_type, std::string> smear_name{
      {smearing_type::FERMI_DIRAC, "Fermi-Dirac"},
      {smearing_type::COLD, "Cold"},
      {smearing_type::GAUSS, "Gauss"},
      {smearing_type::METHFESSEL_PAXTON, "Methfessel-Paxton"},
      {smearing_type::GAUSSIAN_SPLINE, "Gaussian-spline"}};

  auto& logger = Logger::GetInstance();

  logger.detach_stdout();
  logger.attach_file_master("nlcg.out");
  remove("nlcg.json");

  solver.compute();

  logger << "nlcglib parameters\n"
         << std::setw(10) << "T "
         << ": " << T << "\n"
         << std::setw(10) << "smearing "
         << ": " << smear_name.at(smearing_t) << "\n"
         << std::setw(10) << "maxiter"
         << ": " << maxiter << "\n"
         << std::setw(10) << "tol"
         << ": " << tol << "\n"
         << std::setw(10) << "kappa"
         << ": " << kappa << "\n"
         << std::setw(10) << "tau"
         << ": " << tau << "\n"
         << std::setw(10) << "restart"
         << ": " << restart << "\n";

  int Ne = energy_base.nelectrons();
  logger << "num electrons: " << Ne << "\n";
  logger << "tol = " << tol << "\n";

  solver.initialize_orbitals();

  line_search ls;
  ls.t_trial = 0.2;
  ls.tau = tau;
  logger << std::setw(15) << std::left << "Iteration" << std::setw(15) << std::left << "Free energy"
         << "\t" << std::setw(15) << std::left << "Residual"
         << "\n";

  auto slope = solver.initial_direction();

  // CG related variables
  slope_t fr = slope;  // Fletcher-Reeves numerator
  cg_state state = cg_state::CG;

  auto write_json = [&](int step, int freq = 10) { solver.write_json(slope, step, freq); };

  for (int cg_iter = 1; cg_iter < maxiter + 1; ++cg_iter) {
    logger.flush();
    if (std::abs(slope.x + slope.eta) < tol) {
      info = print_info(solver.get_F(),
                        solver.ks_energy(),
                        solver.get_entropy(),
                        slope,
                        solver.get_chemical_potential(),
                        cg_iter);
      write_json(cg_iter);

      solver.ehandle().print_info();  // print magnetization
      logger << TO_STDOUT << "kT * S   : " << std::setprecision(13) << solver.get_entropy() << "\n"
             << "F        : " << std::setprecision(13) << solver.get_F() << "\n"
             << "KS-energy: " << std::setprecision(13) << solver.get_F() - solver.get_entropy()
             << "\n"
             << "NLCG SUCCESS\n";
      logger.flush();

      info.converged = true;

      return info;
    }
    auto g = [&](double t) { return solver.evaluate(t); };

    write_json(cg_iter);


    info = print_info(solver.get_F(),
                      solver.ks_energy(),
                      solver.get_entropy(),
                      slope,
                      solver.get_chemical_potential(),
                      cg_iter);
    solver.ehandle().print_info();  // print magnetization

    timer.start();
    auto ls_result = ls(g, solver, slope.x + slope.eta);
    auto tlap = timer.stop();
    logger << "line search took: " << tlap << " seconds\n";
    logger.flush();

    if (!ls_result) solver.discard_trial();

    /* search direction is not a descent direction */
    if ((!ls_result && ls_result.error() == LineSearchErrors::SlopeError &&
         state == cg_state::CG)) {
      // attempt preconditioned SD
      log_restart_warning(cg_iter, slope);
      slope = solver.restarted();
      fr = slope;
      state = cg_state::pSD;
      continue;
    }

    if (!ls_result && ls_result.error() == LineSearchErrors::SlopeError && state == cg_state::pSD) {
      // attempt steepest descent
      log_restart_warning(cg_iter, slope);
      slope = solver.restarted_sd();
      fr = slope;
      state = cg_state::SD;
      continue;
    }

    if (!ls_result && ls_result.error() == LineSearchErrors::SlopeError && state == cg_state::SD) {
      write_json(cg_iter, 1);
      throw std::runtime_error("unrecoverable error");
    }

    /* backtracking failed */
    if (!ls_result && ls_result.error() == LineSearchErrors::DescentError &&
        state == cg_state::SD) {
      // abort!
      write_json(cg_iter, 1);
      throw std::runtime_error("Backtracking failed in steepest descent. Abort!");
    }

    if (!ls_result && ls_result.error() == LineSearchErrors::DescentError &&
        state == cg_state::pSD) {
      // continue with unpreconditioned SD step
      logger << "i=" << cg_iter << ": backtracking failed -> steepest descent\n";
      slope = solver.restarted_sd();
      fr = slope;
      state = cg_state::SD;
      continue;
    }

    if (!ls_result && ls_result.error() == LineSearchErrors::DescentError &&
        state == cg_state::CG) {
      // continue with preconditioned SD step
      logger << "i=" << cg_iter << ": backtracking failed -> restart\n";
      slope = solver.restarted();
      fr = slope;
      state = cg_state::pSD;
      continue;
    }

    if (cg_iter % restart == 0) {
      log_restart(cg_iter, restart);
      slope = solver.restarted();
      fr = slope;
      state = cg_state::pSD;
    }

    if (ls_result) {
      std::tie(fr, slope) = solver.accept(fr);
      state = cg_state::CG;
    } else {
      throw std::runtime_error("Unhandled line-search error occured.");
    }
  }
  return info;
}


nlcg_info
nlcg_us_cpu(EnergyBase& energy_base,
            UltrasoftPrecondBase& us_precond_base,
            OverlapBase& overlap_base,
            InverseOverlapBase& inverse_overlap_base,
            smearing_type smearing,
            double temp,
            double tol,
            double kappa,
            double tau,
            int maxiter,
            int restart)
{
  return nlcg_us(execution_backend::host,
                 energy_base,
                 us_precond_base,
                 overlap_base,
                 inverse_overlap_base,
                 smearing,
                 temp,
                 maxiter,
                 tol,
                 kappa,
                 tau,
                 restart);
}

nlcg_info
nlcg_us_device([[maybe_unused]] EnergyBase& energy_base,
               [[maybe_unused]] UltrasoftPrecondBase& us_precond_base,
               [[maybe_unused]] OverlapBase& overlap_base,
               [[maybe_unused]] InverseOverlapBase& inverse_overlap_base,
               [[maybe_unused]] smearing_type smearing,
               [[maybe_unused]] double temp,
               [[maybe_unused]] double tol,
               [[maybe_unused]] double kappa,
               [[maybe_unused]] double tau,
               [[maybe_unused]] int maxiter,
               [[maybe_unused]] int restart)
{
#if defined(__NLCGLIB__CUDA) || defined(__NLCGLIB__ROCM)
  return nlcg_us(execution_backend::device,
                 energy_base,
                 us_precond_base,
                 overlap_base,
                 inverse_overlap_base,
                 smearing,
                 temp,
                 maxiter,
                 tol,
                 kappa,
                 tau,
                 restart);
#else
  throw std::runtime_error("recompile nlcglib with CUDA or ROCM.");
#endif
}

// norm conserving implementation is missing at the moment
nlcg_info
nlcg_mvp2_cpu([[maybe_unused]] EnergyBase& energy_base,
              [[maybe_unused]] smearing_type smearing,
              [[maybe_unused]] double temp,
              [[maybe_unused]] double tol,
              [[maybe_unused]] double kappa,
              [[maybe_unused]] double tau,
              [[maybe_unused]] int maxiter,
              [[maybe_unused]] int restart)
{
  throw std::runtime_error("temporarily unavailable!");
}

nlcg_info
nlcg_mvp2_device([[maybe_unused]] EnergyBase& energy_base,
                 [[maybe_unused]] smearing_type smearing,
                 [[maybe_unused]] double temp,
                 [[maybe_unused]] double tol,
                 [[maybe_unused]] double kappa,
                 [[maybe_unused]] double tau,
                 [[maybe_unused]] int maxiter,
                 [[maybe_unused]] int restart)
{
  throw std::runtime_error("temporarily unavailable!");
}

nlcg_info
nlcg_mvp2_cpu_device([[maybe_unused]] EnergyBase& energy_base,
                     [[maybe_unused]] smearing_type smearing,
                     [[maybe_unused]] double temp,
                     [[maybe_unused]] double tol,
                     [[maybe_unused]] double kappa,
                     [[maybe_unused]] double tau,
                     [[maybe_unused]] int maxiter,
                     [[maybe_unused]] int restart)
{
  throw std::runtime_error("temporarily unavailable!");
}

nlcg_info
nlcg_mvp2_device_cpu([[maybe_unused]] EnergyBase& energy_base,
                     [[maybe_unused]] smearing_type smearing,
                     [[maybe_unused]] double temp,
                     [[maybe_unused]] double tol,
                     [[maybe_unused]] double kappa,
                     [[maybe_unused]] double tau,
                     [[maybe_unused]] int maxiter,
                     [[maybe_unused]] int restart)
{
  throw std::runtime_error("temporarily unavailable!");
}


nlcg_info
nlcg_us_device_cpu(EnergyBase& energy_base,
                   UltrasoftPrecondBase& us_precond_base,
                   OverlapBase& overlap_base,
                   InverseOverlapBase& inverse_overlap_base,
                   smearing_type smear,
                   double T,
                   double tol,
                   double kappa,
                   double tau,
                   int maxiter,
                   int restart)
{
  // this is now the same as `nlcg_us_cpu`, since everything is copied to host before returning to
  // nlcglib
  return nlcg_us_cpu(energy_base,
                     us_precond_base,
                     overlap_base,
                     inverse_overlap_base,
                     smear,
                     T,
                     tol,
                     kappa,
                     tau,
                     maxiter,
                     restart);
}

nlcg_info
nlcg_us_cpu_device(EnergyBase& energy_base,
                   UltrasoftPrecondBase& us_precond_base,
                   OverlapBase& overlap_base,
                   InverseOverlapBase& inverse_overlap_base,
                   smearing_type smear,
                   double T,
                   double tol,
                   double kappa,
                   double tau,
                   int maxiter,
                   int restart)
{
  // this is now the same as `nlcg_us_device`, since everything is copied to host before returning
  // to nlcglib
  return nlcg_us_device(energy_base,
                        us_precond_base,
                        overlap_base,
                        inverse_overlap_base,
                        smear,
                        T,
                        tol,
                        kappa,
                        tau,
                        maxiter,
                        restart);
}

}  // namespace nlcglib
