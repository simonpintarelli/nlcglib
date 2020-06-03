#include <nlcglib.hpp>
#include <iostream>
#include <iomanip>
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "la/layout.hpp"
#include "la/map.hpp"
#include "la/mvector.hpp"
#include "preconditioner.hpp"
#include "smearing.hpp"
#include "traits.hpp"
#include "pseudo_hamiltonian/grad_eta.hpp"
#include "free_energy.hpp"
#include "exec_space.hpp"
#include "geodesic.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"
#include "mvp2.hpp"
#include "linesearch/linesearch.hpp"

typedef std::complex<double> complex_double;

namespace nlcglib {

template <class memspace, class xspace=memspace>
void nlcg(EnergyBase& energy_base, smearing_type smear, double T, int maxiter, double tol, double kappa, double tau, int restart)
{
  Timer timer;
  FreeEnergy<memspace, xspace> free_energy(T, energy_base, smear);
  std::map<smearing_type, std::string> smear_name{
      {smearing_type::FERMI_DIRAC, "Fermi-Dirac"},
      {smearing_type::GAUSSIAN_SPLINE, "Gaussian-spline"}};

  auto& logger = Logger::GetInstance();
  logger.detach_stdout();
  logger.attach_file_master("nlcg.out");

  free_energy.compute();

  logger << "F (initial) =  " << std::setprecision(8) << free_energy.get_F() << "\n";
  logger << "KS (initial) =  " << std::setprecision(8) << free_energy.ks_energy() << "\n";
  logger << "nlcglib parameters\n"
           << std::setw(10) << "T "
           << ": " << T << "\n"
           << std::setw(10) << "smearing "
           << ": " << smear_name.at(smear) << "\n"
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

  auto ek = free_energy.get_ek();
  auto wk = free_energy.get_wk();
  auto commk = wk.commk();
  Smearing smearing = free_energy.get_smearing();
  // set fn = f_D(ek)
  auto fn = smearing.fn(ek);
  auto X0 = free_energy.get_X();

  free_energy.compute(X0, fn);
  auto Hx = free_energy.get_HX();
  auto X = copy(free_energy.get_X());

  PreconditionerTeter<xspace> Prec(free_energy.get_gkvec_ekin());
  GradEta grad_eta(T, kappa);

  auto eta = eval_threaded(tapply(make_diag(), ek));
  auto Hij = eval_threaded(tapply(inner_(), X, Hx, wk));
  auto g_eta = grad_eta.g_eta(Hij, wk, ek, fn, free_energy.occupancy());
  auto delta_eta = grad_eta.delta_eta(Hij, ek, wk);

  auto Xll = lagrange_multipliers(X, Hx, Prec);
  auto g_X = gradX(X, Hx, fn, Xll, wk);
  auto delta_x = precondGradX(X, Hx, Prec, Xll);

  // initial search direction Z_.
  auto Z_x = copy(delta_x);
  auto Z_eta = copy(delta_eta);

  double slope = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);

  if (slope >= 0) {
    throw std::runtime_error("ascending slope detected. Abort!");
  }

  double fr = compute_slope(g_X, delta_x, g_eta, delta_eta, commk);
  line_search ls;
  ls.t_trial = 0.2;
  ls.tau = tau;
  logger << std::setw(15) << std::left << "Iteration"
           << std::setw(15) << std::left << "Free energy" << "\t"
           << std::setw(15) << std::left << "Residual" << "\n";

  for (int i = 1; i < maxiter+1; ++i) {
    logger << "Iteration " << i << "\n";
    timer.start();

    // check for convergence
    if (std::abs(slope) < tol) {
      logger << std::setw(15) << std::left << i << std::setw(15) << std::left << std::scientific
             << std::setprecision(12) << free_energy.get_F() << "\t" << std::setw(15) << std::left
             << std::scientific << std::setprecision(12) << slope << "\n";

      logger << "kT * S: " << std::setprecision(10) << free_energy.get_entropy() << "\n";
      logger << "F     : " << std::setprecision(10) << free_energy.get_F() << "\n";
      logger << "NLCG SUCCESS\n";
      return;
    }

    // main loop
    auto ek_ul = ls(
        [&](auto& ef) { return [&](double t) { return geodesic(ef, X, eta, Z_x, Z_eta, t); }; },
        free_energy,
        slope);

    auto ek = std::get<0>(ek_ul);
    auto u = std::get<1>(ek_ul);

    // obtain new H@x, compute g_X, g_eta, delta_x, delta_eta
    Hx = free_energy.get_HX();
    X = copy(free_energy.get_X());
    // updated fn is missing!!
    auto fni = free_energy.get_fn();

    eta = eval_threaded(tapply(make_diag(), ek));

    auto Hij = eval_threaded(tapply(inner_(), X, Hx, wk));
    auto g_eta = grad_eta.g_eta(Hij, wk, ek, fni, free_energy.occupancy());
    auto delta_eta = grad_eta.delta_eta(Hij, ek, wk);

    auto Xll = lagrange_multipliers(X, Hx, Prec);
    auto g_X = gradX(X, Hx, fni, Xll, wk);
    auto delta_x = precondGradX(X, Hx, Prec, Xll);

    // rotate previous search direction ..
    // TODO: only needed if not doing restart ...
    auto Z_Xp = rotateX(Z_x, u);
    auto Z_etap = rotateEta(Z_eta, u);

    // conjugate directions
    double fr_new = compute_slope(g_X, delta_x, g_eta, delta_eta, commk);
    if (fr_new > 0) {
      throw std::runtime_error("Error: increasing slope !!!, <.,.> = "  + std::to_string(fr_new));
    }
    double gamma = fr_new / fr;
    fr = fr_new;
    if (!(i % restart == 0))
      logger << "\t CG gamma = " << gamma << "\n";

    if (i % restart == 0) {
      logger << "CG restart\n";
      // overwrites Z_xp
      Z_x = copy(delta_x);
      // overwrite Z_etap
      Z_eta = copy(delta_eta);
    } else {
      // overwrites Z_xp
      Z_x = eval_threaded(conjugatex(delta_x, Z_Xp, X, gamma));
      // overwrite Z_etap
      Z_eta = eval_threaded(conjugateeta(delta_eta, Z_etap, gamma));
    }

    slope = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);

    if (slope >= 0) {
      if (i % restart == 0) throw std::runtime_error("no descent direction could be found, abort!");
      logger << ">> slope > 0, force restart.\n";
      Z_x = copy(delta_x);
      Z_eta = copy(delta_eta);

      slope = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);
    }
    logger << std::setw(15) << std::left << i
           << std::setw(15) << std::left << std::scientific << std::setprecision(12) << free_energy.get_F() << "\t"
           << std::setw(15) << std::left << std::scientific << std::setprecision(12) << slope << "\n";

    auto tlap = timer.stop();
    logger << "cg iteration took " << tlap << " s\n";
  }
}


template <class memspace>
void nlcg_check_gradient(EnergyBase& energy_base)
{
  double T = 300;
  double kappa = 1;
  FreeEnergy<memspace> free_energy(T, energy_base, smearing_type::FERMI_DIRAC);

  free_energy.compute();
  Logger() << "F (initial) =  " << std::setprecision(8) << free_energy.get_F() << "\n";
  int Ne = energy_base.nelectrons();
  Logger() << "num electrons: " << Ne << "\n";

  auto X = free_energy.get_X();

  auto ek = free_energy.get_ek();
  auto wk = free_energy.get_wk();
  auto commk = wk.commk();

  Logger() << "test call smearing" << "\n";
  Smearing smearing = free_energy.get_smearing();
  // set fn = f_D(ek)
  auto fn = smearing.fn(ek);

  // compute and retrieve new wrappers
  free_energy.compute(X, fn);
  Logger() << "F (initial must NOT change) =  " << free_energy.get_F() << std::setprecision(8)
           << "\n";
  // retrieve new objs because ptr might have changed
  X = free_energy.get_X();
  auto Hx = free_energy.get_HX(); // obtain Hx
  PreconditionerTeter<memspace> Prec(free_energy.get_gkvec_ekin());
  GradEta grad_eta(T, kappa);
  auto Hij = eval_threaded(tapply(inner_(), X, Hx, wk));

  auto xnorm = eval_threaded(tapply(innerh_tr(), X, X));
  Logger() << "l2norm(X)"
            << "\n";
  print(xnorm);

  auto Xll = lagrange_multipliers(X, Hx, Prec);
  auto g_X = gradX(X, Hx, fn, Xll, wk);
  auto delta_x = precondGradX(X, Hx, Prec, Xll);

  // check that overlap is zero
  auto no = eval_threaded(tapply_async(
      [](auto x, auto delta_x) {
        auto ss = inner_()(x, eval(delta_x));
        return innerh_tr()(ss, ss);
      },
      X,
      delta_x));
  Logger() << "<X, G>: \n";
  print(no);

  auto X_new = copy(free_energy.get_X());

  std::cout << "new F = " << std::scientific << std::setprecision(8) << free_energy.get_F() << "\n";

  std::cout << " ---- geodesic ----" << "\n";
  /// call geodesic (aka line evaluator)
  auto eta = eval_threaded(tapply(make_diag(), ek));
  auto delta_eta = grad_eta.delta_eta(Hij, ek, wk);
  std::cout << "|delta_eta| = " << l2norm(delta_eta) << "\n";

  // compute slope in X
  auto g_eta = grad_eta.g_eta(Hij, wk, ek, fn, free_energy.occupancy());
  double slope = compute_slope(g_X, delta_x, g_eta, delta_eta, commk);
  Logger() << "slope (all): " << std::setprecision(8) << slope << "\n";

  // compute at t=0, because fn will change, e.g. fn=f_n(ek)
  geodesic(free_energy, X_new, eta, delta_x, delta_eta, 0);
  // oops now HX is wrong ...
  double F0 = free_energy.get_F();
  Logger() << "F0: " << std::scientific << std::setprecision(11) << F0 << "\n";
  for (double dt : {1e-5, 1e-6, 1e-7}) {
    std::cout << "dt: " << dt << "\n";
    geodesic(free_energy, X_new, eta, delta_x, delta_eta, dt);
    double F1 = free_energy.get_F();
    Logger() << "F1: " << std::scientific << std::setprecision(11) << F1 << "\n";

    auto dFdt = (F1 - F0) / dt;
    Logger() << "slope (fd) = " << std::setprecision(8) << dFdt << "\n";
  }
}

void
nlcg_check_gradient_host(EnergyBase& energy)
{
#ifdef __CLANG
  Kokkos::initialize();
  nlcg_check_gradient<Kokkos::HostSpace>(energy);
  Kokkos::finalize();
#endif
}


void
nlcg_check_gradient_cuda(EnergyBase& energy)
{
#ifdef __CLANG
  Kokkos::initialize();
  nlcg_check_gradient<Kokkos::CudaSpace>(energy);
  Kokkos::finalize();
#endif
}


void
nlcg_mvp2_cpu(EnergyBase& energy_base, smearing_type smearing, double temp, double tol, double kappa, double tau, int maxiter, int restart)
{
  Kokkos::initialize();
  nlcg<Kokkos::HostSpace>(energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);
  Kokkos::finalize();
}

void
nlcg_mvp2_device(EnergyBase& energy_base, smearing_type smearing, double temp, double tol, double kappa, double tau, int maxiter, int restart)
{
  Kokkos::initialize();
  nlcg<Kokkos::CudaSpace>(energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);
  Kokkos::finalize();
}

/**
 * obtain |psi> and H |psi> on device, but execute on host
 */
void
nlcg_mvp2_device_cpu(EnergyBase& energy_base,
                      smearing_type smearing,
                      double temp,
                      double tol,
                      double kappa,
                      double tau,
                      int maxiter,
                      int restart)
{
  Kokkos::initialize();
  nlcg<Kokkos::CudaSpace, Kokkos::HostSpace>(energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);
  Kokkos::finalize();
}

/**
 * obtain |psi> and H |psi> on host, but execute on device
 */
void
nlcg_mvp2_cpu_device(EnergyBase& energy_base,
                      smearing_type smearing,
                      double temp,
                      double tol,
                      double kappa,
                      double tau,
                      int maxiter,
                      int restart)
{
  Kokkos::initialize();
  nlcg<Kokkos::HostSpace, Kokkos::CudaSpace>(
      energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);
  Kokkos::finalize();
}


}  // namespace nlcglib
