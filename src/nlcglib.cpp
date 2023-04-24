#include <omp.h>
#include <Kokkos_Core.hpp>
#include <cfenv>
#include <iomanip>
#include <ios>
#include <iostream>
#include <nlcglib.hpp>
#include <set>
#include "exec_space.hpp"
#include "free_energy.hpp"
#include "geodesic.hpp"
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "la/layout.hpp"
#include "la/map.hpp"
#include "la/mvector.hpp"
#include "la/utils.hpp"
#include "linesearch/linesearch.hpp"
#include "mvp2.hpp"
#include "overlap.hpp"
#include "preconditioner.hpp"
#include "pseudo_hamiltonian/grad_eta.hpp"
#include "smearing.hpp"
#include "traits.hpp"
#include "ultrasoft_precond.hpp"
#include "utils/format.hpp"
#include "utils/logger.hpp"
#include "utils/step_logger.hpp"
#include "utils/timer.hpp"
#include <cstdio>

typedef std::complex<double> complex_double;

namespace nlcglib {

void
initialize()
{
#if KOKKOS_VERSION < 30700
  Kokkos::InitArguments args;
  args.disable_warnings = true;
#ifdef USE_OPENMP
  args.num_threads = omp_get_max_threads();
#endif /* endif USE_OPENMP */
#else  /* KOKKOS_VERSION >= 3.7.00 */
  Kokkos::InitializationSettings args;
  args.set_disable_warnings(true);
#endif /* endif KOKKOS VERSION */
#ifdef USE_OPENMP
  args.num_threads = omp_get_max_threads();
#endif

#ifdef __NLCGLIB__MAGMA
  nlcg_init_magma();
#endif

  Kokkos::initialize(args);
}


void finalize()
{
  Kokkos::finalize();
}

auto
print_info(double free_energy,
           double ks_energy,
           double entropy,
           double slope_x,
           double slope_eta,
           int step)
{
  auto& logger = Logger::GetInstance();
  logger << TO_STDOUT << std::setw(15) << std::left << step << std::setw(15) << std::left
         << std::fixed << std::setprecision(13) << free_energy << "\t" << std::setw(15) << std::left
         << std::scientific << std::setprecision(13) << slope_x << " " << std::scientific
         << std::setprecision(13) << slope_eta << "\n"
         << "\t kT * S   : " << std::fixed << std::setprecision(13) << entropy << "\n"
         << "\t KS energy: " << std::fixed << std::setprecision(13) << ks_energy << "\n";

  nlcg_info info;
  info.F = free_energy;
  info.S = entropy;
  info.tolerance = slope_x + slope_eta;
  info.iter = step;

  return info;
}

template <class T1, class T2, class T3>
void
cg_write_step_json(double free_energy,
                   double ks_energy,
                   double entropy,
                   double slope_x,
                   double slope_eta,
                   T1&& ek,
                   T2&& fn,
                   T3&& hii,
                   std::map<std::string, double> energy_components,
                   Communicator& commk,
                   int step)
{
  StepLogger logger(step, "nlcg.json", commk.rank() == 0);
  logger.log("F", free_energy);
  logger.log("EKS", ks_energy);
  logger.log("entropy", entropy);
  logger.log("slope_x", slope_x);
  logger.log("slope_eta", slope_eta);
  logger.log("ks_energy_comps", energy_components);

  if (step % 10 == 0) {
    auto ek_host =
        eval_threaded(tapply(
                          [](auto&& x) {
                            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
                          }, ek)) .allgather(commk);

    auto fn_host =
        eval_threaded(tapply(
                          [](auto&& x) {
                            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
                          }, fn)) .allgather(commk);

    // auto hii_host =
    //     eval_threaded(tapply(
    //                       [](auto&& x) {
    //                         return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    //                       }, hii)) .allgather(commk);

    logger.log("eta", ek_host);
    logger.log("fn", fn_host);
    // logger.log("hii", hii_host);
  }
}

template <class memspace, class xspace = memspace>
nlcg_info
nlcg(EnergyBase& energy_base,
     smearing_type smear,
     double T,
     int maxiter,
     double tol,
     double kappa,
     double tau,
     int restart)
{
  // std::feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(~FE_UNDERFLOW);  // Enable all floating point exceptions but FE_INEXACT
  nlcg_info info;

  Timer timer;
  FreeEnergy<memspace, xspace> free_energy(T, energy_base, smear);
  std::map<smearing_type, std::string> smear_name{
      {smearing_type::FERMI_DIRAC, "Fermi-Dirac"},
      {smearing_type::GAUSSIAN_SPLINE, "Gaussian-spline"}};

  auto& logger = Logger::GetInstance();
  logger.detach_stdout();
  logger.attach_file_master("nlcg.out");

  free_energy.compute();

  logger << "F (initial) =  " << std::setprecision(13) << free_energy.get_F() << "\n";
  logger << "KS (initial) =  " << std::setprecision(13) << free_energy.ks_energy() << "\n";
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

  auto Xll = lagrange_multipliers(X, X, Hx, Prec);
  auto g_X = gradX(X, Hx, fn, Xll, wk);
  auto delta_x = precondGradX(X, Hx, Prec, Xll);

  // initial search direction Z_.
  auto Z_x = copy(delta_x);
  auto Z_eta = copy(delta_eta);


  auto slope_x_eta = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);
  double slope = std::get<0>(slope_x_eta) + std::get<1>(slope_x_eta);

  if (slope >= 0) {
    throw std::runtime_error("ascending slope detected. Abort!");
  }

  double fr = compute_slope_single(g_X, delta_x, g_eta, delta_eta, commk);
  line_search ls;
  ls.t_trial = 0.2;
  ls.tau = tau;
  logger << std::setw(15) << std::left << "Iteration" << std::setw(15) << std::left << "Free energy"
         << "\t" << std::setw(15) << std::left << "Residual"
         << "\n";

  bool force_restart = false;
  if (commk.rank() == 0) {
    std::ofstream fout("nlcg.json", std::ios_base::out);  // clear `nlcg.json`
  }
  for (int i = 1; i < maxiter + 1; ++i) {
    logger << "Iteration " << i << "\n";
    timer.start();

    // check for convergence
    if (std::abs(slope) < tol) {
      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        std::get<0>(slope_x_eta),
                        std::get<1>(slope_x_eta),
                        i);

      logger << TO_STDOUT << "kT * S   : " << std::setprecision(13) << free_energy.get_entropy()
             << "\n"
             << "F        : " << std::setprecision(13) << free_energy.get_F() << "\n"
             << "KS-energy: " << std::setprecision(13)
             << free_energy.get_F() - free_energy.get_entropy() << "\n"
             << "NLCG SUCCESS\n";
      return info;
    }

    // main loop
    try {
      auto ek_ul = ls(
          [&](auto& ef) { return [&](double t) { return geodesic(ef, X, eta, Z_x, Z_eta, t); }; },
          free_energy,
          slope,
          force_restart);

      ek = std::get<0>(ek_ul);
      auto u = std::get<1>(ek_ul);

      // obtain new H@x, compute g_X, g_eta, delta_x, delta_eta
      Hx = free_energy.get_HX();
      X = copy(free_energy.get_X());
      // updated fn is missing!!
      auto fni = free_energy.get_fn();

      eta = eval_threaded(tapply(make_diag(), ek));

      Hij = eval_threaded(tapply(inner_(), X, Hx, wk));
      g_eta = grad_eta.g_eta(Hij, wk, ek, fni, free_energy.occupancy());
      delta_eta = grad_eta.delta_eta(Hij, ek, wk);

      Xll = lagrange_multipliers(X, X, Hx, Prec);
      g_X = gradX(X, Hx, fni, Xll, wk);
      delta_x = precondGradX(X, Hx, Prec, Xll);

      // rotate previous search direction ..
      auto Z_Xp = rotateX(Z_x, u);
      auto Z_etap = rotateEta(Z_eta, u);
      // conjugate directions
      double fr_new = compute_slope_single(g_X, delta_x, g_eta, delta_eta, commk);
      if (fr_new > 0) {
        throw std::runtime_error("Error: increasing slope !!!, <.,.> = " + format("%.5g", fr_new));
      }
      double gamma = fr_new / fr;
      fr = fr_new;
      if (!(i % restart == 0 && !force_restart)) logger << "\t CG gamma = " << gamma << "\n";

      if (i % restart == 0 || force_restart) {
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

      slope_x_eta = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);
      slope = std::get<0>(slope_x_eta) + std::get<1>(slope_x_eta);

      if (slope >= 0) {
        if (i % restart == 0 || force_restart)
          throw std::runtime_error("no descent direction could be found, abort!");
        logger << ">> slope > 0, force restart.\n";
        Z_x = copy(delta_x);
        Z_eta = copy(delta_eta);

        slope_x_eta = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);
        slope = std::get<0>(slope_x_eta) + std::get<1>(slope_x_eta);
      }

      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        std::get<0>(slope_x_eta),
                        std::get<1>(slope_x_eta),
                        i);
      free_energy.ehandle().print_info();  // print magnetization

      auto tlap = timer.stop();

      cg_write_step_json(free_energy.get_F(),
                         free_energy.ks_energy(),
                         free_energy.get_entropy(),
                         std::get<0>(slope_x_eta),
                         std::get<1>(slope_x_eta),
                         ek,
                         fni,
                         eval_threaded(tapply([](auto&& x) { return diag(x); }, Hij)),
                         free_energy.ks_energy_components(),
                         commk,
                         i);

      logger << "cg iteration took " << tlap << " s\n";
      logger.flush();
    } catch (DescentError&) {
      logger << "WARNING: No descent direction found, nlcg didn't reach final tolerance\n";
      return info;
    }
  }

  return info;
}

template <class memspace>
void
check_overlap(EnergyBase& e, OverlapBase& Sb, OverlapBase& Sib)
{
  FreeEnergy<memspace, memspace> Energy(100, e, smearing_type::FERMI_DIRAC);

  auto X = copy(Energy.get_X());
  Overlap S(Sb);
  Overlap Sinv(Sib);

  std::cout << "l2norm(X) = " << l2norm(X) << "\n";

  auto SX = tapply_op(S, X);
  auto SinvX = tapply_op(Sinv, X);
  std::cout << "l2norm(SX): " << l2norm(SX) << "\n";
  std::cout << "l2norm(SinvX): " << l2norm(SinvX) << "\n";

  auto tr = innerh_reduce(X, SX);
  std::cout << "tr(XSX): " << tr << "\n";
  auto Xref = tapply(
      [](auto x, auto s, auto si) {
        auto sx = s(x);
        auto x2 = si(sx);
        return x2;
      }, X, S, Sinv);
  auto Xref2 = tapply(
      [](auto x, auto s, auto si) {
        auto six = si(x);
        auto x2 = s(six);
        return x2;
      }, X, S, Sinv);

  auto error = tapply(
      [](auto x, auto y) {
        auto z = copy(x);
        add(z, y, -1, 1);
        return z;
      }, X, Xref);

  double diff = l2norm(error);
  std::cout << "** check: S(S_inv(x)), error: " << diff << "\n";
}

void
nlcheck_overlap(EnergyBase& e, OverlapBase& s, OverlapBase& si)
{
  check_overlap<Kokkos::HostSpace>(e, s, si);
}


struct minus
{
  template <class X>
  auto operator()(X&& x)
  {
    auto res = empty_like()(x);
    add(res, x, -1, 0);
    return res;
  }
};


template <class memspace, class xspace = memspace>
nlcg_info
nlcg_us(EnergyBase& energy_base,
        UltrasoftPrecondBase& us_precond_base,
        OverlapBase& overlap_base,
        smearing_type smear,
        double T,
        int maxiter,
        double tol,
        double kappa,
        double tau,
        int restart)
{
  // std::feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT &
                 ~FE_UNDERFLOW);  // Enable all floating point exceptions but FE_INEXACT
  nlcg_info info;

  auto S = Overlap(overlap_base);
  auto P = USPreconditioner(us_precond_base);

  Timer timer;
  FreeEnergy<memspace, xspace> free_energy(T, energy_base, smear);
  std::map<smearing_type, std::string> smear_name{
      {smearing_type::FERMI_DIRAC, "Fermi-Dirac"},
      {smearing_type::GAUSSIAN_SPLINE, "Gaussian-spline"}};

  auto& logger = Logger::GetInstance();
  logger.detach_stdout();
  logger.attach_file_master("nlcg.out");
  remove("nlcg.json");

  free_energy.compute();

  logger << "F (initial) =  " << std::setprecision(13) << free_energy.get_F() << "\n";
  logger << "KS (initial) =  " << std::setprecision(13) << free_energy.ks_energy() << "\n";
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

  auto fn = smearing.fn(ek);
  auto X0 = free_energy.get_X();
  free_energy.compute(X0, fn);

  auto Hx = free_energy.get_HX();
  auto X = copy(free_energy.get_X());

  // PreconditionerTeter<xspace> Prec(free_energy.get_gkvec_ekin());
  GradEta grad_eta(T, kappa);

  auto eta = eval_threaded(tapply(make_diag(), ek));
  auto Hij = eval_threaded(tapply(inner_(), X, Hx, wk));
  auto g_eta = grad_eta.g_eta(Hij, wk, ek, fn, free_energy.occupancy());
  auto delta_eta = grad_eta.delta_eta(Hij, ek, wk);

  auto SX = tapply_op(S, X);
  auto Xll = lagrange_multipliers(X, SX, Hx, P);
  auto g_X = gradX(SX, Hx, fn, Xll, wk);
  auto delta_x = precondGradX_us(SX, Hx, P, Xll);

  // initial search direction Z_.
  auto Z_x = copy(delta_x);
  auto Z_eta = copy(delta_eta);

  auto slope_x_eta = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);
  double slope = std::get<0>(slope_x_eta) + std::get<1>(slope_x_eta);

  if (slope >= 0) {
    throw std::runtime_error("ascending slope detected. Abort!");
  }

  double fr = compute_slope_single(g_X, delta_x, g_eta, delta_eta, commk);
  line_search ls;
  ls.t_trial = 0.2;
  ls.tau = tau;
  logger << std::setw(15) << std::left << "Iteration" << std::setw(15) << std::left << "Free energy"
         << "\t" << std::setw(15) << std::left << "Residual"
         << "\n";

  bool force_restart = false;

  if (commk.rank() == 0) {
    std::ofstream fout("nlcg.json", std::ios_base::out);  // clear `nlcg.json`
  }
  for (int i = 1; i < maxiter + 1; ++i) {
    logger << "Iteration " << i << "\n";
    timer.start();

    // check for convergence
    if (std::abs(slope) < tol) {
      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        std::get<0>(slope_x_eta),
                        std::get<1>(slope_x_eta),
                        i);

      logger << TO_STDOUT << "kT * S   : " << std::setprecision(13) << free_energy.get_entropy()
             << "\n"
             << "F        : " << std::setprecision(13) << free_energy.get_F() << "\n"
             << "KS-energy: " << std::setprecision(13)
             << free_energy.get_F() - free_energy.get_entropy() << "\n"
             << "NLCG SUCCESS\n";
      return info;
    }

    // main loop
    try {
      auto ek_ul = ls(
          [&](auto& ef) {
            return [&](double t) { return geodesic_us(ef, X, eta, Z_x, Z_eta, S, t); };
          }, free_energy, slope, force_restart);

      ek = std::get<0>(ek_ul);
      auto u = std::get<1>(ek_ul);
      auto ls_info = std::get<2>(ek_ul);

      // obtain new H@x, compute g_X, g_eta, delta_x, delta_eta
      Hx = free_energy.get_HX();
      X = copy(free_energy.get_X());
      auto fni = free_energy.get_fn();

      eta = eval_threaded(tapply(make_diag(), ek));

      Hij = eval_threaded(tapply(inner_(), X, Hx, wk));
      g_eta = grad_eta.g_eta(Hij, wk, ek, fni, free_energy.occupancy());
      delta_eta = grad_eta.delta_eta(Hij, ek, wk);

      SX = tapply_op(S, X);
      // this is SXll
      Xll = lagrange_multipliers(X, SX, Hx, P);
      g_X = gradX(SX, Hx, fni, Xll, wk);
      delta_x = precondGradX_us(SX, Hx, P, Xll);

      // rotate previous search direction ..
      auto Z_Xp = rotateX(Z_x, u);
      auto Z_etap = rotateEta(Z_eta, u);
      // conjugate directions
      double fr_new = compute_slope_single(g_X, delta_x, g_eta, delta_eta, commk);
      if (fr_new > 0) {
        throw std::runtime_error("Error: increasing slope !!!, <.,.> = " + format("%.5g", fr_new));
      }
      double gamma = fr_new / fr;
      fr = fr_new;
      if (!(i % restart == 0 && !force_restart)) logger << "\t CG gamma = " << gamma << "\n";

      if (i % restart == 0 || force_restart) {
        logger << "CG restart\n";
        // overwrites Z_xp
        Z_x = copy(delta_x);
        // overwrite Z_etap
        Z_eta = copy(delta_eta);
      } else {
        // overwrites Z_xp
        Z_x = eval_threaded(conjugatex_us(delta_x, Z_Xp, X, SX, gamma));
        // overwrite Z_etap
        Z_eta = eval_threaded(conjugateeta(delta_eta, Z_etap, gamma));
      }

      slope_x_eta = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);
      slope = std::get<0>(slope_x_eta) + std::get<1>(slope_x_eta);

      if (slope >= 0) {
        if (i % restart == 0 || force_restart)
          throw std::runtime_error("no descent direction could be found, abort!");
        logger << ">> slope > 0, force restart.\n";
        Z_x = copy(delta_x);
        Z_eta = copy(delta_eta);

        slope_x_eta = compute_slope(g_X, Z_x, g_eta, Z_eta, commk);
        slope = std::get<0>(slope_x_eta) + std::get<1>(slope_x_eta);
      }

      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        std::get<0>(slope_x_eta),
                        std::get<1>(slope_x_eta),
                        i);
      free_energy.ehandle().print_info();

      auto tlap = timer.stop();

      cg_write_step_json(free_energy.get_F(),
                         free_energy.ks_energy(),
                         free_energy.get_entropy(),
                         std::get<0>(slope_x_eta),
                         std::get<1>(slope_x_eta),
                         ek,
                         fni,
                         eval_threaded(tapply([](auto&& x) { return diag(x); }, Hij)),
                         free_energy.ks_energy_components(),
                         commk,
                         i);

      logger << "cg iteration took " << tlap << " s\n";
      logger.flush();
    } catch (DescentError&) {
      auto zero = tapply_async(zeros_like(), Z_eta);
      auto slope_x_eta = compute_slope(g_X, Z_x, g_eta, zero, commk);
      double slope_x = std::get<0>(slope_x_eta);

      auto Ft = [&](double t) { return geodesic_us(free_energy, X, eta, Z_x, zero, S, t); };

      logger << "--- DEBUG bt search failed, print energies along Z_X ---\n";
      auto uniform_range = linspace(0, 0.5, 10);
      std::vector<double> ts(uniform_range.begin(), uniform_range.end());
      for (auto ti : {1e-4, 1e-5, 1e-6, 1e-7}) ts.push_back(ti);

      std::sort(ts.begin(), ts.end());

      std::vector<double> Fs;

      for (double t : ts) {
        Ft(t);
        double Ef = free_energy.get_F();
        Fs.push_back(Ef);
        logger << "t: " << std::scientific << std::setprecision(5) << t << ", Ef: " << std::fixed
               << std::setprecision(13) << Ef << "\n";
      }
      /* compute slopes */
      logger << "slope (<G_X, Z_X>): " << std::setprecision(5) << std::scientific << slope_x
             << "\n";
      double F0 = Fs[0];
      for (auto i = 1u; i < Fs.size(); ++i) {
        double Fi = Fs[i];
        double dt = ts[i] - ts[0];
        double dF = Fi - F0;
        logger << "slope (dt=" << std::setprecision(5) << std::scientific << dt
               << "): " << std::setprecision(5) << std::scientific << (dF / dt) << "\n";
      }

      {
        /* steepest descent in eta  */
        logger << " --- DEBUG, steepest descent in Eta ---\n";
        auto zerox = tapply_async(zeros_like(), Z_x);
        // -g_eta
        auto zg_eta = tapply_async(minus(), g_eta);

        auto slope_x_eta = compute_slope(g_X, zerox, g_eta, zg_eta, commk);
        double slope_eta = std::get<1>(slope_x_eta);

        auto Ft_eta = [&](double t) {
          return geodesic_us(free_energy, X, eta, zerox, zg_eta, S, t);
        };

        logger << "slope (<g_eta, -g_eta>): " << std::setprecision(5) << std::scientific
               << slope_eta << "\n";
        std::vector<double> Fs;
        for (double t : ts) {
          Ft_eta(t);
          double Ef = free_energy.get_F();
          Fs.push_back(Ef);
          logger << "t: " << std::scientific << std::setprecision(5) << t << ", Ef: " << std::fixed
                 << std::setprecision(13) << Ef << "\n";
        }
      }
      {
        /* steepest descent in eta  */
        logger << " --- DEBUG, search along Z_eta (preconditioned) ---\n";
        auto zerox = tapply_async(zeros_like(), Z_x);
        auto slope_x_eta = compute_slope(g_X, zerox, g_eta, Z_eta, commk);
        double slope_eta = std::get<1>(slope_x_eta);

        logger << "slope (<g_eta, Z_eta>): " << std::setprecision(5) << std::scientific << slope_eta
               << "\n";

        auto Ft_zeta = [&](double t) {
          return geodesic_us(free_energy, X, eta, zerox, Z_eta, S, t);
        };
        std::vector<double> Fs;
        for (double t : ts) {
          Ft_zeta(t);
          double Ef = free_energy.get_F();
          Fs.push_back(Ef);
          logger << "t: " << std::scientific << std::setprecision(5) << t << ", Ef: " << std::fixed
                 << std::setprecision(13) << Ef << "\n";
        }
      }

      logger << "----------\n";

      logger << "WARNING: No descent direction found, nlcg didn't reach final tolerance\n";
      return info;
    }
  }

  return info;
}



nlcg_info
nlcg_mvp2_cpu(EnergyBase& energy_base,
              smearing_type smearing,
              double temp,
              double tol,
              double kappa,
              double tau,
              int maxiter,
              int restart)
{
  auto info =
      nlcg<Kokkos::HostSpace>(energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);

  return info;
}

nlcg_info
nlcg_mvp2_device(EnergyBase& energy_base,
                 smearing_type smearing,
                 double temp,
                 double tol,
                 double kappa,
                 double tau,
                 int maxiter,
                 int restart)
{
#ifdef __NLCGLIB__CUDA
  auto info =
      nlcg<Kokkos::CudaSpace>(energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);
  return info;
#else
  throw std::runtime_error("recompile nlcglib with CUDA.");
#endif
}

/**
 * obtain |psi> and H |psi> on device, but execute on host
 */
nlcg_info
nlcg_mvp2_device_cpu(EnergyBase& energy_base,
                     smearing_type smearing,
                     double temp,
                     double tol,
                     double kappa,
                     double tau,
                     int maxiter,
                     int restart)
{
#ifdef __NLCGLIB__CUDA
  auto info = nlcg<Kokkos::CudaSpace, Kokkos::HostSpace>(
      energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);
  return info;
#else
  throw std::runtime_error("recompile nlcglib with CUDA.");
#endif
}

/**
 * obtain |psi> and H |psi> on host, but execute on device
 */
nlcg_info
nlcg_mvp2_cpu_device(EnergyBase& energy_base,
                     smearing_type smearing,
                     double temp,
                     double tol,
                     double kappa,
                     double tau,
                     int maxiter,
                     int restart)
{
#ifdef __NLCGLIB__CUDA
  auto info = nlcg<Kokkos::HostSpace, Kokkos::CudaSpace>(
      energy_base, smearing, temp, maxiter, tol, kappa, tau, restart);
  return info;
#else
  throw std::runtime_error("recompile nlcglib with CUDA.");
#endif
}

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
               int restart)
{
#ifdef __NLCGLIB__CUDA
  auto info = nlcg_us<Kokkos::CudaSpace>(
      energy_base, us_precond_base, overlap_base, smear, T, maxiter, tol, kappa, tau, restart);
  return info;
#else
  throw std::runtime_error("recompile nlcglib with CUDA.");
#endif
}

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
            int restart)
{
  auto info = nlcg_us<Kokkos::HostSpace>(
      energy_base, us_precond_base, overlap_base, smear, T, maxiter, tol, kappa, tau, restart);

  return info;
}

/**
 * obtain |psi> and H |psi> on device, but execute on host
 */
nlcg_info
nlcg_us_device_cpu(EnergyBase& energy_base,
                   UltrasoftPrecondBase& us_precond_base,
                   OverlapBase& overlap_base,
                   smearing_type smearing,
                   double temp,
                   double tol,
                   double kappa,
                   double tau,
                   int maxiter,
                   int restart)
{
#ifdef __NLCGLIB__CUDA
  auto info = nlcg_us<Kokkos::CudaSpace, Kokkos::HostSpace>(energy_base,
                                                            us_precond_base,
                                                            overlap_base,
                                                            smearing,
                                                            temp,
                                                            maxiter,
                                                            tol,
                                                            kappa,
                                                            tau,
                                                            restart);
  return info;
#else
  throw std::runtime_error("recompile nlcglib with CUDA.");
#endif
}

nlcg_info
nlcg_us_cpu_device(EnergyBase& energy_base,
                   UltrasoftPrecondBase& us_precond_base,
                   OverlapBase& overlap_base,
                   smearing_type smearing,
                   double temp,
                   double tol,
                   double kappa,
                   double tau,
                   int maxiter,
                   int restart)
{
#ifdef __NLCGLIB__CUDA
  auto info = nlcg_us<Kokkos::HostSpace, Kokkos::CudaSpace>(energy_base,
                                                            us_precond_base,
                                                            overlap_base,
                                                            smearing,
                                                            temp,
                                                            maxiter,
                                                            tol,
                                                            kappa,
                                                            tau,
                                                            restart);
  return info;
#else
  throw std::runtime_error("recompile nlcglib with CUDA.");
#endif
}

}  // namespace nlcglib
