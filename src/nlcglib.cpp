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


#include "descent_direction.hpp"

typedef std::complex<double> complex_double;

namespace nlcglib {

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
  StepLogger logger(step);
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
                          },
                          ek))
            .allgather(commk);

    auto fn_host =
        eval_threaded(tapply(
                          [](auto&& x) {
                            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
                          },
                          fn))
            .allgather(commk);

    auto hii_host =
        eval_threaded(tapply(
                          [](auto&& x) {
                            return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
                          },
                          hii))
            .allgather(commk);

    logger.log("eta", ek_host);
    logger.log("fn", fn_host);
    logger.log("hii", hii_host);
  }
}


template <class memspace>
void
check_overlap(EnergyBase& e, OverlapBase& Sb, OverlapBase& Sib)
{
  FreeEnergy Energy(100, e, smearing_type::FERMI_DIRAC);

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
      },
      X,
      S,
      Sinv);
  auto Xref2 = tapply(
      [](auto x, auto s, auto si) {
        auto six = si(x);
        auto x2 = s(six);
        return x2;
      },
      X,
      S,
      Sinv);

  auto error = tapply(
      [](auto x, auto y) {
        auto z = copy(x);
        add(z, y, -1, 1);
        return z;
      },
      X,
      Xref);

  double diff = l2norm(error);
  std::cout << "** check: S(S_inv(x)), error: " << diff << "\n";
}

void
nlcheck_overlap(EnergyBase& e, OverlapBase& s, OverlapBase& si)
{
  Kokkos::initialize();
  check_overlap<Kokkos::HostSpace>(e, s, si);
  Kokkos::finalize();
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


/// xspace -> memory space where nlcg is executed
template <class xspace>
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
  FreeEnergy free_energy(T, energy_base, smear);
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

  auto Hx = copy(free_energy.get_HX());
  auto X = copy(free_energy.get_X());

  // double fr = compute_slope_single(g_X, delta_x, g_eta, delta_eta, commk);
  line_search ls;
  ls.t_trial = 0.2;
  ls.tau = tau;
  logger << std::setw(15) << std::left << "Iteration" << std::setw(15) << std::left << "Free energy"
         << "\t" << std::setw(15) << std::left << "Residual"
         << "\n";

  // auto HX_c = copy(Hx);
  descent_direction dd(T, kappa);

  auto eta = eval_threaded(tapply(make_diag(), ek));
  auto slope_zx_zeta = dd.restarted(xspace(), X, ek, fn, Hx, wk, S, P, free_energy);
  double slope = std::get<0>(slope_zx_zeta);
  auto z_x = std::get<1>(slope_zx_zeta);
  auto z_eta = std::get<2>(slope_zx_zeta);
  // allocate rotation matrices
  auto ul = eval_threaded(tapply([](auto&& z) { return empty_like()(z); }, z_eta));

  // CG related variables
  double fr = slope;  // Fletcher-Reeves numerator
  bool force_restart{false};

  for (int cg_iter = 0; cg_iter < maxiter; ++cg_iter) {
    if (std::abs(slope) < tol) {
      return info;
    }
    try {
      // line search
      auto g = [x = X, eta, z_x, z_eta, &S, &smearing, &free_energy](double t) {
        auto ek_ul_xnext = geodesic(xspace(), x, eta, z_x, z_eta, S, t);
        auto ek = std::get<0>(ek_ul_xnext);
        auto X = std::get<2>(ek_ul_xnext);
        auto fn = smearing.fn(ek);

        free_energy.compute(X, fn);

        return ek_ul_xnext;
      };

      timer.start();
      auto ek_ul_x = ls(g, free_energy, slope, force_restart);
      auto tlap = timer.stop();

      // update (X, fn(ek), ul, Hx) after line-search
      ek = std::get<0>(ek_ul_x);
      ul = std::get<1>(ek_ul_x);
      X = std::get<2>(ek_ul_x);
      fn = free_energy.get_fn();
      Hx = copy(free_energy.get_HX());

      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        slope /* slope in X and eta, temporarily */,
                        -1 /* need to separate the two slopes first */,
                        cg_iter);
      logger << "line search took: " << tlap << " seconds\n";

      if ((cg_iter % restart == 0) || force_restart) {
        /* compute directions for steepest descent */
        timer.start();

        auto slope_zx_zeta = dd.restarted(xspace(), X, ek, fn, Hx, wk, S, P, free_energy);
        slope = std::get<0>(slope_zx_zeta);
        fr = slope;
        z_x = std::get<1>(slope_zx_zeta);
        z_eta = std::get<2>(slope_zx_zeta);

        auto tlap = timer.stop();
        logger << "steepest descent took: " << tlap << " seconds\n";
      } else {
        /* compute directions for cg */
        timer.start();

        auto fr_slope_z_x_z_eta =
            dd.conjugated(xspace(), fr, X, ek, fn, Hx, z_x, z_eta, ul, wk, S, P, free_energy);
        fr = std::get<0>(fr_slope_z_x_z_eta);
        slope = std::get<1>(fr_slope_z_x_z_eta);
        z_x = std::get<2>(fr_slope_z_x_z_eta);
        z_eta = std::get<3>(fr_slope_z_x_z_eta);

        auto tlap = timer.stop();
        logger << "conjugated descent took: " << tlap << " seconds\n";
      }
      logger.flush();
    } catch (DescentError&) {
      // CG failed abort
      logger << "WARNING: No descent direction found, nlcg didn't reach final tolerance\n";
      return info;
    }
  }
  return info;
}


nlcg_info
nlcg_us_cpu(EnergyBase& energy_base,
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
  Kokkos::initialize();
  auto info = nlcg_us<Kokkos::HostSpace>(energy_base,
                                         us_precond_base,
                                         overlap_base,
                                         smearing,
                                         temp,
                                         maxiter,
                                         tol,
                                         kappa,
                                         tau,
                                         restart);
  Kokkos::finalize();
  return info;
#else
  throw std::runtime_error("recompile nlcglib with CUDA.");
#endif
}

nlcg_info
nlcg_us_device(EnergyBase& energy_base,
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
  Kokkos::initialize();
  auto info = nlcg_us<Kokkos::CudaSpace>(energy_base,
                                         us_precond_base,
                                         overlap_base,
                                         smearing,
                                         temp,
                                         maxiter,
                                         tol,
                                         kappa,
                                         tau,
                                         restart);
  Kokkos::finalize();
  return info;
}

// norm conserving implementation is missing at the moment
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
  throw std::runtime_error("temporarily unavailable!");
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
  throw std::runtime_error("temporarily unavailable!");
}

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
  throw std::runtime_error("temporarily unavailable!");
}

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
  throw std::runtime_error("temporarily unavailable!");
}


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
                   int restart)
{
  // this is now the same as `nlcg_us_cpu`, since everything is copied to host before returning to
  // nlcglib
  return nlcg_us_cpu(
      energy_base, us_precond_base, overlap_base, smear, T, tol, kappa, tau, maxiter, restart);
}

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
                   int restart)
{
  // this is now the same as `nlcg_us_device`, since everything is copied to host before returning
  // to nlcglib
  return nlcg_us_device(
      energy_base, us_precond_base, overlap_base, smear, T, tol, kappa, tau, maxiter, restart);
}


}  // namespace nlcglib
