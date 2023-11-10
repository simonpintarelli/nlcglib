#include <Kokkos_Core.hpp>
// #include <Kokkos_Parallel.hpp>
#include <cfenv>
#include <cstdio>
#include <iomanip>
#include <ios>
#include <iostream>
#include <nlcglib.hpp>
#include "exec_space.hpp"
#include "free_energy.hpp"
#include "geodesic.hpp"
#include "interface.hpp"
#include "la/dvector.hpp"
#include "la/lapack.hpp"
#include "la/layout.hpp"
#include "la/magma.hpp"
#include "la/map.hpp"
#include "la/mvector.hpp"
#include "la/utils.hpp"
#include "linesearch/linesearch.hpp"
#include "mpi/communicator.hpp"
#include "mvp2/descent_direction.hpp"
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

void
finalize()
{
  Kokkos::finalize();
#ifdef __NLCGLIB__MAGMA
  nlcg_finalize_magma();
#endif
}

auto
print_info(double free_energy,
           double ks_energy,
           double entropy,
           double slope_x,
           double slope_eta,
           double efermi,
           int step)
{
  auto& logger = Logger::GetInstance();
  logger << TO_STDOUT << std::setw(15) << std::left << step << std::setw(15) << std::left
         << std::fixed << std::setprecision(13) << free_energy << "\t" << std::setw(15) << std::left
         << std::scientific << std::setprecision(13) << slope_x << " " << std::scientific
         << std::setprecision(13) << slope_eta << "\n"
         << "\t kT * S       : " << std::fixed << std::setprecision(13) << entropy << "\n"
         << "\t Fermi energy : " << std::fixed << std::setprecision(13) << efermi << "\n"
         << "\t KS energy    : " << std::fixed << std::setprecision(13) << ks_energy << "\n";

  nlcg_info info;
  info.F = free_energy;
  info.S = entropy;
  info.tolerance = slope_x + slope_eta;
  info.iter = step;

  return info;
}

template <class T1, class T2>
void
cg_write_step_json(double free_energy,
                   double ks_energy,
                   double entropy,
                   double slope_x,
                   double slope_eta,
                   double efermi,
                   T1&& ek,
                   T2&& fn,
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
  logger.log("fermi_energy", efermi);
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

    logger.log("eta", ek_host);
    logger.log("fn", fn_host);
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


/// xspace -> memory space where nlcg is executed
template <class xspace, enum smearing_type smearing_t>
nlcg_info
nlcg_us(EnergyBase& energy_base,
        UltrasoftPrecondBase& us_precond_base,
        OverlapBase& overlap_base,
        double T,
        int maxiter,
        double tol,
        double kappa,
        double tau,
        int restart)
{
  // std::feclearexcept(FE_ALL_EXCEPT);
  // feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT &
  //                ~FE_UNDERFLOW);  // Enable all floating point exceptions but FE_INEXACT
  nlcg_info info;

  Communicator comm_world(energy_base.comm_world());

  auto S = Overlap(overlap_base);
  auto P = USPreconditioner(us_precond_base);

  Timer timer;
  FreeEnergy free_energy(T, energy_base, smearing_t);
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

  free_energy.compute();

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

  auto ek = free_energy.get_ek();
  auto wk = free_energy.get_wk();
  auto commk = wk.commk();
  Smearing smearing = free_energy.get_smearing();

  auto mu_fn = smearing.fn(ek);
  double mu = std::get<0>(mu_fn);
  auto fn = std::get<1>(mu_fn);
  auto X0 = free_energy.get_X();
  free_energy.compute(X0, fn, ek, mu);

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
  descent_direction<smearing_t> dd(T, kappa);

  auto eta = eval_threaded(tapply(make_diag(), ek));
  auto slope_zx_zeta = dd.restarted(xspace(), X, ek, fn, Hx, wk, mu, S, P, free_energy);
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
      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        slope,
                        -1,
                        free_energy.get_chemical_potential(),
                        cg_iter);
      cg_write_step_json(free_energy.get_F(),
                         free_energy.ks_energy(),
                         free_energy.get_entropy(),
                         slope,
                         -1,
                         free_energy.get_chemical_potential(),
                         ek,
                         fn,
                         free_energy.ks_energy_components(),
                         comm_world,
                         cg_iter);

      free_energy.ehandle().print_info();  // print magnetization
      logger << TO_STDOUT << "kT * S   : " << std::setprecision(13) << free_energy.get_entropy()
             << "\n"
             << "F        : " << std::setprecision(13) << free_energy.get_F() << "\n"
             << "KS-energy: " << std::setprecision(13)
             << free_energy.get_F() - free_energy.get_entropy() << "\n"
             << "NLCG SUCCESS\n";
      logger.flush();

      info.converged = true;

      return info;
    }
    try {
      // line search
      // TODO: capture variables explicitly
      auto g = [&](double t) {
        auto ek_ul_xnext = geodesic(xspace(), X, eta, z_x, z_eta, S, t);
        auto ek = std::get<0>(ek_ul_xnext);
        auto Xn = std::get<2>(ek_ul_xnext);
        auto mu_fn = smearing.fn(ek);
        double mu = std::get<0>(mu_fn);

        free_energy.compute(Xn, std::get<1>(mu_fn), ek, mu);

        return std::tuple_cat(ek_ul_xnext, std::make_tuple(mu));
      };

      cg_write_step_json(free_energy.get_F(),
                         free_energy.ks_energy(),
                         free_energy.get_entropy(),
                         slope,
                         -1,
                         free_energy.get_chemical_potential(),
                         ek,
                         fn,
                         free_energy.ks_energy_components(),
                         comm_world,
                         cg_iter);

      timer.start();

      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        slope /* slope in X and eta, temporarily */,
                        -1 /* need to separate the two slopes first */,
                        free_energy.get_chemical_potential(),
                        cg_iter);
      free_energy.ehandle().print_info();  // print magnetization

      auto ek_ul_x_mu = ls(g, free_energy, slope, force_restart);
      auto tlap = timer.stop();
      logger << "line search took: " << tlap << " seconds\n";

      // update (X, fn(ek), ul, Hx) after line-search
      ek = std::get<0>(ek_ul_x_mu);
      ul = std::get<1>(ek_ul_x_mu);
      X = std::get<2>(ek_ul_x_mu);
      double mu = std::get<3>(ek_ul_x_mu);
      eta = eval_threaded(tapply(make_diag(), ek));
      fn = free_energy.get_fn();
      Hx = copy(free_energy.get_HX());

      if ((cg_iter % restart == 0) || force_restart) {
        /* compute directions for steepest descent */
        timer.start();
        auto slope_zx_zeta = dd.restarted(xspace(), X, ek, fn, Hx, wk, mu, S, P, free_energy);
        slope = std::get<0>(slope_zx_zeta);  // no need to catch slope > 0 -> linesearch will throw
        fr = slope;
        z_x = std::get<1>(slope_zx_zeta);
        z_eta = std::get<2>(slope_zx_zeta);

        auto tlap = timer.stop();
        logger << "steepest descent took: " << tlap << " seconds\n";
      } else {
        /* compute directions for cg */
        timer.start();

        auto fr_slope_z_x_z_eta =
            dd.conjugated(xspace(), fr, X, ek, fn, Hx, z_x, z_eta, ul, wk, mu, S, P, free_energy);
        fr = std::get<0>(fr_slope_z_x_z_eta);
        slope = std::get<1>(fr_slope_z_x_z_eta);
        z_x = std::get<2>(fr_slope_z_x_z_eta);
        z_eta = std::get<3>(fr_slope_z_x_z_eta);

        if (slope > 0) {
          // force restart
          logger << "i=" << cg_iter << ": slope > 0 detected -> restart\n";
          auto slope_zx_zeta = dd.restarted(xspace(), X, ek, fn, Hx, wk, mu, S, P, free_energy);
          slope = std::get<0>(
              slope_zx_zeta);  // no need to catch slope > 0 again -> linesearch will throw
          fr = slope;
          z_x = std::get<1>(slope_zx_zeta);
          z_eta = std::get<2>(slope_zx_zeta);

          force_restart = true;
        }

        auto tlap = timer.stop();
        logger << "conjugated descent took: " << tlap << " seconds\n";
      }
      logger.flush();
    } catch (DescentError&) {
      // CG failed abort
      logger << "[NLCG] Error: No descent direction found, nlcg didn't reach final tolerance\n";
      return info;
    } catch (SlopeError&) {
      logger << "[NLCG] Error: slope > 0 after CG-restart. Abort.\n";
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
  switch (smearing) {
    case smearing_type::FERMI_DIRAC: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::FERMI_DIRAC>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::GAUSSIAN_SPLINE>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::GAUSS: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::GAUSS>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::METHFESSEL_PAXTON>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::COLD: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::COLD>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    default:
      throw std::runtime_error("invalid smearing type given");
  }
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
#ifdef __NLCGLIB__CUDA
  switch (smearing) {
    case smearing_type::FERMI_DIRAC: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::FERMI_DIRAC>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::GAUSSIAN_SPLINE>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::GAUSS: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::GAUSS>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::METHFESSEL_PAXTON>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::COLD: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::COLD>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }

    default:
      throw std::runtime_error("invalid smearing type given");
  }
#elif defined __NLCGLIB__ROCM
  switch (smearing) {
    case smearing_type::FERMI_DIRAC: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::FERMI_DIRAC>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::GAUSSIAN_SPLINE>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::GAUSS: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::GAUSS>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::METHFESSEL_PAXTON>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }
    case smearing_type::COLD: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::COLD>(
          energy_base, us_precond_base, overlap_base, temp, maxiter, tol, kappa, tau, restart);
      return info;
    }

    default:
      throw std::runtime_error("invalid smearing type given");
  }

#else
  throw std::runtime_error("recompile nlcglib with CUDA or ROCM.");
#endif
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
