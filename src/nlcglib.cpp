#include <fmt/format.h>
#include <Kokkos_Core.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include <nlcglib.hpp>
#include "free_energy.hpp"
#include "geodesic.hpp"
#include "interface.hpp"
#include "la/lapack.hpp"
#include "la/mvector.hpp"
#include "la/utils.hpp"
#include "linesearch/linesearch.hpp"
#include "mpi/communicator.hpp"
#include "mvp2/descent_direction.hpp"
#include "overlap.hpp"
#include "smearing.hpp"
#include "ultrasoft_precond.hpp"
#include "utils/logger.hpp"
#include "utils/step_logger.hpp"
#include "utils/timer.hpp"

typedef std::complex<double> complex_double;

enum class cg_state
{
  CG,   // conjugate gradient
  pSD,  // preconditioned restart
  SD,   // steepest descent
};

namespace nlcglib {

void
initialize()
{
  Kokkos::InitializationSettings args;
  args.set_disable_warnings(true);
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
print_info(
    double free_energy, double ks_energy, double entropy, slope_t slope, double efermi, int step)
{
  double slope_tot = slope.x + slope.eta;
  auto& logger = Logger::GetInstance();
  //                       fmt::arg("slope_tot", slope_tot));
  logger << TO_STDOUT
         << fmt::format(
                "{iter:<6d}"
                R"(Etot     : {F:20.10f} [Ha]
      Residual : {slope_tot:>20.5e}
      kT * S   : {entropy:>20.8f} [Ha]
      Efermi   : {efermi:>20.8f} [Ha]
      KS energy: {ks_energy:>20.8f} [Ha]
      slope x  : {slope_x:>20.5e}
      slope eta: {slope_eta:>20.5e}
)",
                fmt::arg("iter", step),
                fmt::arg("F", free_energy),
                fmt::arg("slope_tot", slope_tot),
                fmt::arg("entropy", entropy),
                fmt::arg("efermi", efermi),
                fmt::arg("ks_energy", ks_energy),
                fmt::arg("slope_x", slope.x),
                fmt::arg("slope_eta", slope.eta));

  nlcg_info info;
  info.F = free_energy;
  info.S = entropy;
  info.tolerance = slope.x + slope.eta;
  info.iter = step;

  return info;
}

template <class T1, class T2, class T3>
void
cg_write_step_json(double free_energy,
                   double ks_energy,
                   double entropy,
                   slope_t slope,
                   double efermi,
                   T1&& ek,
                   T2&& fn,
                   T3&& wk,
                   std::map<std::string, double> energy_components,
                   Communicator& commk,
                   int step,
                   int freq)
{
  StepLogger logger(step, "nlcg.json", commk.rank() == 0);
  logger.log("F", free_energy);
  logger.log("EKS", ks_energy);
  logger.log("entropy", entropy);
  logger.log("slope_x", slope.x);
  logger.log("slope_eta", slope.eta);
  logger.log("fermi_energy", efermi);
  logger.log("ks_energy_comps", energy_components);
  if (step == 0) {
    logger.log("wk", wk);
  }

  if (step % freq == 0) {
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


/// xspace -> memory space where nlcg is executed
template <class xspace, enum smearing_type smearing_t>
nlcg_info
nlcg_us(EnergyBase& energy_base,
        UltrasoftPrecondBase& us_precond_base,
        OverlapBase& overlap_base,
        InverseOverlapBase& inverse_overlap_base,
        double T,
        int maxiter,
        double tol,
        double kappa,
        double tau,
        int restart)
{
  PROFILE("nlcglib");
  nlcg_info info;

  Communicator comm_world(energy_base.comm_world());

  auto S = Overlap(overlap_base);
  auto Sinv = InverseOverlap(inverse_overlap_base);
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
  auto [slope, z_x, z_eta] = dd.restarted(xspace(), X, ek, fn, Hx, wk, mu, S, P, free_energy);

  // allocate rotation matrices
  auto ul = eval_threaded(tapply([](auto&& z) { return empty_like()(z); }, z_eta));

  // CG related variables
  slope_t fr = slope;  // Fletcher-Reeves numerator
  cg_state state = cg_state::CG;

  auto write_json = [&](int step, int freq = 10) {
    cg_write_step_json(free_energy.get_F(),
                       free_energy.ks_energy(),
                       free_energy.get_entropy(),
                       slope,
                       free_energy.get_chemical_potential(),
                       ek,
                       fn,
                       wk,
                       free_energy.ks_energy_components(),
                       comm_world,
                       step,
                       freq);
  };

  for (int cg_iter = 1; cg_iter < maxiter + 1; ++cg_iter) {
    logger.flush();
    if (std::abs(slope.x + slope.eta) < tol) {
      info = print_info(free_energy.get_F(),
                        free_energy.ks_energy(),
                        free_energy.get_entropy(),
                        slope,
                        free_energy.get_chemical_potential(),
                        cg_iter);
      write_json(cg_iter);

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
    auto g = [&](double t) {
      auto ek_ul_xnext = geodesic(xspace(), X, eta, z_x, z_eta, S, t);
      auto ek = std::get<0>(ek_ul_xnext);
      auto Xn = std::get<2>(ek_ul_xnext);
      auto mu_fn = smearing.fn(ek);
      double mu = std::get<0>(mu_fn);

      free_energy.compute(Xn, std::get<1>(mu_fn), ek, mu);

      return std::tuple_cat(ek_ul_xnext, std::make_tuple(mu));
    };

    write_json(cg_iter);


    info = print_info(free_energy.get_F(),
                      free_energy.ks_energy(),
                      free_energy.get_entropy(),
                      slope,
                      free_energy.get_chemical_potential(),
                      cg_iter);
    free_energy.ehandle().print_info();  // print magnetization

    timer.start();
    auto ls_result = ls(g, free_energy, slope.x + slope.eta);
    auto tlap = timer.stop();
    logger << "line search took: " << tlap << " seconds\n";
    logger.flush();

    /* search direction is not a descent direction */
    if ((!ls_result && ls_result.error() == LineSearchErrors::SlopeError &&
         state == cg_state::CG)) {
      // attempt preconditioned SD
      logger << fmt::format(
          "WARNING: iter={:d} slope={:.5e} ({:.5e},{:.5e}) > 0 detected -> restart\n",
          cg_iter,
          slope.x + slope.eta,
          slope.x,
          slope.eta);
      std::tie(slope, z_x, z_eta) =
          dd.restarted(xspace(), X, ek, fn, Hx, wk, mu, S, P, free_energy);
      fr = slope;
      state = cg_state::pSD;
      continue;
    }

    if (!ls_result && ls_result.error() == LineSearchErrors::SlopeError && state == cg_state::pSD) {
      // attempt steepest descent
      logger << fmt::format(
          "WARNING: iter={:d} slope={:.5e} ({:.5e},{:.5e}) > 0 detected -> restart\n",
          cg_iter,
          slope.x + slope.eta,
          slope.x,
          slope.eta);
      std::tie(slope, z_x, z_eta) =
          dd.restarted_sd(xspace(), X, ek, fn, Hx, wk, mu, S, Sinv, P, free_energy);
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
      std::tie(slope, z_x, z_eta) =
          dd.restarted_sd(xspace(), X, ek, fn, Hx, wk, mu, S, Sinv, P, free_energy);
      fr = slope;
      state = cg_state::SD;
      continue;
    }

    if (!ls_result && ls_result.error() == LineSearchErrors::DescentError &&
        state == cg_state::CG) {
      // continue with preconditioned SD step
      logger << "i=" << cg_iter << ": backtracking failed -> restart\n";
      std::tie(slope, z_x, z_eta) =
          dd.restarted(xspace(), X, ek, fn, Hx, wk, mu, S, P, free_energy);
      fr = slope;
      state = cg_state::pSD;
      continue;
    }

    if (cg_iter % restart == 0) {
      logger << fmt::format("i={:d} cg_restart({:d})\n", cg_iter, restart);
      std::tie(slope, z_x, z_eta) =
          dd.restarted(xspace(), X, ek, fn, Hx, wk, mu, S, P, free_energy);
      fr = slope;
      state = cg_state::pSD;
    }

    if (ls_result) {
      auto ek_ul_x_mu = ls_result.value();
      ek = std::get<0>(ek_ul_x_mu);
      ul = std::get<1>(ek_ul_x_mu);
      X = std::get<2>(ek_ul_x_mu);
      double mu = std::get<3>(ek_ul_x_mu);
      eta = eval_threaded(tapply(make_diag(), ek));
      fn = free_energy.get_fn();
      Hx = copy(free_energy.get_HX());

      std::tie(fr, slope, z_x, z_eta) =
          dd.conjugated(xspace(), fr, X, ek, fn, Hx, z_x, z_eta, ul, wk, mu, S, P, free_energy);
      state = cg_state::CG;
    } else {
      throw std::runtime_error(fmt::format("Unhandled line-search error occured."));
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
  switch (smearing) {
    case smearing_type::FERMI_DIRAC: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::FERMI_DIRAC>(energy_base,
                                                                         us_precond_base,
                                                                         overlap_base,
                                                                         inverse_overlap_base,
                                                                         temp,
                                                                         maxiter,
                                                                         tol,
                                                                         kappa,
                                                                         tau,
                                                                         restart);
      return info;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::GAUSSIAN_SPLINE>(energy_base,
                                                                             us_precond_base,
                                                                             overlap_base,
                                                                             inverse_overlap_base,
                                                                             temp,
                                                                             maxiter,
                                                                             tol,
                                                                             kappa,
                                                                             tau,
                                                                             restart);
      return info;
    }
    case smearing_type::GAUSS: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::GAUSS>(energy_base,
                                                                   us_precond_base,
                                                                   overlap_base,
                                                                   inverse_overlap_base,
                                                                   temp,
                                                                   maxiter,
                                                                   tol,
                                                                   kappa,
                                                                   tau,
                                                                   restart);
      return info;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::METHFESSEL_PAXTON>(energy_base,
                                                                               us_precond_base,
                                                                               overlap_base,
                                                                               inverse_overlap_base,
                                                                               temp,
                                                                               maxiter,
                                                                               tol,
                                                                               kappa,
                                                                               tau,
                                                                               restart);
      return info;
    }
    case smearing_type::COLD: {
      auto info = nlcg_us<Kokkos::HostSpace, smearing_type::COLD>(energy_base,
                                                                  us_precond_base,
                                                                  overlap_base,
                                                                  inverse_overlap_base,
                                                                  temp,
                                                                  maxiter,
                                                                  tol,
                                                                  kappa,
                                                                  tau,
                                                                  restart);
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
               InverseOverlapBase& inverse_overlap_base,
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
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::FERMI_DIRAC>(energy_base,
                                                                         us_precond_base,
                                                                         overlap_base,
                                                                         inverse_overlap_base,
                                                                         temp,
                                                                         maxiter,
                                                                         tol,
                                                                         kappa,
                                                                         tau,
                                                                         restart);
      return info;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::GAUSSIAN_SPLINE>(energy_base,
                                                                             us_precond_base,
                                                                             overlap_base,
                                                                             inverse_overlap_base,
                                                                             temp,
                                                                             maxiter,
                                                                             tol,
                                                                             kappa,
                                                                             tau,
                                                                             restart);
      return info;
    }
    case smearing_type::GAUSS: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::GAUSS>(energy_base,
                                                                   us_precond_base,
                                                                   overlap_base,
                                                                   inverse_overlap_base,
                                                                   temp,
                                                                   maxiter,
                                                                   tol,
                                                                   kappa,
                                                                   tau,
                                                                   restart);
      return info;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::METHFESSEL_PAXTON>(energy_base,
                                                                               us_precond_base,
                                                                               overlap_base,
                                                                               inverse_overlap_base,
                                                                               temp,
                                                                               maxiter,
                                                                               tol,
                                                                               kappa,
                                                                               tau,
                                                                               restart);
      return info;
    }
    case smearing_type::COLD: {
      auto info = nlcg_us<Kokkos::CudaSpace, smearing_type::COLD>(energy_base,
                                                                  us_precond_base,
                                                                  overlap_base,
                                                                  inverse_overlap_base,
                                                                  temp,
                                                                  maxiter,
                                                                  tol,
                                                                  kappa,
                                                                  tau,
                                                                  restart);
      return info;
    }

    default:
      throw std::runtime_error("invalid smearing type given");
  }
#elif defined __NLCGLIB__ROCM
  switch (smearing) {
    case smearing_type::FERMI_DIRAC: {
      auto info =
          nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::FERMI_DIRAC>(energy_base,
                                                                              us_precond_base,
                                                                              overlap_base,
                                                                              inverse_overlap_base,
                                                                              temp,
                                                                              maxiter,
                                                                              tol,
                                                                              kappa,
                                                                              tau,
                                                                              restart);
      return info;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::GAUSSIAN_SPLINE>(
          energy_base,
          us_precond_base,
          overlap_base,
          inverse_overlap_base,
          temp,
          maxiter,
          tol,
          kappa,
          tau,
          restart);
      return info;
    }
    case smearing_type::GAUSS: {
      auto info =
          nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::GAUSS>(energy_base,
                                                                        us_precond_base,
                                                                        overlap_base,
                                                                        inverse_overlap_base,
                                                                        temp,
                                                                        maxiter,
                                                                        tol,
                                                                        kappa,
                                                                        tau,
                                                                        restart);
      return info;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::METHFESSEL_PAXTON>(
          energy_base,
          us_precond_base,
          overlap_base,
          inverse_overlap_base,
          temp,
          maxiter,
          tol,
          kappa,
          tau,
          restart);
      return info;
    }
    case smearing_type::COLD: {
      auto info = nlcg_us<Kokkos::Experimental::HIPSpace, smearing_type::COLD>(energy_base,
                                                                               us_precond_base,
                                                                               overlap_base,
                                                                               inverse_overlap_base,
                                                                               temp,
                                                                               maxiter,
                                                                               tol,
                                                                               kappa,
                                                                               tau,
                                                                               restart);
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
