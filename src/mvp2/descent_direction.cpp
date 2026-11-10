#include "descent_direction.h"
#include "descent_direction_conjugate.hpp"
#include "descent_direction_restart.hpp"
#include "descent_direction_sd.hpp"
#include "free_energy.h"
#include "overlap.hpp"
#include "ultrasoft_precond.hpp"
#include "utils/logger.hpp"

namespace nlcglib {

template <class MemorySpace>
descent_direction<MemorySpace>::descent_direction(double T, double kappa, smearing_type smear)
    : T(T)
    , kappa(kappa)
    , smear(smear)
{
}

template <class MemorySpace>
std::tuple<slope_t, slope_t, mvector<host_matrix_t>, mvector<host_matrix_t>>
descent_direction<MemorySpace>::conjugated(slope_t fr_old,
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
                                           FreeEnergy& free_energy)
{
  double mo = free_energy.occupancy();
  /* always executed on CPU */
  double dFdmu = GradEtaHelper::dFdmu(smear, free_energy.get_ek(), en, fn, wk, mu, T, mo);
  double sumfn = GradEtaHelper::dmu_deta(smear, en, wk, mu, T, mo);

  auto commk = wk.commk();

  descent_direction_conjugate<MemorySpace> functor(
      MemorySpace{}, mu, dFdmu, sumfn, T, kappa, mo, smear);

  auto [m_fr, delta_x, delta_eta, z_x, z_eta, m_slope_zp] =
      unzip(tapply_eager(functor, X, en, fn, hx, zxp, zetap, ul, S, P, wk));

  slope_t fr = sum(m_fr, commk);
  double gamma = (fr.x + fr.eta) / (fr_old.x + fr_old.eta);
  Logger::GetInstance() << " CG gamma " << std::setprecision(3) << gamma << "\n";

  slope_t slope_zp = sum(m_slope_zp, commk);

  /* this is tr{<Z|g>} = tr{<Δ + γ*Z(n-1)|g>} = tr{<Δ |g>} + γ * tr{<Z(n-1)|g>}
   *            ^                                   ^                  ^
   *          slope  =                             fr    + γ *     slope_zp
   */
  slope_t slope = fr + gamma * slope_zp;

  // This operation is in-place and overwrites z_x, z_eta.
  tapply_eager(
      [gamma](auto delta_x, auto delta_eta, auto z_x, auto z_eta) {
        // Z <- gamma*Z + delta
        (void)add(z_x, delta_x, 1.0, gamma);
        (void)add(z_eta, delta_eta, 1.0, gamma);
      },
      delta_x,
      delta_eta,
      z_x,
      z_eta);

  return std::make_tuple(fr, slope, z_x, z_eta);
}

template <class MemorySpace>
std::tuple<slope_t, mvector<host_matrix_t>, mvector<host_matrix_t>>
descent_direction<MemorySpace>::restarted(const mvector<host_matrix_t>& X,
                                          const mvector<host_vector_t>& en,
                                          const mvector<host_vector_t>& fn,
                                          const mvector<host_matrix_t>& hx,
                                          const mvector<double>& wk,
                                          double mu,
                                          const Overlap& S,
                                          const USPreconditioner& P,
                                          FreeEnergy& free_energy)
{
  double mo = free_energy.occupancy();
  double dFdmu = GradEtaHelper::dFdmu(smear, free_energy.get_ek(), en, fn, wk, mu, T, mo);
  double sumfn = GradEtaHelper::dmu_deta(smear, en, wk, mu, T, mo);

  auto commk = wk.commk();

  descent_direction_restart<MemorySpace> functor(
      MemorySpace{}, mu, dFdmu, sumfn, T, kappa, mo, smear);

  auto [m_fr, z_x, z_eta] = unzip(tapply_eager(functor, X, en, fn, hx, S, P, wk));
  // auto ures = unzip(res);

  slope_t fr = sum(m_fr, commk);

  return std::make_tuple(fr, z_x, z_eta);
}

template <class MemorySpace>
std::tuple<slope_t, mvector<host_matrix_t>, mvector<host_matrix_t>>
descent_direction<MemorySpace>::restarted_sd(const mvector<host_matrix_t>& X,
                                             const mvector<host_vector_t>& en,
                                             const mvector<host_vector_t>& fn,
                                             const mvector<host_matrix_t>& hx,
                                             const mvector<double>& wk,
                                             double mu,
                                             const Overlap& S,
                                             const Overlap& Sinv,
                                             const USPreconditioner& P [[maybe_unused]],
                                             FreeEnergy& free_energy)
{
  double mo = free_energy.occupancy();
  double dFdmu = GradEtaHelper::dFdmu(smear, free_energy.get_ek(), en, fn, wk, mu, T, mo);
  double sumfn = GradEtaHelper::dmu_deta(smear, en, wk, mu, T, mo);

  auto commk = wk.commk();

  descent_direction_sd<MemorySpace> functor(MemorySpace{}, mu, dFdmu, sumfn, T, kappa, mo, smear);

  auto [m_fr, z_x, z_eta] = unzip(tapply_eager(functor, X, en, fn, hx, S, Sinv, wk));
  slope_t fr = sum(m_fr, commk);

  return std::make_tuple(slope_t{.x = fr.x, .eta = fr.eta}, z_x, z_eta);
}


template class descent_direction<Kokkos::HostSpace>;
#ifdef __NLCGLIB__CUDA
template class descent_direction<Kokkos::CudaSpace>;
#elif defined(__NLCGLIB__ROCM)
template class descent_direction<Kokkos::HIPSpace>;
#endif

}  // namespace nlcglib
