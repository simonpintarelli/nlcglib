#pragma once

#include "descent_direction_impl.hpp"

namespace nlcglib {

template <enum smearing_type SMEARING_TYPE>
class descent_direction
{
public:
  descent_direction(double T, double kappa)
      : T(T)
      , kappa(kappa)
  {
  }

  template <class mem_t,
            class x_t,
            class e_t,
            class f_t,
            class hx_t,
            class zxp_t,
            class zetap_t,
            class ul_t,
            class op_t,
            class prec_t,
            class F>
  auto conjugated(const mem_t& memspc,
                  double fr_old,
                  const mvector<x_t>& X,
                  const mvector<e_t>& en,
                  const mvector<f_t>& fn,
                  const mvector<hx_t>& hx,
                  const mvector<zxp_t>& zxp,
                  const mvector<zetap_t>& zetap,
                  const mvector<ul_t>& ul,
                  const mvector<double>& wk,
                  op_t&& S,
                  prec_t&& P,
                  F&& free_energy);

  /// restarted CG step or steepest descent
  template <class mem_t,
            class x_t,
            class e_t,
            class f_t,
            class hx_t,
            class op_t,
            class prec_t,
            class F>
  std::tuple<double, mvector<to_layout_left_t<x_t>>, mvector<to_layout_left_t<x_t>>> restarted(
      const mem_t& memspc,
      const mvector<x_t>& X,
      const mvector<e_t>& en,
      const mvector<f_t>& fn,
      const mvector<hx_t>& hx,
      const mvector<double>& wk,
      op_t&& S,
      prec_t&& P,
      F&& free_energy);

private:
  double T;
  double kappa;
};

template <enum smearing_type SMEARING_TYPE>
template <class mem_t,
          class x_t,
          class e_t,
          class f_t,
          class hx_t,
          class zxp_t,
          class zetap_t,
          class ul_t,
          class op_t,
          class prec_t,
          class F>
auto
descent_direction<SMEARING_TYPE>::conjugated(const mem_t& memspc,
                                             double fr_old,
                                             const mvector<x_t>& X,
                                             const mvector<e_t>& en,
                                             const mvector<f_t>& fn,
                                             const mvector<hx_t>& hx,
                                             const mvector<zxp_t>& zxp,
                                             const mvector<zetap_t>& zetap,
                                             const mvector<ul_t>& ul,
                                             const mvector<double>& wk,
                                             op_t&& S,
                                             prec_t&& P,
                                             F&& free_energy)
{
  double mo = free_energy.occupancy();
  /* always executed on CPU */
  double dFdmu = GradEtaHelper::dFdmu(free_energy.get_ek(), en, fn, wk, mo);
  double sumfn = GradEtaHelper::dmu_deta(fn, wk, mo);

  auto commk = wk.commk();

  descent_direction_impl<mem_t, SMEARING_TYPE> functor(memspc, dFdmu, sumfn, T, kappa, mo);

  auto res = eval_threaded(tapply_async(functor, X, en, fn, hx, zxp, zetap, ul, S, P, wk));

  auto ures = unzip(res);

  double fr = sum(std::get<0>(ures), commk);

  double gamma = fr / fr_old;

  Logger::GetInstance() << " CG gamma " << std::setprecision(3) << gamma << "\n";

  auto delta_x = std::get<1>(ures);
  auto delta_eta = std::get<2>(ures);

  auto z_x = std::get<3>(ures);
  auto z_eta = std::get<4>(ures);
  double slope_zp = sum(std::get<5>(ures), commk);

  /* this is tr{<Z|g>} = tr{<Δ + γ*Z(n-1)|g>} = tr{<Δ |g>} + γ * tr{<Z(n-1)|g>}
   *            ^                                   ^                  ^
   *          slope  =                             fr    + γ *     slope_zp
   */
  double slope = fr + gamma * slope_zp;

  eval_threaded(
      // note: this operation is in-place and overwrite z_x, z_eta
      tapply_async(
          [gamma](auto delta_x, auto delta_eta, auto z_x, auto z_eta) {
            // Z <- gamma*Z + delta
            (void)add(z_x, delta_x, 1.0, gamma);
            (void)add(z_eta, delta_eta, 1.0, gamma);
            return "void";
          },
          delta_x,
          delta_eta,
          z_x,
          z_eta));

  return std::make_tuple(fr, slope, z_x, z_eta);
}

template <enum smearing_type SMEARING_TYPE>
template <class mem_t,
          class x_t,
          class e_t,
          class f_t,
          class hx_t,
          class op_t,
          class prec_t,
          class F>
std::tuple<double, mvector<to_layout_left_t<x_t>>, mvector<to_layout_left_t<x_t>>>
descent_direction<SMEARING_TYPE>::restarted(const mem_t& memspc,
                                            const mvector<x_t>& X,
                                            const mvector<e_t>& en,
                                            const mvector<f_t>& fn,
                                            const mvector<hx_t>& hx,
                                            const mvector<double>& wk,
                                            op_t&& S,
                                            prec_t&& P,
                                            F&& free_energy)
{
  double mo = free_energy.occupancy();
  double dFdmu = GradEtaHelper::dFdmu(free_energy.get_ek(), en, fn, wk, mo);
  double sumfn = GradEtaHelper::dmu_deta(fn, wk, mo);

  auto commk = wk.commk();

  descent_direction_impl<mem_t, SMEARING_TYPE> functor(memspc, dFdmu, sumfn, T, kappa, mo);

  auto res = eval_threaded(tapply_async(functor, X, en, fn, hx, S, P, wk));
  auto ures = unzip(res);

  double fr = sum(std::get<0>(ures), commk);
  auto z_x = std::get<1>(ures);
  auto z_eta = std::get<2>(ures);

  return std::make_tuple(fr, z_x, z_eta);
}


}  // namespace nlcglib
