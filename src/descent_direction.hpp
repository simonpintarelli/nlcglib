#pragma once

#include <Kokkos_Core.hpp>
#include "la/dvector.hpp"
#include "la/mvector.hpp"
#include "mvp2.hpp"
#include "pseudo_hamiltonian/grad_eta.hpp"


namespace nlcglib {

template <class memspace_t>
class descent_direction_impl
{
public:
  descent_direction_impl(
      const memspace_t& memspc, double dFdmu, double sumfn, double T, double kappa, double mo)
      : memspc(memspc)
      , dFdmu(dFdmu)
      , sumfn(sumfn)
      , T(T)
      , kappa(kappa)
      , mo(mo)
  {
  }

  /* interface routine, does memory transfers if needed */
  template <class x_t,
            class e_t,
            class f_t,
            class hx_t,
            class zxp_t,
            class zetap_t,
            class ul_t,
            class op_t,
            class prec_t>
  auto operator()(x_t&& X,
                  e_t&& en,
                  f_t&& fn,
                  hx_t&& hx,
                  zxp_t&& zxp,
                  zetap_t&& zetap,
                  ul_t&& ul,
                  op_t&& S,
                  prec_t&& P,
                  double wk);

  /* interface routine, does memory transfers if needed, for CG restart (steepest descent) */
  template <class x_t,
            class e_t,
            class f_t,
            class hx_t,
            class op_t,
            class prec_t>
  auto operator()(x_t&& X,
                  e_t&& en,
                  f_t&& fn,
                  hx_t&& hx,
                  op_t&& S,
                  prec_t&& P,
                  double wk);

  template <class x_t,
            class e_t,
            class f_t,
            class hx_t,
            class op_t,
            class prec_t,
            class zxp_t,
            class zetap_t,
            class ul_t>
  auto exec_spc(x_t&& x,
                e_t&& e,
                f_t&& f,
                hx_t&& hx,
                op_t&& s,
                prec_t&& p,
                zxp_t&& zxp,
                zetap_t&& zetap,
                ul_t&& ul,
                double wk);

  template <class x_t, class sx_t, class zxp_t, class zetap_t, class ul_t, class gx_t, class geta_t>
  auto exec_conjugate(
      x_t&& x, sx_t&& sx, zxp_t&& zxp, zetap_t&& zetap, ul_t&& ul, gx_t&& gx, geta_t&& geta);

  template <class x_t,
            class e_t,
            class f_t,
            class hx_t,
            class op_t,
            class prec_t>
  auto exec_spc(x_t&& x,
                e_t&& e,
                f_t&& f,
                hx_t&& hx,
                op_t&& s,
                prec_t&& p,
                double wk);

private:
  memspace_t memspc;
  double dFdmu;
  double sumfn;
  double T;
  double kappa;
  double mo;
};


template <class memspc_t>
template <class x_t,
          class e_t,
          class f_t,
          class hx_t,
          class op_t,
          class prec_t,
          class zxp_t,
          class zetap_t,
          class ul_t>
auto
descent_direction_impl<memspc_t>::exec_spc(x_t&& x,
                                           e_t&& e,
                                           f_t&& f,
                                           hx_t&& hx,
                                           op_t&& s,
                                           prec_t&& p,
                                           zxp_t&& zxp,
                                           zetap_t&& zetap,
                                           ul_t&& ul,
                                           double wk)
{
  auto sx = s(x);
  auto llm = local::lmult()(x, sx, hx, p);
  auto gx = local::gradx()(sx, hx, f, llm, wk);
  auto delta_x = local::precondgx_us()(sx, hx, p, llm);
  auto hij = inner_()(x, hx, wk);
  // // std::cout << dFdmu << ", " << sumfn << "\n";

  GradEta grad_eta(this->T, this->kappa);
  auto g_eta = grad_eta.g_eta(hij, wk, e, f, this->sumfn, this->dFdmu, this->mo);
  auto delta_eta = GradEta::_delta_eta(this->kappa)(hij, e, wk);

  double fr_x = 2 * innerh_tr()(gx, delta_x).real();
  double fr_eta = innerh_tr()(g_eta, delta_eta).real();
  double fr = fr_x + fr_eta;

  // CG contributions
  auto res_conj = this->exec_conjugate(x, sx, zxp, zetap, ul, gx, g_eta);
  double slope_zp = std::get<0>(res_conj);
  auto z_x = std::get<1>(res_conj);
  auto z_eta = std::get<2>(res_conj);

  return std::make_tuple(fr, delta_x, delta_eta, z_x, z_eta, slope_zp);
}


template <class memspc_t>
template <class x_t, class sx_t, class zxp_t, class zetap_t, class ul_t, class gx_t, class geta_t>
auto
descent_direction_impl<memspc_t>::exec_conjugate(
    x_t&& x, sx_t&& sx, zxp_t&& zxp, zetap_t&& zetap, ul_t&& ul, gx_t&& gx, geta_t&& geta)
{
  auto zx_tmp = local::rotatex()(zxp, ul);
  auto zeta = local::rotateeta()(zetap, ul);

  // apply Lagrange multipliers to zx
  auto zx = local::conjugatex()(zx_tmp, x, sx);

  auto slope_x_loc = 2 * innerh_tr()(zx, gx).real();
  auto slope_eta_loc = innerh_tr()(zeta, geta).real();
  double slope_loc = slope_x_loc + slope_eta_loc;

  return std::make_tuple(slope_loc, zx, zeta);
}


template <class memspc_t>
template <class x_t,
          class e_t,
          class f_t,
          class hx_t,
          class op_t,
          class prec_t>
auto
descent_direction_impl<memspc_t>::exec_spc(x_t&& x,
                                           e_t&& e,
                                           f_t&& f,
                                           hx_t&& hx,
                                           op_t&& s,
                                           prec_t&& p,
                                           double wk)
{
  auto sx = s(x);
  auto llm = local::lmult()(x, sx, hx, p);
  auto gx = local::gradx()(sx, hx, f, llm, wk);
  auto delta_x = local::precondgx_us()(sx, hx, p, llm);
  auto hij = inner_()(x, hx, wk);

  GradEta grad_eta(this->T, this->kappa);
  auto g_eta = grad_eta.g_eta(hij, wk, e, f, this->sumfn, this->dFdmu, this->mo);
  auto delta_eta = GradEta::_delta_eta(this->kappa)(hij, e, wk);

  double fr_x = 2 * innerh_tr()(gx, delta_x).real();
  double fr_eta = innerh_tr()(g_eta, delta_eta).real();
  double fr = fr_x + fr_eta;

  return std::make_tuple(fr, delta_x, delta_eta);

}


template <class memspc_t>
template <class x_t,
          class e_t,
          class f_t,
          class hx_t,
          class zxp_t,
          class zetap_t,
          class ul_t,
          class op_t,
          class prec_t>
auto
descent_direction_impl<memspc_t>::operator()(x_t&& X_h,
                                             e_t&& en_h,
                                             f_t&& fn_h,
                                             hx_t&& hx_h,
                                             zxp_t&& zxp_h,
                                             zetap_t&& zetap_h,
                                             ul_t&& ul_h,
                                             op_t&& S,
                                             prec_t&& P,
                                             double wk)
{
  auto X = create_mirror_view_and_copy(memspc, X_h);
  auto en = Kokkos::create_mirror_view_and_copy(memspc, en_h);
  auto fn = Kokkos::create_mirror_view_and_copy(memspc, fn_h);
  auto HX = create_mirror_view_and_copy(memspc, hx_h);

  // previous search directions
  auto ZXp = create_mirror_view_and_copy(memspc, zxp_h);
  auto Zetap = create_mirror_view_and_copy(memspc, zetap_h);
  auto ul = create_mirror_view_and_copy(memspc, ul_h);

  auto res = this->exec_spc(X, en, fn, HX, S, P, ZXp, Zetap, ul, wk);

  // steepest descent vars
  double fr = std::get<0>(res);
  auto delta_x = std::get<1>(res);
  auto delta_eta = std::get<2>(res);

  // CG vars
  auto z_x = std::get<3>(res);
  auto z_eta = std::get<4>(res);
  double slope_zp = std::get<5>(res);

  // copy Δ to host
  auto delta_x_h = create_mirror_view_and_copy(Kokkos::HostSpace(), delta_x);
  auto delta_eta_h = create_mirror_view_and_copy(Kokkos::HostSpace(), delta_eta);

  // copy Z to host
  auto z_x_h = create_mirror_view_and_copy(Kokkos::HostSpace(), z_x);
  auto z_eta_h = create_mirror_view_and_copy(Kokkos::HostSpace(), z_eta);

  /// return slopes and Δ, Z (host memeory)
  return std::make_tuple(fr, delta_x_h, delta_eta_h, z_x_h, z_eta_h, slope_zp);
}

template <class memspc_t>
template <class x_t,
          class e_t,
          class f_t,
          class hx_t,
          class op_t,
          class prec_t>
auto
descent_direction_impl<memspc_t>::operator()(x_t&& X_h,
                                             e_t&& en_h,
                                             f_t&& fn_h,
                                             hx_t&& hx_h,
                                             op_t&& S,
                                             prec_t&& P,
                                             double wk)
{
  auto X = create_mirror_view_and_copy(memspc, X_h);
  auto en = Kokkos::create_mirror_view_and_copy(memspc, en_h);
  auto fn = Kokkos::create_mirror_view_and_copy(memspc, fn_h);
  auto HX = create_mirror_view_and_copy(memspc, hx_h);

  auto res = this->exec_spc(X, en, fn, HX, S, P, wk);

  // steepest descent vars
  double fr = std::get<0>(res);
  auto delta_x = std::get<1>(res);
  auto delta_eta = std::get<2>(res);

  // copy Δ to host
  auto delta_x_h = create_mirror_view_and_copy(Kokkos::HostSpace(), delta_x);
  auto delta_eta_h = create_mirror_view_and_copy(Kokkos::HostSpace(), delta_eta);

  /// return slopes and Δ, Z (host memeory)
  return std::make_tuple(fr, delta_x_h, delta_eta_h);
}



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
  template <class mem_t, class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t, class F>
  auto restarted(const mem_t& memspc,
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
    double dFdmu = GradEtaHelper::dFdmu(free_energy.get_ek(), en, fn, wk);
    double sumfn = GradEtaHelper::dmu_deta(fn, wk, mo);

    auto commk = wk.commk();

    descent_direction_impl<mem_t> functor(
        memspc, dFdmu, sumfn, T, kappa, mo);

    auto res = eval_threaded(tapply_async(functor, X, en, fn, hx, S, P, wk));
    auto ures = unzip(res);

    double fr = sum(std::get<0>(ures), commk);
    auto z_x = std::get<1>(ures);
    auto z_eta = std::get<2>(ures);

    return std::make_tuple(fr, z_x, z_eta);
  }

private:
  double T;
  double kappa;
};

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
descent_direction::conjugated(const mem_t& memspc,
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
  double dFdmu = GradEtaHelper::dFdmu(free_energy.get_ek(), en, fn, wk);
  double sumfn = GradEtaHelper::dmu_deta(fn, wk, mo);

  auto commk = wk.commk();

  descent_direction_impl<mem_t> functor(
      memspc, dFdmu, sumfn, T, kappa, mo);

  auto res = eval_threaded(tapply_async(functor, X, en, fn, hx, zxp, zetap, ul, S, P, wk));

  auto ures = unzip(res);

  double fr = sum(std::get<0>(ures), commk);

  double gamma = fr / fr_old;
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
}  // namespace nlcglib
