#pragma once

#include "descent_direction_impl.hpp"
#include "la/dvector.hpp"
#include "mvp2.hpp"
#include "pseudo_hamiltonian/grad_eta.hpp"
#include "utils/profile.hpp"

namespace nlcglib {
/// Restart (preconditoned) CG
template <class memspace_t, enum smearing_type smearing_t>
class descent_direction_restart : public descent_direction_base<memspace_t, smearing_t>
{
public:
  descent_direction_restart(const memspace_t& memspc,
                            double mu,
                            double dFdmu,
                            double sumfn,
                            double T,
                            double kappa,
                            double mo)
      : descent_direction_base<memspace_t, smearing_t>::descent_direction_base(
            memspc, mu, dFdmu, sumfn, T, kappa, mo)
  {
  }

  /* interface routine, does memory transfers if needed, for CG restart (steepest descent) */
  template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
  auto operator()(x_t&& X, e_t&& en, f_t&& fn, hx_t&& hx, op_t&& S, prec_t&& P, double wk);

private:
  /* CG restart gradients */
  template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
  std::tuple<slope_t, to_layout_left_t<x_t>, to_layout_left_t<x_t>> exec_spc(
      x_t&& x, e_t&& e, f_t&& f, hx_t&& hx, op_t&& s, prec_t&& p, double wk);

private:
  using descent_direction_base<memspace_t, smearing_t>::memspc;
  using descent_direction_base<memspace_t, smearing_t>::mu;
  using descent_direction_base<memspace_t, smearing_t>::dFdmu;
  using descent_direction_base<memspace_t, smearing_t>::sumfn;
  using descent_direction_base<memspace_t, smearing_t>::T;
  using descent_direction_base<memspace_t, smearing_t>::kappa;
  using descent_direction_base<memspace_t, smearing_t>::mo;
};


/// restarted
template <class memspc_t, enum smearing_type smearing_t>
template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
std::tuple<slope_t, to_layout_left_t<x_t>, to_layout_left_t<x_t>>
descent_direction_restart<memspc_t, smearing_t>::exec_spc(
    x_t&& x, e_t&& e, f_t&& f, hx_t&& hx, op_t&& s, prec_t&& p, double wk)
{
  PROFILE("exec");
  auto sx = s(x);
  auto llm = local::lmult()(x, sx, hx, p);
  auto gx = local::gradx()(sx, hx, f, llm, wk);
  auto delta_x = local::precondgx_us()(sx, hx, p, llm);
  auto hij = inner_()(x, hx, wk);

  GradEta<smearing_t> grad_eta(this->T, this->kappa);
  auto g_eta = grad_eta.g_eta(hij, mu, wk, e, f, this->sumfn, this->dFdmu, this->mo);
  auto delta_eta = _delta_eta(this->kappa)(hij, e, wk);

  double fr_x = 2 * innerh_tr()(gx, delta_x).real();
  double fr_eta = innerh_tr()(g_eta, delta_eta).real();
  slope_t fr{.x = fr_x, .eta = fr_eta};

  return std::make_tuple(fr, delta_x, delta_eta);
}


template <class memspc_t, enum smearing_type smearing_t>
template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
auto
descent_direction_restart<memspc_t, smearing_t>::operator()(
    x_t&& X_h, e_t&& en_h, f_t&& fn_h, hx_t&& hx_h, op_t&& S, prec_t&& P, double wk)
{
  PROFILE("nlcglib::cg::restart");
  // namespace of input an result
  using input_memspc = typename std::remove_reference_t<x_t>::storage_t::memory_space;

  auto X = create_mirror_view_and_copy(memspc, X_h);
  auto en = Kokkos::create_mirror_view_and_copy(memspc, en_h);
  auto fn = Kokkos::create_mirror_view_and_copy(memspc, fn_h);
  auto HX = create_mirror_view_and_copy(memspc, hx_h);

  auto [fr, delta_x, delta_eta] = this->exec_spc(X, en, fn, HX, S, P, wk);

  // // steepest descent vars
  // double fr = std::get<0>(res);
  // auto delta_x = std::get<1>(res);
  // auto delta_eta = std::get<2>(res);

  // copy Δ to host
  auto delta_x_h = create_mirror_view_and_copy(input_memspc(), delta_x);
  auto delta_eta_h = create_mirror_view_and_copy(input_memspc(), delta_eta);

  /// return slopes and Δ, Z (host memeory)
  return std::make_tuple(fr, delta_x_h, delta_eta_h);
}


}  // namespace nlcglib
