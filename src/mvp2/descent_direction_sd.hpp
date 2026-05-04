#pragma once

#include "descent_direction_impl.hpp"
#include "la/dvector.hpp"
#include "mvp2.hpp"
#include "pseudo_hamiltonian/grad_eta.hpp"
#include "utils/profile.hpp"

// (unpreconditoned) steepest descent
namespace nlcglib {
template <class memspace_t, enum smearing_type smearing_t>
class descent_direction_sd : public descent_direction_base<memspace_t, smearing_t>
{
public:
  descent_direction_sd(const memspace_t& memspc,
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
  template <class x_t, class e_t, class f_t, class hx_t, class op1_t, class op2_t>
  auto operator()(x_t&& X, e_t&& en, f_t&& fn, hx_t&& hx, op1_t&& S, op2_t&& Sinv, double wk);

private:
  /* CG restart gradients */
  template <class x_t, class e_t, class f_t, class hx_t, class op1_t, class op2_t>
  std::tuple<slope_t, to_layout_left_t<x_t>, to_layout_left_t<x_t>> exec_spc(
      x_t&& x, e_t&& e, f_t&& f, hx_t&& hx, op1_t&& s, op2_t&& sinv, double wk);

private:
  using descent_direction_base<memspace_t, smearing_t>::memspc;
  using descent_direction_base<memspace_t, smearing_t>::mu;
  using descent_direction_base<memspace_t, smearing_t>::dFdmu;
  using descent_direction_base<memspace_t, smearing_t>::sumfn;
  using descent_direction_base<memspace_t, smearing_t>::T;
  using descent_direction_base<memspace_t, smearing_t>::kappa;
  using descent_direction_base<memspace_t, smearing_t>::mo;
};

template <class memspc_t, enum smearing_type smearing_t>
template <class x_t, class e_t, class f_t, class hx_t, class op1_t, class op2_t>
std::tuple<slope_t, to_layout_left_t<x_t>, to_layout_left_t<x_t>>
descent_direction_sd<memspc_t, smearing_t>::exec_spc(
    x_t&& x, e_t&& e, f_t&& f, hx_t&& hx, op1_t&& s, op2_t&& sinv, double wk)
{
  PROFILE("exec");
  auto hij = inner_()(x, hx, wk);
  auto cgx = sinv(hx);
  auto sx = s(x);
  auto sxhij = transform_alloc(sx, hij, 1.0);
  // compute contravairant gradient Sinv(Hx) - X * Hij
  // 10.1016/j.cpc.2005.07.011 Eq 3.
  // cgx <- 1.0*cgx - x*hij
  transform(cgx, Kokkos::complex(1.0), Kokkos::complex(-1.0), x, hij);
  // covariant gradient
  auto gx = local::gradx()(sx, hx, f, sxhij, wk);

  GradEta<smearing_t> grad_eta(this->T, this->kappa);
  auto g_eta = grad_eta.g_eta(hij, mu, wk, e, f, this->sumfn, this->dFdmu, this->mo);

  double fr_x = -2.0 * innerh_tr()(gx, cgx).real();
  double fr_eta = -1.0 * innerh_tr()(g_eta, g_eta).real();
  slope_t fr{.x = fr_x, .eta = fr_eta};

  // apply minus 1
  scale(cgx, cgx, -1);
  scale(g_eta, g_eta, -1);
  return std::make_tuple(fr, gx, g_eta);
}


template <class memspc_t, enum smearing_type smearing_t>
template <class x_t, class e_t, class f_t, class hx_t, class op1_t, class op2_t>
auto
descent_direction_sd<memspc_t, smearing_t>::operator()(
    x_t&& X_h, e_t&& en_h, f_t&& fn_h, hx_t&& hx_h, op1_t&& S, op2_t&& Sinv, double wk)
{
  PROFILE("nlcglib::cg::steepest_descent");
  // namespace of input an result
  using input_memspc = typename std::remove_reference_t<x_t>::storage_t::memory_space;

  auto X = create_mirror_view_and_copy(memspc, X_h);
  auto en = Kokkos::create_mirror_view_and_copy(memspc, en_h);
  auto fn = Kokkos::create_mirror_view_and_copy(memspc, fn_h);
  auto HX = create_mirror_view_and_copy(memspc, hx_h);

  auto [fr, delta_x, delta_eta] = this->exec_spc(X, en, fn, HX, S, Sinv, /* P, */ wk);

  // copy Δ to host
  auto delta_x_h = create_mirror_view_and_copy(input_memspc(), delta_x);
  auto delta_eta_h = create_mirror_view_and_copy(input_memspc(), delta_eta);

  /// return slopes and Δ, Z (host memeory)
  return std::make_tuple(fr, delta_x_h, delta_eta_h);
}


}  // namespace nlcglib
