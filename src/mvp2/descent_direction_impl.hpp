#pragma once

#include <Kokkos_Core.hpp>
#include "la/dvector.hpp"
#include "mpi/communicator.hpp"
#include "mvp2.hpp"
#include "pseudo_hamiltonian/grad_eta.hpp"


namespace nlcglib {

struct slope_t
{
  double x, eta;
};

inline slope_t
operator+(const slope_t& lhs, const slope_t& rhs)
{
  return slope_t{.x = lhs.x + rhs.x, .eta = lhs.eta + rhs.eta};
}

inline slope_t
operator*(const slope_t& slope, double f)
{
  return slope_t{.x = slope.x * f, .eta = slope.eta * f};
}

inline slope_t
operator*(double f, const slope_t& slope)
{
  return slope * f;
}

inline slope_t
sum(const mvector<slope_t>& m_slope, const Communicator& comm)
{
  // sum locally
  slope_t slope{0, 0};
  for (auto& elem : m_slope) {
    slope = slope + elem.second;
  }
  // mpi reduction
  std::array<double, 2> x{slope.x, slope.eta};
  sum_inplace(x, comm);
  return slope_t{.x=x[0], .eta=x[1]};
}

/** Helper class for CG algorithm.
    @param memspace_t     execution memory space
    @param smearing_type  Smearing type
 */
template <class memspace_t, enum smearing_type smearing_t>
class descent_direction_impl
{
public:
  descent_direction_impl(const memspace_t& memspc,
                         double mu,
                         double dFdmu,
                         double sumfn,
                         double T,
                         double kappa,
                         double mo)
      : memspc(memspc)
      , mu(mu)
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
  template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
  auto operator()(x_t&& X, e_t&& en, f_t&& fn, hx_t&& hx, op_t&& S, prec_t&& P, double wk);

private:
  /* computations in execution space for conjugated gradients  */
  template <class x_t,
            class e_t,
            class f_t,
            class hx_t,
            class op_t,
            class prec_t,
            class zxp_t,
            class zetap_t,
            class ul_t>
  std::tuple<slope_t,
             to_layout_left_t<x_t>,
             to_layout_left_t<zetap_t>,
             to_layout_left_t<x_t>,
             to_layout_left_t<zetap_t>,
             slope_t>
  exec_spc(x_t&& x,
           e_t&& e,
           f_t&& f,
           hx_t&& hx,
           op_t&& s,
           prec_t&& p,
           zxp_t&& zxp,
           zetap_t&& zetap,
           ul_t&& ul,
           double wk);

  /* CG conjugated direction gradients */
  template <class x_t, class sx_t, class zxp_t, class zetap_t, class ul_t, class gx_t, class geta_t>
  std::tuple<slope_t, to_layout_left_t<zxp_t>, to_layout_left_t<zetap_t>> exec_conjugate(
      x_t&& x, sx_t&& sx, zxp_t&& zxp, zetap_t&& zetap, ul_t&& ul, gx_t&& gx, geta_t&& geta);


  /* CG restart gradients */
  template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
  std::tuple<slope_t, to_layout_left_t<x_t>, to_layout_left_t<x_t>> exec_spc(
      x_t&& x, e_t&& e, f_t&& f, hx_t&& hx, op_t&& s, prec_t&& p, double wk);

private:
  memspace_t memspc;
  double mu;
  double dFdmu;
  double sumfn;
  double T;
  double kappa;
  double mo;
};

/// conjugated
template <class memspc_t, enum smearing_type smearing_t>
template <class x_t,
          class e_t,
          class f_t,
          class hx_t,
          class op_t,
          class prec_t,
          class zxp_t,
          class zetap_t,
          class ul_t>
std::tuple<slope_t,
           to_layout_left_t<x_t>,
           to_layout_left_t<zetap_t>,
           to_layout_left_t<x_t>,
           to_layout_left_t<zetap_t>,
           slope_t>
descent_direction_impl<memspc_t, smearing_t>::exec_spc(x_t&& x,
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

  GradEta<smearing_t> grad_eta(this->T, this->kappa);
  auto g_eta = grad_eta.g_eta(hij, mu, wk, e, f, this->sumfn, this->dFdmu, this->mo);
  auto delta_eta = _delta_eta(this->kappa)(hij, e, wk);

  double fr_x = 2 * innerh_tr()(gx, delta_x).real();
  double fr_eta = innerh_tr()(g_eta, delta_eta).real();
  slope_t fr{.x = fr_x, .eta = fr_eta};

  // CG contributions
  auto [slope_zp, z_x, z_eta] = this->exec_conjugate(x, sx, zxp, zetap, ul, gx, g_eta);

  return std::make_tuple(fr, delta_x, delta_eta, z_x, z_eta, slope_zp);
}


template <class memspc_t, enum smearing_type smearing_t>
template <class x_t, class sx_t, class zxp_t, class zetap_t, class ul_t, class gx_t, class geta_t>
std::tuple<slope_t, to_layout_left_t<zxp_t>, to_layout_left_t<zetap_t>>
descent_direction_impl<memspc_t, smearing_t>::exec_conjugate(
    x_t&& x, sx_t&& sx, zxp_t&& zxp, zetap_t&& zetap, ul_t&& ul, gx_t&& gx, geta_t&& geta)
{
  auto zx_tmp = local::rotatex()(zxp, ul);
  auto zeta = local::rotateeta()(zetap, ul);

  // apply Lagrange multipliers to zx
  auto zx = local::conjugatex()(zx_tmp, x, sx);

  auto slope_x_loc = 2 * innerh_tr()(zx, gx).real();
  auto slope_eta_loc = innerh_tr()(zeta, geta).real();
  slope_t slope_loc{.x = slope_x_loc, .eta = slope_eta_loc};

  return std::make_tuple(slope_loc, zx, zeta);
}

/// restarted
template <class memspc_t, enum smearing_type smearing_t>
template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
std::tuple<slope_t, to_layout_left_t<x_t>, to_layout_left_t<x_t>>
descent_direction_impl<memspc_t, smearing_t>::exec_spc(
    x_t&& x, e_t&& e, f_t&& f, hx_t&& hx, op_t&& s, prec_t&& p, double wk)
{
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
descent_direction_impl<memspc_t, smearing_t>::operator()(x_t&& X_h,
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
  // namespace of input an result
  using input_memspc = typename std::remove_reference_t<x_t>::storage_t::memory_space;

  auto X = create_mirror_view_and_copy(memspc, X_h);
  auto en = Kokkos::create_mirror_view_and_copy(memspc, en_h);
  auto fn = Kokkos::create_mirror_view_and_copy(memspc, fn_h);
  auto HX = create_mirror_view_and_copy(memspc, hx_h);

  // previous search directions
  auto ZXp = create_mirror_view_and_copy(memspc, zxp_h);
  auto Zetap = create_mirror_view_and_copy(memspc, zetap_h);
  auto ul = create_mirror_view_and_copy(memspc, ul_h);

  // auto res = this->exec_spc(X, en, fn, HX, S, P, ZXp, Zetap, ul, wk);
  auto [fr, delta_x, delta_eta, z_x, z_eta, slope_zp] =
      this->exec_spc(X, en, fn, HX, S, P, ZXp, Zetap, ul, wk);

  // // steepest descent vars
  // double fr = std::get<0>(res);
  // auto delta_x = std::get<1>(res);
  // auto delta_eta = std::get<2>(res);

  // // CG vars
  // auto z_x = std::get<3>(res);
  // auto z_eta = std::get<4>(res);
  // double slope_zp = std::get<5>(res);

  // copy Δ to host
  auto delta_x_h = create_mirror_view_and_copy(input_memspc(), delta_x);
  auto delta_eta_h = create_mirror_view_and_copy(input_memspc(), delta_eta);

  // copy Z to host
  auto z_x_h = create_mirror_view_and_copy(input_memspc(), z_x);
  auto z_eta_h = create_mirror_view_and_copy(input_memspc(), z_eta);

  /// return slopes and Δ, Z (host memeory)
  return std::make_tuple(fr, delta_x_h, delta_eta_h, z_x_h, z_eta_h, slope_zp);
}

template <class memspc_t, enum smearing_type smearing_t>
template <class x_t, class e_t, class f_t, class hx_t, class op_t, class prec_t>
auto
descent_direction_impl<memspc_t, smearing_t>::operator()(
    x_t&& X_h, e_t&& en_h, f_t&& fn_h, hx_t&& hx_h, op_t&& S, prec_t&& P, double wk)
{
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
