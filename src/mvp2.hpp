#pragma once

#include "la/lapack.hpp"
#include "la/utils.hpp"
#include "mpi/communicator.hpp"

namespace nlcglib {

namespace local {

struct lmult
{
  template<class x_t, class hx_t, class prec_t>
  to_layout_left_t<std::remove_reference_t<x_t>>
  operator()(x_t&& x, hx_t&& hx, prec_t&& prec)
  {
    // Lagrange multipliers
    // compute ll = (xkx)^{-1} @ xKhx
    auto xkx = inner_()(x, prec(x));
    auto xkhx = inner_()(x, prec(hx));
    auto khx = prec(hx);
    solve_sym(xkx, xkhx);
    auto ll = xkhx;
    // X @ ll
    auto xll = transform_alloc(x, ll);
    return xll;
  }
};

struct gradx
{
  template <class x_t, class hx_t, class fn_t, class ll_t, class wk_t>
  to_layout_left_t<std::remove_reference_t<x_t>>
  operator()(x_t&& x, hx_t&& hx, fn_t&& fn, ll_t&& xll, wk_t&& wk)
  {
    auto g_x = empty_like()(x);
    scale(g_x, hx, fn, wk);
    add(g_x, eval(xll), -wk);
    return g_x;
  }
};

struct precondgx
{
  template <class x_t, class hx_t, class prec_t, class ll_t>
  to_layout_left_t<std::remove_reference_t<x_t>>
  operator()(x_t&& x, hx_t&& hx, prec_t&& prec, ll_t&& xll)
  {
    auto delta_x = empty_like()(x);
    // delta_x <- -hx
    add(delta_x, hx, -1.0);
    // delta_x <- += X @ ll
    add(delta_x, eval(xll), 1.0);
    prec.apply_in_place(delta_x);
    return delta_x;
  }
};

struct rotatex
{
  template <class x_t, class u_t>
  to_layout_left_t<std::remove_reference_t<x_t>>
  operator()(x_t&& x, u_t&& u)
  {
    return transform_alloc(x, eval(u));
  }
};

struct rotateeta
{
  template<class eta_t, class u_t>
  to_layout_left_t<std::remove_reference_t<eta_t>>
  operator()(eta_t&& eta, u_t&& u)
  {
    auto etau = transform_alloc(eta, eval(u));
    return inner_()(eval(u), etau);
  }
};

struct slope
{
  template<class gx_t, class zx_t, class ge_t, class ze_t>
  Kokkos::complex<double>
  operator()(gx_t&& gx, zx_t&& zx, ge_t&& geta, ze_t&& zeta)
  {
    auto slope_eta = innerh_tr()(eval(geta), eval(zeta));
    auto slope = 2 * innerh_tr()(gx, zx) + slope_eta;
    return slope;
  }
};

struct conjugatex
{
  conjugatex(double gamma) : gamma(gamma) {}

  /**
   * Overwrites zxp
   */
  template<class dx_t, class zxp_t, class x_t>
  to_layout_left_t<std::remove_reference_t<zxp_t>>
  operator()(dx_t&& dx, zxp_t&& zxp, x_t&& x)
  {
    // Zxp needs orthogonality updated
    auto tmp = inner_()(x, zxp);
    // corr = X @ X^H @ Z_X^{(i-1)}
    auto corr = transform_alloc(x, tmp);
    add(zxp, corr, -gamma, gamma);
    add(zxp, dx, 1);
    return zxp;
  }

  double gamma;
};

struct conjugateeta
{
  conjugateeta(double gamma) : gamma(gamma) {}

  /**
   * Overwrites zep
   */
  template <class deta_t, class zetap_t>
  to_layout_left_t<std::remove_reference_t<zetap_t>>
  operator()(deta_t&& deta, zetap_t&& zep)
  {
    add(zep, deta, 1.0, gamma);
    return zep;
  }

  double gamma;
};

}  // local

/// Lagrange multipliers
template <class X_t, class Hx_t, class Prec_t>
auto
lagrange_multipliers(const X_t& X, const Hx_t& Hx, const Prec_t& Prec)
{
  return tapply_async(local::lmult(), X, Hx, Prec);
}

/// gradient
template <class X_t, class Hx_t, class fn_t, class ll_t, class wk_t>
auto gradX(const X_t& X, const Hx_t& Hx, const fn_t& fn, const ll_t& Xll, const wk_t& wk)
{
  return tapply_async(local::gradx(), X, Hx, fn, Xll, wk);
}

/// delta_x
template<class X_t, class Hx_t, class Prec_t, class ll_t>
auto precondGradX(const X_t& X, const Hx_t& Hx, const Prec_t& Prec, const ll_t& Xll)
{
  return tapply_async(local::precondgx(), X, Hx, Prec, Xll);
}

/// apply subspace rotation on X
template<class X_t, class U_t>
auto rotateX(const X_t& X, const U_t& U)
{
  return tapply_async(local::rotatex(), X, U);
}

/// apply subpsace rotation on eta
template<class Eta_t, class U_t>
auto rotateEta(const Eta_t& Eta, const U_t& U)
{
  return tapply_async(local::rotateeta(), Eta, U);
}

template <class gx_t, class zx_t, class ge_t, class ze_t>
double
compute_slope(const gx_t& gx, const zx_t& zx, const ge_t& geta, ze_t& zeta, const Communicator& commk)
{
  return sum(eval_threaded(tapply(local::slope(), gx, zx, geta, zeta)), commk) .real();
}

template <class dx_t, class zxp_t, class x_t>
auto conjugatex(dx_t&& dx, zxp_t&& zxp, x_t&& x, double gamma)
{
  return tapply_async(local::conjugatex(gamma), dx, zxp, x);
}

template<class deta_t, class zetap_t>
auto conjugateeta(deta_t&& deta, zetap_t&& zep, double gamma)
{
  return tapply_async(local::conjugateeta(gamma), deta, zep);
}

}  // nlcglib
