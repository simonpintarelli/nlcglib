#pragma once

#include <Kokkos_Core.hpp>

#include "la/utils.hpp"
#include "la/lapack.hpp"

namespace nlcglib {

template <class T>
struct pp
{
};

namespace local {
struct advance_eta
{
  advance_eta(double t) : t(t) {}

  template <class eta_t, class d_eta_t>
  KokkosDVector<Kokkos::complex<double>**, nlcglib::SlabLayoutV, Kokkos::LayoutLeft,
                typename std::remove_reference_t<eta_t>::storage_t::memory_space>
  operator()(eta_t&& eta, d_eta_t&& d_eta)
  {
    auto eta_next = empty_like()(d_eta);
    deep_copy(eta_next, eta);
    add(eta_next, eval(d_eta), t);
    return eta_next;
  }

  double t;
};

struct eigvals_and_vectors
{
  template<class eta_t>
  std::tuple<Kokkos::View<double*, typename eta_t::storage_t::memory_space>, to_layout_left_t<eta_t>>
  operator()(const eta_t& eta)
  {
    auto Ul = empty_like()(eta);
    using memspace = typename decltype(Ul)::storage_t::memory_space;
    Kokkos::View<double*, memspace> ek("eigvals, eta", Ul.map().ncols());
    eigh(Ul, ek, eta);
    return std::make_tuple(ek, Ul);
  }
};

struct advance_x
{
  advance_x(double t) : t(t) {}

  template<class x_t, class dx_t, class ul_t>
  to_layout_left_t<std::remove_reference_t<dx_t>>
  // auto
  operator()(x_t&& x, dx_t&& dx, ul_t&& ul)
  {
    // pp<to_layout_left_t<dx_t>>::foo;
    auto x_next = empty_like()(x);
    deep_copy(x_next, x);
    add(x_next, eval(dx), t);
    x_next = loewdin(x_next);
    return transform_alloc(x_next, eval(ul));
  }

  double t;
};

}  // local

template <class energy_t, class X_t, class eta_t, class g_x_t, class g_eta_t>
auto
geodesic(energy_t& F, X_t& X, const eta_t& eta, const g_x_t& g_x, const g_eta_t& g_eta, double t)
{
  // compute eta_next <- eta + t* g_eta
  auto eta_next = tapply_async(local::advance_eta(t), eta, g_eta);
  // get eigenvalues and eigenvectors of next eta
  auto ek_Ul = tapply_async(local::eigvals_and_vectors(), eta_next);
  // grab results
  auto ek = eval_threaded(tapply([](auto ek_ul) { return std::get<0>(eval(ek_ul));}, ek_Ul));
  auto Ul = eval_threaded(tapply([](auto ek_ul) { return std::get<1>(eval(ek_ul));}, ek_Ul));
  // obtain occupation numbers
  auto fn = F.get_smearing().fn(ek);

  // X <- ortho((X + t*g_X) @ Ul)
  auto x_next = tapply_async(local::advance_x(t), X, g_x, Ul);

  F.compute(eval_threaded(x_next), fn);

  return std::make_tuple(ek, Ul);
}

}  // nlcglib
