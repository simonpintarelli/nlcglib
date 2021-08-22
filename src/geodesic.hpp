#pragma once

#include <Kokkos_Core.hpp>

#include "la/utils.hpp"
#include "la/lapack.hpp"

namespace nlcglib {

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

template<class exec_space>
struct advance_x_eta
{
  // also needs overlap operator
  advance_x_eta(double t) : t(t) {}

  template<class x_t, class eta_t, class g_X_t, class g_eta_t, class S_t>
  void operator()(x_t&& X, eta_t&& eta, g_X_t&& g_X, g_eta_t&& g_eta, S_t&& S) {
    // Kokko
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

  // nc case
  template<class x_t, class dx_t, class ul_t>
  to_layout_left_t<std::remove_reference_t<dx_t>>
  operator()(x_t&& x, dx_t&& dx, ul_t&& ul)
  {
    // pp<to_layout_left_t<dx_t>>::foo;
    auto x_next = empty_like()(x);
    deep_copy(x_next, x);
    add(x_next, eval(dx), t);
    x_next = loewdin(x_next);
    return transform_alloc(x_next, eval(ul));
  }

  // ultra-soft case
  template <class x_t, class dx_t, class ul_t, class op_t>
  to_layout_left_t<std::remove_reference_t<dx_t>>
  operator()(x_t&& x, dx_t&& dx, ul_t&& ul, op_t&& s)
  {
    // pp<to_layout_left_t<dx_t>>::foo;
    auto x_next = empty_like()(x);
    deep_copy(x_next, x);
    add(x_next, eval(dx), t);
    x_next = loewdin(x_next, eval(s(x_next)));
    return transform_alloc(x_next, eval(ul));
  }

  double t;
};

}  // local

// template <class energy_t, class X_t, class eta_t, class g_x_t, class g_eta_t>
// auto
// geodesic(energy_t& F, X_t& X, const eta_t& eta, const g_x_t& g_x, const g_eta_t& g_eta, double t)
// {
//   // compute eta_next <- eta + t* g_eta
//   auto eta_next = tapply_async(local::advance_eta(t), eta, g_eta);
//   // get eigenvalues and eigenvectors of next eta
//   auto ek_Ul = tapply_async(local::eigvals_and_vectors(), eta_next);
//   // grab results
//   auto ek = eval_threaded(tapply([](auto ek_ul) { return std::get<0>(eval(ek_ul));}, ek_Ul));
//   auto Ul = eval_threaded(tapply([](auto ek_ul) { return std::get<1>(eval(ek_ul));}, ek_Ul));
//   // obtain occupation numbers
//   auto fn = F.get_smearing().fn(ek);

//   // X <- ortho((X + t*g_X) @ Ul)
//   auto x_next = tapply_async(local::advance_x(t), X, g_x, Ul);

//   F.compute(eval_threaded(x_next), fn);

//   return std::make_tuple(ek, Ul);
// }

// template <class energy_t, class X_t, class eta_t, class g_x_t, class g_eta_t, class Op_t>
// auto
// geodesic_us(energy_t& F, X_t& X, const eta_t& eta, const g_x_t& g_x, const g_eta_t& g_eta, const Op_t& S, double t)
// {
//   // compute eta_next <- eta + t* g_eta
//   auto eta_next = tapply_async(local::advance_eta(t), eta, g_eta);
//   // get eigenvalues and eigenvectors of next eta
//   auto ek_Ul = tapply_async(local::eigvals_and_vectors(), eta_next);
//   // grab results
//   auto ek = eval_threaded(tapply([](auto ek_ul) { return std::get<0>(eval(ek_ul)); }, ek_Ul));
//   auto Ul = eval_threaded(tapply([](auto ek_ul) { return std::get<1>(eval(ek_ul)); }, ek_Ul));
//   // obtain occupation numbers
//   auto fn = F.get_smearing().fn(ek);

//   // X <- ortho((X + t*g_X) @ Ul)
//   auto x_next = tapply_async(local::advance_x(t), X, g_x, Ul, S);

//   F.compute(eval_threaded(x_next), fn);

//   return std::make_tuple(ek, Ul);
// }


namespace impl {
template <class X_t, class eta_t, class g_x_t, class g_eta_t, class Op_t>
auto
geodesic_us(
    X_t& X, const eta_t& eta, const g_x_t& z_x, const g_eta_t& z_eta, const Op_t& S, double t)
{
  // compute eta_next <- eta + t* g_eta
  auto eta_next = local::advance_eta(t)(eta, z_eta);
  // get eigenvalues and eigenvectors of next eta
  auto ek_Ul = local::eigvals_and_vectors()(eta_next);
  // grab results
  auto ek = std::get<0>(ek_Ul);
  auto Ul = std::get<1>(ek_Ul);

  // X <- ortho((X + t*g_X) @ Ul)
  auto x_next = local::advance_x(t)(X, z_x, Ul, S);

  return std::make_tuple(ek, Ul, x_next);
}


template <class mem_space_t>
struct geodesic_us_functor
{
  geodesic_us_functor(const mem_space_t& mem_space, double t)
      : mem_space(mem_space)
      , t(t)
  {
  }

  template <class X_t, class eta_t, class z_x_t, class z_eta_t, class Op_t>
  auto operator()(
      const X_t& X_h, const eta_t& eta_h, const z_x_t& z_x_h, const z_eta_t& z_eta_h, const Op_t& S)

  {
    // todo
    auto X = create_mirror_view_and_copy(mem_space, X_h);
    auto eta = create_mirror_view_and_copy(mem_space, eta_h);
    auto z_x = create_mirror_view_and_copy(mem_space, z_x_h);
    auto z_eta = create_mirror_view_and_copy(mem_space, z_eta_h);

    auto result = eval(impl::geodesic_us(X, eta, z_x, z_eta, S, t));

    // copy results to host
    auto ek = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), std::get<0>(result));
    auto Ul = create_mirror_view_and_copy(Kokkos::HostSpace(), std::get<1>(result));
    auto x_next_h = create_mirror_view_and_copy(Kokkos::HostSpace(), std::get<2>(result));
    return std::make_tuple(ek, Ul, x_next_h);
  }

  mem_space_t mem_space;
  double t;
};

}  // namespace impl

/// Geodesic for Ultrasoft PP formulation
/// returns tuple<ek, Ul, X>
template <class mem_space_t, class X_t, class eta_t, class z_x_t, class z_eta_t, class Op_t>
auto
geodesic(const mem_space_t& mem_space,
         const X_t& X_h,
         const eta_t& eta_h,
         const z_x_t& z_x_h,
         const z_eta_t& z_eta_h,
         const Op_t& S,
         double t)
{
  impl::geodesic_us_functor<mem_space_t> functor(mem_space, t);

  auto res = tapply_async(functor, X_h, eta_h, z_x_h, z_eta_h, S);

  return unzip(eval_threaded(res));
}


}  // namespace nlcglib
