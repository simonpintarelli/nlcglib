#pragma once

#include <Kokkos_Core.hpp>
#include <functional>
#include <future>
#include <vector>
#include <la/dvector.hpp>
#include <exec_space.hpp>
#include <traits.hpp>

namespace nlcglib {

template <class>
class mvector;  // forward declaration
template <class, class>
class mvector_base;  // forward declaration

template <class... ANY>
struct to_layout_left
{
};

/// unwrap multi-vector
template <class T>
struct to_layout_left<mvector<T>> : public to_layout_left<T>
{
};

template <class T>
struct to_layout_left<mvector<T>&> : public to_layout_left<T>
{
};

template <class T>
struct to_layout_left<const mvector<T>&> : public to_layout_left<T>
{
};

template <class aT, class LAYOUT, class... KOKKOS>
struct to_layout_left<KokkosDVector<aT, LAYOUT, KOKKOS...>>
{
  using input_type = KokkosDVector<aT, LAYOUT, KOKKOS...>;
  using memspace = typename input_type::storage_t::memory_space;
  using result = KokkosDVector<aT, LAYOUT, Kokkos::LayoutLeft, memspace>;
};

template <class aT, class LAYOUT, class... KOKKOS>
struct to_layout_left<KokkosDVector<aT, LAYOUT, KOKKOS...>&&>
{
  using input_type = KokkosDVector<aT, LAYOUT, KOKKOS...>;
  using memspace = typename input_type::storage_t::memory_space;
  using result = KokkosDVector<aT, LAYOUT, Kokkos::LayoutLeft, memspace>;
};


template <class aT, class LAYOUT, class... KOKKOS>
struct to_layout_left<const KokkosDVector<aT, LAYOUT, KOKKOS...>&> : to_layout_left<KokkosDVector<aT, LAYOUT, KOKKOS...>> {};


template <class aT, class LAYOUT, class... KOKKOS>
struct to_layout_left<const KokkosDVector<aT, LAYOUT, KOKKOS...>>
    : to_layout_left<KokkosDVector<aT, LAYOUT, KOKKOS...>>
{
};


template <typename T>
using to_layout_left_t = typename to_layout_left<T>::result;


template <class T, class LAYOUT, class... KOKKOS_ARGS>
auto
_zeros_like(const KokkosDVector<T**, LAYOUT, KOKKOS_ARGS...>& input)
{
  using mat_t = to_layout_left_t<KokkosDVector<T**, LAYOUT, KOKKOS_ARGS...>>;
  mat_t zeros(input.map());
  return zeros;
}

template <class T, class LAYOUT, class... KOKKOS_ARGS>
auto
_identity_like(const KokkosDVector<T**, LAYOUT, KOKKOS_ARGS...>& input)
{
  throw std::runtime_error("not implemented");
}

inline std::vector<double>
linspace(double begin, double end, int npoints)
{
  std::vector<double> vec(npoints);
  double dx{0};
  if (npoints > 1) dx = (end - begin) / (npoints - 1);
  for (int i = 0; i < npoints; ++i) {
    vec[i] = begin + i * dx;
  }

  return vec;
}

struct zeros_like
{
  template <class T>
  auto operator()(T&& t) const
  {
    return _zeros_like(std::forward<T>(t));
  }
};


template <class T, class... ARGS>
Kokkos::View<T*, ARGS...>
_empty_like(const Kokkos::View<T*, ARGS...>& other)
{
  // return Kokkos::View<T*, ARGS...>("tmp", other.size());
  auto ret = Kokkos::View<T*, ARGS...>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "tmp"), other.size());
#ifdef DEBUG
  // initialize with NAN to throw an error immediately if not overwritten
  using memspc = typename Kokkos::View<T*, ARGS...>::memory_space;
  Kokkos::parallel_for(Kokkos::RangePolicy<exec_t<memspc>>(0, other.size()), KOKKOS_LAMBDA(int i) {
      ret(i) = NAN;
      });
#endif
  return ret;
}


template <class T, class LAYOUT, class... ARGS>
to_layout_left_t<KokkosDVector<T, LAYOUT, ARGS...>>
_empty_like(const KokkosDVector<T, LAYOUT, ARGS...>& other)
{
  using return_type = to_layout_left_t<KokkosDVector<T, LAYOUT, ARGS...>>;
  auto ret = return_type(other.map(), Kokkos::view_alloc(Kokkos::WithoutInitializing, "tmp"));
#ifdef DEBUG
  using memspc = typename return_type::storage_t::memory_space;
  // initialize with NAN to throw an error immediately if not overwritten
  auto mDST = ret.array();
  int m = mDST.extent(0);
  int n = mDST.extent(1);
  typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_t<memspc>> mdrange_policy;
  Kokkos::parallel_for(
      "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
        mDST(i, j) = NAN;
      });
#endif
  return ret;

  // return return_type(other.map(), "tmp");
}


struct empty_like
{
  template <class T>
  auto operator()(T&& t) const
  {
    return _empty_like(std::forward<T>(t));
  }
};


template <class T, class LAYOUT, class... ARGS>
auto
copy(const KokkosDVector<T, LAYOUT, ARGS...>& other)
{
  auto ret = empty_like()(other);
  deep_copy(ret, other);
  return ret;
}


template <class T, class... ARGS>
void
print(const Kokkos::View<T*, ARGS...>& x)
{
  using vector_t = Kokkos::View<T*, ARGS...>;
  static_assert(vector_t::dimension::rank == 1, "1d array expected");
  auto xh = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(xh, x);
  for (int i = 0; i < xh.extent(0); ++i) {
    std::cout << x(i) << "\t";
  }
  std::cout << "\n";
}

/// threaded apply over mvector
template <class FUNCTOR, class ARG, class... ARGS>
auto
tapply(FUNCTOR&& fun, const ARG& arg0, const ARGS&... args)
{
  using R = decltype(fun(eval(std::declval<typename ARG::value_type>()),
                         eval(std::declval<typename ARGS::value_type>())...));
  mvector<std::function<R()>> result(arg0.commk());
  for (auto& elem : arg0) {
    auto key = elem.first;
    auto get_key = [key](auto container) { return container.at(key); };
    result[key] = std::bind(fun, eval(get_key(arg0)), eval(get_key(args))...);
  }
  return result;
}

template <class T>
class find_type
{
  using t = typename T::fjsadlfjasf;
};

/// tapply for packed operators
template <class OP, class ARG0, class... ARGS>
auto
tapply_op(OP&& op, const ARG0& arg0, const ARGS&... args)
{
  // TODO: would need op for a single key here ...
  // using R = decltype(op(eval(std::declval<ARG0>()), eval(std::declval<ARGS>())...));
  using R = decltype(empty_like()(eval(std::declval<typename ARG0::value_type>())));

  mvector<std::function<R()>> result(arg0.commk());
  for (auto& elem : arg0) {
    auto key = elem.first;
    // find_type<decltype(key)>::t;
    auto get_key = [key](auto container) { return container.at(key); };
    auto fun = op.at(key);
    result[key] = [=](){ return fun(eval(get_key(arg0)), eval(get_key(args))...); };
  }
  return result;
}

/// threaded apply over mvector
template <class FUNCTOR, class ARG, class... ARGS>
auto
tapply_async(FUNCTOR&& fun, const ARG& arg0, const ARGS&... args)
{
  using R = decltype(fun(eval(std::declval<typename ARG::value_type>()),
                         eval(std::declval<typename ARGS::value_type>())...));
  mvector<std::shared_future<R>> result(arg0.commk());
  for (auto& elem : arg0) {
    auto key = elem.first;
    auto get_key = [key](auto container) { return container.at(key); };
    result[key] = std::async(std::launch::deferred,
                             std::bind(fun, eval(get_key(arg0)), eval(get_key(args))...))
                      .share();
  }
  return result;
}


}  // namespace nlcglib
