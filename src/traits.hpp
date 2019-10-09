#pragma once

#include <Kokkos_View.hpp>
#include <iostream>
#include <functional>
#include <utility>
#include <future>

namespace nlcglib {

template <class X>
struct identity
{
  typedef X type;
};

template <class X>
using identity_t = typename identity<X>::type;

template <class X>
struct _is_kokkos_view : std::false_type
{
};

template <class... args>
struct _is_kokkos_view<Kokkos::View<args...>> : std::true_type
{
  using type = Kokkos::View<args...>;
};

template <class X>
struct is_kokkos_view : _is_kokkos_view<std::remove_cv_t<std::remove_reference_t<X>>>
{
};


template <class X>
struct _is_future : std::false_type
{};

template <class X>
struct _is_future<std::shared_future<X>> : std::true_type
{
  using type = std::shared_future<X>;
};

template <class X>
struct is_future : _is_future<std::remove_cv_t<std::remove_reference_t<X>>>
{
};


// template <class X>
// struct is_kokkos_view : _is_kokkos_view<std::remove_reference_t<std::remove_cv_t<X>>>
// {
// };

template <typename... Ts>
struct make_void
{
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class, class = void_t<>>
struct is_callable : std::false_type
{
};

template <class X>
struct is_callable<X, void_t<decltype(std::declval<X>().operator()())>> : std::true_type
{
};

// forward declaration mvector
template <class>
class mvector;

template <class X>
constexpr auto&&
eval(X&& x, std::enable_if_t<(!is_callable<X>::value && !is_future<X>::value) || is_kokkos_view<X>::value> * = nullptr)
{
  return std::forward<X>(x);
}

template <class X>
constexpr auto
eval(X &&x, std::enable_if_t<is_callable<X>::value && !is_kokkos_view<X>::value> * = nullptr)
{
  return x();
}

template <class X>
constexpr auto
eval(std::shared_future<X>&& x)
{
  return x.get();
}

template <class X>
constexpr auto
eval(const std::shared_future<X>& x)
{
  return x.get();
}

template<class X>
struct eval_type {
  using type = std::remove_reference_t<decltype(eval(std::declval<X>()))>;
};

template <class X>
using eval_t = typename eval_type<X>::type;


template<class X>
struct result_of
{
  using type = decltype(eval(std::declval<X>()));
};

template<class X>
struct result_of<mvector<X>>
{
  using type = typename result_of<X>::type;
};

/// compute result of expression, unpacks mvector
template <class X>
using result_of_t = typename result_of<X>::type;

}  // namespace nlcglib
