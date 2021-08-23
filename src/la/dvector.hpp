#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_View.hpp>
#include <string>
#include <type_traits>
#include <utility>
#include <complex>
#include <iomanip>
#include "map.hpp"
#include "nlcglib.hpp"

namespace nlcglib {

namespace _local {

using view_alloc_no_init_t =
    decltype(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string{}));

}  // _local

// forward declaration
template <class T, class LAYOUT, class... X>
class KokkosDVector;

template<class X>
class memory_space {};

template <class... X>
struct memory_space<KokkosDVector<X...>>
{
  using memspace = typename KokkosDVector<X...>::storage_t::memory_space;
};

template <class... X>
struct memory_space<Kokkos::View<X...>>
{
  using memspace = typename Kokkos::View<X...>::memory_space;
};

template<class X>
using memory_t = typename memory_space<std::remove_cv_t<std::remove_reference_t<X>>>::memspace;

template <class T>
struct is_on_host : std::integral_constant<bool, Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_t<T>>::accessible>
{};

#ifdef __NLCGLIB_CUDA
template <class T>
struct is_on_device
    : std::integral_constant<bool, Kokkos::SpaceAccessibility<Kokkos::CudaSpace, memory_t<T>>::accessible>
{
};
#else
template <class T>
struct is_on_device
  : std::integral_constant<bool, false>
{
};
#endif

/// get memory_type enum of x
template <typename X>
constexpr memory_type
get_mem_type(X&& x)
{
  memory_type mem_t{memory_type::none};
  bool is_host = is_on_host<X>::value;
  bool is_device = is_on_device<X>::value;

  if (is_host && !is_device) {
    mem_t = memory_type::host;
  } else if (is_device && !is_host) {
    mem_t = memory_type::device;
  } else {
    throw std::runtime_error("memory type unknown!");
  }

  return mem_t;
}


template <class T, class... ARGS>
buffer_protocol<std::complex<double>, 2>
as_buffer_protocol(const KokkosDVector<T**, ARGS...>& kokkosdvec)
{
  using type = KokkosDVector<T**, ARGS...>;
  return as_buffer_protocol(const_cast<type&>(kokkosdvec));
}


template <class T, class... ARGS>
buffer_protocol<std::complex<double>, 2> as_buffer_protocol(KokkosDVector<T**, ARGS...>& kokkosdvec)
{
  using vector_t = KokkosDVector<T, ARGS...>;
  using numeric_t = typename vector_t::storage_t::value_type;
  static_assert(std::is_same<numeric_t, Kokkos::complex<double>>::value, "todo: remove this limitation");

  auto mem_t = get_mem_type(kokkosdvec);
  std::array<int, 2> strides;
  strides[0] = kokkosdvec.array().stride(0);
  strides[1] = kokkosdvec.array().stride(1);

  std::array<int, 2> sizes;
  sizes[0] = kokkosdvec.array().extent(0);
  sizes[1] = kokkosdvec.array().extent(1);

  // TODO: is MPI_COMM_SELF always correct here?
  return buffer_protocol<std::complex<double>, 2>(strides, sizes,
                                                  reinterpret_cast<std::complex<double>*>(kokkosdvec.array().data()),
                                                  mem_t, MPI_COMM_SELF);
}

/// Distributed vector based on Kokkos
template <class T, class LAYOUT = SlabLayoutV, class... KOKKOS_ARGS>
class KokkosDVector
{
public:
  using dtype = T;
  using layout_t = LAYOUT;
  using storage_t = Kokkos::View<T, KOKKOS_ARGS...>;
  // figure out dimension of the underlying array
  static const int dim = storage_t::dimension::rank;
  using numeric_t = typename storage_t::value_type;

public:
  KokkosDVector(const Map<layout_t>& map, std::string label = std::string{})
      : map_(map)
      , kokkos_(label, map.nrows(), map.ncols())
  {
  }

  KokkosDVector(const Map<layout_t>& map, storage_t array);

  /// ViewCtorProp has a variadic template constructor which isn't declared explicit
  /// this constructor is thus ambigous unless kokkos will fix it's ViewCtorProp constructor
  KokkosDVector(const Map<layout_t>& map, _local::view_alloc_no_init_t&& vaw)
      : map_(map)
      , kokkos_(vaw, map.nrows(), map.ncols())
  {
  }

  KokkosDVector(KokkosDVector&& other)
      : map_(std::move(other.map_))
      , kokkos_(std::move(other.kokkos_))
  {
  }

  KokkosDVector(const KokkosDVector& other)
      : map_(other.map_)
      , kokkos_(other.kokkos_)
  {
  }

  KokkosDVector() = default;

  KokkosDVector& operator=(const KokkosDVector& other) = default;
  KokkosDVector& operator=(KokkosDVector&& other) = default;

  /// initialize from pointers
  template<class NUMERIC_T>
  KokkosDVector(const Map<layout_t>& map, const buffer_protocol<NUMERIC_T, 2>& buffer);

  /// local number of elements
  int lsize() const { return kokkos_.size(); }

  const storage_t& array() const { return kokkos_; }
  storage_t& array() { return kokkos_; }
  const Map<layout_t>& map() const { return map_; }

  KokkosDVector copy(std::string label = std::string{}) const;

private:
  Map<layout_t> map_;
  storage_t kokkos_;
};


template<class T1, class T2>
struct numeric {
  static_assert(std::is_same<T1,T2>::value, "requires same type");
  template<typename T>
  static T* map(T* x) {
    static_assert(std::is_same<std::remove_cv_t<T>*,T1*>::value, "invalid type");
    return x;
  }
};


template<>
struct numeric<std::complex<double>, Kokkos::complex<double>>
{
  static Kokkos::complex<double>* map(std::complex<double>* x)
  {
    return reinterpret_cast<Kokkos::complex<double>*>(x);
  }

  static const Kokkos::complex<double>* map(const std::complex<double>* x)
  {
   return reinterpret_cast<const Kokkos::complex<double>*>(x);
  }
};


template <class T, class LAYOUT, class... KOKKOS_ARGS>
template <class NUMERIC_T>
KokkosDVector<T, LAYOUT, KOKKOS_ARGS...>::KokkosDVector(const Map<LAYOUT>& map,
                                                        const buffer_protocol<NUMERIC_T, 2>& buffer)
    : map_(map)
    , kokkos_(
        numeric<NUMERIC_T, numeric_t>::map(buffer.data),
        Kokkos::LayoutStride(buffer.size[0], buffer.stride[0], buffer.size[1], buffer.stride[1]))
{
  static_assert(std::is_same<typename storage_t::memory_traits, Kokkos::MemoryUnmanaged>::value, "must be unmanaged");
  static_assert(dim == 2, "constructor is only valid for a 2-dimensional array");
}

template <class T, class LAYOUT, class... KOKKOS_ARGS>
KokkosDVector<T, LAYOUT, KOKKOS_ARGS...>::KokkosDVector(const Map<LAYOUT>& map,
                                                        storage_t array)
    : map_(map), kokkos_(array)
{

}

    template <class T, class LAYOUT, class... KOKKOS_ARGS>
    KokkosDVector<T, LAYOUT, KOKKOS_ARGS...> KokkosDVector<T, LAYOUT, KOKKOS_ARGS...>::copy(
        std::string label) const
{
  static_assert(!std::is_same<typename storage_t::memory_traits, Kokkos::MemoryUnmanaged>::value, "not yet implemented");

  KokkosDVector Result(this->map_, Kokkos::view_alloc(Kokkos::WithoutInitializing, label));

  Kokkos::deep_copy(Result.array(), this->array());
  return Result;
}


template <class T1, class L1, class... KOKKOS1, class T2, class L2, class... KOKKOS2>
inline void
deep_copy(KokkosDVector<T1, L1, KOKKOS1...>& dst, const KokkosDVector<T2, L2, KOKKOS2...>& src)
{
  static_assert(std::is_same<L1, L2>::value, "deep_copy requires identical layouts");
  Kokkos::deep_copy(dst.array(), src.array());
}


template <class KokkosSpace, class T2, class L2, class... KOKKOS2>
inline auto
create_mirror_view_and_copy(const KokkosSpace& Space, const KokkosDVector<T2, L2, KOKKOS2...>& src)
{
  // TODO: we are hardcoding LayoutLeft for return type here.
  using ret = KokkosDVector<T2, L2, Kokkos::LayoutLeft, KokkosSpace>;
  auto dst = Kokkos::create_mirror_view_and_copy(Space, src.array());
  return ret(src.map(), dst);
}

template <class... T>
struct to_kokkos_dvector {};


template <class T, class LAYOUT, class... KOKKOS_ARGS>
struct to_kokkos_dvector<T, LAYOUT, Kokkos::View<T, KOKKOS_ARGS...>>
{
  using type = KokkosDVector<T, LAYOUT, KOKKOS_ARGS...>;
};


template <class T, class LAYOUT, class... KOKKOS_ARGS>
using to_kokkos_dvector_t = typename to_kokkos_dvector<T, LAYOUT, Kokkos::View<T, KOKKOS_ARGS...>>::type;


template <class T, class LAYOUT, class... KOKKOS_ARGS>
auto
create_host_mirror(KokkosDVector<T**, LAYOUT, KOKKOS_ARGS...>& other)
{
  using return_t = KokkosDVector<T**, LAYOUT, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  return_t out(other.map());
  deep_copy(out, other);
  return out;
}


template <class T, class LAYOUT, class... KOKKOS_ARGS>
auto
create_host_mirror(const KokkosDVector<T*, LAYOUT, KOKKOS_ARGS...>& other)
{
  using return_t = KokkosDVector<T*, LAYOUT, Kokkos::HostSpace>;
  return_t out(other.map());
  deep_copy(out, other);
  return out;
}


template <class T, class LAYOUT, class... KOKKOS_ARGS>
auto
create_mirror_view_and_copy(const KokkosDVector<T*, LAYOUT, KOKKOS_ARGS...>& other)
{
  using return_t = KokkosDVector<T*, LAYOUT, Kokkos::HostSpace>;
  return_t out(other.map());
  deep_copy(out, other);
  return out;
}


template<class T, class... ARGS, class O>
std::enable_if_t<KokkosDVector<T**>::storage_t::dimension::rank==2>
print(const KokkosDVector<T**, ARGS...>& mat, O&& out, int precision = 4)
{
  double tol = 1e-14;
  auto hmat = create_host_mirror(mat);
  auto& harr = hmat.array();
  for (int i = 0; i < harr.extent(0); ++i) {
    for (int j = 0; j < harr.extent(1); ++j) {
      if (Kokkos::abs(harr(i,j)) > tol)
        out << std::setprecision(precision) << harr(i, j) << " ";
      else
        out << T{0,0} << " ";
    }
    out << "\n";
  }
}


}  // namespace nlcglib
