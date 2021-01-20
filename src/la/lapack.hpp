#pragma once

#include <functional>
#include <utility>
#include <la/map.hpp>
#include "lapack_cpu.hpp"
#ifdef __NLCGLIB__CUDA
#include "lapack_cuda.hpp"
#endif
#include "mvector.hpp"
#include "traits.hpp"
#include "utils.hpp"
#include "exec_space.hpp"

namespace nlcglib {

// /// extract the diagonal, returns a KokkosView
// template <class T, class LAYOUT, class... KOKKOS>
// Kokkos::View < T*, typename KokkosDVector<T**, LAYOUT, KOKKOS...>::storage_t::memory_space>
// diag(const KokkosDVector<T**, LAYOUT, KOKKOS...>& X)
// {
//   using vector_t = KokkosDVector<T**, LAYOUT, KOKKOS...>;
//   using memspace = typename vector_t::storage_t::memory_space;
//   int n = std::min(X.array().extent(0), X.array().extent(1));
//   Kokkos::View<T*, memspace> d("diag", n);

//   if (Kokkos::SpaceAccessibility<Kokkos::Cuda, memspace>::accessible) {
//     typedef Kokkos::RangePolicy<Kokkos::Cuda> range_policy;
//     auto Xm = X.array();
//     Kokkos::parallel_for(
//         "diag", range_policy(0, n), KOKKOS_LAMBDA(int i) { d(i) = Xm(i, i); });

//   } else if (Kokkos::SpaceAccessibility<Kokkos::Serial, memspace>::accessible) {
//     typedef Kokkos::RangePolicy<Kokkos::Serial> range_policy;
//     auto Xm = X.array();
//     Kokkos::parallel_for(
//         "diag", range_policy(0, n), KOKKOS_LAMBDA(int i) { d(i) = Xm(i, i); });
//   } else {
//     // raise exception
//     throw std::runtime_error("no suitable ExecutionSpace found.");
//   }
//   return d;
// }

#ifdef __NLCGLIB__CUDA
/// diag (on CUDA-GPU)
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<
    Kokkos::SpaceAccessibility<
        Kokkos::Cuda,
        typename KokkosDVector<T**, LAYOUT, KOKKOS...>::storage_t::memory_space>::accessible,
    Kokkos::View<T*, typename KokkosDVector<T**, LAYOUT, KOKKOS...>::storage_t::memory_space>>
diag(const KokkosDVector<T**, LAYOUT, KOKKOS...>& X)
{
  using vector_t = KokkosDVector<T**, LAYOUT, KOKKOS...>;
  using memspace = typename vector_t::storage_t::memory_space;
  int n = std::min(X.array().extent(0), X.array().extent(1));
  Kokkos::View<T*, memspace> d("diag", n);

  typedef Kokkos::RangePolicy<Kokkos::Cuda> range_policy;
  auto Xm = X.array();
  Kokkos::parallel_for(
      "diag", range_policy(0, n), KOKKOS_LAMBDA(int i) { d(i) = Xm(i, i); });

  return d;
}
#endif

/// diag (on HOST)
template <class T, class LAYOUT, class... KOKKOS>
std::enable_if_t<
    Kokkos::SpaceAccessibility<
        Kokkos::Serial,
        typename KokkosDVector<T**, LAYOUT, KOKKOS...>::storage_t::memory_space>::accessible,
    Kokkos::View<T*, typename KokkosDVector<T**, LAYOUT, KOKKOS...>::storage_t::memory_space>>
diag(const KokkosDVector<T**, LAYOUT, KOKKOS...>& X)
{
  using vector_t = KokkosDVector<T**, LAYOUT, KOKKOS...>;
  using memspace = typename vector_t::storage_t::memory_space;
  int n = std::min(X.array().extent(0), X.array().extent(1));
  Kokkos::View<T*, memspace> d("diag", n);

  typedef Kokkos::RangePolicy<Kokkos::Serial> range_policy;
  auto Xm = X.array();
  Kokkos::parallel_for(
      "diag", range_policy(0, n), KOKKOS_LAMBDA(int i) { d(i) = Xm(i, i); });
  return d;
}


/**
 * Make a diagonal matrix from the given diagonal entries.
 */
struct make_diag
{
  template <class T, class... ARGS>
  KokkosDVector<T**,
                SlabLayoutV,
                Kokkos::LayoutLeft,
                typename Kokkos::View<T*, ARGS...>::memory_space>
  operator()(const Kokkos::View<T*, ARGS...>& x)
  {
    using vector_t = Kokkos::View<T*, ARGS...>;
    static_assert(vector_t::dimension::rank == 1, "dimension mismatch");
    using memspace = typename vector_t::memory_space;
    using matrix_t = KokkosDVector<T**, SlabLayoutV, Kokkos::LayoutLeft, memspace>;

    int n = x.extent(0);
    matrix_t out{Map<>(Communicator(), SlabLayoutV({{0, 0, n, n}}))};
    auto& out_arr = out.array();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<exec_t<memspace>>(0, n),
        KOKKOS_LAMBDA(int i) { out_arr(i, i) = x(i); });
    return out;
  }
};

/**
 * Scales column of a matrix by a vector.
 *
 * Computes: dst <- beta * dst + alpha * src * x
 *
 * dst, src are matrix valued
 * x is vector valued
 * alpha, beta are scalars
 */
template <class M0,
          class M1,
          class T2,
          class... KOKKOS2>
M0&
scale(M0& dst,
      const M1& src,
      const Kokkos::View<T2*, KOKKOS2...>& x,
      double alpha,
      double beta = 0)
{
  auto mDST = dst.array();
  auto mSRC = src.array();
  int m = mSRC.extent(0);
  int n = mSRC.extent(1);

  using vector_t = M0;
  using memspace = typename vector_t::storage_t::memory_space;
  typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_t<memspace>> mdrange_policy;
  if (src.array().stride(0) == 1) {
    if (beta == 0)
      Kokkos::parallel_for(
          "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
            mDST(i, j) = alpha * x(j) * mSRC(i, j);
          });
    else
      Kokkos::parallel_for(
          "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
            mDST(i, j) = mDST(i, j) * beta + alpha * x(j) * mSRC(i, j);
          });
  } else {
    throw std::runtime_error("invalid stride");
  }

  return dst;
}

/**
 * Scales matrix by a scalar.
 *
 * Computes: dst <- beta * dst + alpha * src
 *
 * dst, src are matrix valued
 * alpha, beta are scalars
 */
template <class M1, class M2>
M1&
scale(M1& dst,
      const M2& src,
      double alpha,
      double beta)
{
  auto mDST = dst.array();
  auto mSRC = src.array();
  int m = mSRC.extent(0);
  int n = mSRC.extent(1);

  using vector_t = M1;
  using memspace = typename vector_t::storage_t::memory_space;
  typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_t<memspace>> mdrange_policy;
  if (src.array().stride(0) == 1) {
    if (beta == 0)
      Kokkos::parallel_for(
          "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
            mDST(i, j) =  alpha * mSRC(i, j);
          });

    else
      Kokkos::parallel_for(
          "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
            mDST(i, j) = mDST(i, j) * beta + alpha * mSRC(i, j);
          });
  } else {
    throw std::runtime_error("no suitable ExecutionSpace found.");
  }
  return dst;
}

/**
 * Scales matrix by a scalar.
 *
 * Computes: dst <- alpha * src
 *
 * dst, src are matrix valued
 * alpha, beta are scalars
 */
template <class M1, class M2>
M1&
scale(M1& dst,
      const M2& src,
      double alpha)
{
  auto mDST = dst.array();
  auto mSRC = src.array();
  int m = mSRC.extent(0);
  int n = mSRC.extent(1);

  using vector_t = M1;
  using memspace = typename vector_t::storage_t::memory_space;
  typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_t<memspace>> mdrange_policy;
  if (src.array().stride(0) == 1) {
    Kokkos::parallel_for(
        "scale", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
          mDST(i, j) = alpha * mSRC(i, j);
        });
  } else {
    throw std::runtime_error("invalid strides.");
  }
  return dst;
}


/**
 * Scales column of a matrix by a vector.
 *
 * Returns alpha * src * x
 * src is matrix valued
 * x is vector valued
 * alpha is a scalar
 */
template <class T, class M1, class LAYOUT, class... KOKKOS1>
to_layout_left_t<M1>
scale_alloc(const M1& src,
            const Kokkos::View<double*, KOKKOS1...>& x,
            double alpha = 1)
{
  auto mSRC = src.array();
  int n = src.map().nrows();
  int m = src.map().ncols();
  assert(x.extent(0) == m);
  using vector_t = M1;
  // using memspace = typename vector_t::storage_t::memory_space;
  Map<SlabLayoutV> map(src.map().comm(), SlabLayoutV({{0, 0, n, m}}));
  // TODO do not initialize memory in dst
  to_layout_left_t<vector_t> dst(src.map());
  scale(dst, src, x, alpha, 0);
  return dst;
}


/// dst <- ϐ * dst  + α * src
template <class T1, class M2, class LAYOUT0, class... KOKKOS0>
void
_add(KokkosDVector<T1**, LAYOUT0, KOKKOS0...>& dst,
    M2& src,
    const identity_t<T1>& alpha,
    const identity_t<T1>& beta = T1{1.0})
{
  using T = decltype(std::declval<T1>() + std::declval<typename M2::numeric_t>());
  using vector_t =
      KokkosDVector<T**, LAYOUT0, KOKKOS0...>;
  using memspace = typename vector_t::storage_t::memory_space;

  auto mDST = dst.array();
  auto mSRC = src.array();
  int m = mSRC.extent(0);
  int n = mSRC.extent(1);
  assert(mSRC.extent(0) == mDST.extent(0));
  assert(mSRC.extent(1) == mDST.extent(1));
  typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_t<memspace>> mdrange_policy;
  if (beta == T1{0})
    Kokkos::parallel_for(
        "add", mdrange_policy({0, 0}, {m, n}), KOKKOS_LAMBDA(int i, int j) {
          mDST(i, j) = alpha * mSRC(i, j);
        });
  else
    Kokkos::parallel_for(
        "add", mdrange_policy({0, 0}, {m, n}), KOKKOS_LAMBDA(int i, int j) {
          mDST(i, j) = mDST(i, j) * beta + alpha * mSRC(i, j);
        });
}

struct inner_ {
  /// Inner product allocating the returned matrix
  template <class T, class M1, class... KOKKOS2>
  to_layout_left_t<M1>
  operator()(const M1& A,
             const KokkosDVector<T**, SlabLayoutV, KOKKOS2...>& B,
             const identity_t<T>& alpha = T{1.0},
             const identity_t<T>& beta = T{0.})
  {
    int n = A.map().ncols();
    int m = B.map().ncols();
    Map<SlabLayoutV> map(A.map().comm(), SlabLayoutV({{0, 0, n, m}}));
    to_layout_left_t<M1> C(map);
    inner(C, A, B, alpha, beta);
    return C;
  }
};

/// Hermitian inner product, summed
struct innerh_tr
{
#ifdef __NLCGLIB__CUDA
  template <class M1, class M2>
  std::enable_if_t<
    Kokkos::SpaceAccessibility<Kokkos::Cuda, typename M1::storage_t::memory_space>::accessible,
    typename M1::numeric_t>
  operator()(const M1& X, const M2& Y)
  {
    int nrows = X.array().extent(0);
    int ncols = X.array().extent(1);

    using matrix_t = M1;
    using T = typename M1::numeric_t;

    using memory_space = typename matrix_t::storage_t::memory_space;

    Kokkos::View<T*, memory_space> tmp("", nrows);

    auto x = X.array();
    auto y = Y.array();

    T sum{0};

    // inner_reduce along rows
    Kokkos::parallel_for(
        "", Kokkos::RangePolicy<Kokkos::Cuda>(0, nrows), KOKKOS_LAMBDA(int i) {
          for (int j = 0; j < ncols; ++j) {
            tmp(i) += x(i, j) * Kokkos::conj(y(i, j));
          }
        });
    // sum vector
    Kokkos::parallel_reduce(
        "",
        Kokkos::RangePolicy<Kokkos::Cuda>(0, nrows),
        KOKKOS_LAMBDA(int i, T& lsum) { lsum += tmp(i); },
        sum);
      return sum;
  }
#endif

  template <class M1, class M2>
  std::enable_if_t<
      Kokkos::SpaceAccessibility<Kokkos::Serial, typename M1::storage_t::memory_space>::accessible,
      typename M1::numeric_t>
  operator()(const M1& X, const M2& Y)
  {
    int nrows = X.array().extent(0);
    int ncols = X.array().extent(1);

    using matrix_t = M1;
    using T = typename M1::numeric_t;

    using memory_space = typename matrix_t::storage_t::memory_space;

    Kokkos::View<T*, memory_space> tmp("", nrows);

    auto x = X.array();
    auto y = Y.array();

    T sum{0};
    // compute on host
    Kokkos::parallel_for(
        "", Kokkos::RangePolicy<exec_t<memory_space>>(0, nrows), KOKKOS_LAMBDA(int i) {
          for (int j = 0; j < ncols; ++j) {
            tmp(i) += x(i, j) * Kokkos::conj(y(i, j));
          }
        });
    Kokkos::parallel_reduce(
        "",
        Kokkos::RangePolicy<Kokkos::Serial>(0, nrows),
        KOKKOS_LAMBDA(int i, T& lsum) { lsum += tmp(i); },
        sum);

    return sum;
  }
};

template <class X, class Y>
Kokkos::complex<double>
innerh_reduce(const mvector<X>& x, const mvector<Y>& y)
{
  auto tmp = eval_threaded(tapply(innerh_tr(), x, y));
  auto z = sum(tmp);
  return Kokkos::real(z);
}


template<class X>
double l2norm(const mvector<X>& x) {
  auto tmp = eval_threaded(tapply(innerh_tr(), x, x));
  auto z = sum(tmp);
  if (std::abs(Kokkos::imag(z)) > 1e-10) {
    throw std::runtime_error("invalid value");
  }
  return Kokkos::real(z);
}

#ifdef __NLCGLIB__CUDA
template<class memspace>
std::enable_if_t<Kokkos::SpaceAccessibility<Kokkos::Cuda, memspace>::accessible>
loewdin_aux(Kokkos::View<double*, memspace>& w)
{
  // compute on device
  Kokkos::parallel_for(
      "scale", Kokkos::RangePolicy<Kokkos::Cuda>(0, w.size()), KOKKOS_LAMBDA(int i) {
        w(i) = 1.0 / sqrt(w(i));
      });
}
#endif

template <class memspace>
std::enable_if_t<Kokkos::SpaceAccessibility<Kokkos::Serial, memspace>::accessible>
loewdin_aux(Kokkos::View<double*, memspace>& w)
{
  Kokkos::parallel_for(
      "scale", Kokkos::RangePolicy<Kokkos::Serial>(0, w.size()), KOKKOS_LAMBDA(int i) {
        w(i) = 1.0 / sqrt(w(i));
      });
}


template <class T, class LAYOUT, class... KOKKOS>
to_layout_left_t<KokkosDVector<T**, LAYOUT, KOKKOS...>>
loewdin(const KokkosDVector<T**, LAYOUT, KOKKOS...>& X)
{
  using matrix_t = KokkosDVector<T**, KOKKOS...>;
  using memspace = typename matrix_t::storage_t::memory_space;

  auto S = inner_()(X, X);
  Kokkos::View<double*, memspace> w("eigvals, loewdin", X.array().extent(1));
  auto U = empty_like()(S);
  eigh(U, w, S);

  loewdin_aux(w);
  // // w <- 1 / sqrt(w)
  // if (Kokkos::SpaceAccessibility<Kokkos::Cuda, memspace>::accessible) {
  //   // compute on device
  //   Kokkos::parallel_for("scale", Kokkos::RangePolicy<Kokkos::Cuda>(0, w.size()), KOKKOS_LAMBDA(int i) {
  //       w(i) = 1.0 / sqrt(w(i));
  //     });
  // } else if (Kokkos::SpaceAccessibility<Kokkos::Serial, memspace>::accessible) {
  //   // compute on host
  //   Kokkos::parallel_for(
  //       "scale", Kokkos::RangePolicy<Kokkos::Serial>(0, w.size()), KOKKOS_LAMBDA(int i) {
  //         w(i) = 1.0 / sqrt(w(i));
  //       });
  // }
  // Kokkos::fence();
  // S <- U / sqrt(eigvals)
  scale(S, U, w, 1, 0);
  auto R = zeros_like()(U);
  // R <- S @ U.H
  outer(R, S, U);

  auto Y = zeros_like()(X);
  transform(Y, Kokkos::complex<double>{0.0}, Kokkos::complex<double>{1.0}, X, R);

  return Y;
}




/// Inner product allocating the returned matrix
template <class M1, class M2>
to_layout_left_t<M1>
transform_alloc(
    const M1& A,
    const M2& B,
    const identity_t<typename M1::numeric_t>& alpha = identity_t<typename M1::numeric_t>{1.0},
    const identity_t<typename M1::numeric_t>& beta = identity_t<typename M1::numeric_t>{0.})
{
  to_layout_left_t<M1> C(A.map());
  transform(C, beta, alpha, A, B);
  return C;
}

}  // namespace nlcglib
