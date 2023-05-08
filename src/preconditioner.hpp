#pragma once

#include <Kokkos_Core.hpp>

#include "la/dvector.hpp"
#include "la/mvector.hpp"
#include "la/lapack.hpp"
#include "exec_space.hpp"

namespace nlcglib {

template <class SPACE>
using view_t = Kokkos::View<double*, SPACE>;


template <class SPACE>
class diagonal_preconditioner
{
public:
  diagonal_preconditioner(const view_t<SPACE>& entries)
      : entries(entries)
  {}

  template <class M1,
            class M2,
            class... KOKKOS_ARGS3>
  static void apply(M1& dst,
                    const M2& src,
                    const Kokkos::View<double*, KOKKOS_ARGS3...>& entries)
  {
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_t<SPACE>> mdrange_policy;
    int m = dst.array().extent(0);
    int n = dst.array().extent(1);
    auto mdst = dst.array();
    auto msrc = src.array();

    Kokkos::parallel_for(
        "teter preconditioner", mdrange_policy({{0, 0}}, {{m, n}}), KOKKOS_LAMBDA(int i, int j) {
          mdst(i, j) = entries(i) * msrc(i, j);
        });
  }

  template<typename X>
  auto operator()(const X& x)
  {
    auto y = empty_like()(x);
    diagonal_preconditioner::apply(y, x, entries);
    return y;
  }

  template<typename X>
  void apply_in_place(X& x)
  {
    diagonal_preconditioner::apply(x, x, entries);
  }

private:
  view_t<SPACE> entries;
};

/**
 *  Payne, M. C., Teter, M. P., Allan, D. C., Arias, T. A., & Joannopoulos, J.
 *  D., Iterative minimization techniques for ab initio total-energy
 *  calculations: molecular dynamics and conjugate gradients.
 *  http://dx.doi.org/10.1103/RevModPhys.64.1045
 */
template <class SPACE>
class PreconditionerTeter //: public mvector_base<PreconditionerTeter<SPACE>, diagonal_preconditioner<SPACE>>
{
public:
  using memspace = SPACE;
  using value_type = diagonal_preconditioner<SPACE>;
public:
  PreconditionerTeter(std::shared_ptr<VectorBaseZ> ekin)
  {
    auto ekin_vector = make_mmvector<Kokkos::HostSpace>(ekin);
    for (auto& elem : ekin_vector) {
      auto& key = elem.first;
      auto& ekin_loc = elem.second;
      int n = ekin_loc.size();
      auto result = Kokkos::View<double*, Kokkos::HostSpace>("", n);
      Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, n), [ekin_loc, result](int i) {
        double T = ekin_loc(i);
        double T2 = T * T;
        double T3 = T2 * T;
        double T4 = T2 * T2;
        double tp = 16 * T4 / (27 + 18 * T + 12 * T2 + 8 * T3);
        result(i) = 1 / (1 + tp);
      });
      // store result in mvector
      kinetic_diag_precond[key] = Kokkos::create_mirror(memspace(), result);
      Kokkos::deep_copy(kinetic_diag_precond.at(key), result);
    }
  }


  template<class key_t>
  auto operator[](const key_t& key) const
  {
    return diagonal_preconditioner<memspace>(kinetic_diag_precond.at(key));
  }

  template<class key_t>
  auto at(const key_t& key) const
  {
    return this->operator[](key);
  }

private:
  mvector<view_t<memspace>> kinetic_diag_precond;
};


}  // namespace nlcglib
