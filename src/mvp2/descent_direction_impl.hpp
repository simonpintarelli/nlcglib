#pragma once

#include <Kokkos_Core.hpp>
#include "descent_direction.h"
#include "la/lapack.hpp"
#include "la/mvector.hpp"
#include "mpi/communicator.hpp"

namespace nlcglib {

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
  return slope_t{.x = x[0], .eta = x[1]};
}

template <class memspace_t>
class descent_direction_base
{
public:
  descent_direction_base(const memspace_t& memspc,
                         double mu,
                         double dFdmu,
                         double sumfn,
                         double T,
                         double kappa,
                         double mo,
                         smearing_type smear)
      : memspc(memspc)
      , mu(mu)
      , dFdmu(dFdmu)
      , sumfn(sumfn)
      , T(T)
      , kappa(kappa)
      , mo(mo)
      , smear(smear)
  {
  }

protected:
  memspace_t memspc;
  double mu;
  double dFdmu;
  double sumfn;
  double T;
  double kappa;
  double mo;
  smearing_type smear;
};


struct _delta_eta
{
  _delta_eta(double kappa)
      : kappa(kappa)
  {
  }
  template <class hij_t, class ek_t, class wk_t>
  to_layout_left_t<std::remove_reference_t<hij_t>> operator()(const hij_t& hij,
                                                              const ek_t& ek,
                                                              const wk_t& wk) const
  {
    using Kokkos::RangePolicy;
    using Kokkos::Rank;

    auto d_eta = empty_like()(hij);
    using memspc = typename hij_t::storage_t::memory_space;
    static_assert(std::is_same<memspc, typename ek_t::memory_space>::value,
                  "memory spaces must match");
    scale(d_eta, hij, kappa / wk);
    int n = ek.extent(0);
    auto d_eta_array = d_eta.array();

    // CUDA will crash if it tries to access a member variable in a lambda capture ...
    double local_kappa = kappa;
    Kokkos::parallel_for(
        RangePolicy<exec_t<memspc>>(0, n),
        KOKKOS_LAMBDA(int i) { d_eta_array(i, i) = d_eta_array(i, i) - local_kappa * ek(i); });

    return d_eta;
  }

  double kappa;
};


}  // namespace nlcglib
