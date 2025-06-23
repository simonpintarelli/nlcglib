#pragma once

#include <Kokkos_Core.hpp>
#include "la/mvector.hpp"
#include "mpi/communicator.hpp"

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
  return slope_t{.x = x[0], .eta = x[1]};
}

template <class memspace_t, enum smearing_type smearing_t>
class descent_direction_base
{
public:
  descent_direction_base(const memspace_t& memspc,
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

protected:
  memspace_t memspc;
  double mu;
  double dFdmu;
  double sumfn;
  double T;
  double kappa;
  double mo;
};


}  // namespace nlcglib
