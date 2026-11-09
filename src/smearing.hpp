#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include "constants.hpp"
#include "dft/newton_minimization_smearing.hpp"
#include "interface.hpp"
#include "la/mvector.hpp"
#include "la/utils.hpp"
#include "smearing/kernels.h"
#include "utils/env.hpp"
#include "utils/logger.hpp"

namespace nlcglib {


template <class Fun>
double
find_chemical_potential(Fun&& fun, double mu0, double tol)
{
  double mu = mu0;
  double de = 0.1;
  int sp{1};
  int s{1};
  int nmax{1000};
  int counter{0};
  while (std::abs(fun(mu)) > tol && counter < nmax) {
    sp = s;
    if (fun(mu) > 0)
      s = 1;
    else
      s = -1;
    if (s == sp)
      de *= 1.25;
    else
      de *= 0.25;
    mu += s * de;
    counter++;
  }

  if (!(std::abs(fun(mu)) < tol)) {
    throw std::runtime_error("couldn't find chemical potential f(mu) = " + std::to_string(fun(mu)) +
                             ", mu = " + std::to_string(mu));
  }

  return mu;
}

// outside because nvcc refuses to compile otherwise
template <class X>
struct sum_func
{
  template <class... KOKKOS_ARGS>
  static double call(const Kokkos::View<double*, KOKKOS_ARGS...>& ek,
                     double mu,
                     double T,
                     double mo,
                     double (*func_ptr)(double, double))
  {
    int n = ek.extent(0);

    double kT = physical_constants::kb * T;
    double lsum{0};
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::Serial>(0, n),
        KOKKOS_LAMBDA(int i, double& v) { v += (*func_ptr)(-1.0 * (ek(i) - mu) / kT, mo); },
        lsum);
    return lsum;
  }
};

template <class base_class>
struct summed
{
private:
  // previously sum_func
public:
  template <class... ARGS>
  static double sum_delta(const Kokkos::View<double*, ARGS...>& ek, double mu, double T, double mo)
  {
    return sum_func<base_class>::call(ek, mu, T, mo, &base_class::delta);
  }


  template <class... ARGS>
  static double sum_fn(const Kokkos::View<double*, ARGS...>& ek, double mu, double T, double mo)
  {
    return sum_func<base_class>::call(ek, mu, T, mo, &base_class::fn);
  }

  template <class... ARGS>
  static double sum_entropy(const Kokkos::View<double*, ARGS...>& ek,
                            double mu,
                            double T,
                            double mo)
  {
    return sum_func<base_class>::call(ek, mu, T, mo, &base_class::entropy);
  }

  template <class... ARGS>
  static double sum_dxdelta(const Kokkos::View<double*, ARGS...>& ek,
                            double mu,
                            double T,
                            double mo)
  {
    return sum_func<base_class>::call(ek, mu, T, mo, &base_class::dxdelta);
  }
};

struct non_monotonous
{
};

struct fermi_dirac : fermi_dirac_kernel, summed<fermi_dirac>
{
};

struct gaussian_spline : gaussian_spline_kernel, summed<gaussian_spline>
{
};

struct cold_smearing : cold_smearing_kernel, summed<cold_smearing>, non_monotonous
{
};

struct methfessel_paxton_smearing : methfessel_paxton_smearing_kernel,
                                    summed<methfessel_paxton_smearing>,
                                    non_monotonous
{
};

struct gauss_smearing : gauss_smearing_kernel, summed<gauss_smearing>
{
};

/* smearing aliases */
template <enum smearing_type smearing_t>
class smearing;

template <>
class smearing<smearing_type::FERMI_DIRAC> : public fermi_dirac
{
};

template <>
class smearing<smearing_type::GAUSSIAN_SPLINE> : public gaussian_spline
{
};

template <>
class smearing<smearing_type::GAUSS> : public gauss_smearing
{
};

template <>
class smearing<smearing_type::COLD> : public cold_smearing
{
};

template <>
class smearing<smearing_type::METHFESSEL_PAXTON> : public methfessel_paxton_smearing
{
};

// Find occuptions for Fermi-Dirac, Gauss, Gaussian-Spline smearing.
template <class SMEARING, class X, class scalar_vec_t>
auto
occupation_from_mvector(double T,
                        const mvector<X>& x,
                        double kT,
                        double occ,
                        int Ne,
                        const scalar_vec_t& wk,
                        double tol)
{
  auto x_host = eval_threaded(tapply(
      [](auto x) {
        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
        return x_host;
      },
      x));

  auto x_all = x_host.allgather(wk.commk());
  auto wk_all = wk.allgather();

  double mu = find_chemical_potential(
      [&x = x_all, &wk = wk_all, &Ne = Ne, T = T, occ = occ](double mu) {
        double sum = 0;
        for (auto& wki : wk) {
          // sum over k-points
          auto& key = wki.first;
          // sum_fn corresponds to ∑_i f(i), for i = 0..nbnands
          sum += wki.second * SMEARING::sum_fn(x[key], mu, T, occ);
        }
        return Ne - sum;
      },
      0, /* mu0 */
      tol /* tolerance */);

  // call eval on x_host (x_host stores only the local k-points)
  auto fn_host = eval_threaded(tapply(
      [mu = mu, kT = kT, occ = occ](auto ek) {
        using memspace = typename decltype(ek)::memory_space;
        static_assert(std::is_same<memspace, Kokkos::HostSpace>::value, "must be host space");
        int n = ek.size();
        Kokkos::View<double*, Kokkos::HostSpace> out(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "fn"), n);

        for (int i = 0; i < n; ++i) {
          out(i) = SMEARING::fn((mu - ek(i)) / kT, occ);
        }
        return out;
      },
      x_host));

  using target_memspc = typename X::memory_space;
  // copy back to target memory space
  auto fn = eval_threaded(tapply(
      [](auto fn_host) {
        auto fn = Kokkos::View<double*, target_memspc>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "fn"), fn_host.size());
        Kokkos::deep_copy(fn, fn_host);
        return fn;
      },
      fn_host));

  return std::make_tuple(mu, fn);
}

// Find occupations non-monotonous smearing types, i.e. Methfessel-Paxton and cold smearing.
template <class SMEARING, class X, class scalar_vec_t>
auto
occupation_from_mvector_newton(double T,
                               const mvector<X>& x,
                               double kT,
                               double occ,
                               int Ne,
                               const scalar_vec_t& wk,
                               double tol)
{
  auto x_host = eval_threaded(tapply(
      [](auto x) {
        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
        return x_host;
      },
      x));

  auto x_all = x_host.allgather(wk.commk());
  auto wk_all = wk.allgather();

  // find initial value for the Newton minimization using Gauss smearing
  double mu0 = find_chemical_potential(
      [&x = x_all, &wk = wk_all, &Ne = Ne, T = T, occ = occ](double mu) {
        double sum{0};
        for (auto& wki : wk) {
          // sum over k-points
          auto& key = wki.first;
          // sum_fn corresponds to ∑_i f(i), for i = 0..nbnands
          sum += wki.second * gauss_smearing::sum_fn(x[key], mu, T, occ);
        }
        return Ne - sum;
      },
      0, /* mu0 */
      tol /* tolerance */);

  auto N = [&x = x_all, &wk = wk_all, occ = occ, T = T](double mu) {
    double sum{0};
    for (auto& wki : wk) {
      auto& key = wki.first;
      sum += wki.second * SMEARING::sum_fn(x[key], mu, T, occ);
    }
    return sum;
  };
  auto dN = [&x = x_all, &wk = wk_all, T, occ, kT](double mu) {
    double fsum{0};
    for (auto& wki : wk) {
      auto& key = wki.first;
      fsum += wki.second / kT * SMEARING::sum_delta(x[key], mu, T, occ);
    }
    return fsum;
  };
  auto ddN = [&x = x_all, &wk = wk_all, T, occ, kT](double mu) {
    double fsum{0};
    for (auto& wki : wk) {
      auto& key = wki.first;
      fsum += wki.second / (kT * kT) * SMEARING::sum_dxdelta(x[key], mu, T, occ);
    }
    return fsum;
  };

  // // Newton minimization using mu0 as initial value
  double mu;
  try {
    mu = newton_minimization_chemical_potential(N, dN, ddN, mu0, Ne, tol);
  } catch (failed_to_converge&) {
    Logger::GetInstance()
        << "Warning: newton minimization for Fermi energy failed, fallback to bisection search.\n";
    mu = find_chemical_potential(
        [&x = x_all, &wk = wk_all, &Ne = Ne, T = T, occ = occ](double mu) {
          double sum = 0;
          for (auto& wki : wk) {
            // sum over k-points
            auto& key = wki.first;
            // sum_fn corresponds to ∑_i f(i), for i = 0..nbnands
            sum += wki.second * SMEARING::sum_fn(x[key], mu, T, occ);
          }
          return Ne - sum;
        },
        0, /* mu0 */
        tol /* tolerance */);
  }

  // call eval on x_host (x_host stores only the local k-points)
  auto fn_host = eval_threaded(tapply(
      [mu = mu, kT = kT, occ = occ](auto ek) {
        using memspace = typename decltype(ek)::memory_space;
        static_assert(std::is_same<memspace, Kokkos::HostSpace>::value, "must be host space");
        int n = ek.size();
        Kokkos::View<double*, Kokkos::HostSpace> out(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "fn"), n);

        for (int i = 0; i < n; ++i) {
          out(i) = SMEARING::fn((mu - ek(i)) / kT, occ);
        }
        return out;
      },
      x_host));

  using target_memspc = typename X::memory_space;
  // copy back to target memory space
  auto fn = eval_threaded(tapply(
      [](auto fn_host) {
        auto fn = Kokkos::View<double*, target_memspc>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "fn"), fn_host.size());
        Kokkos::deep_copy(fn, fn_host);
        return fn;
      },
      fn_host));

  Logger::GetInstance().flush();
  return std::make_tuple(mu, fn);
}


template <class smearing_t, class X, class scalar_vec_t>
auto
occupation_from_mvector1(
    double T, const mvector<X>& x, double occ, int Ne, const scalar_vec_t& wk, double tol)
{
  bool skip_newton = env::get_skip_newton_efermi();

  // std::cout << " non-monotonous smearing? " << std::is_base_of<non_monotonous, smearing_t>::value
  // << "\n";

  // check if newton should be ignored of env.
  double kT = physical_constants::kb * T;
  if (!skip_newton && std::is_base_of<non_monotonous, smearing_t>::value) {
    return occupation_from_mvector_newton<smearing_t>(T, x, kT, occ, Ne, wk, tol);
  } else {
    return occupation_from_mvector<smearing_t>(T, x, kT, occ, Ne, wk, tol);
  }
}


class Smearing
{
public:
  Smearing(double T,
           int num_electrons,
           double max_occ,
           const mvector<double>& wk,
           enum smearing_type smearing_t)
      : T(T)
      , Ne(num_electrons)
      , occ(max_occ)
      , wk(wk)
      , smearing_t(smearing_t)
  {
    if (T == 0) {
      throw std::runtime_error("Temperature must be > 0.");
    }
    kT = T * physical_constants::kb;
  }

  Smearing() = delete;

  template <class X>
  auto fn(const mvector<X>& ek);

  template <class X>
  auto ek(const mvector<X>& fn);

  template <class X>
  double entropy(const mvector<X>& en, double mu);


protected:
  /// Temperature in Kelvin
  double T;
  /// number of electrons
  int Ne;
  /// max occupancy 1 or 2
  double occ;
  /// kb * T
  double kT;
  /// TODO hardcoded constant
  double tol{1e-12};

  mvector<double> wk;
  smearing_type smearing_t;
};

template <class X>
auto
Smearing::fn(const mvector<X>& x)
{
  switch (smearing_t) {
    case smearing_type::FERMI_DIRAC: {
      auto mu_fn = occupation_from_mvector1<fermi_dirac>(
          this->T, x, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto mu_fn = occupation_from_mvector1<gaussian_spline>(
          this->T, x, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::GAUSS: {
      auto mu_fn = occupation_from_mvector1<gauss_smearing>(
          this->T, x, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto mu_fn = occupation_from_mvector1<methfessel_paxton_smearing>(
          this->T, x, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::COLD: {
      auto mu_fn = occupation_from_mvector1<cold_smearing>(
          this->T, x, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    default:
      throw std::runtime_error("invalid smearing given");
      break;
  }
}


template <class X>
double
Smearing::entropy(const mvector<X>& en, double mu)
{
  // static_assert(is_on_host<X>::value, "fn must reside in host memory");
  static_assert(is_on_host<X>::value, "en must reside in host memory");
  // this sum goes over all k-points, since wk * tapply(..) will inherit wk's communicator
  switch (smearing_t) {
    case smearing_type::FERMI_DIRAC: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc =
                                           smearing<smearing_type::FERMI_DIRAC>::sum_entropy(
                                               enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc =
                                           smearing<smearing_type::GAUSSIAN_SPLINE>::sum_entropy(
                                               enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }
    case smearing_type::GAUSS: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc = smearing<smearing_type::GAUSS>::sum_entropy(
                                           enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }

    case smearing_type::COLD: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc = smearing<smearing_type::COLD>::sum_entropy(
                                           enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc =
                                           smearing<smearing_type::METHFESSEL_PAXTON>::sum_entropy(
                                               enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }
    default:
      throw std::runtime_error("invalid smearing type");
  }
}


}  // namespace nlcglib
