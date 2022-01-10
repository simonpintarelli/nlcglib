#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <valarray>
#include "constants.hpp"
#include "dft/newton_minimization_smearing.hpp"
#include "interface.hpp"
#include "la/mvector.hpp"
#include "la/utils.hpp"
#include "utils/env.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"

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

/// Fermi-Dirac smearing
struct fermi_dirac : summed<fermi_dirac>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    if (x < -35) {
      return double{0};
    }
    if (x > 40) {
      return mo;
    }
    return mo - mo / (1 + std::exp(x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    // double fni = fn(x, mo);
    // return -1 * fni * (mo-fni) / mo;
    if (std::abs(x) > 35) {
      return 0;
    }
    double denom = std::exp(-x / 2) + std::exp(x / 2);
    denom *= denom;
    return mo / denom;
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    if (std::abs(x) > 40) {
      return 0;
    }
    double expx = std::exp(x);
    return -mo * (expx * (expx - 1)) / std::pow(1 + expx, 3);
  }


  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    if (std::abs(x) > 40) {
      return 0;
    }
    double expx = std::exp(x);
    return mo * (std::log(1 + expx) - expx * x / (1 + expx));
  }
};

/// Gaussian-spline smearing
struct gaussian_spline : summed<gaussian_spline>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    if (x > 8) return mo;
    if (x < -8) return 0;
    double sq2 = std::sqrt(2.0);
    if (x <= 0) {
      return mo / 2 * std::exp(x * (sq2 - x));
    } else {
      return mo * (1 - 0.5 * std::exp(-x * (sq2 + x)));
    }
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    if (std::abs(x) > 7) return 0;
    double sqrt2 = std::sqrt(2.0);
    if (x <= 0) {
      return mo * 0.5 * std::exp((sqrt2 - x) * x) * (sqrt2 - 2 * x);
    } else {
      return mo * 0.5 * std::exp(-x * (sqrt2 + x)) * (sqrt2 + 2 * x);
    }
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    if (std::abs(x) > 7) return 0;
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    double sqrte = std::exp(0.5);

    if (x > 0) {
      return 0.25 *
             (2 * std::exp(-x * (sqrt2 + x)) * x + sqrte * sqrtpi * std::erfc(1 / sqrt2 + x));
    } else {
      return 0.25 *
             (-2 * std::exp(x * (sqrt2 - x)) * x + sqrte * sqrtpi * std::erfc(1 / sqrt2 - x));
    }
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    double sqrt2 = std::sqrt(2);

    if (x > 8 || x < -8) return 0;

    if (x <= 0) {
      return -2 * mo * std::exp((sqrt2 - x) * x) * (sqrt2 - x);
    } else {
      return -2 * mo * std::exp(-x * (sqrt2 + x)) * x * (sqrt2 + x);
    }
  }
};

/// Cold smearing
struct cold_smearing : summed<cold_smearing>, non_monotonous
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    if (x > 8) return mo;
    if (x < -8) return 0;
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    return mo *
           (std::exp(-0.5 + (sqrt2 - x) * x) * sqrt2 / sqrtpi + 0.5 * std::erfc(1 / sqrt2 - x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    if (x < -8) return 0;
    if (x > 10) return 0;

    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    double z = (x - 1 / sqrt2);
    return mo * std::exp(-z * z) * (2 - sqrt2 * x) / sqrtpi;
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    if (x < -8) return 0;
    if (x > 10) return 0;
    double sqrt2 = std::sqrt(2.0);
    return mo * std::exp(-0.5 + sqrt2 * x - x * x) * (sqrt2 - 6 * x + 2 * sqrt2 * x * x) /
           std::sqrt(constants::pi);
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    if (x < -8) return 0;
    if (x > 10) return 0;
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    return mo * std::exp(-0.5 + (sqrt2 - x) * x) * (1 - sqrt2 * x) / 2 / sqrtpi;
  }
};

/// first order MP smearing
struct methfessel_paxton_smearing : summed<methfessel_paxton_smearing>, non_monotonous
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    double x2 = x * x;
    double sqrtpi = std::sqrt(constants::pi);
    return mo / 2 * (1 + std::exp(-x2) * x / sqrtpi + std::erf(x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    double x2 = x * x;
    double sqrtpi = std::sqrt(constants::pi);
    return mo * std::exp(-x2) * (1 + 0.25 * (2 - 4 * x2)) / sqrtpi;
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    double sqrtpi = std::sqrt(constants::pi);
    return mo * std::exp(-x * x) * (2 * x * x - 5) / sqrtpi;
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    double x2 = x * x;
    double sqrtpi = std::sqrt(constants::pi);
    return mo * std::exp(-x2) * (1 - 2 * x2) / 4 / sqrtpi;
  }
};

struct gauss_smearing : summed<gauss_smearing>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    return mo / 2 * (1 + std::erf(x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    return mo * std::exp(-x * x) / std::sqrt(constants::pi);
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    return mo / 2 * std::exp(-x * x) / std::sqrt(constants::pi);
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    return -2 * mo * std::exp(-x * x) * x / std::sqrt(constants::pi);
  }
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

  // // TODO: start Newton minimization for cold and m-p smearing.
  // if (std::is_same<SMEARING, cold_smearing>::value || std::is_same<SMEARING,
  // methfessel_paxton_smearing>::value) {
  //   auto N = [&x = x_all, &wk = wk_all, kT = kT, occ = occ, T=T](double mu) {
  //     /// TODO f(..) must sum over all x[key]
  //     double sum = 0;
  //     for (auto& wki : wk) {
  //       auto& key = wki.first;
  //       sum += wki.second * SMEARING::sum_fn(x[key], mu, T, occ);
  //     }
  //     return sum;
  //   };
  //   auto dN = [&x = x_all, &wk = wk_all, T, occ](double mu) {

  //     double fsum = 0;
  //     for (auto& wki : wk) {
  //       auto& key = wki.first;
  //       fsum += wki.second * SMEARING::sum_delta(x[key], mu, T, occ);
  //     }
  //     return fsum;
  //   };
  //   auto ddN = [&x = x_all, &wk = wk_all, T, occ](double mu) {
  //     double fsum = 0;
  //     for (auto& wki : wk) {
  //       auto& key = wki.first;
  //       fsum += wki.second * SMEARING::sum_dxdelta(x[key], mu, T, occ);
  //     }
  //     return fsum;
  //   };
  //   // // Newton minimization using mu as initial value
  //   double mu = newton_minimization_chemical_potential(N, dN, ddN, mu, Ne, 1e-10);
  // }

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
        double sum = 0;
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
    double sum = 0;
    for (auto& wki : wk) {
      auto& key = wki.first;
      sum += wki.second * SMEARING::sum_fn(x[key], mu, T, occ);
    }
    return sum;
  };
  auto dN = [&x = x_all, &wk = wk_all, T, occ, kT](double mu) {
    double fsum = 0;
    for (auto& wki : wk) {
      auto& key = wki.first;
      fsum += wki.second / kT * SMEARING::sum_delta(x[key], mu, T, occ);
    }
    return fsum;
  };
  auto ddN = [&x = x_all, &wk = wk_all, T, occ, kT](double mu) {
    double fsum = 0;
    for (auto& wki : wk) {
      auto& key = wki.first;
      fsum += wki.second / (kT * kT) * SMEARING::sum_dxdelta(x[key], mu, T, occ);
    }
    return fsum;
  };

  // // Newton minimization using mu as initial value
  double mu;
  try {
    mu = newton_minimization_chemical_potential(N, dN, ddN, mu0, Ne, tol);
  } catch (failed_to_converge) {
    Logger::GetInstance()
        << "Warning: newton minimization for Fermi energy failed, fallback to bisection search.\n";
    // TODO print a warning that fallback to bisection search was used
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

  template <class X, class Y>
  double entropy(const mvector<X>& fn, const mvector<Y>& en, double mu);


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
  double tol{1e-11};

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
auto
Smearing::ek(const mvector<X>& fn)
{
  switch (smearing_t) {
    case smearing_type::FERMI_DIRAC: {
      auto ek = eval_threaded(tapply(
          [occ = occ, kT = kT](auto fi) {
            auto x = inverse_fermi_dirac(fi, occ);
            using exec = typename decltype(fi)::execution_space;
            Kokkos::parallel_for(Kokkos::RangePolicy<exec>(0, fi.size()),
                                 [=](int i) { x(i) = x(i) * kT; });
            return x;
          },
          fn));
      /// copy only entries that are present in ek
      return ek;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto ek = tapply(
          [occ = occ, kT = kT](auto fn) {
            auto x = inverse_gaussian_spline(fn, occ);
            using exec = typename decltype(fn)::execution_space;
            Kokkos::parallel_for(Kokkos::RangePolicy<exec>(0, fn.size()),
                                 [=](int i) { x(i) = x(i) * kT; });
            return x;
          },
          fn);
      return eval_threaded(ek);
    }
    case smearing_type::METHFESSEL_PAXTON: {
      throw std::runtime_error("smearing_type::METHFESSEL_PAXTON not yet implemented");
      break;
    }
    case smearing_type::COLD: {
      throw std::runtime_error("smearing_type::COLD not yet implemented");
      break;
    }
    default:
      throw std::runtime_error("smearing::ek invalid smearing type given");
  }
}


template <class X, class Y>
double
Smearing::entropy(const mvector<X>& fn, const mvector<Y>& en, double mu)
{
  static_assert(is_on_host<X>::value, "fn must reside in host memory");
  static_assert(is_on_host<Y>::value, "en must reside in host memory");
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
