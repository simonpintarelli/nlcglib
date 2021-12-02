#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <valarray>
#include "constants.hpp"
#include "la/mvector.hpp"
#include "la/utils.hpp"
#include "utils/logger.hpp"
#include "utils/timer.hpp"
#include "interface.hpp"

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
    if ( fun (mu) > 0)
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
  static double sum_fn(const Kokkos::View<double*, ARGS...>& ek,
                            double mu,
                            double T,
                            double mo)
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


template<class base_class>
struct sum_entropy_base
{
  template<class... ARGS>
  static double call(const Kokkos::View<double*, ARGS...>& ek, double mu, double T, double mo) {
    int n = ek.extent(0);

    double kT = physical_constants::kb * T;
    double entropy{0};
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Kokkos::Serial>(0, n),
                            KOKKOS_LAMBDA(int i, double& v) {
                              v += base_class::entropy( -1.0 * (ek(i)-mu)/kT, mo);
                            }, entropy);
    return entropy;
  }
};



/// Fermi-Dirac smearing
struct fermi_dirac : sum_entropy_base<fermi_dirac>, summed<fermi_dirac>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    if ( x < -35) {
      return double{0};
    }
    if ( x > 40) {
      return mo;
    }
    return mo - mo / (1 + std::exp(x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo){
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
    if(std::abs(x) > 40) {
      return 0;
    }
    double expx = std::exp(x);
    return -mo * (expx *(expx-1)) / std::pow(1+expx, 3);
  }


  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    if(std::abs(x) > 40) {
      return 0;
    }
    double expx = std::exp(x);
    return mo*(std::log(1 + expx) - expx * x / (1 + expx));
  }
};

/// Gaussian-spline smearing
struct gaussian_spline : sum_entropy_base<gaussian_spline>, summed<gaussian_spline>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    double sq2 = std::sqrt(2.0);
    if (x > 0) {
      return mo*(1 - 0.5 * std::exp(-x * (sq2 + x)));
    } else {
      return mo/2 * std::exp(x*(sq2-x));
    }
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    double sqrt2 = std::sqrt(2.0);
    if (x <= 0) {
      return mo * 0.5 * std::exp((sqrt2 - x) * x) * (sqrt2 - 2 * x);
    } else {
      return mo * 0.5 * std::exp(-x * (sqrt2 + x)) * (sqrt2 + 2 * x);
    }
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 =  std::sqrt(2.0);
    double sqrte = std::exp(0.5);

    if (x > 0) {
      return 0.25 *
             (2 * std::exp(-x * (sqrt2 + x)) * x + sqrte * sqrtpi * std::erfc(1 / sqrt2 + x));
    } else {
      return 0.25 *
             (-2 * std::exp(x * (sqrt2 - x)) * x + sqrte * sqrtpi * std::erfc(1 / sqrt2 - x));
    }
  }
};

/// Cold smearing
struct cold_smearing : sum_entropy_base<cold_smearing>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 =  std::sqrt(2.0);
    return mo*(std::exp(-0.5 + ( sqrt2 - x) * x) * sqrt2 / sqrtpi + 0.5*std::erfc(1/sqrt2 -x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    double z = (x - 1 / sqrt2);
    return mo * std::exp(-z*z) * (2-sqrt2*x) / sqrtpi;
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    double sqrt2 = std::sqrt(2.0);
    return std::exp(-0.5 + sqrt2 * x - x * x) * (sqrt2 - 6 * x + 2 * sqrt2 * x * x) / std::sqrt(constants::pi);
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    return mo*std::exp(-0.5 + (sqrt2-x) * x) * (1 - sqrt2 *x) / 2 / sqrtpi;
  }
};

/// first order MP smearing
struct methfessel_paxton_smearing : sum_entropy_base<methfessel_paxton_smearing>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    double x2 = x * x;
    double sqrtpi = std::sqrt(constants::pi);
    return mo/2 * (1 + std::exp(-x2) * x / sqrtpi + std::erf(x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    double x2 = x*x;
    double sqrtpi = std::sqrt(constants::pi);
    return mo * std::exp(-x2) * (1 + 0.25*(2-4*x2)) / sqrtpi;
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    double sqrtpi = std::sqrt(constants::pi);
    return mo * std::exp(-x*x) * (2*x*x -5) / sqrtpi;
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    double x2 = x * x;
    double sqrtpi = std::sqrt(constants::pi);
    return mo * std::exp(-x2) * (1-2*x2) / 4 / sqrtpi;
  }
};

struct gauss_smearing : sum_entropy_base<gauss_smearing>
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    return mo / 2 * (1 + std::erf(x));
  }

  KOKKOS_INLINE_FUNCTION static double delta(double x, double mo)
  {
    return mo * std::exp(-x*x) / std::sqrt(constants::pi);
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    return mo / 2 * std::exp(-x*x) / std::sqrt(constants::pi);
  }
};

/* smearing aliases */
template <enum smearing_type smearing_t>
class smearing;

template <>
class smearing<smearing_type::FERMI_DIRAC> : public fermi_dirac
{};

template <>
class smearing<smearing_type::GAUSSIAN_SPLINE> : public gaussian_spline
{};

template <>
class smearing<smearing_type::GAUSS> : public gauss_smearing
{
};

template <>
class smearing<smearing_type::COLD> : public cold_smearing
{};

template <>
class smearing<smearing_type::METHFESSEL_PAXTON> : public methfessel_paxton_smearing
{};

template <class SMEARING, class X, class scalar_vec_t>
auto
occupation_from_mvector(
    const mvector<X>& x, double kT, double occ, int Ne, const scalar_vec_t& wk, double tol)
{
  auto x_host = eval_threaded(tapply(
      [](auto x) {
        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
        return x_host;
      },
      x));

  auto fsum = [kT = kT, occ = occ](auto ek, double mu) {
    using memspace = typename decltype(ek)::memory_space;
    static_assert(std::is_same<memspace, Kokkos::HostSpace>::value, "must be host space");

    int n = ek.size();
    double sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += SMEARING::fn((mu - ek(i)) / kT, occ);
    }
    return sum;
  };

  auto x_all = x_host.allgather(wk.commk());
  auto wk_all = wk.allgather();

  double mu = find_chemical_potential(
      [&x = x_all, &wk = wk_all, &Ne = Ne, &fsum](double mu) {
        double sum = 0;
        for (auto& wki : wk) {
          auto& key = wki.first;
          sum += wki.second * fsum(x[key], mu);
        }
        return Ne - sum;
      },
      0, /* mu0 */
      tol /* tolerance */);

  // TODO: start Newton minimization for cold and m-p smearing.
  if (std::is_same<SMEARING, cold_smearing>::value || std::is_same<SMEARING, methfessel_paxton_smearing>::value) {
    // auto N = [&x = x_all, &wk = wk_all, &f = SMEARING::fn, kT = kT, occ = occ](double mu) {
    //   /// TODO f(..) must sum over all x[key]
    //   double sum = 0;
    //   for (auto& wki : wk) {
    //     auto& key = wki.first;
    //     sum += wki.second * f(x[key], mu);
    //   }
    //   return sum;
    // };
    // auto dN = [&x = x_all, &wk = wk_all, &f = SMEARING::delta](double mu) {

    //   double fsum = 0;
    //   for (auto& wki : wk) {
    //     auto& key = wki.first;
    //     fsum += wki.second * f(x[key], mu);
    //   }
    //   return fsum;
    // };
    // auto ddN = [&x = x_all, &wk = wk_all, &f = SMEARING::dxdelta](double mu) {
    //   double fsum = 0;
    //   for (auto& wki : wk) {
    //     auto& key = wki.first;
    //     fsum += wki.second * f(x[key], mu);
    //   }
    //   return fsum;
    // };
    // Newton minimization using mu as initial value
  }

  // call eval on x_host (x_host stores only the local k-points)
  auto fn_host = eval_threaded(tapply(
      [mu = mu, kT = kT, occ = occ](auto ek) {
        using memspace = typename decltype(ek)::memory_space;
        static_assert(std::is_same<memspace, Kokkos::HostSpace>::value, "must be host space");
        int n = ek.size();
        Kokkos::View<double*, Kokkos::HostSpace> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "fn"), n);

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


class Smearing
{
public:
  Smearing(double T, int num_electrons, double max_occ, const mvector<double>& wk, enum smearing_type smearing_t)
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
      auto mu_fn = occupation_from_mvector<fermi_dirac>(
          x, this->kT, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto mu_fn = occupation_from_mvector<gaussian_spline>(
          x, this->kT, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::GAUSS: {
      auto mu_fn = occupation_from_mvector<gauss_smearing>(
          x, this->kT, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      auto mu_fn = occupation_from_mvector<methfessel_paxton_smearing>(
          x, this->kT, this->occ, this->Ne, this->wk, this->tol);
      return mu_fn;
    }
    case smearing_type::COLD: {
      auto mu_fn = occupation_from_mvector<cold_smearing>(
          x, this->kT, this->occ, this->Ne, this->wk, this->tol);
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
            Kokkos::parallel_for(
                Kokkos::RangePolicy<exec>(0, fi.size()),
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
            Kokkos::parallel_for(
                Kokkos::RangePolicy<exec>(0, fn.size()),
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
  static_assert(is_on_host<X>::value, "fn needs to be in host memory");
  static_assert(is_on_host<Y>::value, "en needs to be in host memory");
  // this sum goes over all k-points, since wk * tapply(..) will inherit wk's communicator
  switch (smearing_t) {
    case smearing_type::FERMI_DIRAC: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc = smearing<smearing_type::FERMI_DIRAC>::call(
                                           enk, mu, T, occ);
                                       double foo = smearing<smearing_type::FERMI_DIRAC>::sum_entropy(enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc =
                                           smearing<smearing_type::GAUSSIAN_SPLINE>::call(
                                               enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }
    case smearing_type::GAUSS: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc = smearing<smearing_type::GAUSS>::call(
                                           enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }

    case smearing_type::COLD: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc =
                                           smearing<smearing_type::COLD>::call(
                                               enk, mu, T, occ);
                                       return loc;
                                     },
                                     en));
      return S;
    }
    case smearing_type::METHFESSEL_PAXTON: {
      double S = -1.0 * sum(wk * tapply(
                                     [occ = occ, mu = mu, T = T](auto enk) {
                                       double loc = smearing<smearing_type::METHFESSEL_PAXTON>::call(
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
