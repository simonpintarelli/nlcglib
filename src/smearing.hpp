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

/// Fermi-Dirac smearing
struct fermi_dirac
{
  inline static double compute(double x)
  {
    if (x < -50) {
      return 1;
    }

    if (x > 40) {
      return 0;
    }

    return 1. / (1. + std::exp(x));
  }
};

/// Gaussian-spline smearing
struct efermi_spline
{
  inline static double compute(double x)
  {
    double sq2 = std::sqrt(2);
    if (x < 0) {
      return 1 - 0.5 * std::exp(0.5 - std::pow(x - 1 / sq2, 2));
    } else {
      return 0.5 * std::exp(0.5 - std::pow(1 / sq2 + x, 2));
    }
  }
};

template <class... args>
auto
inverse_fermi_dirac(Kokkos::View<double*, args...>& fn, double mo)
{
  using array_type = Kokkos::View<double*, args...>;
  int n = fn.size();
  Kokkos::View<double*, Kokkos::HostSpace> out("ek", n);
  static_assert(
      Kokkos::SpaceAccessibility<typename array_type::memory_space, Kokkos::HostSpace>::accessible,
      "invalid memory space");
  assert(fn.stride(0) == 1);

  std::valarray<double> fn_val(fn.data(), fn.size());
  std::valarray<bool> zero_loc = fn_val < 1e-12;
  std::valarray<bool> ones_loc = fn_val > mo-1e-12;
  // assuming that fn is strictly decreasing!
  assert(std::is_sorted(fn.data(), fn.data() + n, [](auto x, auto y) { return x > y; }));

  auto first_zero = std::find(std::begin(zero_loc), std::end(zero_loc), true);
  int num_zeros = std::end(zero_loc) - first_zero;
  auto last_one = std::find(std::begin(ones_loc), std::end(ones_loc), false);
  int num_ones = (last_one - std::begin(ones_loc)) + 1;

  double scale = 0.1;
  for (int i = 0; i < num_ones; ++i) {
    out(n - (i + 1)) = 50 + scale * (num_ones - (i + 1));
  }

  for (int i = 0; i < num_zeros; ++i) {
    out(i) = 50 - scale * (num_zeros - (i + 1));
  }

  for (int i = 0; i < n; ++i) {
    if ((zero_loc[i] || ones_loc[i]) == false) {
      double fi = fn(i) / mo;
      out(i) = std::log(1.0 / (fi - 1));
    }
  }

  return out;
}


template <class... args>
double
fermi_entropy(const Kokkos::View<double*, args...>& fn, double mo)
{
  auto fn_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fn);
  assert(fn_host.stride(0) == 1);
  if (fn_host.stride(0) != 1)
    throw std::runtime_error("invalid stride");

  std::valarray<double> fn_val(fn_host.data(), fn.size());
  fn_val /= mo;

  std::valarray<bool> zero_loc = fn_val <= 1e-12;
  std::valarray<bool> ones_loc = fn_val >= 1-1e-12;
  std::valarray<double> fni = fn_val[(!zero_loc) && (!ones_loc)];

  return (fni * std::log(fni) + (1.0 - fni) * std::log(1.0 - fni)).sum();
}

template <class... args>
double
gaussian_spline_entropy(const Kokkos::View<double*, args...>& xkokkos)
{
  auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xkokkos);

  std::valarray<double> x(x_host.data(), x_host.size());
  static double sqrt_piexp = std::sqrt(constants::pi * std::exp(1.));

  double S = 0;
  for (double xi : x) {
    double z = std::abs(xi);
    S += 0.25 * (2 * std::exp(-z * (std::sqrt(2) + z)) * z
                 + sqrt_piexp * std::erfc(1. / std::sqrt(2) + z));
  }
  return -1.*S;
}


template <class... args>
Kokkos::View<double*, Kokkos::HostSpace>
inverse_gaussian_spline(const Kokkos::View<double*, args...>& fn_input, double mo)
{
  auto fn_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fn_input);
  std::valarray<double> fn(fn_host.data(), fn_host.size());
  fn /= mo;

  std::valarray<bool> ifu = fn > 0.5;

  std::valarray<double> xi(fn.size());

  // remove numerical noise
  fn[fn<0.0] = 0.0;

  double ub = 8.0;
  double lb = -5.0;
  std::valarray<bool> if0 = efermi_spline::compute(ub) > fn;
  std::valarray<bool> if1 = efermi_spline::compute(lb) < fn;

  std::valarray<bool> ifb = if0 || if1;

  std::valarray<bool> iifu = ifu && !ifb;
  std::valarray<double> fn_iifu = fn[iifu];

  std::valarray<bool> iifl = !ifu && !ifb;
  std::valarray<double> fn_iifl = fn[iifl];

  xi[iifu] = (1. - std::sqrt(1. - 2. * std::log(2. - 2. * fn_iifu))) / std::sqrt(2.);
  xi[iifl] = (-1. + std::sqrt(1. - 2. * std::log(2. * fn_iifl))) / std::sqrt(2.);
  xi[if0] = ub;
  xi[if1] = lb;

  Kokkos::View<double*, Kokkos::HostSpace> out("ek", xi.size());
  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, fn.size()), KOKKOS_LAMBDA(int i) {
      out(i) = xi[i];
    });

  return out;
}

template <class SMEARING, class X, class scalar_vec_t>
auto
get_occupation_numbers(
    const mvector<X>& x, double kT, double occ, int Ne, const scalar_vec_t& wk, double tol)
{
  // todo create xhost
  auto x_host = eval_threaded(tapply(
      [](auto x) {
        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
        return x_host;
      },
      x));

  auto fd = [kT = kT, occ = occ](auto ek, auto fn_scratch, double mu) {
    using memspace = typename decltype(ek)::memory_space;
    static_assert(std::is_same<memspace, Kokkos::HostSpace>::value, "must be host space");

    // auto ek_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ek);

    int n = ek.size();
    double sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += occ * SMEARING::compute((ek(i) - mu) / kT);
    }

    // copy back to original space
    return sum;
    // return Kokkos::create_mirror_view_and_copy(memspace(), out);
  };

  auto x_all = x_host.allgather(wk.commk());
  auto wk_all = wk.allgather();

  auto fn_scratch = eval_threaded(tapply([](auto ek) {
    int n = ek.size();
    Kokkos::View<double*, Kokkos::HostSpace> out(Kokkos::ViewAllocateWithoutInitializing("fn"), n);
    return out;
  }, x_all));

  double mu = find_chemical_potential(
      [&x = x_all, &wk = wk_all, &Ne = Ne, &fn_scratch = fn_scratch, &fd](double mu) {
        double fsum = 0;
        for (auto& wki : wk) {
          auto& key = wki.first;
          fsum += wki.second * fd(x[key], fn_scratch[key], mu);
        }
        return Ne - fsum;
      },
      0, /* mu0 */
      tol /* tolerance */);

  // call eval on x_host (x_host stores only the local k-points)
  auto fn_host = eval_threaded(tapply(
      [mu = mu, kT = kT, occ = occ](auto ek) {
        using memspace = typename decltype(ek)::memory_space;
        static_assert(std::is_same<memspace, Kokkos::HostSpace>::value, "must be host space");
        int n = ek.size();
        Kokkos::View<double*, Kokkos::HostSpace> out(Kokkos::ViewAllocateWithoutInitializing("fn"), n);

        for (int i = 0; i < n; ++i) {
          out(i) = occ * SMEARING::compute((ek(i) - mu) / kT);
        }
        return out;
        },
      x_host));

  using target_memspc = typename X::memory_space;
  // copy back to target memory space
  auto fn = eval_threaded(tapply([](auto fn_host) {
                         auto fn = Kokkos::View<double*, target_memspc>(Kokkos::ViewAllocateWithoutInitializing("fn"), fn_host.size());
                         Kokkos::deep_copy(fn, fn_host);
                         return fn;
                       }, fn_host));

  return fn;
}


class Smearing
{
public:
  Smearing(double T, int num_electrons, double max_occ, const mvector<double>& wk, enum smearing_type smearing)
      : T(T)
      , Ne(num_electrons)
      , occ(max_occ)
      , wk(wk)
      , smearing(smearing)
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
  double entropy(const mvector<X>& fn);


protected:
  /// Temperature in Kelvin
  double T;
  /// number of electrons
  int Ne;
  /// max occupancy 1 or 2
  double occ;
  /// kb * T
  double kT;
  double tol{1e-11};

  mvector<double> wk;
  smearing_type smearing;
};


template <class X>
auto
Smearing::fn(const mvector<X>& x)
{
  switch (smearing) {
    case smearing_type::FERMI_DIRAC: {
      return get_occupation_numbers<fermi_dirac>(x,
                                                 this->kT,
                                                 this->occ,
                                                 this->Ne,
                                                 this->wk,
                                                 this->tol);
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      return get_occupation_numbers<efermi_spline>(x,
                                                   this->kT,
                                                   this->occ,
                                                   this->Ne,
                                                   this->wk,
                                                   this->tol);
    }
    default:
      throw std::runtime_error("invalid smearing given");
  }
}

template <class X>
auto
Smearing::ek(const mvector<X>& fn)
{
  switch (smearing) {
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
  }
}

template <class X>
double
Smearing::entropy(const mvector<X>& fn)
{
  // this sum goes over all k-points, since wk * tapply(..) will inherit wk's communicator
  switch (smearing) {
    case smearing_type::FERMI_DIRAC: {
      return sum(wk * tapply([occ = occ](auto x) { return fermi_entropy(x, occ); }, fn));
    }
    case smearing_type::GAUSSIAN_SPLINE: {
      auto x = tapply([occ=occ] (auto fn) { return inverse_gaussian_spline(fn, occ); }, fn);
      return sum(wk * tapply([](auto x) { return gaussian_spline_entropy(x); }, x));
    }
    default:
      throw std::runtime_error("invalid smearing type");
  }
}


}  // namespace nlcglib
