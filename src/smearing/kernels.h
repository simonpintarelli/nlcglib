#pragma once

#include <Kokkos_Macros.hpp>
#include <cmath>
#include "constants.hpp"

namespace nlcglib {

/// Fermi-Dirac smearing
struct fermi_dirac_kernel
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
struct gaussian_spline_kernel
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
      return 0.25 * mo *
             (2 * std::exp(-x * (sqrt2 + x)) * x + sqrte * sqrtpi * std::erfc(1 / sqrt2 + x));
    } else {
      return 0.25 * mo *
             (-2 * std::exp(x * (sqrt2 - x)) * x + sqrte * sqrtpi * std::erfc(1 / sqrt2 - x));
    }
  }

  KOKKOS_INLINE_FUNCTION static double dxdelta(double x, double mo)
  {
    double sqrt2 = std::sqrt(2.0);

    if (x > 8 || x < -8) return 0;

    if (x <= 0) {
      return -2 * mo * std::exp((sqrt2 - x) * x) * x * (sqrt2 - x);
    } else {
      return -2 * mo * std::exp(-x * (sqrt2 + x)) * x * (sqrt2 + x);
    }
  }
};

/// Cold smearing
/// https://doi.org/10.1103/PhysRevLett.82.3296
struct cold_smearing_kernel
{
  KOKKOS_INLINE_FUNCTION static double fn(double x, double mo)
  {
    if (x > 8) return mo;
    if (x < -8) return 0;
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    return mo *
           (std::exp(-0.5 + (sqrt2 - x) * x) / sqrt2 / sqrtpi + 0.5 * std::erfc(1 / sqrt2 - x));
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
    double z = (x - 1 / sqrt2);
    return mo * std::exp(-z * z) * (sqrt2 - 6 * x + 2 * sqrt2 * x * x) / std::sqrt(constants::pi);
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    if (x < -8) return 0;
    if (x > 10) return 0;
    double sqrtpi = std::sqrt(constants::pi);
    double sqrt2 = std::sqrt(2.0);
    double z = (x - 1 / sqrt2);
    return mo * std::exp(-z * z) * (1 - sqrt2 * x) / 2 / sqrtpi;
  }
};

/// first order MP smearing
/// https://doi.org/10.1103/PhysRevB.40.3616
struct methfessel_paxton_smearing_kernel
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
    return mo * std::exp(-x * x) * x * (2 * x * x - 5) / sqrtpi;
  }

  KOKKOS_INLINE_FUNCTION static double entropy(double x, double mo)
  {
    double x2 = x * x;
    double sqrtpi = std::sqrt(constants::pi);
    return mo * std::exp(-x2) * (1 - 2 * x2) / 4 / sqrtpi;
  }
};

struct gauss_smearing_kernel
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


}  // namespace nlcglib
