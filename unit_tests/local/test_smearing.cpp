#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>
#include "constants.hpp"
#include "smearing.hpp"

using namespace nlcglib;

namespace {

/// Abscissae x = (mu - e) / kT used for the pointwise kernel checks.
std::vector<double>
sample_points()
{
  std::vector<double> xs;
  for (double x = -6.0; x <= 6.0 + 1e-12; x += 0.137) {
    xs.push_back(x);
  }
  return xs;
}

double
central_diff(double (*f)(double, double), double x, double mo, double h = 1e-5)
{
  return (f(x + h, mo) - f(x - h, mo)) / (2 * h);
}

}  // namespace

/* ------------------------------------------------------------------------------------------
 * Pointwise checks of the smearing kernels (fn, delta, dxdelta, entropy).
 * ------------------------------------------------------------------------------------------ */

template <class S>
class SmearingKernel : public ::testing::Test
{
protected:
  static constexpr bool monotonous = !std::is_base_of<non_monotonous, S>::value;
};

using kernel_types = ::testing::
    Types<fermi_dirac, gauss_smearing, cold_smearing, methfessel_paxton_smearing>;

class KernelNames
{
public:
  template <class T>
  static std::string GetName(int)
  {
    if (std::is_same<T, fermi_dirac>::value) return "FermiDirac";
    if (std::is_same<T, gauss_smearing>::value) return "Gauss";
    if (std::is_same<T, cold_smearing>::value) return "Cold";
    if (std::is_same<T, methfessel_paxton_smearing>::value) return "MethfesselPaxton";
    return "unknown";
  }
};

TYPED_TEST_SUITE(SmearingKernel, kernel_types, KernelNames);

/// fn -> 0 for e >> mu and fn -> mo for e << mu.
TYPED_TEST(SmearingKernel, OccupationLimits)
{
  for (double mo : {1.0, 2.0}) {
    EXPECT_NEAR(TypeParam::fn(-50, mo), 0, 1e-12);
    EXPECT_NEAR(TypeParam::fn(50, mo), mo, 1e-12);
  }
}

/// delta = d fn / dx
TYPED_TEST(SmearingKernel, DeltaIsDerivativeOfFn)
{
  for (double mo : {1.0, 2.0}) {
    for (double x : sample_points()) {
      double ref = central_diff(&TypeParam::fn, x, mo);
      EXPECT_NEAR(TypeParam::delta(x, mo), ref, 1e-7 * mo) << "x = " << x << ", mo = " << mo;
    }
  }
}

/// dxdelta = d delta / dx
TYPED_TEST(SmearingKernel, DxDeltaIsDerivativeOfDelta)
{
  for (double mo : {1.0, 2.0}) {
    for (double x : sample_points()) {
      double ref = central_diff(&TypeParam::delta, x, mo);
      EXPECT_NEAR(TypeParam::dxdelta(x, mo), ref, 1e-7 * mo) << "x = " << x << ", mo = " << mo;
    }
  }
}

/// generalized entropy s(x) satisfies s'(x) = -x * delta(x)
TYPED_TEST(SmearingKernel, EntropyDerivative)
{
  for (double mo : {1.0, 2.0}) {
    for (double x : sample_points()) {
      double ref = -x * TypeParam::delta(x, mo);
      EXPECT_NEAR(central_diff(&TypeParam::entropy, x, mo), ref, 1e-7 * mo)
          << "x = " << x << ", mo = " << mo;
    }
  }
}

TYPED_TEST(SmearingKernel, EntropyLimits)
{
  for (double mo : {1.0, 2.0}) {
    EXPECT_NEAR(TypeParam::entropy(-50, mo), 0, 1e-12);
    EXPECT_NEAR(TypeParam::entropy(50, mo), 0, 1e-12);
  }
}

/// \int delta(x) dx = mo
TYPED_TEST(SmearingKernel, DeltaNormalization)
{
  for (double mo : {1.0, 2.0}) {
    double a = -40, b = 40;
    int n = 80000;
    double h = (b - a) / n;
    double integral = 0.5 * (TypeParam::delta(a, mo) + TypeParam::delta(b, mo));
    for (int i = 1; i < n; ++i) {
      integral += TypeParam::delta(a + i * h, mo);
    }
    integral *= h;
    EXPECT_NEAR(integral, mo, 1e-8);
  }
}

/// monotone smearings: 0 <= fn <= mo, delta >= 0, entropy >= 0
TYPED_TEST(SmearingKernel, MonotoneBounds)
{
  if (!TestFixture::monotonous) {
    GTEST_SKIP() << "non-monotonous smearing";
  }
  double mo = 2;
  for (double x : sample_points()) {
    double f = TypeParam::fn(x, mo);
    EXPECT_GE(f, 0) << "x = " << x;
    EXPECT_LE(f, mo) << "x = " << x;
    EXPECT_GE(TypeParam::delta(x, mo), 0) << "x = " << x;
    EXPECT_GE(TypeParam::entropy(x, mo), 0) << "x = " << x;
  }
}

/// Fermi-Dirac against closed form expressions
TEST(FermiDirac, ClosedForm)
{
  for (double mo : {1.0, 2.0}) {
    for (double x : sample_points()) {
      double f = 1.0 / (1.0 + std::exp(-x));
      EXPECT_NEAR(fermi_dirac::fn(x, mo), mo * f, 1e-14);
      EXPECT_NEAR(fermi_dirac::delta(x, mo), mo * f * (1 - f), 1e-14);
      double s = -(f * std::log(f) + (1 - f) * std::log(1 - f));
      EXPECT_NEAR(fermi_dirac::entropy(x, mo), mo * s, 1e-12);
    }
  }
}

/* ------------------------------------------------------------------------------------------
 * Smearing class: chemical potential and occupation numbers.
 * ------------------------------------------------------------------------------------------ */

class SmearingClass : public ::testing::TestWithParam<smearing_type>
{
protected:
  using vec_t = Kokkos::View<double*, Kokkos::HostSpace>;

  void SetUp() override
  {
    for (int ik = 0; ik < nk; ++ik) {
      auto key = std::make_pair(ik, 0);
      wk_[key] = 1.0 / nk;
      vec_t eki("ek" + std::to_string(ik), num_bands);
      // slightly different band structure per k-point
      eki(0) = -10 + 0.1 * ik;
      for (int ib = 1; ib < num_bands; ++ib) {
        eki(ib) = eki(ib - 1) + std::exp(-0.05 * ib);
      }
      ek_[key] = eki;
    }
  }

  static constexpr int nk = 3;
  static constexpr int num_bands = 200;
  static constexpr int num_electrons = 30;
  static constexpr double T = 50000;

  mvector<double> wk_{Communicator()};
  mvector<vec_t> ek_{Communicator()};
};

TEST_P(SmearingClass, ElectronCount)
{
  for (double mo : {1.0, 2.0}) {
    Smearing smearing(T, num_electrons, mo, wk_, GetParam());
    auto [mu, fn] = smearing.fn(ek_);

    double ne = 0;
    for (auto& [key, fnk] : fn) {
      for (size_t ib = 0; ib < fnk.extent(0); ++ib) {
        ne += wk_[key] * fnk(ib);
      }
    }
    EXPECT_NEAR(ne, num_electrons, 1e-8) << "mo = " << mo << ", mu = " << mu;

    bool monotonous = GetParam() == smearing_type::FERMI_DIRAC ||
                      GetParam() == smearing_type::GAUSS;
    if (monotonous) {
      for (auto& [key, fnk] : fn) {
        for (size_t ib = 0; ib < fnk.extent(0); ++ib) {
          EXPECT_GE(fnk(ib), 0);
          EXPECT_LE(fnk(ib), mo);
          if (ib > 0) EXPECT_LE(fnk(ib), fnk(ib - 1)) << "occupations not ordered";
        }
      }
      double S = smearing.entropy(ek_, mu);
      EXPECT_LE(S, 0) << "Smearing::entropy returns -S with S >= 0";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllSmearings,
                         SmearingClass,
                         ::testing::Values(smearing_type::FERMI_DIRAC,
                                           smearing_type::GAUSS,
                                           smearing_type::COLD,
                                           smearing_type::METHFESSEL_PAXTON),
                         [](const auto& info) {
                           switch (info.param) {
                             case smearing_type::FERMI_DIRAC:
                               return std::string("FermiDirac");
                             case smearing_type::GAUSS:
                               return std::string("Gauss");
                             case smearing_type::COLD:
                               return std::string("Cold");
                             case smearing_type::METHFESSEL_PAXTON:
                               return std::string("MethfesselPaxton");
                             default:
                               return std::string("unknown");
                           }
                         });
