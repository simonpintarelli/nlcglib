#pragma once

#include <iomanip>
#include <type_traits>
#include <utility>
#include "utils/expected.hpp"
#include "utils/logger.hpp"
#include "utils/profile.hpp"

namespace nlcglib {

enum class LineSearchErrors
{
  DescentError,
  StepError,
  SlopeError
};

struct line_search_info
{
  std::string type;  // the ls-type used
};

class line_search
{
private:
  template <class GEODESIC, class FREE_ENERGY>
  auto qline(GEODESIC& G, FREE_ENERGY& FE, double slope)
      -> util::expected<decltype(G(std::declval<double>())), LineSearchErrors>;

  template <class GEODESIC, class FREE_ENERGY>
  auto bt_search(GEODESIC& G, FREE_ENERGY& FE, double F0)
      -> util::expected<decltype(G(std::declval<double>())), LineSearchErrors>;

public:
  template <class GEODESIC, class FREE_ENERGY>
  auto operator()(GEODESIC&& G, FREE_ENERGY&& FE, double slope) -> util::expected<
      std::remove_reference_t<decltype(qline(G, FE, std::declval<double>()).value())>,
      LineSearchErrors>;

  /// trial step
  double t_trial{0.2};
  /// parameter for backtracking search
  double tau{0.1};
};

template <class GEODESIC, class FREE_ENERGY>
auto
line_search::operator()(GEODESIC&& G, FREE_ENERGY&& FE, double slope) -> util::expected<
    std::remove_reference_t<decltype(qline(G, FE, std::declval<double>()).value())>,
    LineSearchErrors>
{
  if (slope > 0) {
    return util::unexpected(LineSearchErrors::SlopeError);
  }
  Logger::GetInstance() << "line search t_trial = " << std::scientific << t_trial << "\n";
  double F0 = FE.get_F();
  auto qline_result = qline(G, FE, slope);
  if (!qline_result && qline_result.error() == LineSearchErrors::StepError) {
    // handle StepError
    auto bt_result = bt_search(G, FE, F0);
    // also check error
    if (!bt_result && bt_result.error() == LineSearchErrors::DescentError) {
      G(0);  // reset state, gradients, etc
      return util::unexpected(bt_result.error());
    }
    return bt_result.value();
  }
  return qline_result.value();
}

/**
 * Backtracking search, reduce step size until lower energy is found.
 */
template <class GEODESIC, class FREE_ENERGY>
auto
line_search::bt_search(GEODESIC& G, FREE_ENERGY& FE, double F0)
    -> util::expected<decltype(G(std::declval<double>())), LineSearchErrors>
{
  PROFILE("nlcglib::line_search::bt_search");
  double t = t_trial;
  while (t > 1e-8) {
    auto ek_ul = G(t);
    double Fp = FE.get_F();
    Logger::GetInstance() << "fd slope: " << std::scientific << std::setprecision(3)
                          << (Fp - F0) / t << " t: " << t << " F:" << std::fixed
                          << std::setprecision(13) << Fp << "\n";
    if (Fp < F0) {
      Logger::GetInstance() << "fd slope: " << std::scientific << std::setprecision(3)
                            << (Fp - F0) / t << "\n";
      return ek_ul;
    }
    t *= tau;
    Logger::GetInstance() << "\tbacktracking search tau = " << std::scientific
                          << std::setprecision(5) << t << "\n";
  }
  // TODO: let logger print state
  Logger::GetInstance().flush();
  return util::unexpected(LineSearchErrors::DescentError);
}

/**
 * Quadratic line search.
 *
 * Returns tuple (ek, Ul)
 */
template <class GEODESIC, class FREE_ENERGY>
auto
line_search::qline(GEODESIC& G, FREE_ENERGY& FE, double slope)
    -> util::expected<decltype(G(std::declval<double>())), LineSearchErrors>
{
  PROFILE("nlcglib::line_search::qline");
  double F0 = FE.get_F();

  // // DEBUG check slope
  // {
  //   double dt = 1e-6;
  //   G(dt);
  //   double F1 = FE.get_F();
  //   double fd_slope = (F1-F0)/dt;
  //   Logger::GetInstance() << "\t DEBUG qline slope = " << std::setprecision(6) << slope << ",
  //   fd_slope = " << fd_slope << "\n";
  // }

  // (END) DEBUG check slope

  double tsearch = t_trial;
  double a, b, c, F1, t_min;
  while (true) {
    c = F0;
    b = slope;

    // evaluate at trial point and obtain new F
    G(tsearch);
    F1 = FE.get_F();

    a = (F1 - b * tsearch - c) / (tsearch * tsearch);

    t_min = -b / (2 * a);

    // check curvature, might need to increase trial point
    if (a < 0) {
      Logger::GetInstance() << "\t in line-search increase t_trial by *5 \n";
      tsearch *= 5;
    } else {
      break;
    }
  }

  double F_pred = -b * b / (4 * a) + c;

  // evaluate FE at predicted minimum
  auto ek_ul = G(t_min);
  double F_min = FE.get_F();
  Logger::GetInstance() << "\t t_min = " << t_min << " q line prediction error: " << std::scientific
                        << std::setprecision(8) << (F_pred - F_min) << " dE: " << std::scientific
                        << std::setprecision(8) << (F0 - F_min) << "\n";

  if (F_min > F0) {
    Logger::GetInstance() << std::setprecision(13) << "\t quadratic line search failed:\n"
                          << "\t - F_min: " << F_min << "\n"
                          << "\t - F0:    " << F0 << "\n\n";
    return util::unexpected(LineSearchErrors::StepError);
  }

  return ek_ul;
}

}  // namespace nlcglib
