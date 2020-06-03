#pragma once

#include "exceptions.hpp"
#include "utils/logger.hpp"
#include <iomanip>

namespace nlcglib {

class line_search
{
private:
  template <class GEODESIC, class FREE_ENERGY>
  auto qline(GEODESIC& G_base, FREE_ENERGY& FE, double slope);

  template <class GEODESIC, class FREE_ENERGY>
  auto bt_search(GEODESIC& G_base, FREE_ENERGY& FE, double F0);

public:
  template <class GEODESIC, class FREE_ENERGY>
  auto operator()(GEODESIC&& G_base, FREE_ENERGY&& FE, double slope)
  {
    Logger::GetInstance() << "line search t_trial = " << std::scientific << t_trial << "\n";
    double F0 = FE.get_F();
    try {
      return qline(G_base, FE, slope);
    } catch (StepError& step_error) {
      Logger::GetInstance() << "\t"
                            << "quadratic line search failed -> backtracking search\n";
      return bt_search(G_base, FE, F0);
    }
  }

  /// trial step
  double t_trial{0.2};
  /// parameter for backtracking search
  double tau{0.1};
};


/**
 * Search any admissible lower energy
 */
template <class GEODESIC, class FREE_ENERGY>
auto
line_search::bt_search(GEODESIC& G_base, FREE_ENERGY& FE, double F0)
{
  auto G = G_base(FE);

  if (tau >= 1) {
    throw std::runtime_error("invalid value");
  }

  double t = t_trial;
  while (t > 1e-8) {
    auto ek_ul = G(t);
    double Fp = FE.get_F();
    if (Fp <= F0) {
      return ek_ul;
    }
    t *= tau;
    Logger::GetInstance() << "\tbacktracking search tau = " << std::scientific << t << "\n";
  }
  // TODO: let logger print state

  throw std::runtime_error("bt_search could NOT find a new minimum");
}

/**
 * Quadratic line search.
 *
 * Returns tuple (ek, Ul)
 */
template <class GEODESIC, class FREE_ENERGY>
auto
line_search::qline(GEODESIC& G_base, FREE_ENERGY& FE, double slope)
{
  // G(t)
  auto G = G_base(FE);

  double F0 = FE.get_F();

  // // DEBUG check slope
  // {
  //   double dt = 1e-6;
  //   G(dt);
  //   double F1 = FE.get_F();
  //   double fd_slope = (F1-F0)/dt;
  //   Logger::GetInstance() << "\t DEBUG qline slope = " << std::setprecision(6) << slope << ", fd_slope = " << fd_slope << "\n";
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
  Logger::GetInstance() << "\t t_min = " << t_min <<  ", q line prediction error: " << std::scientific << std::setprecision(8) << (F_pred - F_min) <<  "\n";

  if (F_min > F0) {
    throw StepError();
  }

  // TODO: update t_trial by t_min


  return ek_ul;
}



// template <class GEODESIC, class FREE_ENERGY>
// auto
// line_search(GEODESIC&& G_base, FREE_ENERGY&& FE, double slope, double t_trial = 0.2)
// {
//   double F0 = FE.get_F();
//   try {
//     return qline(G_base, FE, slope, t_trial);
//   } catch (StepError& step_error) {
//     Logger::GetInstance() << "\t"
//              << "quadratic line search failed -> backtracking search\n";
//     return bt_search(G_base, FE, F0, t_trial, 0.1);
//   }
// }


}  // namespace nlcglib
