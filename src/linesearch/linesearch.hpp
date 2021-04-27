#pragma once

#include "exceptions.hpp"
#include "utils/logger.hpp"
#include <iomanip>

namespace nlcglib {

class line_search
{
private:
  template <class GEODESIC, class FREE_ENERGY>
  auto qline(GEODESIC& G_base, FREE_ENERGY& FE, double slope, bool& force_restart);

  template <class GEODESIC, class FREE_ENERGY>
  auto bt_search(GEODESIC& G_base, FREE_ENERGY& FE, double F0, bool& force_restart);

public:
  template <class GEODESIC, class FREE_ENERGY>
  auto operator()(GEODESIC&& G_base, FREE_ENERGY&& FE, double slope, bool& force_restart)
  {
    Logger::GetInstance() << "line search t_trial = " << std::scientific << t_trial << "\n";
    double F0 = FE.get_F();
    try {
      return qline(G_base, FE, slope, force_restart);
    } catch (StepError& step_error) {
      Logger::GetInstance() << "\t"
                            << "quadratic line search failed -> backtracking search\n";
      return bt_search(G_base, FE, F0, force_restart);
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
line_search::bt_search(GEODESIC& G_base, FREE_ENERGY& FE, double F0, bool& force_restart)
{
  auto G = G_base(FE);

  if (tau >= 1) {
    throw std::runtime_error("invalid value");
  }

  double t = t_trial;
  while (t > 1e-8) {
    auto ek_ul = G(t);
    double Fp = FE.get_F();
    Logger::GetInstance() << "fd slope: " << std::scientific << std::setprecision(3) << (Fp - F0) / t << " t: " << t
                          << " F:" << std::fixed << std::setprecision(13) << Fp << "\n";
    if (Fp < F0) {
      Logger::GetInstance() << "fd slope: " << std::scientific << std::setprecision(3) << (Fp - F0)/t << "\n";
      force_restart = false;
      return ek_ul;
    }
    t *= tau;
    Logger::GetInstance() << "\tbacktracking search tau = " << std::scientific << std::setprecision(5) << t << "\n";
  }
  // TODO: let logger print state
  Logger::GetInstance().flush();
  if (force_restart)  {
    throw DescentError();
  } else {
    force_restart = true;
    return G(0);  // reset gradient
  }
}

/**
 * Quadratic line search.
 *
 * Returns tuple (ek, Ul)
 */
template <class GEODESIC, class FREE_ENERGY>
auto
line_search::qline(GEODESIC& G_base, FREE_ENERGY& FE, double slope, bool& force_restart)
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
    Logger::GetInstance() << std::scientific << std::setprecision(12)
                          << "F_min: " << F_min << "\n"
                          << "F0:    " << F0 << "\n";
    throw StepError();
  }

  // reset force_restart
  force_restart = false;

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
