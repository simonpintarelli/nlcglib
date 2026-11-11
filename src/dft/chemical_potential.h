#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

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

}  // namespace nlcglib
