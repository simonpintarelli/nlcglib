#pragma once

#include <memory>
#include <tuple>
#include "interface.hpp"
#include "la/vector_types.h"

namespace nlcglib {

/// Host-side occupation and chemical-potential solver.
class Smearing
{
public:
  Smearing(double T,
           int num_electrons,
           double max_occ,
           const mvector<double>& wk,
           smearing_type smearing_t);
  ~Smearing();

  std::tuple<double, mvector<host_vector_t>> fn(const mvector<host_vector_t>& ek);
  double entropy(const mvector<host_vector_t>& en, double mu);

private:
  double T;
  int Ne;
  double occ;
  double kT;
  double tol{1e-12};
  std::unique_ptr<mvector<double>> wk;
  smearing_type smearing_t;
};

}  // namespace nlcglib
