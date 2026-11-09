#pragma once

#include <memory>
#include <tuple>
#include "interface.hpp"
#include "la/types.h"

namespace nlcglib {

class Smearing;

class FreeEnergy
{
public:
  FreeEnergy(double T, EnergyBase& energy, smearing_type smear);
  virtual ~FreeEnergy();

  void compute(const mvector<host_matrix_t>& X,
               const mvector<host_vector_t>& fn,
               const mvector<host_vector_t>& en,
               double mu);
  void compute(const mvector<host_matrix_view_t>& X,
               const mvector<host_vector_t>& fn,
               const mvector<host_vector_t>& en,
               double mu);
  void compute();

  std::tuple<double, mvector<host_vector_t>> occupations(const mvector<host_vector_t>& en);

  mvector<host_matrix_view_t> get_X();
  mvector<host_matrix_view_t> get_HX();
  mvector<host_vector_t> get_fn();
  mvector<host_vector_t> get_ek();
  mvector<double> get_wk();
  std::shared_ptr<VectorBaseZ> get_gkvec_ekin();
  double occupancy();
  double ks_energy();
  std::map<std::string, double> ks_energy_components();

  double get_F() const { return free_energy; }
  double get_entropy() const { return entropy; }
  const auto& ehandle() const { return energy; }
  double get_chemical_potential() const { return energy.get_chemical_potential(); }

private:
  template <class Matrix>
  void compute_impl(const mvector<Matrix>& X,
                    const mvector<host_vector_t>& fn,
                    const mvector<host_vector_t>& en,
                    double mu);

  double T;
  double free_energy;
  double entropy;
  EnergyBase& energy;
  std::unique_ptr<Smearing> smearing;
};

}  // namespace nlcglib
