#pragma once

#include <memory>
#include <utility>
#include "execution_backend.h"
#include "interface.hpp"
#include "mvp2/slope.h"

namespace nlcglib {

/// Own the matrix state and backend work behind a scalar control interface.
class solver_state
{
public:
  solver_state(execution_backend backend,
               EnergyBase& energy,
               UltrasoftPrecondBase& preconditioner,
               OverlapBase& overlap,
               InverseOverlapBase& inverse_overlap,
               smearing_type smear,
               double T,
               double kappa);
  ~solver_state();

  void compute();
  void initialize_orbitals();
  slope_t initial_direction();
  slope_t restarted();
  slope_t restarted_sd();

  /// Evaluate and retain a trial; return its step for the line search.
  double evaluate(double step);
  void discard_trial();
  /// Commit the last accepted trial and update the conjugate direction.
  std::pair<slope_t, slope_t> accept(slope_t fr);

  double get_F() const;
  double get_entropy() const;
  double ks_energy() const;
  double get_chemical_potential() const;
  const EnergyBase& ehandle() const;
  void write_json(slope_t slope, int step, int frequency);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

}  // namespace nlcglib
