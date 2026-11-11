#include "free_energy.h"
#include "constants.hpp"
#include "la/mvector.hpp"
#include "smearing.h"
#include "utils/profile.hpp"

namespace nlcglib {

// Independent of the wavefunction matrix type.
struct FreeEnergy::Occupations
{
  explicit Occupations(const mvector<host_vector_t>& fn)
  {
    for (const auto& fi : fn) {
      const auto& view = fi.second;
      std::vector<double> value(view.size());
      std::copy(view.data(), view.data() + view.size(), value.data());
      values.push_back(std::move(value));
      keys.push_back(fi.first);
    }
  }

  std::vector<std::vector<double>> values;
  std::vector<std::pair<int, int>> keys;
};

FreeEnergy::FreeEnergy(double T, EnergyBase& energy, smearing_type smear)
    : T(T)
    , energy(energy)
    , smearing(std::make_unique<Smearing>(T,
                                          energy.nelectrons(),
                                          energy.occupancy(),
                                          make_mmscalar(energy.get_kpoint_weights()),
                                          smear))
{
  /* empty */
}

template <class Matrix>
void
FreeEnergy::compute_impl(const mvector<Matrix>& X,
                         const mvector<host_vector_t>& fn,
                         const mvector<host_vector_t>& en,
                         double mu)
{
  PROFILE("nlcglib::FreeEnergy::compute");
  const Occupations occupations(fn);

  auto Xsirius = make_mmatrix<Kokkos::HostSpace>(this->energy.get_C(memory_type::host));
  for (auto& xi : Xsirius) {
    PROFILE("host unpinned -> pinned");
    const auto& x = X.at(xi.first);
    // Both overloads have host storage, including the provider's unmanaged buffer.
    Kokkos::deep_copy(xi.second.array(), x.array());
  }

  compute_energy(occupations, en, mu);
}

void
FreeEnergy::compute_energy(const Occupations& fn, const mvector<host_vector_t>& en, double mu)
{
  energy.set_fn(fn.keys, fn.values);
  energy.compute();

  // update fermi energy in SIRIUS (no effect here, but make sure to leave SIRIUS in a consistent
  // state)
  energy.set_chemical_potential(mu);

  double etot = energy.get_total_energy();
  double S = smearing->entropy(en, mu);

  entropy = physical_constants::kb * T * S;
  free_energy = etot + entropy;
}

mvector<host_vector_t>
FreeEnergy::get_fn()
{
  return make_mmvector<Kokkos::HostSpace>(this->energy.get_fn());
}

mvector<host_matrix_view_t>
FreeEnergy::get_X()
{
  return make_mmatrix<Kokkos::HostSpace>(this->energy.get_C(memory_type::host));
}


mvector<host_matrix_view_t>
FreeEnergy::get_HX()
{
  return make_mmatrix<Kokkos::HostSpace>(this->energy.get_hphi(memory_type::host));
}


mvector<host_vector_t>
FreeEnergy::get_ek()
{
  return make_mmvector<Kokkos::HostSpace>(this->energy.get_ek());
}

mvector<double>
FreeEnergy::get_wk()
{
  return make_mmscalar(this->energy.get_kpoint_weights());
}

std::shared_ptr<VectorBaseZ>
FreeEnergy::get_gkvec_ekin()
{
  return this->energy.get_gkvec_ekin();
}

double
FreeEnergy::occupancy()
{
  return this->energy.occupancy();
}

double
FreeEnergy::ks_energy()
{
  return this->energy.get_total_energy();
}

std::map<std::string, double>
FreeEnergy::ks_energy_components()
{
  return this->energy.get_energy_components();
}

void
FreeEnergy::compute()
{
  energy.compute();
}


FreeEnergy::~FreeEnergy() = default;

void
FreeEnergy::compute(const mvector<host_matrix_t>& X,
                    const mvector<host_vector_t>& fn,
                    const mvector<host_vector_t>& en,
                    double mu)
{
  compute_impl(X, fn, en, mu);
}

void
FreeEnergy::compute(const mvector<host_matrix_view_t>& X,
                    const mvector<host_vector_t>& fn,
                    const mvector<host_vector_t>& en,
                    double mu)
{
  compute_impl(X, fn, en, mu);
}

std::tuple<double, mvector<host_vector_t>>
FreeEnergy::occupations(const mvector<host_vector_t>& en)
{
  return smearing->fn(en);
}

}  // namespace nlcglib
