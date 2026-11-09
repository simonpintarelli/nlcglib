#include "free_energy.h"
#include "constants.hpp"
#include "smearing.hpp"
#include "utils/profile.hpp"

namespace nlcglib {

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
  // convert fn to std::vector
  auto map_fn = tapply(
      [](auto fi) {
        auto fi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fi);
        int n = fi_host.size();
        std::vector<double> vec_fi(n);
        std::copy(fi_host.data(), fi_host.data() + n, vec_fi.data());
        return vec_fi;
      },
      fn);
  std::vector<std::vector<double>> vec_fn;
  std::vector<std::pair<int, int>> key_fn;
  for (auto& fi : map_fn) vec_fn.push_back(eval(fi.second));
  for (auto& fi : map_fn) key_fn.push_back(eval(fi.first));

  auto Xsirius = make_mmatrix<Kokkos::HostSpace>(this->energy.get_C(memory_type::host));
  execute(tapply(
      [](auto x_sirius, auto x) {
        PROFILE("host unpinned -> pinned");
        auto xh = Kokkos::create_mirror(x.array());
        // copy to Kokkos owned host mirror,
        // since  Kokkos refuses to copy device, managed -> host, unmanaged
        Kokkos::deep_copy(xh, x.array());
        Kokkos::deep_copy(x_sirius.array(), xh);
      },
      Xsirius,
      X));

  energy.set_fn(key_fn, vec_fn);
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
