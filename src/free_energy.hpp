#pragma once

#include <Kokkos_Core.hpp>
#include "constants.hpp"
#include "interface.hpp"
#include "smearing.hpp"

namespace nlcglib {

class FreeEnergy
{
public:
  FreeEnergy(double T, EnergyBase& energy, smearing_type smear);
  virtual ~FreeEnergy() {}

  template <class tF, class tX, class tE>
  void compute(const mvector<tX>& X, const mvector<tF>& fn, const mvector<tE>& en, double mu);

  void compute();

  auto get_X();
  auto get_HX();
  // auto get_SX();
  auto get_fn();
  auto get_ek();
  auto get_wk();
  auto get_gkvec_ekin();
  double occupancy();
  double ks_energy();

  std::map<std::string, double> ks_energy_components();

  double get_F() const { return free_energy; }
  double get_entropy() const { return entropy; }
  const auto& ehandle() const { return energy; }

  Smearing& get_smearing() { return smearing; }
  double get_chemical_potential() const { return energy.get_chemical_potential(); }

private:
  double T;
  double free_energy;
  double entropy;
  EnergyBase& energy;
  Smearing smearing;
};


FreeEnergy::FreeEnergy(double T, EnergyBase& energy, smearing_type smear)
    : T(T)
    , energy(energy)
    , smearing(T,
               energy.nelectrons(),
               energy.occupancy(),
               make_mmscalar(energy.get_kpoint_weights()),
               smear)
{
  /* empty */
}

template <class tF, class tX, class tE>
void
FreeEnergy::compute(const mvector<tX>& X, const mvector<tF>& fn, const mvector<tE>& en, double mu)
{
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

  // update fermi energy in SIRIUS (no effect here, but make sure to leave SIRIUS in a consistent state)
  energy.set_chemical_potential(mu);

  double etot = energy.get_total_energy();
  double S = smearing.entropy(fn, en, mu);

  entropy = physical_constants::kb * T * S;
  free_energy = etot + entropy;
}

auto
FreeEnergy::get_fn()
{
  return make_mmvector<Kokkos::HostSpace>(this->energy.get_fn());
}

auto
FreeEnergy::get_X()
{
  return make_mmatrix<Kokkos::HostSpace>(this->energy.get_C(memory_type::host));
}


auto
FreeEnergy::get_HX()
{
  return make_mmatrix<Kokkos::HostSpace>(this->energy.get_hphi(memory_type::host));
}


// auto
// FreeEnergy::get_SX()
// {
//   return make_mmatrix<Kokkos::HostSpace>(this->energy.get_sphi(memory_type::host));
// }

auto
FreeEnergy::get_ek()
{
  return make_mmvector<Kokkos::HostSpace>(this->energy.get_ek());
}

auto
FreeEnergy::get_wk()
{
  return make_mmscalar(this->energy.get_kpoint_weights());
}

auto
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


}  // namespace nlcglib
