#pragma once

#include <Kokkos_Core.hpp>
#include "constants.hpp"
#include "smearing.hpp"
#include "interface.hpp"

namespace nlcglib {

template <class MEMSPACE, class XMEMSPACE=MEMSPACE>
class FreeEnergy
{
public:
  FreeEnergy(double T, EnergyBase& energy, smearing_type smear);
  virtual ~FreeEnergy() {}

  template <class tF, class tX>
  void compute(const mvector<tX>& X, const mvector<tF>& fn);

  void compute();

  auto get_X();
  auto get_HX();
  auto get_SX();
  auto get_fn();
  auto get_ek();
  auto get_wk();
  auto get_gkvec_ekin();
  double occupancy();
  double ks_energy();

  double get_F() const { return free_energy; }
  double get_entropy() const { return entropy; }

  Smearing& get_smearing() { return smearing; }

private:
  double T;
  double free_energy;
  double entropy;
  EnergyBase& energy;
  Smearing smearing;
};


template <class MEMSPACE, class XMEMSPACE>
FreeEnergy<MEMSPACE, XMEMSPACE>::FreeEnergy(double T, EnergyBase& energy, smearing_type smear)
    : T(T)
    , energy(energy)
    , smearing(
        T, energy.nelectrons(), energy.occupancy(), make_mmscalar(energy.get_kpoint_weights()), smear)
{
  /* empty */
}


template <class MEMSPACE, class XMEMSPACE>
template <class tF, class tX>
void
FreeEnergy<MEMSPACE, XMEMSPACE>::compute(const mvector<tX>& X, const mvector<tF>& fn)
{
  // convert fn to std::vector
  auto map_fn = tapply(
      [](auto fi) {
        auto fi_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fi);
        int n = fi_host.size();
        std::vector<double> vec_fi(n);
        std::copy(fi_host.data(), fi_host.data() + n, vec_fi.data());
        return vec_fi;
      }, fn);
  std::vector<std::vector<double>> vec_fn;
  for (auto& fi : map_fn) vec_fn.push_back(eval(fi.second));

  auto Xsirius = make_mmatrix<Kokkos::HostSpace>(this->energy.get_C(memory_type::host));
  execute(tapply(
      [](auto x_sirius, auto x) {
        auto xh = Kokkos::create_mirror(x.array());
        // copy to Kokkos owned host mirror,
        // since  Kokkos refuses to copy device, managed -> host, unmanaged
        Kokkos::deep_copy(xh, x.array());
        Kokkos::deep_copy(x_sirius.array(), xh);
      }, Xsirius, X));

  energy.set_fn(vec_fn);
  energy.compute();

  double etot = energy.get_total_energy();
  double S = smearing.entropy(fn);

  entropy = physical_constants::kb * T * S;
  free_energy = etot + entropy;
}


template <class MEMSPACE, class XMEMSPACE>
void
FreeEnergy<MEMSPACE, XMEMSPACE>::compute()
{
  energy.compute();
  double etot = energy.get_total_energy();
  double S = smearing.entropy(this->get_fn());

  entropy = physical_constants::kb * T * S;
  free_energy = etot + entropy;
}


template <class MEMSPACE, class XMEMSPACE>
auto
FreeEnergy<MEMSPACE, XMEMSPACE>::get_X()
{
  // memory type none -> take what sirius has as default
  return make_mmatrix<MEMSPACE, XMEMSPACE>(this->energy.get_C(memory_type::none));
}


template <class MEMSPACE, class XMEMSPACE>
auto
FreeEnergy<MEMSPACE, XMEMSPACE>::get_HX()
{
  return make_mmatrix<MEMSPACE, XMEMSPACE>(this->energy.get_hphi());
}


template <class MEMSPACE, class XMEMSPACE>
auto
FreeEnergy<MEMSPACE, XMEMSPACE>::get_SX()
{
  return make_mmatrix<MEMSPACE, XMEMSPACE>(this->energy.get_sphi());
}

template <class MEMSPACE, class XMEMSPACE>
auto
FreeEnergy<MEMSPACE, XMEMSPACE>::get_fn()
{
  return make_mmvector<XMEMSPACE>(this->energy.get_fn());
}

template <class MEMSPACE, class XMEMSPACE>
auto
FreeEnergy<MEMSPACE, XMEMSPACE>::get_ek()
{
  return make_mmvector<XMEMSPACE>(this->energy.get_ek());
}

template <class MEMSPACE, class XMEMSPACE>
auto
FreeEnergy<MEMSPACE, XMEMSPACE>::get_wk()
{
  return make_mmscalar(this->energy.get_kpoint_weights());
}

template <class MEMSPACE, class XMEMSPACE>
auto
FreeEnergy<MEMSPACE, XMEMSPACE>::get_gkvec_ekin()
{
  return this->energy.get_gkvec_ekin();
}

template <class MEMSPACE, class XMEMSPACE>
double
FreeEnergy<MEMSPACE, XMEMSPACE>::occupancy()
{
  return this->energy.occupancy();
}

template <class MEMSPACE, class XMEMSPACE>
double
FreeEnergy<MEMSPACE, XMEMSPACE>::ks_energy()
{
  return this->energy.get_total_energy();
}

}  // namespace nlcglib
