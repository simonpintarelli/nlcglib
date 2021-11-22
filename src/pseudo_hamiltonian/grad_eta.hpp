#pragma once

#include <Kokkos_Core.hpp>
#include "constants.hpp"
#include "exec_space.hpp"
#include "la/mvector.hpp"
#include "smearing.hpp"

namespace nlcglib {

template <enum smearing_type smearing_t>
struct GradEtaHelper
{
  /// NOTE: the factor 1/ kT isn't included.
  template <class array1_t, class array2_t, class array3_t, class array4_t>
  static double dFdmu(const mvector<array1_t>& Hii,
                      const mvector<array2_t>& en,
                      const mvector<array3_t>& fn,
                      const mvector<array4_t>& wk,
                      double mu,
                      double T,
                      double mo)
  {
    static_assert(
        is_on_host<array1_t>::value && is_on_host<array2_t>::value && is_on_host<array3_t>::value,
        "GradEtaHelper::dFdmu expects host memory input");
    auto commk = wk.commk();
    double dFdmu_loc{0};
    for (auto& elem : Hii) {
      auto hii = elem.second;
      auto key = elem.first;
      int nbands = hii.size();
      auto en_loc = en[key];
      auto fn_loc = fn[key];
      // int nbands =
      double kT = physical_constants::kb * T;
      Kokkos::complex<double> v{0};
      Kokkos::parallel_reduce(
          "dFdmu",
          Kokkos::RangePolicy<Kokkos::Serial>(0, nbands),
          KOKKOS_LAMBDA(int i, Kokkos::complex<double>& result) {
            double delta = smearing<smearing_t>::delta((en_loc(i) - mu) / kT, mo);
            result += (hii(i) - en_loc(i)) * (-1.0 * delta);
          },
          v);
      dFdmu_loc += v.real() * wk[key];  // note that hii is real-valued
    }
    auto dFdmu = commk.allreduce(dFdmu_loc, mpi_op::sum);
    return dFdmu;
  }

  /** w_k * fn (1-fn) summed over all k-points
     \f[
         \sum_{n, k'} w_{k'} f_n (mo-f_n)
     \f]
     \$mo\$ is 1 for spin-polarized calucations and 2 otherwise.
  */
  template <class array1_t>
  static double dmu_deta(
      const mvector<array1_t>& en, const mvector<double>& wk, double mu, double T, double mo)
  {
    static_assert(is_on_host<array1_t>::value, "GradEtaHelper::dmu_deta expects host memory input");

    auto commk = wk.commk();

    double kT = physical_constants::kb * T;

    double v{0};
    for (auto& vwki : wk) {
      auto key = vwki.first;
      double w_k = vwki.second;  // k-point weight
      int nbands = en[key].size();
      auto en_loc = en[key];
      for (int i = 0; i < nbands; ++i) {
        double delta = smearing<smearing_t>::delta((en_loc(i) - mu) / kT, mo);
        v += -1 * delta * w_k;
      }
    }

    return commk.allreduce(v, mpi_op::sum);
  }
};

struct _delta_eta
{
  _delta_eta(double kappa)
      : kappa(kappa)
  {
  }
  template <class hij_t, class ek_t, class wk_t>
  to_layout_left_t<std::remove_reference_t<hij_t>> operator()(const hij_t& hij,
                                                              const ek_t& ek,
                                                              const wk_t& wk) const
  {
    using Kokkos::RangePolicy;
    using Kokkos::Rank;

    auto d_eta = empty_like()(hij);
    using memspc = typename hij_t::storage_t::memory_space;
    static_assert(std::is_same<memspc, typename ek_t::memory_space>::value,
                  "memory spaces must match");
    scale(d_eta, hij, kappa / wk);
    int n = ek.extent(0);
    auto d_eta_array = d_eta.array();

    // CUDA will crash if it tries to access a member variable in a lambda capture ...
    double local_kappa = kappa;
    Kokkos::parallel_for(
        RangePolicy<exec_t<memspc>>(0, n),
        KOKKOS_LAMBDA(int i) { d_eta_array(i, i) = d_eta_array(i, i) - local_kappa * ek(i); });

    return d_eta;
  }

  double kappa;
};


template <enum smearing_type smearing_t>
class GradEta
{
public:
  GradEta(double T, double kappa)
      : kappa(kappa)
  {
    kT = physical_constants::kb * T;
  }

  /**
   * Gradient of η
   */
  template <class matrix_t, class array1_t, class array2_t>
  to_layout_left_t<matrix_t> g_eta(const matrix_t& Hij,
                                   double mu,
                                   double wk,
                                   const array1_t& ek,
                                   const array2_t& fn,
                                   double dmu_deta,
                                   double dFdmu,
                                   double mo)
  {
    // TODO: add static assert Hij, ek, fn must all have the same memory space
    auto gETA = zeros_like()(Hij);

    using SPACE = typename matrix_t::storage_t::memory_space;
    using exec_space = exec_t<SPACE>;
    auto mgETA = gETA.array();

    auto mHij = Hij.array();
    int nbands = mHij.extent(0);
    double kT_loc = kT;
    Kokkos::parallel_for(
        "gEta (1)", Kokkos::RangePolicy<exec_space>(0, nbands), KOKKOS_LAMBDA(int i) {
          double delta = smearing<smearing_t>::delta((ek(i) - mu) / kT_loc, mo);
          mgETA(i, i) = -1 / kT_loc * (mHij(i, i) - wk * ek(i)) * (-1.0 * delta);
        });

    if (std::abs(dmu_deta) < 1e-12) {
      // zero contribution
    } else {
      Kokkos::parallel_for(
          "gEta (2)", Kokkos::RangePolicy<exec_space>(0, nbands), KOKKOS_LAMBDA(int i) {
            // sumfn dmuFn
            double delta = smearing<smearing_t>::delta((ek(i) - mu) / kT_loc, mo);
            mgETA(i, i) += wk * (-1.0 * delta) / dmu_deta * (dFdmu / kT_loc);
          });
    }

    Kokkos::parallel_for(
        "gEta(3)",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>({{0, 0}}, {{nbands, nbands}}),
        KOKKOS_LAMBDA(int i, int j) {
          if (i == j) {
            // do nothing
          } else {
            double ej = ek(j);
            double ei = ek(i);
            if (std::abs(ej - ei) < 1e-10) {
              // zero contribution
            } else {
              double II = (fn(j) - fn(i)) / (ej - ei);
              mgETA(i, j) += II * mHij(i, j);
            }
          }
        });
    return gETA;
  }


  /**
   * Preconditioned gradient of η
   */
  template <class matrix_t, class vector_t, class vector2_t>
  // mvector<std::shared_future<to_layout_left_t<matrix_t>>>
  auto delta_eta(const mvector<matrix_t>& Hij,
                 const mvector<vector_t>& ek,
                 const mvector<vector2_t>& wk)
  {
    // delta_eta = kappa * (hij - diag(ek))
    return tapply_async(_delta_eta(kappa), Hij, ek, wk);
  }


private:
  // temperature (in Kelvin)
  double kappa;
  double kT;
};


}  // namespace nlcglib
