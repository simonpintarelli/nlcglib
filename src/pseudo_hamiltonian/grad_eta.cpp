#include "grad_eta.h"
#include "constants.hpp"
#include "exec_space.hpp"
#include "la/lapack.hpp"
#include "smearing.hpp"
#include "utils/profile.hpp"

namespace nlcglib {
namespace {

// Dispatch once on the host, before launching a statically specialized kernel.
template <class Function>
decltype(auto)
dispatch_smearing(smearing_type smear, Function&& function)
{
  switch (smear) {
    case smearing_type::FERMI_DIRAC:
      return function.template operator()<smearing_type::FERMI_DIRAC>();
    case smearing_type::GAUSSIAN_SPLINE:
      return function.template operator()<smearing_type::GAUSSIAN_SPLINE>();
    case smearing_type::GAUSS:
      return function.template operator()<smearing_type::GAUSS>();
    case smearing_type::METHFESSEL_PAXTON:
      return function.template operator()<smearing_type::METHFESSEL_PAXTON>();
    case smearing_type::COLD:
      return function.template operator()<smearing_type::COLD>();
    default:
      throw std::runtime_error("invalid smearing type given");
  }
}

template <enum smearing_type smearing_t>
struct GradEtaHelperImpl
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
            double delta = smearing<smearing_t>::delta((mu - en_loc(i)) / kT, mo);
            result += (hii(i) - en_loc(i)) * (delta);
          },
          v);
      dFdmu_loc += v.real() * wk[key];  // note that hii is real-valued
    }
    auto dFdmu = commk.allreduce(dFdmu_loc, mpi_op::sum);
    return dFdmu;
  }

  /** w_k * fn (1-fn) summed over all k-points
     \f[
         \sum_{n, k'} w_{k'} delta( (μ - e_{n,k'}) / kT )
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
        double delta = smearing<smearing_t>::delta((mu - en_loc(i)) / kT, mo);
        v += delta * w_k;
      }
    }

    return commk.allreduce(v, mpi_op::sum);
  }
};

template <enum smearing_type smearing_t>
class GradEtaKernel
{
public:
  explicit GradEtaKernel(double kT)
      : kT(kT)
  {
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
    PROFILE("nlcglib::g_eta");
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
          double delta = smearing<smearing_t>::delta((mu - ek(i)) / kT_loc, mo);
          mgETA(i, i) = -1 / kT_loc * (mHij(i, i) - wk * ek(i)) * (delta);
        });

    if (std::abs(dmu_deta) < 1e-12) {
      // zero contribution
    } else {
      Kokkos::parallel_for(
          "gEta (2)", Kokkos::RangePolicy<exec_space>(0, nbands), KOKKOS_LAMBDA(int i) {
            // sumfn dmuFn
            double delta = smearing<smearing_t>::delta((mu - ek(i)) / kT_loc, mo);
            mgETA(i, i) += wk * (delta) / dmu_deta * (dFdmu / kT_loc);
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


private:
  double kT;
};

}  // namespace

double
GradEtaHelper::dFdmu(smearing_type smear,
                     const mvector<host_vector_t>& Hii,
                     const mvector<host_vector_t>& en,
                     const mvector<host_vector_t>& fn,
                     const mvector<double>& wk,
                     double mu,
                     double T,
                     double mo)
{
  return dispatch_smearing(smear, [&]<smearing_type Kind>() {
    return GradEtaHelperImpl<Kind>::dFdmu(Hii, en, fn, wk, mu, T, mo);
  });
}

double
GradEtaHelper::dmu_deta(smearing_type smear,
                        const mvector<host_vector_t>& en,
                        const mvector<double>& wk,
                        double mu,
                        double T,
                        double mo)
{
  return dispatch_smearing(smear, [&]<smearing_type Kind>() {
    return GradEtaHelperImpl<Kind>::dmu_deta(en, wk, mu, T, mo);
  });
}

template <class MemorySpace>
GradEta<MemorySpace>::GradEta(double T, smearing_type smear)
    : kT(physical_constants::kb * T)
    , smear(smear)
{
}

template <class MemorySpace>
dense_matrix_t<MemorySpace>
GradEta<MemorySpace>::g_eta(const dense_matrix_t<MemorySpace>& Hij,
                            double mu,
                            double wk,
                            const real_vector_t<MemorySpace>& ek,
                            const real_vector_t<MemorySpace>& fn,
                            double dmu_deta,
                            double dFdmu,
                            double mo)
{
  return dispatch_smearing(smear, [&]<smearing_type Kind>() {
    return GradEtaKernel<Kind>(kT).g_eta(Hij, mu, wk, ek, fn, dmu_deta, dFdmu, mo);
  });
}

template class GradEta<Kokkos::HostSpace>;
#ifdef __NLCGLIB__CUDA
template class GradEta<Kokkos::CudaSpace>;
#elif defined(__NLCGLIB__ROCM)
template class GradEta<Kokkos::HIPSpace>;
#endif

}  // namespace nlcglib
