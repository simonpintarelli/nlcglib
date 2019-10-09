#pragma once

#include <Kokkos_Core.hpp>
#include "la/mvector.hpp"
#include "constants.hpp"
#include "exec_space.hpp"

namespace nlcglib {

class GradEta
{
public:
  // template <typename memspace>
  // struct to_space
  // {
  // };

  // template <>
  // struct to_space<Kokkos::Cuda>
  // {
  //   using type = Kokkos::CudaSpace;
  // };

  // template <>
  // struct to_space<Kokkos::Serial>
  // {
  //   using type = Kokkos::HostSpace;
  // };

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
  mvector<to_layout_left_t<matrix_t>>
  g_eta(const mvector<matrix_t>& Hij,
        const mvector<double>& wk,
        const mvector<array1_t>& ek,
        const mvector<array2_t>& fn,
        double mo = 1.0)
  {
    // types:
    // Hij -- is a matrix type
    // wk  -- is mvector<double>
    // ek  -- is mvector<Kokkos::View<double*, ...>
    // fn  -- is mvector<Kokkos::View<double*, ...>

    // copy inputs to EXEC space memory?

    auto gETA = eval_threaded(tapply(zeros_like(), Hij));
    // TODO
    // copy everything to device ...

    // dFdmu  = 1 / kT sum(diag(Hij)-kw(ek) * fn * (mo-fn))
    // tmp  = sum(diag(Hij)-kw(ek) * fn * (mo-fn))
    // using SPACE = typename to_space<EXEC>::type;
    using SPACE = typename matrix_t::storage_t::memory_space;
    using complex_array_t = Kokkos::View<Kokkos::complex<double>*, SPACE>;
    mvector<Kokkos::View<Kokkos::complex<double>*, SPACE>> tmp;
    using exec_space = exec_t<SPACE>;

    std::vector<std::complex<double>> dFdmu_k;
    std::vector<double> sumfn_k; // sumfn_k = k_w * f_n (1-f_n)
    for (auto& elem : Hij) {
      auto key = elem.first;
      auto hij = diag(eval(Hij[key]));
      auto fn_loc = fn[key];
      double wk_loc = wk[key];
      auto ek_loc = ek[key];
      auto gETA_loc = gETA[key].array();
      // dFdmu_k
      int nbands = hij.size();
      tmp[key] = complex_array_t("", nbands);
      auto& tmp_loc = tmp[key];
      // for some unknown reason cuda lambda capture of class variable kT is not working (invalid memaccess)
      // create a local variable
      double kT_loc = kT;
      Kokkos::parallel_for(
          "gEta (1,2)", Kokkos::RangePolicy<exec_space>(0, nbands), KOKKOS_LAMBDA(int i) {
            double fni = fn_loc(i);
            tmp_loc(i) = (hij(i) - wk_loc * ek_loc(i)) * fni * (mo - fni);
            Kokkos::complex<double> res = -1 / kT_loc * tmp_loc(i);
            gETA_loc(i, i) = res;
          });

      Kokkos::complex<double> dFdmu_ki{0};
      Kokkos::parallel_reduce(
          "gEta (2)",
          Kokkos::RangePolicy<exec_space>(0, nbands),
          KOKKOS_LAMBDA(int i, Kokkos::complex<double>& loc_sum) { loc_sum += tmp_loc(i); },
          dFdmu_ki);
      // std::cout << "dFdmu_ki: " << dFdmu_ki.real() << ", " << dFdmu_ki.imag() << "\n";
      dFdmu_k.push_back(std::complex<double>{dFdmu_ki.real(), dFdmu_ki.imag()});

      double sumfn_ki{0};
      Kokkos::parallel_reduce(
          "sumfn",
          Kokkos::RangePolicy<exec_space>(0, nbands),
          KOKKOS_LAMBDA(int i, double& loc_sum) {
            double fni = fn_loc(i);
            loc_sum += wk_loc * fni * (mo - fni);
          },
          sumfn_ki);
      sumfn_k.push_back(sumfn_ki);
    }
    // allgather and compute sum locally...
    auto& comm = wk.commk();
    dFdmu_k = flatten(comm.allgather(dFdmu_k));
    Kokkos::complex<double> dFdmu =
        1 / kT * std::accumulate(dFdmu_k.begin(), dFdmu_k.end(), std::complex<double>{0});
    // same for sumfn
    sumfn_k = flatten(comm.allgather(sumfn_k));
    double sumfn = std::accumulate(sumfn_k.begin(), sumfn_k.end(), 0.0);

    for (auto& elem : gETA) {
      // update term gETA2
      auto key = elem.first;
      auto& fn_loc = fn[key];
      double wk_loc = wk[key];
      auto gETA_loc = gETA[key].array();
      int nbands = fn_loc.size();
      if (std::abs(sumfn) < 1e-10) {
        // zero contribution
      } else {
        Kokkos::parallel_for(
            "gEta (2)",
            Kokkos::RangePolicy<exec_space>(0, nbands),
            KOKKOS_LAMBDA(int i) {
              // sumfn dmuFn
              double fni = fn_loc(i);
              gETA_loc(i, i) += wk_loc * fni * (mo - fni) / sumfn * dFdmu;
            });
      }
    }

    for (auto& elem : gETA) {
      auto key = elem.first;
      int n = elem.second.array().extent(0);
      auto& f = fn[key];
      auto e = ek[key];
      auto mgETA = gETA[key].array();
      auto Hij_loc = Hij[key].array();

      Kokkos::parallel_for(
          "gEta, outer",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>({{0, 0}}, {{n, n}}),
          KOKKOS_LAMBDA(int i, int j) {
            if (i == j) {
              // do nothing
            } else {
              double ej = e(j);
              double ei = e(i);
              if (std::abs(ej - ei) < 1e-10) {
                // zero contribution
              } else {
                double II = (f(j) - f(i)) / (ej - ei);
                mgETA(i, j) += II * Hij_loc(i, j);
              }
            }
          });
    }
    return gETA;
  }

  struct _delta_eta
  {
    _delta_eta(double kappa) : kappa(kappa) {}
    template<class hij_t, class ek_t, class wk_t>
    to_layout_left_t<std::remove_reference_t<hij_t>>
    operator()(const hij_t& hij, const ek_t& ek, const wk_t& wk) const
    {
      using Kokkos::RangePolicy;
      using Kokkos::Rank;

      auto d_eta = empty_like()(hij);
      using memspc = typename hij_t::storage_t::memory_space;
      static_assert(std::is_same<memspc, typename ek_t::memory_space>::value, "memory spaces must match");
      scale(d_eta, hij, kappa / wk);
      int n = ek.extent(0);
      auto d_eta_array = d_eta.array();

      // CUDA will crash if it tries to access a member variable in a lambda capture ...
      double local_kappa = kappa;
      Kokkos::parallel_for(
          RangePolicy<exec_t<memspc>>(0, n),
          KOKKOS_LAMBDA(int i) {
            d_eta_array(i, i) = d_eta_array(i, i) - local_kappa * ek(i);
          });

      return d_eta;
    }

    double kappa;
  };

  /**
   * Preconditioned gradient of η
   */
  template <class matrix_t, class vector_t, class vector2_t>
  // mvector<std::shared_future<to_layout_left_t<matrix_t>>>
  auto
  delta_eta(const mvector<matrix_t>& Hij, const mvector<vector_t>& ek, const mvector<vector2_t>& wk)
  {
    // delta_eta = kappa * (hij - diag(ek))
    _delta_eta functor(kappa);
    return tapply_async(_delta_eta(kappa), Hij, ek, wk);
  }


private:
  // temperature (in Kelvin)
  double kappa;
  double kT;
};


}  // nlcglib
