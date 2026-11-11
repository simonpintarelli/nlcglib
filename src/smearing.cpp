#include "smearing.h"

#include <Kokkos_Core.hpp>
#include <stdexcept>
#include "constants.hpp"
#include "dft/chemical_potential.h"
#include "dft/newton_minimization_smearing.hpp"
#include "la/mvector.hpp"
#include "smearing/kernels.h"
#include "utils/env.hpp"
#include "utils/logger.hpp"

namespace nlcglib {
namespace {

enum class quantity
{
  fn,
  delta,
  dxdelta,
  entropy
};

// Specialize only the band loops. Runtime dispatch happens once per band vector.
template <class Kernel, quantity Q>
double
sum_bands(const host_vector_t& ek, double mu, double T, double occ)
{
  double kT = physical_constants::kb * T;
  double result{0};
  for (size_t i = 0; i < ek.extent(0); ++i) {
    double x = -1.0 * (ek(i) - mu) / kT;
    if constexpr (Q == quantity::fn) {
      result += Kernel::fn(x, occ);
    } else if constexpr (Q == quantity::delta) {
      result += Kernel::delta(x, occ);
    } else if constexpr (Q == quantity::dxdelta) {
      result += Kernel::dxdelta(x, occ);
    } else {
      result += Kernel::entropy(x, occ);
    }
  }
  return result;
}

template <class Kernel>
void
fill_occupations(const host_vector_t& ek, const host_vector_t& fn, double mu, double kT, double occ)
{
  for (size_t i = 0; i < ek.size(); ++i) {
    fn(i) = Kernel::fn((mu - ek(i)) / kT, occ);
  }
}

struct kernel_functions
{
  using sum_type = double (*)(const host_vector_t&, double, double, double);
  sum_type fn;
  sum_type entropy;
  std::optional<sum_type> delta;
  std::optional<sum_type> dxdelta;
  void (*fill)(const host_vector_t&, const host_vector_t&, double, double, double);
};

template <class Kernel, bool non_monotonous = false>
kernel_functions
make_kernels()
{
  kernel_functions kernels{&sum_bands<Kernel, quantity::fn>,
                           &sum_bands<Kernel, quantity::entropy>,
                           nullptr,
                           nullptr,
                           &fill_occupations<Kernel>};
  // Monotone smearings never use the Newton derivatives.
  if constexpr (non_monotonous) {
    kernels.delta = &sum_bands<Kernel, quantity::delta>;
    kernels.dxdelta = &sum_bands<Kernel, quantity::dxdelta>;
  }
  return kernels;
}

kernel_functions
select_kernels(smearing_type type)
{
  switch (type) {
    case smearing_type::FERMI_DIRAC:
      return make_kernels<fermi_dirac_kernel>();
    case smearing_type::GAUSSIAN_SPLINE:
      return make_kernels<gaussian_spline_kernel>();
    case smearing_type::GAUSS:
      return make_kernels<gauss_smearing_kernel>();
    case smearing_type::COLD:
      return make_kernels<cold_smearing_kernel, true>();
    case smearing_type::METHFESSEL_PAXTON:
      return make_kernels<methfessel_paxton_smearing_kernel, true>();
    default:
      throw std::runtime_error("invalid smearing type");
  }
}

// Shared across all smearings: gather, chemical-potential iteration and assembly
// do not depend on the scalar kernel type.
struct electron_count
{
  const mvector<host_vector_t>& ek;
  const mvector<double>& wk;
  double T;
  double occ;
  double kT;

  double sum(kernel_functions::sum_type kernel, double mu, int derivative = 0) const
  {
    double result{0};
    for (const auto& wki : wk) {
      double weight = wki.second;
      if (derivative == 1) weight /= kT;
      if (derivative == 2) weight /= kT * kT;
      result += weight * kernel(ek.at(wki.first), mu, T, occ);
    }
    return result;
  }
};

double
chemical_potential(const electron_count& count,
                   const kernel_functions& kernels,
                   int Ne,
                   double tol,
                   bool use_newton)
{
  auto missing_electrons = [&](double mu) { return Ne - count.sum(kernels.fn, mu); };
  if (!use_newton) return find_chemical_potential(missing_electrons, 0, tol);

  if (use_newton && (!kernels.delta || !kernels.dxdelta)) {
    throw std::runtime_error(
        "Chemical potential: requested use_newton=true, but delta or dxdelta has not been defined");
  }

  double mu0 = find_chemical_potential(
      [&](double mu) {
        return Ne - count.sum(&sum_bands<gauss_smearing_kernel, quantity::fn>, mu);
      },
      0,
      tol);
  try {
    // check that delta, dxdelta has been defined
    return newton_minimization_chemical_potential(
        [&](double mu) { return count.sum(kernels.fn, mu); },
        [&](double mu) { return count.sum(*kernels.delta, mu, 1); },
        [&](double mu) { return count.sum(*kernels.dxdelta, mu, 2); },
        mu0,
        Ne,
        tol);
  } catch (failed_to_converge&) {
    Logger::GetInstance()
        << "Warning: newton minimization for Fermi energy failed, fallback to bisection search.\n";
    return find_chemical_potential(missing_electrons, 0, tol);
  }
}

}  // namespace

Smearing::Smearing(double T,
                   int num_electrons,
                   double max_occ,
                   const mvector<double>& wk,
                   smearing_type smearing_t)
    : T(T)
    , Ne(num_electrons)
    , occ(max_occ)
    , kT(T * physical_constants::kb)
    , wk(std::make_unique<mvector<double>>(wk))
    , smearing_t(smearing_t)
{
  if (T == 0) throw std::runtime_error("Temperature must be > 0.");
}

Smearing::~Smearing() = default;

std::tuple<double, mvector<host_vector_t>>
Smearing::fn(const mvector<host_vector_t>& x)
{
  auto kernels = select_kernels(smearing_t);
  bool skip_newton = env::get_skip_newton_efermi();
  bool use_newton = !skip_newton && kernels.delta != nullptr;

  mvector<host_vector_t> x_host;
  for (const auto& ek : x) x_host[ek.first] = ek.second;
  auto x_all = x_host.allgather(wk->commk());
  auto wk_all = wk->allgather();
  electron_count count{x_all, wk_all, T, occ, kT};
  double mu = chemical_potential(count, kernels, Ne, tol, use_newton);

  mvector<host_vector_t> fn;
  for (const auto& ek : x_host) {
    host_vector_t out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "fn"), ek.second.size());
    kernels.fill(ek.second, out, mu, kT, occ);
    fn[ek.first] = out;
  }

  if (use_newton) Logger::GetInstance().flush();
  return std::make_tuple(mu, fn);
}

double
Smearing::entropy(const mvector<host_vector_t>& en, double mu)
{
  auto kernels = select_kernels(smearing_t);
  // Keep the weights' communicator for the global k-point sum.
  double result{0};
  for (const auto& wki : *wk) {
    result += wki.second * kernels.entropy(en.at(wki.first), mu, T, occ);
  }
  return -1.0 * wk->commk().allreduce(result, mpi_op::sum);
}

}  // namespace nlcglib
