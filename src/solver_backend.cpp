#include "solver_state.h"

#include <Kokkos_Core.hpp>
#include <iostream>
#include <tuple>
#include "free_energy.h"
#include "la/lapack.hpp"
#include "la/mvector.hpp"
#include "la/types.h"
#include "mvp2/descent_direction.h"
#include "nlcglib.hpp"
#include "overlap.hpp"
#include "ultrasoft_precond.hpp"
#include "utils/profile.hpp"
#include "utils/step_logger.h"
#ifdef __NLCGLIB__MAGMA
#include "la/magma.hpp"
#endif

namespace nlcglib {
namespace {

using geodesic_result =
    std::tuple<mvector<host_vector_t>, mvector<host_matrix_t>, mvector<host_matrix_t>>;

template <class MemorySpace>
auto
advance(const dense_matrix_t<MemorySpace>& X,
        const real_matrix_t<MemorySpace>& eta,
        const dense_matrix_t<MemorySpace>& z_x,
        const dense_matrix_t<MemorySpace>& z_eta,
        const applicator<OverlapBase>& S,
        double t)
{
  PROFILE("geodesic_us::exec");
  auto eta_next = empty_like()(z_eta);
  deep_copy(eta_next, eta);
  add(eta_next, z_eta, t);

  auto Ul = empty_like()(eta_next);
  real_vector_t<MemorySpace> ek("eigvals, eta", Ul.map().ncols());
  eigh(Ul, ek, eta_next);

  auto x_next = empty_like()(X);
  deep_copy(x_next, X);
  add(x_next, z_x, t);
  x_next = loewdin(x_next, S(x_next));
  x_next = transform_alloc(x_next, Ul);
  return std::make_tuple(ek, Ul, x_next);
}

template <class MemorySpace>
geodesic_result
geodesic(const MemorySpace& mem_space,
         const mvector<host_matrix_t>& X_h,
         const mvector<host_real_matrix_t>& eta_h,
         const mvector<host_matrix_t>& z_x_h,
         const mvector<host_matrix_t>& z_eta_h,
         const Overlap& S,
         double t)
{
  PROFILE("geodesic");
  auto advance_local = [mem_space, t](const host_matrix_t& X_h,
                                      const host_real_matrix_t& eta_h,
                                      const host_matrix_t& z_x_h,
                                      const host_matrix_t& z_eta_h,
                                      const applicator<OverlapBase>& S) {
    PROFILE("geodesic_us");
    auto X = create_mirror_view_and_copy(mem_space, X_h);
    auto eta = create_mirror_view_and_copy(mem_space, eta_h);
    auto z_x = create_mirror_view_and_copy(mem_space, z_x_h);
    auto z_eta = create_mirror_view_and_copy(mem_space, z_eta_h);
    auto result = advance(X, eta, z_x, z_eta, S, t);

    host_vector_t ek =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), std::get<0>(result));
    auto Ul = create_mirror_view_and_copy(Kokkos::HostSpace(), std::get<1>(result));
    auto x_next = create_mirror_view_and_copy(Kokkos::HostSpace(), std::get<2>(result));
    return std::make_tuple(ek, Ul, x_next);
  };
  return unzip(tapply_eager(advance_local, X_h, eta_h, z_x_h, z_eta_h, S));
}

template geodesic_result
geodesic(const Kokkos::HostSpace&,
         const mvector<host_matrix_t>&,
         const mvector<host_real_matrix_t>&,
         const mvector<host_matrix_t>&,
         const mvector<host_matrix_t>&,
         const Overlap&,
         double);
#ifdef __NLCGLIB__CUDA
template geodesic_result
geodesic(const Kokkos::CudaSpace&,
         const mvector<host_matrix_t>&,
         const mvector<host_real_matrix_t>&,
         const mvector<host_matrix_t>&,
         const mvector<host_matrix_t>&,
         const Overlap&,
         double);
#elif defined(__NLCGLIB__ROCM)
template geodesic_result
geodesic(const Kokkos::HIPSpace&,
         const mvector<host_matrix_t>&,
         const mvector<host_real_matrix_t>&,
         const mvector<host_matrix_t>&,
         const mvector<host_matrix_t>&,
         const Overlap&,
         double);
#endif

geodesic_result
geodesic(execution_backend backend,
         const mvector<host_matrix_t>& X,
         const mvector<host_real_matrix_t>& eta,
         const mvector<host_matrix_t>& z_x,
         const mvector<host_matrix_t>& z_eta,
         const Overlap& S,
         double t)
{
  return dispatch_backend(backend, [&]<class MemorySpace>() {
    return geodesic(MemorySpace{}, X, eta, z_x, z_eta, S, t);
  });
}

mvector<host_matrix_t>
copy_orbitals(const mvector<host_matrix_view_t>& X)
{
  return copy(X);
}

mvector<host_matrix_t>
empty_orbitals_like(const mvector<host_matrix_t>& X)
{
  return tapply_eager([](const host_matrix_t& x) { return empty_like()(x); }, X);
}

mvector<host_real_matrix_t>
diagonal_matrix(const mvector<host_vector_t>& ek)
{
  return tapply_eager(make_diag(), ek);
}

}  // namespace

void
nlcheck_overlap(EnergyBase& e, OverlapBase& Sb, OverlapBase& Sib)
{
  FreeEnergy Energy(100, e, smearing_type::FERMI_DIRAC);

  auto X = copy(Energy.get_X());
  Overlap S(Sb);
  Overlap Sinv(Sib);

  std::cout << "l2norm(X) = " << l2norm(X) << "\n";

  auto SX = tapply_op(S, X);
  auto SinvX = tapply_op(Sinv, X);
  std::cout << "l2norm(SX): " << l2norm(SX) << "\n";
  std::cout << "l2norm(SinvX): " << l2norm(SinvX) << "\n";

  auto tr = innerh_reduce(X, SX);
  std::cout << "tr(XSX): " << tr << "\n";
  auto Xref = tapply(
      [](auto x, auto s, auto si) {
        auto sx = s(x);
        auto x2 = si(sx);
        return x2;
      },
      X,
      S,
      Sinv);

  auto error = tapply(
      [](auto x, auto y) {
        auto z = copy(x);
        add(z, y, -1, 1);
        return z;
      },
      X,
      Xref);

  double diff = l2norm(error);
  std::cout << "** check: S(S_inv(x)), error: " << diff << "\n";
}

struct solver_state::Impl
{
  Impl(execution_backend backend,
       EnergyBase& energy,
       UltrasoftPrecondBase& preconditioner,
       OverlapBase& overlap,
       InverseOverlapBase& inverse_overlap,
       smearing_type smear,
       double T,
       double kappa)
      : backend(backend)
      , comm_world(energy.comm_world())
      , S(overlap)
      , Sinv(inverse_overlap)
      , P(preconditioner)
      , energy(T, energy, smear)
      , direction(backend, T, kappa, smear)
  {
  }

  execution_backend backend;
  Communicator comm_world;
  Overlap S, Sinv;
  USPreconditioner P;
  FreeEnergy energy;
  descent_direction direction;
  mvector<host_vector_t> ek, fn;
  mvector<double> wk;
  mvector<host_matrix_t> X, Hx, z_x, z_eta, ul;
  mvector<host_real_matrix_t> eta;
  // Retain the initial chemical potential used by the existing restart paths.
  double initial_mu;
  geodesic_result trial;
  double trial_mu;
};

solver_state::solver_state(execution_backend backend,
                           EnergyBase& energy,
                           UltrasoftPrecondBase& preconditioner,
                           OverlapBase& overlap,
                           InverseOverlapBase& inverse_overlap,
                           smearing_type smear,
                           double T,
                           double kappa)
    : impl(std::make_unique<Impl>(
          backend, energy, preconditioner, overlap, inverse_overlap, smear, T, kappa))
{
}

solver_state::~solver_state() = default;

void
solver_state::compute()
{
  impl->energy.compute();
}

void
solver_state::initialize_orbitals()
{
  auto& s = *impl;
  s.ek = s.energy.get_ek();
  s.wk = s.energy.get_wk();
  auto mu_fn = s.energy.occupations(s.ek);
  s.initial_mu = std::get<0>(mu_fn);
  // Non-const lvalue assignment only evaluates existing mvector keys.
  s.fn = std::move(std::get<1>(mu_fn));
  auto X0 = s.energy.get_X();
  s.energy.compute(X0, s.fn, s.ek, s.initial_mu);
  s.Hx = copy_orbitals(s.energy.get_HX());
  s.X = copy_orbitals(s.energy.get_X());
}

slope_t
solver_state::initial_direction()
{
  auto& s = *impl;
  s.eta = diagonal_matrix(s.ek);
  auto slope = restarted();
  s.ul = empty_orbitals_like(s.z_eta);
  return slope;
}

slope_t
solver_state::restarted()
{
  auto& s = *impl;
  auto [slope, z_x, z_eta] =
      s.direction.restarted(s.X, s.ek, s.fn, s.Hx, s.wk, s.initial_mu, s.S, s.P, s.energy);
  s.z_x = std::move(z_x);
  s.z_eta = std::move(z_eta);
  return slope;
}

slope_t
solver_state::restarted_sd()
{
  auto& s = *impl;
  auto [slope, z_x, z_eta] = s.direction.restarted_sd(
      s.X, s.ek, s.fn, s.Hx, s.wk, s.initial_mu, s.S, s.Sinv, s.P, s.energy);
  s.z_x = std::move(z_x);
  s.z_eta = std::move(z_eta);
  return slope;
}

double
solver_state::evaluate(double step)
{
  auto& s = *impl;
  s.trial = geodesic(s.backend, s.X, s.eta, s.z_x, s.z_eta, s.S, step);
  auto ek = std::get<0>(s.trial);
  auto Xn = std::get<2>(s.trial);
  auto mu_fn = s.energy.occupations(ek);
  s.trial_mu = std::get<0>(mu_fn);
  s.energy.compute(Xn, std::get<1>(mu_fn), ek, s.trial_mu);
  return step;
}

void
solver_state::discard_trial()
{
  impl->trial = geodesic_result{};
}

std::pair<slope_t, slope_t>
solver_state::accept(slope_t fr)
{
  auto& s = *impl;
  s.ek = std::get<0>(s.trial);
  s.ul = std::get<1>(s.trial);
  s.X = std::get<2>(s.trial);
  s.eta = diagonal_matrix(s.ek);
  s.fn = s.energy.get_fn();
  s.Hx = copy_orbitals(s.energy.get_HX());
  auto [next_fr, slope, z_x, z_eta] = s.direction.conjugated(
      fr, s.X, s.ek, s.fn, s.Hx, s.z_x, s.z_eta, s.ul, s.wk, s.trial_mu, s.S, s.P, s.energy);
  s.z_x = std::move(z_x);
  s.z_eta = std::move(z_eta);
  discard_trial();
  return {next_fr, slope};
}

double
solver_state::get_F() const
{
  return impl->energy.get_F();
}

double
solver_state::get_entropy() const
{
  return impl->energy.get_entropy();
}

double
solver_state::ks_energy() const
{
  return impl->energy.ks_energy();
}

double
solver_state::get_chemical_potential() const
{
  return impl->energy.get_chemical_potential();
}

const EnergyBase&
solver_state::ehandle() const
{
  return impl->energy.ehandle();
}

void
solver_state::write_json(slope_t slope, int step, int frequency)
{
  auto& s = *impl;
  cg_write_step_json(s.energy.get_F(),
                     s.energy.ks_energy(),
                     s.energy.get_entropy(),
                     slope.x,
                     slope.eta,
                     s.energy.get_chemical_potential(),
                     s.ek,
                     s.fn,
                     s.wk,
                     s.energy.ks_energy_components(),
                     s.comm_world,
                     step,
                     frequency);
}

void
initialize()
{
  Kokkos::InitializationSettings args;
  args.set_disable_warnings(true);
#ifdef USE_OPENMP
  args.num_threads = omp_get_max_threads();
#endif
#ifdef __NLCGLIB__MAGMA
  nlcg_init_magma();
#endif
  Kokkos::initialize(args);
}

void
finalize()
{
  Kokkos::finalize();
#ifdef __NLCGLIB__MAGMA
  nlcg_finalize_magma();
#endif
}

}  // namespace nlcglib
