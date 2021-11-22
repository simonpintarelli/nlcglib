#include "smearing.hpp"
#include <Kokkos_Core.hpp>

using namespace nlcglib;


void
run(smearing_type smearing_t)
{
  using cont = typename mvector<double>::container_t;

  int nk = 2;
  // int num_electrons = 100;
  // int num_bands = 200;

  int num_electrons = 30;
  int num_bands = 1500;


  Communicator comm(MPI_COMM_WORLD);

  int nranks = comm.size();
  int pid = comm.rank();

  int nk_loc = nk / comm.size();

  if (pid == nranks-1) {
    nk_loc = nk - (nranks - 1) * nk_loc;
  }


  mvector<double> wk(comm);
  {
    double wk_ = 1. / nk;

    for (int i = 0; i < nk_loc; ++i) {
      wk[std::make_pair(pid * nk / nranks + i, 0)] = wk_;
    }

    double check = sum(wk, comm);
    if (std::abs(check-1) > 1e-10)  {
      std::cout << sum(wk,comm) << "\n";
      throw std::runtime_error("wrong weights");
    }
  }

  if(pid == nranks-1) {
    print(wk);
  }

  std::cout << "before wk all gather"
            << "\n";
  auto wk_all = wk.allgather();
  std::cout << "all weights"
            << "\n";
  print(wk_all);

  double T{50000};
  Smearing smearing(T, num_electrons, 1, wk, smearing_t);

  using vec_t = Kokkos::View<double *, Kokkos::HostSpace>;

  mvector<vec_t> ek;

  for (int i = 0; i < nk_loc; ++i) {

    auto key = std::make_pair(pid * nk / nranks + i, 0);

    double lb = -10;
    vec_t eki("ek" + std::to_string(i), num_bands);
    eki(0) = lb;
    for (int ib = 1; ib < num_bands; ++ib) {
      eki(ib) = eki(ib-1) + std::exp(-0.05*ib);
    }
    ek[key] = eki;
    if (i == 0)
    print(ek);
  }

  auto mu_fn = smearing.fn(ek);
  double S = smearing.entropy(std::get<1>(mu_fn), ek, std::get<0>(mu_fn));
  double smax = comm.allreduce(S, mpi_op::max);
  if ( S != smax) {
    throw std::runtime_error("entropy differs");
  }
  std::cout << "entropy is " << S << "\n";
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize();
  // run(smearing_type::GAUSSIAN_SPLINE);
  run(smearing_type::GAUSSIAN_SPLINE);
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
