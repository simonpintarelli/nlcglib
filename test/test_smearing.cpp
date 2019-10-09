#include "smearing.hpp"
#include <Kokkos_Core.hpp>

using namespace nlcglib;


void run(smearing_type smearing_t)
{
  using cont = typename mvector<double>::container_t;
  mvector<double> wk;
  wk[std::make_pair<int, int>(0, 0)] = 1;

  Smearing smearing(100., 3, 1, wk, smearing_t);

  using vec_t = Kokkos::View<double *, Kokkos::HostSpace>;


  mvector<vec_t> fn;

  vec_t fni("", 5);
  fni(0) = 1;
  fni(1) = 1;
  fni(2) = 0.7;
  fni(3) = 0.2;
  fni(4) = 0.1;
  fni(5) = 0;

  fn[std::make_pair<int, int>(0, 0)] = fni;
  print(fn);

  auto ek = smearing.ek(fn);
  std::cout << "ek"
            << "\n";
  print(ek);

  // print(ek);
  auto fn2 = smearing.fn(ek);
  std::cout << "fn2"
            << "\n";
  print(fn2);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize();
  run(smearing_type::GAUSSIAN_SPLINE);
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
