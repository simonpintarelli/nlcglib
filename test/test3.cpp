#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include "smearing.hpp"
#include "la/mvector.hpp"


auto
run()
{
  int n = 10;
  using vector_t = Kokkos::View<double*, Kokkos::HostSpace>;
  vector_t a_view("test", n);
  for (int i = 0; i < n; ++i) {
    a_view(i) = i;
  }

  vector_t copy(a_view);

  std::cout << "copy.data: " << copy.data() << "\n";
  std::cout << "a_view.data: " << a_view.data() << "\n";
  for (int i = 0; i < n; ++i) {
    std::cout << " " << copy(i);
  }
  std::cout << "\n";

  // // just checking if it is compiling ...
  // nlcglib::FermiDirac FD(300, 1, nlcglib::mvector<double>());

  // auto fn = FD.fn(a_view);
  // auto ek = FD.ek(a_view);

  return a_view;
}


int
main(int argc, char *argv[])
{
  Kokkos::initialize();
  run();
  Kokkos::finalize();
  return 0;
}
