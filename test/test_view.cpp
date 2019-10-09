#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdlib.h>
#include "traits.hpp"

void run()
{
  if (nlcglib::is_kokkos_view<const Kokkos::View<double *>>::value) {
    std::cout << "yes"
              << "\n";
  } else {
    std::cout << "no"
              << "\n";
  }


  // test ctor
  Kokkos::View<double *, Kokkos::Cuda> b("foo", 10);
  Kokkos::View<double *, Kokkos::Cuda> c(b);
  std::cout << "b: " << b.data() << "\n";
  std::cout << "c: " << c.data() << "\n";
}


int main(int argc, char *argv[])
{
  Kokkos::initialize();
  run();
  Kokkos::finalize();
  return 0;
}
