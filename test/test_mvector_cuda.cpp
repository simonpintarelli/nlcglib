#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include "smearing.hpp"
#include "la/mvector.hpp"

#ifdef __NLCGLIB_CUDA
void run()
{
  int n = 10;
  using vector_t = Kokkos::View<double*, Kokkos::CudaSpace>;
  vector_t a_view("test", n);


  auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_view);

  for (int i = 0; i < n; ++i) {
    host_view(i) = i+1;
  }

  Kokkos::deep_copy(a_view, host_view);
  vector_t copy(a_view);

  double foo = 1;

  Kokkos::parallel_for(
      "scale", Kokkos::RangePolicy<Kokkos::Cuda>(0, a_view.size()), KOKKOS_LAMBDA(int i) {
        a_view(i) = foo / sqrt(a_view(i));
      });
}
#endif


int
main(int argc, char *argv[])
{
  Kokkos::initialize();

#ifdef __NLCGLIB_CUDA
  run();
#endif
  Kokkos::finalize();
  return 0;
}
