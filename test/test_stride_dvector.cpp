#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdlib.h>
#include "la/dvector.hpp"


auto unmanaged()
{
  std::cout << "\nunmanaged\n";
  int n = 10;
  double* A = new double[n*n];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[n*i + j] = n * i + j;
    }
  }

  Kokkos::View<double**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > a_view(
      A, n, n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << a_view(i, j) << " ";
    }
    std::cout << "\n";
  }

  return a_view;
}


auto unmanaged_strided()
{
  std::cout << "\nunmanaged_strided\n";
  int n = 10; // rows
  int m = 10; // cols
  int lda = 12;

  double* A;
  int res = posix_memalign(reinterpret_cast<void**>(&A), 256, lda*m*sizeof(double));
  std::cout << "A: " << A << "\n";
  std::cout << "poisx_memalign: " << res << "\n";
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      A[j*lda + i] = i*n + j;
    }
  }
  nlcglib::buffer_protocol<double, 2> buf{std::array<int, 2>{1, lda}, std::array<int, 2>{n, m}, A, nlcglib::memory_type::host};

  nlcglib::Map<> map(nlcglib::Communicator(), nlcglib::SlabLayoutV({{0, 0, n, m}}));
      nlcglib::KokkosDVector<double**,
                             nlcglib::SlabLayoutV,
                             Kokkos::LayoutStride,
                             Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        dvector(map, buf);

  // check daata
  bool passed = true;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      bool is_same = dvector.array()(i, j) == A[j * lda + i];
      passed = passed && is_same;
    }
  }
  if (passed) std::cout << "worked!\n";
  else std::cout << "failed!\n";

  return dvector;
}


int main(int argc, char *argv[])
{
  Kokkos::initialize();
  auto x = unmanaged();
  auto x2 = unmanaged();
  unmanaged_strided();

  std::cout << x.data() << "\n";
  x = x2;
  std::cout << x.data() << " (after assign x=x2)\n";
  std::cout << x2.data() << "\n";
  Kokkos::finalize();
  return 0;

}
