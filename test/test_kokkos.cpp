#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdlib.h>
/// to have exp available on device
#include <math.h>


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
  // how does it work with strides?

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
  posix_memalign(reinterpret_cast<void**>(&A), 256, lda*m*sizeof(double));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      A[j*lda + i] = i*n + j;
    }
  }
  typedef Kokkos::View<double**, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      vector_t;
  vector_t a_view(A, Kokkos::LayoutStride(n, 1, m, lda));

  // int nelems = a_view.size();

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      std::cout << a_view(i, j) << " ";
    }
    std::cout << "\n";
  }
  return a_view;
}


void kokkos_view_stuff()
{
  Kokkos::View<double*, Kokkos::CudaSpace> view;

  view = Kokkos::View<double*, Kokkos::CudaSpace>("nope", 100);
  Kokkos::View<double*, Kokkos::HostSpace> view2("nope", 100);

  // accessing host memory space from Device will crash
  // at runtime (not caught at compile time)
  Kokkos::parallel_for("foo", Kokkos::RangePolicy<Kokkos::Cuda>(0, 100),
                       KOKKOS_LAMBDA (int i){
                         double f = view2(i) * view(i);
                         view2(i) = f;
                       });
}


template<typename numeric_t>
void kokkos_reduction()
{
  using space = Kokkos::HostSpace;
  using exec_space = Kokkos::Serial;
  // using space = Kokkos::CudaSpace;
  // using exec_space = Kokkos::Cuda;

  int n  = 100;
  Kokkos::View<numeric_t*, space> view("", n);
  Kokkos::parallel_for("foo", Kokkos::RangePolicy<exec_space>(0, n),
                       KOKKOS_LAMBDA (int i)
                       {
                         view(i) = i;
                       }
                      );

  numeric_t sum = 0;
  Kokkos::parallel_reduce("summation", Kokkos::RangePolicy<exec_space>(0, view.size()),
                          KOKKOS_LAMBDA (int i, numeric_t& loc_sum) { loc_sum += view(i); }, sum);
  std::cout << "Sum is: " << sum << "\n";
}

template <typename numeric_t>
void
kokkos_reduction_cuda()
{
  // using space = Kokkos::HostSpace;
  // using exec_space = Kokkos::Serial;
  using space = Kokkos::CudaSpace;
  using exec_space = Kokkos::Cuda;

  int n = 100;
  Kokkos::View<numeric_t*, space> view("", n);
  Kokkos::parallel_for(
      "foo", Kokkos::RangePolicy<exec_space>(0, n), KOKKOS_LAMBDA(int i) { view(i) = i; });

  numeric_t sum = 0;
  Kokkos::parallel_reduce(
      "summation",
      Kokkos::RangePolicy<exec_space>(0, view.size()),
      KOKKOS_LAMBDA(int i, numeric_t& loc_sum) { loc_sum += view(i); },
      sum);
  std::cout << "Sum is: " << sum << "\n";
}


struct fun
{
  __device__ __host__ double operator()(double x) const { return 1 / (1 + exp(x)); }
};

void test_template_lambda()
{
  // using space = Kokkos::HostSpace;
  // using exec_space = Kokkos::Serial;
  using space = Kokkos::CudaSpace;
  using exec_space = Kokkos::Cuda;
  using numeric_t = double;

  // int n = 100;
  // Kokkos::View<numeric_t*, space> view("", n);
  // Kokkos::parallel_for(
  //     "foo", Kokkos::RangePolicy<exec_space>(0, n), [view, f=fun()] __device__ __host__ (int i) {
  //       view(i) = f(i);
  //     });
}

int main(int argc, char *argv[])
{
  Kokkos::initialize();
  auto x = unmanaged();
  auto x2 = unmanaged();
  unmanaged_strided();

  std::cout << "trying reduction on cpu: " << "\n";
  kokkos_reduction<double>();
  std::cout << "trying reduction on gpu: " << "\n";
  kokkos_reduction_cuda<double>();
  // // test
  // kokkos_view_stuff();

  std::cout << x.data() << "\n";
  x = x2;
  std::cout << x.data() << " (after assign x=x2)\n";
  std::cout << x2.data() << "\n";
  Kokkos::finalize();
  return 0;
}
