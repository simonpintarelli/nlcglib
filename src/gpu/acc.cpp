#include <Kokkos_Complex.hpp>

namespace nlcglib {

#define CALL_DEVICE_API(func__, args__)                      \
  {                                                          \
    cudaError_t error;                                       \
    error = func__ args__;                                   \
    if (error != cudaSuccess) {                              \
      std::printf("Error in %s at line %i of file %s: %s\n", \
                  #func__,                                   \
                  __LINE__,                                  \
                  __FILE__,                                  \
                  cudaGetErrorString(error));                \
    }                                                        \
  }

namespace acc {

template <class T>
void
copy(T* dst, const T* src, size_t n)
{
  assert(src != nullptr);
  assert(dst != nullptr);
  CALL_DEVICE_API(cudaMemcpy, (dst, src, n * sizeof(T)));
}

template <>
void
copy(double*, const double*, size_t);

template <>
void
copy(Kokkos::complex<double>*, const Kokkos::complex<double>*, size_t);

}  // acc
}  // namespace nlcglib
