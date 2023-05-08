#ifdef __NLCGLIB__ROCM
#if KOKKOS_VERSION > 30701
#include <HIP/Kokkos_HIP_Space.hpp>
#else
#include <Kokkos_HIP_Space.hpp>
#endif // KOKKOS > 3.7.01
#endif
