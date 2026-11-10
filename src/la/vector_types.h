#pragma once

#include <Kokkos_Core_fwd.hpp>

namespace nlcglib {

template <class T>
class mvector;

template <class MemorySpace>
using real_vector_t = Kokkos::View<double*, MemorySpace>;

using host_vector_t = real_vector_t<Kokkos::HostSpace>;

}  // namespace nlcglib
