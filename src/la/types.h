#pragma once

#include <Kokkos_Core.hpp>
#include "la/vector_types.h"

namespace nlcglib {

class SlabLayoutV;

template <class T, class Layout, class... Properties>
class KokkosDVector;

template <class MemorySpace>
using dense_matrix_t =
    KokkosDVector<Kokkos::complex<double>**, SlabLayoutV, Kokkos::LayoutLeft, MemorySpace>;

using host_matrix_t = dense_matrix_t<Kokkos::HostSpace>;
using host_matrix_view_t = KokkosDVector<Kokkos::complex<double>**,
                                         SlabLayoutV,
                                         Kokkos::LayoutStride,
                                         Kokkos::HostSpace,
                                         Kokkos::MemoryUnmanaged>;

}  // namespace nlcglib
