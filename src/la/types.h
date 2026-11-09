#pragma once

#include "la/dvector.hpp"

namespace nlcglib {

template <class MemorySpace>
using dense_matrix_t =
    KokkosDVector<Kokkos::complex<double>**, SlabLayoutV, Kokkos::LayoutLeft, MemorySpace>;

template <class MemorySpace>
using real_vector_t = Kokkos::View<double*, MemorySpace>;

using host_matrix_t = dense_matrix_t<Kokkos::HostSpace>;
using host_vector_t = real_vector_t<Kokkos::HostSpace>;
using host_matrix_view_t = KokkosDVector<Kokkos::complex<double>**,
                                         SlabLayoutV,
                                         Kokkos::LayoutStride,
                                         Kokkos::HostSpace,
                                         Kokkos::MemoryUnmanaged>;

}  // namespace nlcglib
