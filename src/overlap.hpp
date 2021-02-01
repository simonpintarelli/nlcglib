#pragma once

#include "interface.hpp"
#include "la/mvector.hpp"
#include "la/dvector.hpp"

namespace nlcglib {

/// Wrapper for overlap operation computed by sirius, behaves like mvector in an expression.
class Overlap
{
public:
  // need typedef for value_type
  using value_type = std::function<KokkosDVector<Kokkos::complex<double>**, SlabLayoutV, Kokkos::LayoutLeft, Kokkos::HostSpace>()>;
  using key_t = std::pair<int, int>;

public:
  Overlap(const OverlapBase& overlap_base)
      : overlap_base(overlap_base)
  {
    /* empty */
  }

  auto at(const key_t& key) const;

  template<typename MVEC>
  auto operator()(MVEC&& X)
  {
    return tapply_op(*this, std::forward<MVEC>(X));
  }

private:
  const OverlapBase& overlap_base;
};

inline auto
Overlap::at(const key_t& key) const
{
  auto& ref = overlap_base;
  return [&ref, key](auto X) {
    auto Y = empty_like()(X);
    // passed as (non-const) reference, must be lvalues ...
    auto vY = as_buffer_protocol(Y);
    auto vX = as_buffer_protocol(X);
    ref.apply(key, vY, vX);
    return Y;
  };
}

// class Matrix : public MatrixBaseZ
// {
// public:
//   Matrix(const std::vector<buffer_t>& data,
//          const std::vector<kindex_t>& indices,
//          MPI_Comm mpi_comm = MPI_COMM_SELF)
//       : data(data)
//       , indices(indices)
//       , mpi_comm(mpi_comm)
//   {
//   }

//   Matrix(std::vector<buffer_t>&& data,
//           std::vector<kindex_t>&& indices,
//           MPI_Comm mpi_comm = MPI_COMM_SELF)
//       : data{std::forward<std::vector<buffer_t>>(data)}
//       , indices{std::forward<std::vector<kindex_t>>(indices)}
//       , mpi_comm(mpi_comm)
//   { /* empty */
//   }

//   buffer_t get(int i) override { return data[i]; }
//   const buffer_t get(int i) const override { return data[i]; }


//   int size() const override { return data.size(); };

//   MPI_Comm mpicomm(int i) const override { return data[i].mpi_comm; }

//   MPI_Comm mpicomm() const override { return mpi_comm; }

//   kindex_t kpoint_index(int i) const override { return indices[i]; }

// private:
//   std::vector<buffer_t> data;
//   std::vector<kindex_t> indices;
//   MPI_Comm mpi_comm;
// };

// template <class mvectorX>
// conditional_add_const_t<Matrix, std::is_const<mvectorX>::value>
// make_buffer(mvectorX&& X)
// {
//   //
//   std::vector<Matrix::buffer_t> data;
//   std::vector<Matrix::kindex_t> indices;

//   for (auto elem : X) {
//     auto x_data = elem.second;
//     auto key = elem.first;

//     auto mem_t = get_mem_type(x_data);
//     std::array<int, 2> strides;
//     strides[0] = x_data.array().stride(0);
//     strides[1] = x_data.array().stride(1);

//     std::array<int, 2> sizes;
//     sizes[0] = x_data.array().extent(0);
//     sizes[1] = x_data.array().extent(1);

//     // TODO: is MPI_COMM_SELF always correct here?
//     Matrix::buffer_t buf(strides, sizes, x_data.array().data(), mem_t, MPI_COMM_SELF);
//     indices.push_back(key);
//     data.push_back(buf);
//   }
//   return Matrix(data, indices, X.commk().raw());
// }


}  // namespace nlcglib
