#pragma once

#include "interface.hpp"
#include "la/mvector.hpp"

namespace nlcglib {

class Matrix : public nlcglib::MatrixBaseZ
{
  public:
    Matrix(const std::vector<buffer_t>& data, const std::vector<kindex_t>& indices, MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data(data)
        , indices(indices)
        , mpi_comm(mpi_comm)
    {
    }

    Matrix(std::vector<buffer_t>&& data, std::vector<kindex_t>&& indices, MPI_Comm mpi_comm = MPI_COMM_SELF)
        : data{std::forward<std::vector<buffer_t>>(data)}
        , indices{std::forward<std::vector<kindex_t>>(indices)}
        , mpi_comm(mpi_comm)
    { /* empty */
    }

    buffer_t get(int i) override;
    const buffer_t get(int i) const override;

    int size() const override
    {
        return data.size();
    };

    MPI_Comm mpicomm(int i) const override
    {
        return data[i].mpi_comm;
    }

    MPI_Comm mpicomm() const override
    {
        return mpi_comm;
    }

    kindex_t kpoint_index(int i) const override
    {
        return indices[i];
    }

  private:
    std::vector<buffer_t> data;
    std::vector<kindex_t> indices;
    MPI_Comm mpi_comm;
};


class Overlap
{
public:
  Overlap(const OverlapBase& overlap_base)
      : overlap_base(overlap_base)
  {
    /* empty */
  }

  template<class tX>
  auto compute(const mvector<tX>& X) const;

private:
  const OverlapBase& overlap_base;
};

template<class tX>
auto Overlap::compute(const mvector<tX>& X) const
{
  // allocate return type
  using matrix_t = to_layout_left<tX>;
  auto Y = empty_like(X);
  // extract views from X, Y and call apply
  for (auto item = X.begin(); item != X.end(); ++item) {

  }

  overlap_base.apply(Y, X);
  return Y;
}


}  // nlcglib
