#pragma once

#include <memory>
#include "interface.hpp"
#include "la/mvector.hpp"
#include "la/dvector.hpp"
#include "operator.hpp"
#include "mpi/communicator.hpp"


namespace nlcglib {

namespace local {

template<class iterable_t>
class op_iterator
{
public:
  using key_t = std::pair<int, int>;
  using value_t = std::pair<key_t, typename iterable_t::value_type>;

public:
  op_iterator(const std::vector<key_t>& keys, iterable_t& obj, bool end)
      : keys(keys),
        obj(obj)  {
    if (end) {
      pos = keys.size();
    } else {
      key_t k = keys[0];
      pair = std::make_unique<value_t>(k, obj.at(k));
    }
  }

  op_iterator& operator++ () {
    pos++;
    pair = std::make_unique<value_t>(keys[pos], obj.at(keys[pos]));
    return *this;
  }

  std::pair<key_t, typename iterable_t::value_type>& operator*() {
    // auto key = this->keys[pos];
    // return std::make_pair(key, obj.at(key));
    return *pair;
  }

  bool operator!= (const op_iterator<iterable_t>& other) {
    return this->pos != other.pos;
  }

private:
  std::vector<key_t> keys;
  iterable_t& obj;
  std::unique_ptr<value_t> pair;
  int pos{0};
};


}  // local

/// Wrapper for overlap operation computed by sirius, behaves like mvector in an expression.
class Overlap
{
public:
  // need typedef for value_type
  using value_type = applicator<OverlapBase>;
  using key_t = std::pair<int, int>;

public:
  Overlap(const OverlapBase& overlap_base)
      : overlap_base(overlap_base)
  {
    /* empty */
  }

  auto at(const key_t& key) const -> value_type;

  auto begin() { return local::op_iterator<Overlap> (overlap_base.get_keys(), *this, false); }
  auto end() { return local::op_iterator<Overlap>(overlap_base.get_keys(), *this, true); }
  auto begin() const { return local::op_iterator<const Overlap>(overlap_base.get_keys(), *this, false); }
  auto end() const { return local::op_iterator<const Overlap>(overlap_base.get_keys(), *this, true); }

  // void test()
  // {
  //   key_t k(0,0);
  //   using pair_t = std::pair<key_t, value_type>;
  //   pair_t p(k, this->at(k));
  //   std::unique_ptr<pair_t> x = std::make_unique<pair_t>(k, this->at(k));
  // }

  Communicator commk() const {
    throw std::runtime_error("not implemented");
  }


private:
  const OverlapBase& overlap_base;
};

inline auto
Overlap::at(const key_t& key) const -> value_type
{
  return applicator<OverlapBase>(overlap_base, key);
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
