#pragma once

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include "mpi/communicator.hpp"
#include "traits.hpp"
#include "helper/funcs.hpp"
#include <Kokkos_Core.hpp>
#include <cassert>
#include <memory>
#include <iomanip>
#include "nlcglib.hpp"
#include "la/dvector.hpp"
#include "utils.hpp"
#include "gpu/acc.hpp"

namespace nlcglib {

template<class DERIVED, class ELEM>
class mvector_base
{
};


template<class T>
class mvector : public mvector_base<mvector<T>, T> {
  static_assert(std::is_same<T, std::remove_reference_t<T>>::value, "must have ownership");
public:
  using value_type = T;
  using key_t = std::pair<int, int>;
  using container_t = std::map<key_t, T>;

public :
  mvector(Communicator comm) : comm_(comm) {}
  mvector() = default;
  mvector(const mvector&) = default;
  mvector(mvector&&) = default;
  mvector& operator=(const mvector&) = default;
  mvector(const container_t& data) : data_(data) {}

  T& operator[] (key_t k)
  {
    return data_[k];
  }

  const T& operator[] (key_t k) const
  {
    return data_.at(k);
  }

  T& at(key_t k)
  {
    return data_.at(k);
  }

  const T& at(key_t k) const
  {
    return data_.at(k);
  }

  auto begin()
  {
    return data_.begin();
  }

  auto end()
  {
    return data_.end();
  }

  auto begin() const { return data_.begin(); }

  auto end() const { return data_.end(); }

  auto& data()
  {
    return this->data_;
  }

  auto& data() const
  {
    return this->data_;
  }

  auto find()
  {
    return data_.find();
  }

  auto size() { return data_.size(); }
  auto size() const { return data_.size(); }

  mvector empty_like();

  mvector& operator=(std::map<key_t, T>&& data)
  {
    data_ = std::forward<std::map<key_t, T>>(data);
  }

  template<typename Z>
  mvector& operator=(mvector<Z>& other)
  {
    // iterate over keys and call assign
    for(auto& elem : data_) {
      auto key = elem.first;
      data_[key] = eval(other.at(key));
    }
    return *this;
  }

  const Communicator& commk() const
  {
    return comm_;
  }

  template<class X=T>
  std::enable_if_t<std::is_scalar<X>::value, mvector<X>>
  allgather(Communicator comm = Communicator{MPI_COMM_NULL}) const;

  template<class X=T>
  std::enable_if_t<is_kokkos_view<X>::value, mvector<X>>
  allgather(Communicator comm = Communicator{MPI_COMM_NULL}) const;

private:
  container_t data_;
  Communicator comm_;
};

template<class T>
template<class X>
std::enable_if_t<std::is_scalar<X>::value, mvector<X>>
mvector<T>::allgather(Communicator comm) const
{
  if (comm == Communicator{MPI_COMM_NULL}) comm = comm_;

  if (comm < comm_) {
    throw std::runtime_error("mvector::allgather: most likely gave unintended communicator");
  }

  mvector<T> result(Communicator{MPI_COMM_SELF});
  using value_type = std::pair<key_t, T>;
  int nranks = comm.size();
  int rank = comm.rank();
  std::vector<int> nelems(nranks);
  nelems[rank] = data_.size();
  comm.allgather(nelems.data(), 1);
  std::vector<int> scan(nranks, 0);
  std::partial_sum(nelems.begin(), nelems.end() - 1, scan.begin() + 1);

  int size = std::accumulate(nelems.begin(), nelems.end(), 0);
  std::vector<value_type> serialize_buffer(size);
  std::copy(data_.begin(), data_.end(), serialize_buffer.data() + scan[rank]);

  comm.allgather(serialize_buffer.data(), nelems);

  // copy data into return object
  result.data_ = container_t(serialize_buffer.begin(), serialize_buffer.end());

  return result;
}


template<class VAL>
std::vector<std::vector<VAL>> _allgather(const std::vector<VAL>& values, const Communicator& comm)
{
  int nranks = comm.size();
  std::vector<int> nelems(nranks);
  nelems[comm.rank()] = values.size();
  comm.allgather(nelems.data(), 1);
  int nelems_global = std::accumulate(nelems.begin(), nelems.end(), 0);
  std::vector<int> scan(nranks + 1);
  std::partial_sum(nelems.begin(), nelems.end(), scan.data() + 1);

  std::vector<VAL> sendrecv_buffer(nelems_global);
  std::copy(values.begin(), values.end(), sendrecv_buffer.data() + scan[comm.rank()]);
  comm.allgather(sendrecv_buffer.data(), nelems);

  std::vector<std::vector<VAL>> result(nranks);
  for (int i = 0; i < nranks; ++i) {
    result[i] = std::vector<VAL>(sendrecv_buffer.data() + scan[i], sendrecv_buffer.data() + scan[i+1]);
  }
  return result;
}


/// allgather, copy to host -> communicate -> copy to original memory (if needed)
template<class T>
template<class X>
std::enable_if_t<is_kokkos_view<X>::value, mvector<X>>
mvector<T>::allgather(Communicator comm) const
{
  if (comm == Communicator{MPI_COMM_NULL})
    comm = comm_;
  if (comm < comm_) {
    throw std::runtime_error("mvector::allgather: most likely gave unintended communicator");
  }

  static_assert(X::dimension::rank == 1, "implemented for 1D Views only.");

  using numeric_t = typename X::value_type;

  // using value_type = std::pair<key_t, T>;
  int nranks = comm.size();
  int rank = comm.rank();

  std::vector<key_t> local_keys;
  for (auto& elem : data_) local_keys.push_back(elem.first);

  auto global_keys = _allgather(local_keys, comm);

  // collect total size
  std::vector<int> local_number_of_elements(data_.size());
  std::transform(data_.begin(), data_.end(), local_number_of_elements.data(),
                 [](auto& elem) {return elem.second.size();});
  auto global_number_of_elements = _allgather(local_number_of_elements, comm);

  // collect the offsets as a list [every mpi rank] of list [every block]
  std::vector<std::vector<int>> offsets(nranks);
  int global_size{-1};
  {
    int offset = 0;
    for (int rank = 0; rank < nranks; ++rank) {
      for (auto lsize : global_number_of_elements[rank]) {
        offsets[rank].push_back(offset);
        offset += lsize;
      }
    }
    global_size = offset;
  }

  std::vector<numeric_t> send_recv_buffer(global_size);
  {
    int i{0};
    for (auto& elem : data_) {
      auto arr = elem.second;
      auto host_view = Kokkos::create_mirror_view(arr);
      Kokkos::deep_copy(host_view, arr);
      assert(offsets[rank][i] < send_recv_buffer.size());
      std::copy(host_view.data(), host_view.data() + host_view.size(),
                send_recv_buffer.data() + offsets[rank][i]);
      ++i;
    }
  }

  std::vector<int> recv_counts(comm.size());
  std::transform(global_number_of_elements.begin(), global_number_of_elements.end(),
                 recv_counts.begin(),
                 [](auto& in) { return std::accumulate(in.begin(), in.end(), 0); });
  comm.allgather(send_recv_buffer.data(), recv_counts);

  // copy into results and issue memory transfer if needed.
  mvector<T> result(Communicator{MPI_COMM_SELF});
  for(int rank = 0; rank < nranks; ++rank) {
    for (auto block_id = 0ul; block_id < offsets[rank].size(); ++block_id) {
      int lsize = global_number_of_elements[rank][block_id];
      int offset = offsets[rank][block_id];
      Kokkos::View<numeric_t*, Kokkos::HostSpace> tmp(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, ""),
          lsize);
      std::copy(send_recv_buffer.data() + offset, send_recv_buffer.data() + offset + lsize, tmp.data());
      T dst(Kokkos::view_alloc(Kokkos::WithoutInitializing, ""),
            lsize);
      Kokkos::deep_copy(dst, tmp);
      auto key = global_keys[rank][block_id];
      result[key] = dst;
    }
  }
  return result;
}


template<class mspc, class xspc=mspc, class _=void>
struct make_mmatrix_return_type
{};

template <class mspc, class xspc>
struct make_mmatrix_return_type<mspc, xspc, std::enable_if_t<std::is_same<mspc, xspc>::value>>
{
  using type = KokkosDVector<Kokkos::complex<double>**,
                             SlabLayoutV,
                             Kokkos::LayoutStride,
                             mspc,
                             Kokkos::MemoryUnmanaged>;
};

template <class mspc, class xspc>
struct make_mmatrix_return_type<mspc, xspc, std::enable_if_t<!std::is_same<mspc, xspc>::value>>
{
  using type = KokkosDVector<Kokkos::complex<double>**,
                             SlabLayoutV,
                             Kokkos::LayoutLeft,
                             xspc>;
  using view_type = KokkosDVector<Kokkos::complex<double>**,
                                  SlabLayoutV,
                                  Kokkos::LayoutStride,
                                  mspc,
                                  Kokkos::MemoryUnmanaged>;
};


/// @brief create an mvector from SIRIUS adaptor
/// @tparam T Kokkos memory space
/// @tparam execution memory space
template <class T, class X=T>
mvector<typename make_mmatrix_return_type<T, X>::type>
make_mmatrix(std::shared_ptr<MatrixBaseZ> matrix_base, std::enable_if_t<std::is_same<T, X>::value>* _ = nullptr)
{
  static_assert(std::is_same<T,X>::value, "invalid template parameters");
  using memspace = T;
  typedef typename make_mmatrix_return_type<T>::type matrix_t;
  mvector<matrix_t> mvector(Communicator(matrix_base->mpicomm()));
  // kokkosDvector
  int num_vec = matrix_base->size();
  for (int i = 0; i < num_vec; ++i) {
    auto buffer = matrix_base->get(i);
    auto kindex = matrix_base->kpoint_index(i);
    Communicator comm(buffer.mpi_comm);
#ifdef __NLCGLIB__CUDA
    if (Kokkos::SpaceAccessibility<Kokkos::Cuda, memspace>::accessible) {
      // make sure is memory type device
      if (buffer.memtype != memory_type::device)
        throw std::runtime_error("expected device memory, but got " +
                                 memory_names.at(buffer.memtype));
    }
#endif
    if (Kokkos::SpaceAccessibility<Kokkos::Serial, memspace>::accessible) {
      // make sure is memory type device
      if (buffer.memtype != memory_type::host)
        throw std::runtime_error("expected host memory");
    }
    mvector[kindex] =
      matrix_t(Map<>(comm, SlabLayoutV({{0, 0, buffer.size[0], buffer.size[1]}})), buffer);

  }
  return mvector;
}


/// copy implementation
template<class T, class X>
mvector<typename make_mmatrix_return_type<T, X>::type>
make_mmatrix(std::shared_ptr<MatrixBaseZ> matrix_base, std::enable_if_t<!std::is_same<T, X>::value>* _ =nullptr)
{
  static_assert(!std::is_same<T, X>::value, "invalid template parameters");
  using memspace = T;
  typedef typename make_mmatrix_return_type<T, X>::type matrix_t;
  mvector<matrix_t> mvector(Communicator(matrix_base->mpicomm()));
  // kokkosDvector
  int num_vec = matrix_base->size();
  for (int i = 0; i < num_vec; ++i) {
    auto buffer = matrix_base->get(i);
    auto kindex = matrix_base->kpoint_index(i);
    Communicator comm(buffer.mpi_comm);
#ifdef __NLCGLIB__CUDA
    if (Kokkos::SpaceAccessibility<Kokkos::Cuda, memspace>::accessible) {
      // make sure is memory type device
      if (buffer.memtype != memory_type::device)
        throw std::runtime_error("expected device memory, but got " +
                                 memory_names.at(buffer.memtype));
    }
#endif
    if (Kokkos::SpaceAccessibility<Kokkos::Serial, memspace>::accessible) {
      // make sure is memory type device
      if (buffer.memtype != memory_type::host)
        throw std::runtime_error("expected host memory");
    }
    // copy view T, using cuda memcpy ...
    matrix_t mat(Map<>(comm, SlabLayoutV({{0, 0, buffer.size[0], buffer.size[1]}})));
    // issue memcpy
    acc::copy(mat.array().data(), buffer.data, buffer.size[0]*buffer.size[1]);
    mvector[kindex] = mat;
  }
  return mvector;
}


template<class T>
auto make_mmvector(std::shared_ptr<VectorBaseZ> vector_base)
{
  using memspace = T;
  typedef Kokkos::View<double*, memspace> vector_t;
  mvector<vector_t> mvector(Communicator(vector_base->mpicomm()));
  int num_vec = vector_base->size();
  for (int i = 0; i < num_vec; ++i) {
    // vector_t vector();
    auto buffer = vector_base->get(i);
    if (buffer.memtype == memory_type::device) {
#ifdef __NLCGLIB__CUDA
      Kokkos::View<double*, Kokkos::CudaSpace, Kokkos::MemoryUnmanaged> src(buffer.data, buffer.size[0]);
      vector_t dst("vector", buffer.size[0]);
      Kokkos::deep_copy(dst, src);
      auto kindex = vector_base->kpoint_index(i);
      mvector[kindex] = dst;
#else
      throw std::runtime_error("recompile nlcglib with CUDA support");
#endif
    } else if (buffer.memtype == memory_type::host) {
      Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> src(buffer.data,
                                                                            buffer.size[0]);
      vector_t dst("vector", buffer.size[0]);
      Kokkos::deep_copy(dst, src);
      auto kindex = vector_base->kpoint_index(i);
      mvector[kindex] = dst;
    }
  }
  return mvector;
}


inline auto make_mmscalar(std::shared_ptr<ScalarBaseZ> scalar_base)
{
  mvector<ScalarBaseZ::buffer_t> mvector(Communicator(scalar_base->mpicomm()));
  int num_vec = scalar_base->size();
  for(int i = 0; i < num_vec; ++i) {
    auto v = scalar_base->get(i);
    auto key = scalar_base->kpoint_index(i);
    mvector[key] = v;
  }
  return mvector;
}


template <typename T>
auto
eval_threaded(const mvector<T>& input)
{
  mvector<std::remove_reference_t<decltype(eval(std::declval<T>()))>> result;
  for (auto& elem : input) {
    auto key = elem.first;
    result[key] = eval(elem.second);
  }
  return result;
}


template <typename T>
void execute(const mvector<T>& input)
{
  for (auto& elem : input) {
    eval(elem.second);
  }
}


template <class numeric_t, class... ARGS>
auto sum(const Kokkos::View<numeric_t*, ARGS...>& x)
{
  using view_type = Kokkos::View<numeric_t*, ARGS...>;
  static_assert(view_type::dimension::rank == 1,
                "KokkosView");

  auto host_mirror = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(host_mirror, x);

  return std::accumulate(host_mirror.data(), host_mirror.data()+ host_mirror.size(), 0.0);
}


// template <class T>
// std::enable_if_t<is_kokkos_view<eval_t<T>>::value, mvector<std::function<double()> > >
// sum(const mvector<T>& x)
// {
//   return tapply([](auto xi) { return(sum(eval(xi))); }, x);
// }


template<class T>
std::enable_if_t<std::is_scalar<eval_t<T>>::value || std::is_same<Kokkos::complex<double>, T>::value, eval_t<T>>
sum(const mvector<T>& x, Communicator comm = Communicator{MPI_COMM_NULL})
{
  if (comm == Communicator{MPI_COMM_NULL}) comm = x.commk();

  if (comm < x.commk()) {
    throw std::runtime_error("mvector::allgather: most likely gave unintended communicator");
  }

  eval_t<T> sum = 0;
  for (auto& elem: x) {
    sum += eval(elem.second);
  }
  return comm.allreduce(sum, mpi_op::sum);
}


template<class T1, class T2>
auto operator*(const mvector<T1>& a, const mvector<T2>& b)
{
  return tapply([](auto x, auto y) { return eval(x)*eval(y); }, a, b);
}


template<class numeric_t, class... ARGS>
std::enable_if_t<Kokkos::View<numeric_t*,ARGS...>::dimension::rank == 1>
print(const mvector<Kokkos::View<numeric_t*, ARGS...>>& vec)
{
  for (auto& elem : vec) {
    auto key = elem.first;
    auto& array = elem.second;
    auto host_array = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), array);
    std::cout << "kindex: " << key.first << ", " << key.second << "\n";
    for (auto i = 0ul; i < host_array.size(); ++i) {
      std::cout << std::setprecision(10) << host_array(i) << " ";
    }
    std::cout << "\n";
  }
}


struct do_copy {
  template<class X>
  to_layout_left_t<std::remove_reference_t<X>> operator()(X&& x)
  {
    auto copy = empty_like()(x);
    deep_copy(copy, x);
    return copy;
  }
};


template <class X>
auto copy(const mvector<X>& x)
{
  return eval_threaded(tapply(do_copy(), x));
}


template <class... T>
auto unzip(const mvector<std::tuple<T...>>& V) {
  std::tuple<mvector<T>...> U;

  for (auto& elem : V) {
    auto key = elem.first;
    unzip(elem.second, U, key);
  }

  return U;
}


template <class... T>
auto
unzip(const mvector<std::tuple<T...>>& V, const Communicator& commk)
{
  std::tuple<mvector<T>...> U = std::make_tuple(mvector<T>{commk}...);

  for (auto& elem : V) {
    auto key = elem.first;
    unzip(elem.second, U, key);
  }

  return U;
}


template <int POS>
struct unzip_impl
{
  template <class key_t, class... T>
  static auto apply(const std::tuple<T...>& src, std::tuple<mvector<T>...>& dst, const key_t& key)
  {
    auto& dsti = std::get<POS>(dst);
    dsti[key] = std::get<POS>(src);
    unzip_impl<POS - 1>::apply(src, dst, key);
  }
};

template <>
struct unzip_impl<0>
{
  template <class key_t, class... T>
  static auto apply(const std::tuple<T...>& src, std::tuple<mvector<T>...>& dst, const key_t& key)
  {
    auto& dsti = std::get<0>(dst);
    dsti[key] = std::get<0>(src);
  }
};

template <class key_t, class... T>
auto
unzip(const std::tuple<T...>& src, std::tuple<mvector<T>...>& dst, const key_t& key)
{
  using tuple_t = std::tuple<T...>;
  unzip_impl<std::tuple_size<tuple_t>::value-1>::apply(src, dst , key);
}


template <class numeric_t>
void
print(const mvector<numeric_t>& vec)
{
  for (auto& elem : vec) {
    auto key = elem.first;
    auto& val = elem.second;
    std::cout << "kindex (" << key.first << ", " << key.second << "): " << std::setprecision(10) << val << "\n";
  }
}

}  // nlcglib
