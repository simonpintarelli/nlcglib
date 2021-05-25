#pragma once

#include <array>
#include <complex>
#include <memory>
#include <map>
#include <stdexcept>
#include <functional>
#include <vector>
#include "mpi.h"

namespace nlcglib {

enum class memory_type
{
  none,
  host,
  device
};

static std::map<memory_type, std::string> memory_names = {{memory_type::none, "none"},
                                                          {memory_type::host, "host"},
                                                          {memory_type::device, "device"}};

enum class smearing_type
{
  FERMI_DIRAC,
  GAUSSIAN_SPLINE
};


struct nlcg_info
{
  double tolerance;
  double F;
  double S;
  int iter;
};


template <typename T, int d>
struct buffer_protocol
{
  buffer_protocol() = default;
  buffer_protocol(std::array<int, d> stride,
                  std::array<int, d> size,
                  T* data,
                  enum memory_type memtype,
                  MPI_Comm mpi_comm=MPI_COMM_SELF)
      : stride(std::move(stride))
      , size(std::move(size))
      , data(data)
      , memtype(memtype)
      , mpi_comm(mpi_comm)
  { /* empty */ }

  // 1d constructor
  // template<int k=dim, class=std::enable_if_t<k==1>>
  buffer_protocol(int size,
                  T* data,
                  enum memory_type memtype,
                  MPI_Comm mpi_comm= MPI_COMM_SELF)
      : buffer_protocol({1}, {size}, data, memtype, mpi_comm)
  {
    static_assert(d == 1, "not available.");
  }

  buffer_protocol(buffer_protocol&&) = default;
  buffer_protocol(const buffer_protocol&) = default;

  std::array<int, d> stride;
  std::array<int, d> size;
  T* data;
  enum memory_type memtype;
  MPI_Comm mpi_comm{MPI_COMM_SELF};
};

template<int dim, class numeric_t>
class BufferBase
{
public:
  using buffer_t = buffer_protocol<numeric_t, dim>;
  using kindex_t = std::pair<int, int>;

public:
  /// get buffer description of entry i
  virtual buffer_t get(int i) = 0;
  /// get buffer description of entry i
  virtual const buffer_t get(int i) const = 0;
  /// number of entries
  virtual int size() const = 0;
  /// MPI communicator of i-th entry
  virtual MPI_Comm mpicomm(int i) const = 0;
  /// MPI communicator over which entries are distributed
  virtual MPI_Comm mpicomm() const = 0;
  virtual kindex_t kpoint_index(int i) const = 0;
};

template<class numeric_t>
class BufferBase<0, numeric_t>
{
public:
  using buffer_t = numeric_t;
  using kindex_t = std::pair<int, int>;

public:
  /// get buffer description of entry i
  virtual buffer_t get(int i) = 0;
  /// get buffer description of entry i
  virtual const buffer_t get(int i) const = 0;
  /// number of entries
  virtual int size() const = 0;
  /// MPI communicator of i-th entry
  virtual MPI_Comm mpicomm(int i) const = 0;
  /// MPI communicator over which entries are distributed
  virtual MPI_Comm mpicomm() const = 0;
  virtual kindex_t kpoint_index(int i) const = 0;
};

using MatrixBaseZ = BufferBase<2, std::complex<double>>;
using VectorBaseZ = BufferBase<1, double>;
using ScalarBaseZ = BufferBase<0, double>;

class EnergyBase
{
public:
  EnergyBase() = default;
  virtual void compute() = 0;
  /// return the number of electrons
  virtual int nelectrons() = 0;
  /// maximum number of electrons per orbital
  virtual int occupancy() = 0;
  virtual double get_total_energy() = 0;
  virtual std::map<std::string, double> get_energy_components() = 0;
  virtual std::shared_ptr<MatrixBaseZ> get_hphi() = 0;
  virtual std::shared_ptr<MatrixBaseZ> get_sphi() = 0;
  virtual std::shared_ptr<MatrixBaseZ> get_C(memory_type) = 0;
  virtual std::shared_ptr<VectorBaseZ> get_fn() = 0;
  virtual void set_fn(const std::vector<std::pair<int, int>>&, const std::vector<std::vector<double>>&) = 0;
  virtual std::shared_ptr<VectorBaseZ> get_ek() = 0;
  virtual std::shared_ptr<VectorBaseZ> get_gkvec_ekin() = 0;
  virtual std::shared_ptr<ScalarBaseZ> get_kpoint_weights() = 0;
  virtual void print_info() const = 0;
};

class OpBase
{
public:
  using key_t = std::pair<int, int>;
public:
  virtual void apply(const key_t&,
                     MatrixBaseZ::buffer_t& out,
                     MatrixBaseZ::buffer_t& in) const = 0;
  virtual std::vector<key_t> get_keys() const = 0;
};

class OverlapBase : public OpBase
{
};

class UltrasoftPrecondBase : public OpBase
{
};

}  // namespace nlcglib
