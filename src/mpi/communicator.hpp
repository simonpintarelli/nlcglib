#pragma once

#include <mpi.h>
#include <vector>
#include <numeric>
#include "mpi_type.hpp"
#include <cassert>

#define CALL_MPI(func__, args__)                                        \
  {                                                                     \
    if (func__ args__ != MPI_SUCCESS) {                                 \
      printf("error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
      MPI_Abort(MPI_COMM_WORLD, -1);                                    \
    }                                                                   \
  }


namespace nlcglib {

class Communicator
{
 public:
  explicit Communicator(MPI_Comm mpicomm)
  {
    // duplicator communicator
    // MPI_Comm_dup(mpicomm, &mpicomm_);
    mpicomm_ = mpicomm;
  }

  Communicator()
  {
    mpicomm_ = MPI_COMM_SELF;
  }

  Communicator(const Communicator& other) {
    // MPI_Comm_dup(other.mpicomm_, &mpicomm_);
    mpicomm_ = other.mpicomm_;
  }

  Communicator(Communicator&& other) : mpicomm_(other.mpicomm_) {
    other.mpicomm_ = MPI_COMM_NULL;
  }

  Communicator& operator=(const Communicator& other)
  {
    // MPI_Comm_dup(other.mpicomm_, &mpicomm_);
    mpicomm_ = other.mpicomm_;
    return *this;
  }

  bool operator==(const Communicator& other)
  {
    int result;
    // check if both or one is null
    if (mpicomm_ == MPI_COMM_NULL && other.mpicomm_ == MPI_COMM_NULL)
      return true;
    if (mpicomm_ == MPI_COMM_NULL || other.mpicomm_ == MPI_COMM_NULL)
      return false;

    // compare using MPI_Comm_compare (does not allow to compare MPI_COMM_NULL)
    CALL_MPI(MPI_Comm_compare, (mpicomm_, other.mpicomm_, &result));
    if (result == MPI_IDENT)
      return true;
    else
      return false;
  }

  bool operator<(const Communicator& other)
  {
    return this->size() < other.size();
  }

  bool operator>(const Communicator& other)
  {
    return this->size() > other.size();
  }


  Communicator& operator=(Communicator&& other)
  {
    mpicomm_ = other.mpicomm_;
    other.mpicomm_ = 0;
    return *this;
  }

  static void init(int argc, char* argv[])
  {
    MPI_Init(&argc, &argv);
  }

  static void finalize()
  {
    MPI_Finalize();
  }

  int size() const
  {
    int size;
    CALL_MPI(MPI_Comm_size, (mpicomm_, &size));
    return size;
  }

  int rank() const
  {
    int rank;
    CALL_MPI(MPI_Comm_rank, (mpicomm_, &rank));
    return rank;
  }

  template<class T>
  void allgather(T* buffer, const std::vector<int>& recvcounts, const std::vector<int>& displs) const;

  template <class T>
  void allgather(T* buffer, const std::vector<int>& recvcounts) const;

  template <class T>
  void allgather(T* buffer, int recvcount) const;

  template <class VAL>
  std::vector<std::vector<VAL>> allgather(const std::vector<VAL>& values) const;

  template <class T>
  T allreduce(T val, enum mpi_op op) const;

  void barrier() const
  {
    CALL_MPI(MPI_Barrier, (mpicomm_));
  }

  ~Communicator() {
    // mpicomm_ is coming from outside
    // if (mpicomm_ != 0)
    //   MPI_Comm_free(&mpicomm_);
  }

  MPI_Comm raw() const
  {
    return mpicomm_;
  }

 private:
  MPI_Comm mpicomm_;
};

template <class T>
void
Communicator::allgather(T* buffer,
                        const std::vector<int>& recvcounts) const
{
  int nranks = this->size();
  assert(recvcounts.size() == nranks);
  std::vector<int> displs(nranks, 0);
  std::partial_sum(recvcounts.begin(), recvcounts.end()-1, displs.begin()+1);

  CALL_MPI(MPI_Allgatherv,
           (MPI_IN_PLACE, 0,
            MPI_DATATYPE_NULL,
            buffer,
            recvcounts.data(),
            displs.data(),
            mpi_type<T>::type(),
            mpicomm_));
}

template <class T>
void
Communicator::allgather(T* buffer,
                        const std::vector<int>& recvcounts,
                        const std::vector<int>& displs) const
{
  // put assert statements
  CALL_MPI(MPI_Allgatherv, (MPI_IN_PLACE,
                            0,
                            MPI_DATATYPE_NULL,
                            buffer,
                            recvcounts.data(),
                            displs.data(),
                            mpi_type<T>::type(),
                            mpicomm_));
}

template <class T>
void Communicator::allgather(T* buffer, int recvcount) const
{
  CALL_MPI(MPI_Allgather, (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buffer, recvcount, mpi_type<T>::type(), mpicomm_));
}

template <class VAL>
std::vector<std::vector<VAL>>
Communicator::allgather(const std::vector<VAL>& values) const
{
  int nranks = this->size();
  std::vector<int> nelems(nranks);
  nelems[this->rank()] = values.size();
  this->allgather(nelems.data(), 1);
  int nelems_global = std::accumulate(nelems.begin(), nelems.end(), 0);
  std::vector<int> scan(nranks + 1);
  std::partial_sum(nelems.begin(), nelems.end(), scan.data() + 1);

  std::vector<VAL> sendrecv_buffer(nelems_global);
  std::copy(values.begin(), values.end(), sendrecv_buffer.data() + scan[this->rank()]);
  this->allgather(sendrecv_buffer.data(), nelems);

  std::vector<std::vector<VAL>> result(nranks);
  for (int i = 0; i < nranks; ++i) {
    result[i] =
        std::vector<VAL>(sendrecv_buffer.data() + scan[i], sendrecv_buffer.data() + scan[i + 1]);
  }
  return result;
}

template <class T>
T Communicator::allreduce(T val, enum mpi_op op) const
{
  T result{0};
  switch (op) {
    case mpi_op::sum: {
      CALL_MPI(MPI_Allreduce,
               (&val, &result, 1, mpi_type<T>::type(), mpi_op_<mpi_op::sum>::value(), mpicomm_));
      break;
    }
    case mpi_op::min: {
      CALL_MPI(MPI_Allreduce,
               (&val, &result, 1, mpi_type<T>::type(), mpi_op_<mpi_op::min>::value(), mpicomm_));
      break;
    }
    case mpi_op::max: {
      CALL_MPI(MPI_Allreduce,
               (&val, &result, 1, mpi_type<T>::type(), mpi_op_<mpi_op::max>::value(), mpicomm_));
      break;
    }
    default: {
      throw std::runtime_error("Error: invalid MPI_Op given.");
    }
  }

  return result;
}

}  // namespace nlcglib
