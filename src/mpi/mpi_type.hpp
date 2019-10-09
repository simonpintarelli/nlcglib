#include <mpi.h>
#include <utility>
#include <complex>
#include <Kokkos_Core.hpp>

namespace nlcglib {

#define CALL_MPI(func__, args__)                                                  \
  {                                                                               \
    if (func__ args__ != MPI_SUCCESS) {                                           \
      printf("error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
      MPI_Abort(MPI_COMM_WORLD, -1);                                              \
    }                                                                             \
  }

enum class mpi_op
{
  sum,
  max,
  min
};

template <enum mpi_op>
struct mpi_op_{};

template<>
struct mpi_op_<mpi_op::sum>
{
  static MPI_Op value() { return MPI_SUM; }
};

template <>
struct mpi_op_<mpi_op::max>
{
  static MPI_Op value() { return MPI_MAX; }
};

template <>
struct mpi_op_<mpi_op::min>
{
  static MPI_Op value() { return MPI_MIN; }
};

template <typename T>
struct mpi_type
{
};

template <>
struct mpi_type<double>
{
  static MPI_Datatype type() { return MPI_DOUBLE; }
};

template <>
struct mpi_type<std::complex<double>>
{
  static MPI_Datatype type() { return MPI_CXX_DOUBLE_COMPLEX; }
};

template <>
struct mpi_type<Kokkos::complex<double>>
{
  static MPI_Datatype type() { return MPI_CXX_DOUBLE_COMPLEX; }
};


template <>
struct mpi_type<char>
{
  static MPI_Datatype type() { return MPI_CHAR; }
};

template <>
struct mpi_type<int>
{
  static MPI_Datatype type() { return MPI_INT; }
};

template <class T1, class T2>
struct mpi_type<std::pair<T1, T2>>
{
  static MPI_Datatype type()
  {
    MPI_Datatype result;
    int array_of_block_lengths[2] = {1, 1};
    MPI_Aint array_of_displacements[2] = {0, sizeof(T1)};
    MPI_Datatype types[2] = {mpi_type<T1>::type(), mpi_type<T2>::type()};
    CALL_MPI(
        MPI_Type_create_struct, (2,
                                 array_of_block_lengths,
                                 array_of_displacements,
                                 types,
                                 &result));
    CALL_MPI(MPI_Type_commit, (&result));
    return result;
  }
};


}  // namespace nlcglib
