#include <iostream>
#include <utility>
#include <vector>

#include "mpi/communicator.hpp"

using namespace nlcglib;


void
run_int()
{
  Communicator comm(MPI_COMM_WORLD);


  std::vector<int> buffer = {0, 1, 2, 3};
  if (comm.rank() == 0) {
    buffer[2] = 0;
    buffer[3] = 0;
  } else {
    buffer[0] = 0;
    buffer[1] = 0;
  }

  comm.allgather(buffer.data(), 2);

  for (auto i = 0ul; i < buffer.size(); ++i) {
    std::cout << "rank " << comm.rank() << ": " << buffer[i] << "\n";
  }
}


void
run_pair()
{
  Communicator comm(MPI_COMM_WORLD);
  using type = std::pair<int, int>;

  std::vector<type> buffer(4);
  if (comm.rank() == 0) {
    buffer[0] = std::make_pair(0, 0);
    buffer[1] = std::make_pair(1, 0);
  } else {
    buffer[2] = std::make_pair(2, 0);
    buffer[3] = std::make_pair(3, 0);
  }

  comm.allgather(buffer.data(), 2);

  for (auto i = 0ul; i < buffer.size(); ++i) {
    std::cout << "rank " << comm.rank() << ": " << buffer[i].first << " , " << buffer[i].second << "\n";
  }
}


void
run_pair2()
{
  Communicator comm(MPI_COMM_WORLD);
  using type = std::pair<std::pair<int, int>, int>;

  std::vector<type> buffer(4);
  if (comm.rank() == 0) {
    buffer[0] = std::make_pair(std::make_pair(0, 0), 0);
    buffer[1] = std::make_pair(std::make_pair(1, 0), 10);
  } else {
    buffer[2] = std::make_pair(std::make_pair(2, 0), 20);
    buffer[3] = std::make_pair(std::make_pair(3, 0), 30);
  }

  comm.allgather(buffer.data(), 2);

  for (auto i = 0ul; i < buffer.size(); ++i) {
    auto p1 = buffer[i];
    std::cout << "rank " << comm.rank() << ": ("
              << p1.first.first << " , "
              << p1.first.second << ") , "
              << p1.second << "\n";
  }
}


int
main(int argc, char *argv[])
{
  Communicator::init(argc, argv);

  run_int();
  run_pair();
  run_pair2();

  Communicator::finalize();

  return 0;
}
