#include <mpi.h>

void
nlcglib_consumer_run();

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  nlcglib_consumer_run();
  MPI_Finalize();
  return 0;
}
