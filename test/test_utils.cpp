#include "la/mvector.hpp"

using namespace nlcglib;


template <class T>
struct print_type
{
};

void
test_unzip()
{
  mvector<std::tuple<int, int, double>> Z;
  auto X = unzip(Z);
  auto x0 = std::get<0>(X);
  // print_type<decltype(X)>::info;
}


int
main(int, char*[])
{
  test_unzip();
  return 0;
}
