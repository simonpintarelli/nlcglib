#include <iostream>
#include "traits.hpp"

using namespace nlcglib;

class Foo
{
public:
  Foo() = default;
  Foo(const Foo& ) { std::cout << "copy ctr" << "\n";}
  Foo(Foo&&)
  {
    std::cout << "move ctr" << "\n";
  }
};


int main(int argc, char *argv[])
{
  auto x = eval(Foo());
  return 0;
}
