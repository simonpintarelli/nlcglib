#pragma once

#include "la/mvector.hpp"
#include "la/dvector.hpp"

namespace nlcglib {

template <class T>
class applicator
{
public:
  applicator(const T& op, std::pair<int, int> key)
      : op(op)
      , key(key)
  {
  }

  template <class X_t>
  auto operator()(X_t&& X)
  {
    auto Y = empty_like()(X);
    auto vX = as_buffer_protocol(X);
    auto vY = as_buffer_protocol(Y);
    op.apply(key, vY, vX);
    return Y;
  }

private:
  const T& op;
  std::pair<int, int> key;
};


}  // nlcglib
