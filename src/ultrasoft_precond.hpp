#pragma once

namespace nlcglib {

#include "interface.hpp"
#include "la/dvector.hpp"
#include "la/mvector.hpp"

/// Wrapper for overlap operation computed by sirius, behaves like mvector in an expression.
class USPreconditioner
{

public:
  // TODO: rename to k_index
  using key_t = std::pair<int, int>;

public:
  USPreconditioner(const UltrasoftPrecondBase& us_precond_base)
      : us_precond_base(us_precond_base)
  {
    /* empty */
  }

  auto at(const key_t& key) const;

  template <typename MVEC>
  auto operator()(MVEC&& X)
  {
    return tapply_op(*this, std::forward<MVEC>(X));
  }

private:
  const UltrasoftPrecondBase& us_precond_base;
};

inline auto
USPreconditioner::at(const key_t& key) const
{
  auto& ref = us_precond_base;
  return [&ref, key](auto X) {
    auto Y = empty_like()(X);
    auto vX = as_buffer_protocol(X);
    auto vY = as_buffer_protocol(Y);
    ref.apply(key, vY, vX);
    return Y;
  };
}


}  // nlcglib
