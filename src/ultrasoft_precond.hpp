#pragma once

namespace nlcglib {

#include "interface.hpp"
#include "la/dvector.hpp"
#include "la/mvector.hpp"

/// Wrapper for overlap operation computed by sirius, behaves like mvector in an expression.
class USPreconditioner
{
public:
  USPreconditioner(const UltrasoftPrecondBase& us_precond_base)
      : us_precond_base(us_precond_base)
  {
    /* empty */
  }

  auto at(const key_t& key) const;

private:
  const UltrasoftPrecondBase& us_precond_base;
};

inline auto
USPreconditioner::at(const key_t& key) const
{
  auto& ref = us_precond_base;
  return [&ref, key](auto X) {
    auto Y = empty_like()(X);
    ref.apply(key, as_buffer_protocol(Y), as_buffer_protocol(X));
    return Y;
  };
}


}  // nlcglib
