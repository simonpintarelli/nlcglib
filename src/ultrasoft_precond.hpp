#pragma once

#include "interface.hpp"
#include "la/dvector.hpp"
#include "la/mvector.hpp"
#include "operator.hpp"

namespace nlcglib {

/// Wrapper for overlap operation computed by sirius, behaves like mvector in an expression.
class USPreconditioner
{

public:
  // TODO: rename to k_index
  using key_t = std::pair<int, int>;
  using value_type = applicator<UltrasoftPrecondBase>;

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
  return applicator<UltrasoftPrecondBase>(us_precond_base, key);
}


}  // nlcglib
