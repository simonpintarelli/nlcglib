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

  auto begin() { return local::op_iterator<USPreconditioner> (us_precond_base.get_keys(), *this, false); }
  auto end() { return local::op_iterator<USPreconditioner>(us_precond_base.get_keys(), *this, true); }
  auto begin() const { return local::op_iterator<const USPreconditioner>(us_precond_base.get_keys(), *this, false); }
  auto end() const { return local::op_iterator<const USPreconditioner>(us_precond_base.get_keys(), *this, true); }

private:
  const UltrasoftPrecondBase& us_precond_base;
};

inline auto
USPreconditioner::at(const key_t& key) const
{
  return applicator<UltrasoftPrecondBase>(us_precond_base, key);
}


}  // nlcglib
