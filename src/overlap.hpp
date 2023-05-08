#pragma once

#include <memory>
#include "interface.hpp"
#include "la/mvector.hpp"
#include "la/dvector.hpp"
#include "operator.hpp"
#include "mpi/communicator.hpp"


namespace nlcglib {

/// Wrapper for overlap operation computed by sirius, behaves like mvector in an expression.
class Overlap
{
public:
  // need typedef for value_type
  using value_type = applicator<OverlapBase>;
  using key_t = std::pair<int, int>;

public:
  Overlap(const OverlapBase& overlap_base)
      : overlap_base(overlap_base)
  {
    /* empty */
  }

  auto at(const key_t& key) const -> value_type;

  auto begin() { return local::op_iterator<Overlap> (overlap_base.get_keys(), *this, false); }
  auto end() { return local::op_iterator<Overlap>(overlap_base.get_keys(), *this, true); }
  auto begin() const { return local::op_iterator<const Overlap>(overlap_base.get_keys(), *this, false); }
  auto end() const { return local::op_iterator<const Overlap>(overlap_base.get_keys(), *this, true); }

  Communicator commk() const {
    throw std::runtime_error("not implemented");
  }


private:
  const OverlapBase& overlap_base;
};

inline auto
Overlap::at(const key_t& key) const -> value_type
{
  return applicator<OverlapBase>(overlap_base, key);
}

}  // namespace nlcglib
