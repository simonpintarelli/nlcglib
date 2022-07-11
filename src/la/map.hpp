#pragma once

#include "la/layout.hpp"
#include "mpi/communicator.hpp"

namespace nlcglib {

/**
 * Map representing a distributed layout.
 */
template <class LAYOUT = SlabLayoutV>
class Map
{
public:
  using layout_t = LAYOUT;

public:
  Map(const Communicator& comm, const layout_t& layout = layout_t{})
      : comm_(comm)
      , layout_(layout)
  { /* empty */
  }

  Map(const Map& other) = default;
  Map(Map&& other) = default;
  Map() = default;
  Map& operator=(const Map& other) = default;
  Map& operator=(Map&& other) = default;

  /// global number of rows
  int nrows() const { return layout_.nrows(); }
  /// global number of columns
  int ncols() const { return layout_.ncols(); }
  bool is_local() const { return comm_.size() == 1; }
  Communicator& comm() { return comm_; }
  const Communicator& comm() const { return comm_; }

private:
  Communicator comm_;
  layout_t layout_;
};

}  // namespace nlcglib
