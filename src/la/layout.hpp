#pragma once

#include <mpi.h>
#include <initializer_list>
#include <vector>

namespace nlcglib {

/**
 * A block in a matrix starting at (x, y) of extent (nrows, ncols).
 */
struct Block
{
  Block(int x_, int y_, int nrows_, int ncols_)
      : x(x_)
      , y(y_)
      , nrows(nrows_)
      , ncols(ncols_)
  { /* empty */
  }

  Block() = default;

  /// row begin
  int x;
  /// col begin
  int y;
  int nrows;
  int ncols;
};

/**
 * An arbitrary collection of blocks.
 */
class BlockLayout
{
public:
  using block_t = Block;

public:
  BlockLayout(const std::vector<block_t>& blocks)
      : blocks_(blocks)
  {
  }

  // BlockLayout(const BlockLayout&) = default;
  // BlockLayout(BlockLayout&&) = default;
  BlockLayout() = default;
  // BlockLayout& operator=(const BlockLayout&) = default;
  // BlockLayout& operator=(BlockLayout&&) = default;

  const std::vector<block_t>& blocks() const { return blocks_; }

public:
  /// local number of rows
  int nrows() const { throw std::runtime_error("invalid"); }
  /// local number of columns
  int ncols() const { throw std::runtime_error("invalid"); }

protected:
  int nrow_{-1};
  int ncol_{-1};
  std::vector<block_t> blocks_;
};


/// Vertical slab layout.
class SlabLayoutV : public BlockLayout
{
public:
  SlabLayoutV(const std::vector<block_t>& blocks, int ncols = -1)
      : BlockLayout(blocks)
  {
    ncol_ = ncols;
    if (ncols == -1) ncol_ = blocks[0].ncols;
    nrow_ = 0;
    for (auto& block : blocks) {
      nrow_ += block.nrows;
      if (block.ncols != ncol_ || block.y != 0) {
        throw std::runtime_error("invalid layout\n");
      }
    }
  }

  SlabLayoutV() = default;
  // SlabLayoutV(const SlabLayoutV&) = default;
  // SlabLayoutV(SlabLayoutV&&) = default;
  /// local number of rows
  int nrows() const { return nrow_; }
  /// local number of columns
  int ncols() const { return ncol_; }
};


}  // namespace nlcglib
