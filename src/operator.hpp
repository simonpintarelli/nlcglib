#pragma once

#include "la/dvector.hpp"
#include "la/mvector.hpp"

namespace nlcglib {

namespace local {

template <class iterable_t>
class op_iterator
{
public:
  using key_t = std::pair<int, int>;
  using value_t = std::pair<key_t, typename iterable_t::value_type>;

public:
  op_iterator(const std::vector<key_t>& keys, iterable_t& obj, bool end)
      : keys(keys)
      , obj(obj)
  {
    if (end) {
      pos = keys.size();
    } else {
      key_t k = keys[0];
      pair = std::make_unique<value_t>(k, obj.at(k));
    }
  }

  op_iterator& operator++()
  {
    pos++;
    pair = std::make_unique<value_t>(keys[pos], obj.at(keys[pos]));
    return *this;
  }

  std::pair<key_t, typename iterable_t::value_type>& operator*()
  {
    // auto key = this->keys[pos];
    // return std::make_pair(key, obj.at(key));
    return *pair;
  }

  bool operator!=(const op_iterator<iterable_t>& other) { return this->pos != other.pos; }

private:
  std::vector<key_t> keys;
  iterable_t& obj;
  std::unique_ptr<value_t> pair;
  int pos{0};
};


}  // namespace local


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
  auto operator()(X_t&& X) const
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


}  // namespace nlcglib
