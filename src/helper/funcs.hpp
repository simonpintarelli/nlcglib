#pragma once

#include <vector>

namespace nlcglib {

template <typename T>
auto
flatten(const std::vector<std::vector<T>>& in)
{
  std::vector<T> out;
  for (auto& vec : in) {
    for (auto& elem : vec) {
      out.push_back(elem);
    }
  }
  return out;
}

}  // namespace nlcglib
