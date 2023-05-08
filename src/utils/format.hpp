#pragma once

#include <iostream>
#include <cstdio>

namespace nlcglib {

template<class... ARGS>
std::string format(std::string format_string, ARGS&&... args)
{
  char buf[format_string.size()];
  std::sprintf(buf, format_string.c_str(), args...);

  return std::string(buf);
}

}  // nlcglib
