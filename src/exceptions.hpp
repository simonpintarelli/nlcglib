#pragma once

#include <stdexcept>

namespace nlcglib {

class SlopeError : public std::exception
{
public:
  const char* what() const noexcept { return "Slope error"; }
};


class StepError : public std::exception
{
public:
  const char* what() const noexcept { return "Step error"; }
};


class DescentError : public std::exception
{
public:
  const char* what() const noexcept { return "CG failed try restart."; }
};


}  // nlcglib
