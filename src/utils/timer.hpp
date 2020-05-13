#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace nlcglib {

template<class T>
struct duration_string {};

template <>
struct duration_string<std::chrono::milliseconds>
{
  constexpr static char const* label = "[ms]";
};

template <>
struct duration_string<std::chrono::microseconds>
{
  constexpr static char const* label = "[us]";
};

template <>
struct duration_string<std::chrono::nanoseconds>
{
  constexpr static char const* label = "[ns]";
};

class Timer
{
public:
  typedef std::chrono::milliseconds type_t;

public:
  void start();

  double stop();

private:
  typedef std::chrono::high_resolution_clock::time_point time_point;
  time_point t;
};

void
Timer::start()
{
  this->t = std::chrono::high_resolution_clock::now();
}

double
Timer::stop()
{
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double>>(now - this->t).count();
}

}  // nlcglib
