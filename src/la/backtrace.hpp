#pragma once

#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <cstdio>

inline void
stack_backtrace()
{
  void* array[10];
  char** strings;
  int size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  std::printf("Stack backtrace:\n");
  for (int i = 0; i < size; i++) {
    std::printf("%s\n", strings[i]);
  }
  raise(SIGQUIT);
}
