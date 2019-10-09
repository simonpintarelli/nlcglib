#pragma once

#include <cusolverDn.h>
#include <cusolver_common.h>
#include <execinfo.h>
#include <signal.h>
#include <cstdio>
#include <unistd.h>
#include <iostream>

#include "backtrace.hpp"

#define CALL_CUSOLVER(func__, args__)                                                \
{                                                                                    \
  cusolverStatus_t status;                                                           \
  if ((status = func__ args__) != CUSOLVER_STATUS_SUCCESS) {                         \
    cusolver::error_message(status);                                                 \
    char nm[1024];                                                                   \
    gethostname(nm, 1024);                                                           \
    std::printf("hostname: %s\n", nm);                                               \
    std::printf("Error in %s at line %i of file %s\n", #func__, __LINE__, __FILE__); \
    stack_backtrace();                                                               \
  }                                                                                  \
}


namespace cusolver {

void error_message(cusolverStatus_t status);

struct cusolverDnHandle
{
  static cusolverDnHandle_t& _get()
  {
    static cusolverDnHandle_t handle{nullptr};
    return handle;
  }

  static cusolverDnHandle_t& get()
  {
    auto& handle = _get();
    if (!handle) {
      CALL_CUSOLVER(cusolverDnCreate, (&handle));
    }
    return handle;
  }

  static void destroy()
  {
    if(!_get()) CALL_CUSOLVER(cusolverDnDestroy, (_get()));
    _get() = nullptr;
  }

  static cusolverDnHandle_t handle;
};




}  // namespace cusolver
