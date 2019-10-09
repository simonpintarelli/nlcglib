#pragma once

#include <cublas_v2.h>
#include <iostream>

namespace cublas {
struct cublasHandle
{
private:
  static cublasHandle_t& _get()
  {
    static cublasHandle_t handle{nullptr};
    return handle;
  }
public:
  static cublasHandle_t& get()
  {
    cublasHandle_t& handle = _get();
    if(!handle) {
      cublasCreate(&handle);
    }
    return handle;
  }

  static void destroy()
  {
    if(!_get()) cublasDestroy(_get());
    _get() = nullptr;
  }
};


}  // cublas
