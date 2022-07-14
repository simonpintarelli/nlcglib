#pragma once

#include <rocblas.h>


struct rocblasHandle
{
  static rocblas_handle _get()
  {
    static rocblas_handle handle{nullptr};
    return handle;
  }

  static rocblas_handle get()
  {
    auto handle = _get();
    if (!handle) {
      rocblas_create_handle(&handle);
    }
    return handle;
  }
};


namespace nlcglib {

namespace rocm {



}  // rocm


}  // nlcglib
