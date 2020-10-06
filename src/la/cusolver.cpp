#ifdef __CUDA
#include "cusolver.hpp"
#include <iostream>

namespace cusolver {

void
error_message(cusolverStatus_t status)
{
  switch (status) {
    case CUSOLVER_STATUS_NOT_INITIALIZED: {
      std::printf("the CUDA Runtime initialization failed\n");
      break;
    }
    case CUSOLVER_STATUS_ALLOC_FAILED: {
      std::printf("the resources could not be allocated\n");
      break;
    }
    case CUSOLVER_STATUS_ARCH_MISMATCH: {
      std::printf("the device only supports compute capability 2.0 and above\n");
      break;
    }
    case CUSOLVER_STATUS_INVALID_VALUE: {
      std::printf("An unsupported value or parameter was passed to the function\n");
      break;
    }
    case CUSOLVER_STATUS_EXECUTION_FAILED: {
      std::printf(
          "The GPU program failed to execute. This is often caused by a launch failure of the "
          "kernel on the GPU, which can be caused by multiple reasons.\n");
      break;
    }
    case CUSOLVER_STATUS_INTERNAL_ERROR: {
      std::printf(
          "An internal cuSolver operation failed. This error is usually caused by a "
          "cudaMemcpyAsync() failure.\n");
      break;
    }
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: {
      std::printf(
          "The matrix type is not supported by this function. This is usually caused by passing an "
          "invalid matrix descriptor to the function.\n");
      break;
    }
    default: {
      std::printf("cusolver status unknown\n");
    }
  }
}

}  // namespace cusolver

#endif
