find_package(CUDAToolkit REQUIRED)
if (NOT TARGET nlcglib::cudalibs)
  add_library(nlcglib::cudalibs INTERFACE IMPORTED)
  target_link_libraries(nlcglib::cudalibs INTERFACE CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusolver)
endif()
