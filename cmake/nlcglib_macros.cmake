MACRO(NLCGLIB_SETUP_TARGET _target)
  target_link_libraries(
    ${_target} PUBLIC
    Kokkos::kokkos
    MPI::MPI_CXX
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    $<TARGET_NAME_IF_EXISTS:nlcglib::cudalibs>
    $<TARGET_NAME_IF_EXISTS:nlcglib::rocmlibs>
    $<TARGET_NAME_IF_EXISTS:nlcglib::magma>
    $<TARGET_NAME_IF_EXISTS:roc::hipblas> # only required for magma
    $<TARGET_NAME_IF_EXISTS:roc::hipsparse> # only required for magma
    nlohmann_json::nlohmann_json
    nlcg::cpu_lapack
  )

  target_include_directories(${_target} PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
  )

  if(USE_ROCM)
    target_compile_options(${_target} PUBLIC --offload-arch=gfx90a)
  endif()

  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_OPENMP}>:__USE_OPENMP>)
  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_CUDA}>:__NLCGLIB__CUDA>)
  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_ROCM}>:__NLCGLIB__ROCM>)
  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_MAGMA}>:__NLCGLIB__MAGMA>)
  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_GPU_DIRECT}>:__NLCGLIB__GPU_DIRECT>)
  target_include_directories(${_target} PUBLIC $<TARGET_PROPERTY:Kokkos::kokkoscore,INTERFACE_INCLUDE_DIRECTORIES>)
ENDMACRO()
