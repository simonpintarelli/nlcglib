MACRO(NLCGLIB_SETUP_TARGET _target)
  add_dependencies(${_target} nlcglib_internal)
  # message("INTERNAL LIB LOC: ${nlcglib_internal_location}")
  target_link_libraries(
    ${_target} PRIVATE
    ${nlcglib_internal_location}
    Kokkos::kokkos
    ${LAPACK_LIBRARIES}
    MPI::MPI_CXX
    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    $<TARGET_NAME_IF_EXISTS:nlcglib::cudalibs>
    nlohmann_json::nlohmann_json
    )

  target_include_directories(${_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    )

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    target_compile_definitions(${_target} PRIVATE __CLANG)
  endif()

  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_OPENMP}>:__USE_OPENMP>)
  if(LAPACK_VENDOR MATCHES MKL)
    target_compile_definitions(${_target} PUBLIC __USE_MKL)
    target_link_libraries(${_target}  PRIVATE mkl::mkl_intel_32bit_omp_dyn)
  else()
    target_link_libraries(${_target} PRIVATE my_lapack)
  endif()
  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_OPENMP}>:__USE_OPENMP>)
  target_compile_definitions(${_target} PUBLIC $<$<BOOL:${USE_CUDA}>:__NLCGLIB__CUDA>)
  target_include_directories(${_target} PUBLIC $<TARGET_PROPERTY:Kokkos::kokkoscore,INTERFACE_INCLUDE_DIRECTORIES>)
ENDMACRO()
