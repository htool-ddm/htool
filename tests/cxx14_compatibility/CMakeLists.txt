# Not linked to htool: its cxx_std_17 compile feature would override CXX_STANDARD 14.
add_executable(check_cxx14_compatibility check_cxx14_compatibility.cpp)
target_include_directories(check_cxx14_compatibility PRIVATE $<TARGET_PROPERTY:htool,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(check_cxx14_compatibility PRIVATE $<TARGET_PROPERTY:htool,INTERFACE_COMPILE_DEFINITIONS>)
target_link_libraries(check_cxx14_compatibility PRIVATE MPI::MPI_CXX BLAS::BLAS LAPACK::LAPACK)
if(OpenMP_CXX_FOUND)
    target_link_libraries(check_cxx14_compatibility PRIVATE OpenMP::OpenMP_CXX)
endif()
if(HPDDM_FOUND)
    target_compile_definitions(check_cxx14_compatibility PRIVATE HTOOL_WITH_HPDDM)
endif()
target_compile_options(check_cxx14_compatibility PRIVATE ${HTOOL_WARNING_FLAGS})
if(HTOOL_WITH_STRICT_TESTS)
    target_compile_options(check_cxx14_compatibility PRIVATE -Werror)
endif()
set_target_properties(
    check_cxx14_compatibility
    PROPERTIES CXX_STANDARD 14
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)
add_test(NAME check_cxx14_compatibility COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/check_cxx14_compatibility)
set_tests_properties(check_cxx14_compatibility PROPERTIES LABELS "mpi")
add_dependencies(build-tests check_cxx14_compatibility)
