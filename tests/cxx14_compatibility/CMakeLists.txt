# Not linked to htool: its cxx_std_17 compile feature would override CXX_STANDARD 14.
# Also built without OpenMP, to compile the #else branches of #if defined(_OPENMP).
foreach(target check_cxx14_compatibility check_cxx14_compatibility_without_openmp)
    add_executable(${target} check_cxx14_compatibility.cpp)
    target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:htool,INTERFACE_INCLUDE_DIRECTORIES>)
    target_compile_definitions(${target} PRIVATE $<TARGET_PROPERTY:htool,INTERFACE_COMPILE_DEFINITIONS>)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX BLAS::BLAS LAPACK::LAPACK)
    if(HPDDM_FOUND)
        target_compile_definitions(${target} PRIVATE HTOOL_WITH_HPDDM)
    endif()
    target_compile_options(${target} PRIVATE ${HTOOL_WARNING_FLAGS})
    if(HTOOL_WITH_STRICT_TESTS)
        target_compile_options(${target} PRIVATE -Werror)
    endif()
    set_target_properties(
        ${target}
        PROPERTIES CXX_STANDARD 14
                   CXX_STANDARD_REQUIRED ON
                   CXX_EXTENSIONS OFF)
    add_test(NAME ${target} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${target})
    set_tests_properties(${target} PROPERTIES LABELS "mpi")
    add_dependencies(build-tests ${target})
endforeach()
if(OpenMP_CXX_FOUND)
    target_link_libraries(check_cxx14_compatibility PRIVATE OpenMP::OpenMP_CXX)
    # The include directories of htool come with OpenMP's (e.g. /opt/homebrew/include, also holding mpi.h): keep them system ones
    target_include_directories(check_cxx14_compatibility_without_openmp SYSTEM PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
endif()
