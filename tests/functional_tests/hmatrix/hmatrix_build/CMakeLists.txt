#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_hmatrix_build_${type} test_hmatrix_build_${type}.cpp)
    target_link_libraries(Test_hmatrix_build_${type} htool)
    add_dependencies(build-tests-hmatrix-build Test_hmatrix_build_${type})
    add_test(NAME Test_hmatrix_build_${type}_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmatrix_build_${type})
    set_tests_properties(Test_hmatrix_build_${type}_1 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)
    set_tests_properties(Test_hmatrix_build_${type}_1 PROPERTIES LABELS "mpi")

    add_test(NAME Test_hmatrix_build_${type}_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmatrix_build_${type})
    set_tests_properties(Test_hmatrix_build_${type}_2 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
    set_tests_properties(Test_hmatrix_build_${type}_2 PROPERTIES LABELS "mpi")

    add_test(NAME Test_hmatrix_build_${type}_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmatrix_build_${type})
    set_tests_properties(Test_hmatrix_build_${type}_4 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
    set_tests_properties(Test_hmatrix_build_${type}_4 PROPERTIES LABELS "mpi")

endforeach()
