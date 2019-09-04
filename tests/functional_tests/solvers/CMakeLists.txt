#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(cases "ddm")

set(Test_solvers_ARGS ${CMAKE_SOURCE_DIR}/data/data_test/SPD_mat_example.bin
                              ${CMAKE_SOURCE_DIR}/data/data_test/rhs.bin
                              ${CMAKE_SOURCE_DIR}/data/data_test/gmsh_mesh.msh
                              ${CMAKE_SOURCE_DIR}/data/data_test/sol.bin)

set(Test_solvers_multi_rhs_ARGS ${Test_solvers_ARGS}
                                10)


foreach(case ${cases})
    add_executable(Test_solvers_${case} test_solvers_${case}.cpp)
    target_link_libraries(Test_solvers_${case} htool)
    add_dependencies(build-tests Test_solvers_${case})

    add_test(NAME Test_solvers_${case}_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solvers_${case} ${Test_solvers_ARGS} )
    add_test(NAME Test_solvers_${case}_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solvers_${case} ${Test_solvers_ARGS})
    add_test(NAME Test_solvers_${case}_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solvers_${case} ${Test_solvers_ARGS})


    add_executable(Test_solvers_${case}_multi_rhs test_solvers_${case}_multi_rhs.cpp)
    target_link_libraries(Test_solvers_${case}_multi_rhs htool)
    add_dependencies(build-tests Test_solvers_${case}_multi_rhs)

    add_test(NAME Test_solvers_${case}_multi_rhs_1 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solvers_${case}_multi_rhs ${Test_solvers_multi_rhs_ARGS} )
    add_test(NAME Test_solvers_${case}_multi_rhs_2 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solvers_${case}_multi_rhs ${Test_solvers_multi_rhs_ARGS})
    add_test(NAME Test_solvers_${case}_multi_rhs_4 COMMAND  ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solvers_${case}_multi_rhs ${Test_solvers_multi_rhs_ARGS})
endforeach()
