#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(Test_solver_non_symmetric_ARGS ${CMAKE_SOURCE_DIR}/data/data_test/non_symmetric/)

set(Test_solver_symmetric_ARGS ${CMAKE_SOURCE_DIR}/data/data_test/symmetric/)

add_executable(Test_solver_ddm_non_symmetric_single_rhs test_solver_ddm_non_symmetric_single_rhs.cpp)
target_link_libraries(Test_solver_ddm_non_symmetric_single_rhs htool)
add_dependencies(build-tests Test_solver_ddm_non_symmetric_single_rhs)

add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})

add_executable(Test_solver_ddm_symmetric_single_rhs test_solver_ddm_symmetric_single_rhs.cpp)
target_link_libraries(Test_solver_ddm_symmetric_single_rhs htool)
add_dependencies(build-tests Test_solver_ddm_symmetric_single_rhs)

add_test(NAME Test_solver_ddm_symmetric_single_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_single_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_single_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_single_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})

add_executable(Test_solver_ddm_non_symmetric_multi_rhs test_solver_ddm_non_symmetric_multi_rhs.cpp)
target_link_libraries(Test_solver_ddm_non_symmetric_multi_rhs htool)
add_dependencies(build-tests Test_solver_ddm_non_symmetric_multi_rhs)

add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})

add_executable(Test_solver_ddm_symmetric_multi_rhs test_solver_ddm_symmetric_multi_rhs.cpp)
target_link_libraries(Test_solver_ddm_symmetric_multi_rhs htool)
add_dependencies(build-tests Test_solver_ddm_symmetric_multi_rhs)

add_test(NAME Test_solver_ddm_symmetric_multi_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_multi_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_multi_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_multi_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
