#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

include(FetchContent)
FetchContent_Declare(
    data_test_repository
    GIT_REPOSITORY "https://github.com/PierreMarchand20/htool_generate_data_test"
    GIT_TAG origin/main)
FetchContent_GetProperties(data_test_repository)
if(NOT data_test_repository_POPULATED)
    FetchContent_Populate(data_test_repository)
endif()

set(Test_solver_non_symmetric_ARGS ${data_test_repository_SOURCE_DIR}/data/output_non_sym/)
set(Test_solver_symmetric_ARGS ${data_test_repository_SOURCE_DIR}/data/output_sym/)
message(${Test_solver_symmetric_ARGS})
add_executable(Test_solver_ddm_non_symmetric_single_rhs test_solver_ddm_non_symmetric_single_rhs.cpp)
target_link_libraries(Test_solver_ddm_non_symmetric_single_rhs htool)
add_dependencies(build-tests-solvers Test_solver_ddm_non_symmetric_single_rhs)

add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_single_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_single_rhs ${Test_solver_non_symmetric_ARGS})

add_executable(Test_solver_ddm_symmetric_single_rhs test_solver_ddm_symmetric_single_rhs.cpp)
target_link_libraries(Test_solver_ddm_symmetric_single_rhs htool)
add_dependencies(build-tests-solvers Test_solver_ddm_symmetric_single_rhs)

add_test(NAME Test_solver_ddm_symmetric_single_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_single_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_single_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_single_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_single_rhs ${Test_solver_symmetric_ARGS})

add_executable(Test_solver_ddm_non_symmetric_multi_rhs test_solver_ddm_non_symmetric_multi_rhs.cpp)
target_link_libraries(Test_solver_ddm_non_symmetric_multi_rhs htool)
add_dependencies(build-tests-solvers Test_solver_ddm_non_symmetric_multi_rhs)

add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})
add_test(NAME Test_solver_ddm_non_symmetric_multi_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_non_symmetric_multi_rhs ${Test_solver_non_symmetric_ARGS})

add_executable(Test_solver_ddm_symmetric_multi_rhs test_solver_ddm_symmetric_multi_rhs.cpp)
target_link_libraries(Test_solver_ddm_symmetric_multi_rhs htool)
add_dependencies(build-tests-solvers Test_solver_ddm_symmetric_multi_rhs)

add_test(NAME Test_solver_ddm_symmetric_multi_rhs_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_multi_rhs_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_multi_rhs_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
add_test(NAME Test_solver_ddm_symmetric_multi_rhs_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_ddm_symmetric_multi_rhs ${Test_solver_symmetric_ARGS})
