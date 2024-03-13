#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

include(FetchContent)
FetchContent_Declare(
    data_test_repository
    GIT_REPOSITORY "https://github.com/PierreMarchand20/htool_generate_data_test"
    GIT_TAG origin/main
    GIT_SHALLOW TRUE
    GIT_SUBMODULES_RECURSE FALSE
    GIT_SUBMODULES "")

FetchContent_GetProperties(data_test_repository)
if(NOT data_test_repository_POPULATED)
    FetchContent_Populate(data_test_repository)
endif()

set(Test_solver_ARGS ${data_test_repository_SOURCE_DIR}/data/)

add_executable(Test_solver test_solver.cpp)
target_link_libraries(Test_solver htool)
add_dependencies(build-tests-solvers Test_solver)

add_test(NAME Test_solver_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver ${Test_solver_ARGS})
set_tests_properties(Test_solver_1 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver ${Test_solver_ARGS})
set_tests_properties(Test_solver_2 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver ${Test_solver_ARGS})
set_tests_properties(Test_solver_3 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver ${Test_solver_ARGS})
set_tests_properties(Test_solver_4 PROPERTIES LABELS "mpi")
