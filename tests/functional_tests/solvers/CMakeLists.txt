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
    GIT_SUBMODULES "" SOURCE_SUBDIR AVOID_calling_ADD_SUBDIR_in_subseqeuent_MAKEAVILABLE)
# weird hack, see https://gitlab.kitware.com/cmake/cmake/-/issues/26220 and

FetchContent_MakeAvailable(data_test_repository)

set(Test_solver_ARGS ${data_test_repository_SOURCE_DIR}/data/)

add_executable(Test_solver_double test_solver_double.cpp)
target_link_libraries(Test_solver_double htool)
add_dependencies(build-tests-solvers Test_solver_double)

add_executable(Test_solver_complex_double test_solver_complex_double.cpp)
target_link_libraries(Test_solver_complex_double htool)
add_dependencies(build-tests-solvers Test_solver_complex_double)

add_test(NAME Test_solver_double_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_double_1 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_double_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_double_2 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_double_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_double_3 PROPERTIES LABELS "mpi" TIMEOUT 3000)
add_test(NAME Test_solver_double_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_double_4 PROPERTIES LABELS "mpi" TIMEOUT 3000)

add_test(NAME Test_solver_complex_double_1 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_complex_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_complex_double_1 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_complex_double_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_complex_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_complex_double_2 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_complex_double_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_complex_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_complex_double_3 PROPERTIES LABELS "mpi")
add_test(NAME Test_solver_complex_double_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_complex_double ${Test_solver_ARGS})
set_tests_properties(Test_solver_complex_double_4 PROPERTIES LABELS "mpi")

add_executable(Test_local_hmatrix_solvers test_local_hmatrix_solvers.cpp)
target_link_libraries(Test_local_hmatrix_solvers htool)
add_dependencies(build-tests-solvers Test_local_hmatrix_solvers)
add_test(NAME Test_local_hmatrix_solvers COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_local_hmatrix_solvers)
set_tests_properties(Test_local_hmatrix_solvers PROPERTIES LABELS "mpi")

# Task-based tests run without leak detection (see Test_task_based_hmatrix_factorization)
foreach(nb_threads 1 2 4)
    add_test(NAME Test_local_hmatrix_solvers_task_based_${nb_threads} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_local_hmatrix_solvers task_based)
    set_tests_properties(Test_local_hmatrix_solvers_task_based_${nb_threads} PROPERTIES LABELS "mpi" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;OMP_NUM_THREADS=${nb_threads}")
endforeach()

add_executable(Test_solver_execution_policies test_solver_execution_policies.cpp)
target_link_libraries(Test_solver_execution_policies htool)
add_dependencies(build-tests-solvers Test_solver_execution_policies)
foreach(nb_threads 1 2 4)
    add_test(NAME Test_solver_execution_policies_${nb_threads} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_solver_execution_policies ${Test_solver_ARGS})
    set_tests_properties(Test_solver_execution_policies_${nb_threads} PROPERTIES LABELS "mpi" ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0;OMP_NUM_THREADS=${nb_threads}")
endforeach()
