add_executable(Test_task_based_hmatrix_dependencies test_task_based_hmatrix_dependencies.cpp)
target_link_libraries(Test_task_based_hmatrix_dependencies htool)
add_dependencies(build-tests-hmatrix Test_task_based_hmatrix_dependencies)
add_test(Test_task_based_hmatrix_dependencies Test_task_based_hmatrix_dependencies)

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_task_based_hmatrix_full_${type} test_task_based_hmatrix_full_${type}.cpp)
    target_link_libraries(Test_task_based_hmatrix_full_${type} htool)
    add_dependencies(build-tests-hmatrix Test_task_based_hmatrix_full_${type})

    add_test(Test_task_based_hmatrix_full_${type}_1 Test_task_based_hmatrix_full_${type})
    set_tests_properties(Test_task_based_hmatrix_full_${type}_1 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)

    add_test(Test_task_based_hmatrix_full_${type}_2 Test_task_based_hmatrix_full_${type})
    set_tests_properties(Test_task_based_hmatrix_full_${type}_2 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

    add_test(Test_task_based_hmatrix_full_${type}_4 Test_task_based_hmatrix_full_${type})
    set_tests_properties(Test_task_based_hmatrix_full_${type}_4 PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)
endforeach()
