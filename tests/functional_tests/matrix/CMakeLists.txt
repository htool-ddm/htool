add_custom_target(build-tests-matrix-product)
add_dependencies(build-tests-matrix build-tests-matrix-product)
add_subdirectory(matrix_product)

add_custom_target(build-tests-matrix-triangular-solve)
add_dependencies(build-tests-matrix build-tests-matrix-triangular-solve)
add_subdirectory(matrix_triangular_solve)

add_custom_target(build-tests-matrix-factorization)
add_dependencies(build-tests-matrix build-tests-matrix-factorization)
add_subdirectory(matrix_factorization)

add_executable(Test_matrix test_matrix.cpp)
target_link_libraries(Test_matrix htool)
add_dependencies(build-tests-matrix Test_matrix)
add_test(Test_matrix Test_matrix)

add_executable(Test_matrix_file test_matrix_file.cpp)
target_link_libraries(Test_matrix_file htool)
add_dependencies(build-tests-matrix Test_matrix_file)
add_test(Test_matrix_file Test_matrix_file)
