#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_matrix_factorization_${type} test_matrix_factorization_${type}.cpp)
    target_link_libraries(Test_matrix_factorization_${type} htool)
    add_dependencies(build-tests-matrix-factorization Test_matrix_factorization_${type})
    add_test(Test_matrix_factorization_${type} Test_matrix_factorization_${type})
endforeach()
