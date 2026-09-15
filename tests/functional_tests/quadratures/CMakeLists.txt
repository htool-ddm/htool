#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_quadratures test_quadratures.cpp)
target_link_libraries(Test_quadratures htool)
add_dependencies(build-tests-misc Test_quadratures)
add_test(Test_quadratures Test_quadratures)
