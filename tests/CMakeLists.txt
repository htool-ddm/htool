add_subdirectory(functional_tests)

add_executable(Test_warnings test_warnings.cpp)
target_link_libraries(Test_warnings htool)
target_compile_options(Test_warnings PRIVATE -Werror -Wall -Wextra -Wno-sign-compare)
add_test(Test_warnings Test_warnings)
add_dependencies(build-tests Test_warnings)
