#include "../test_task_based_hmatrix_triangular_solve.hpp"
#include <algorithm>                        // for max
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplexSymm...
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    const int n1            = 400;
    const int number_of_rhs = 100;

    for (auto epsilon : {1e-6}) {
        for (auto side : {'L', 'R'}) {
            for (auto operation : {'N', 'T'}) {
                for (auto diag : {'N', 'U'}) {

                    std::cout << "task based hmatrix triangular solve test case: " << "number_of_rhs = " << number_of_rhs << ", epsilon = " << epsilon << ", n1 = " << n1 << ", side = " << side << ", operation = " << operation << ", diag = " << diag << "\n";

                    TestCaseSolve<double, GeneratorTestDoubleSymmetric> test_case(side, operation, n1, number_of_rhs, 1, -1);

                    is_error = is_error || test_task_based_hmatrix_triangular_solve<double, GeneratorTestDoubleSymmetric, TestCaseSolve<double, GeneratorTestDoubleSymmetric>>(test_case, side, operation, diag, epsilon);
                }
            }
        }
    }

    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    if (is_error) {
        std::cerr << "ERROR: At least one test_task_based_hmatrix_triangular_solve case failed." << std::endl;
        return 1;

    } else {
        std::cout << "SUCCESS: All test_task_based_hmatrix_triangular_solve cases passed." << std::endl;
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    return 0;
}
