#include "../test_task_based_hmatrix_factorization.hpp" // for test_task_based_lu_factorization, test_task_based_cholesky_factorization
#include <algorithm>                                    // for max
#include <htool/testing/generator_test.hpp>             // for GeneratorTestComplexSymm...
#include <initializer_list>                             // for initializer_list
#include <iostream>                                     // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    const int n1 = 1100;
    const int n2 = 1000;

    for (auto epsilon : {1e-4, 1e-8}) {
        for (auto trans : {'N', 'T'}) {

            std::cout << "task based LU factorization test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", trans = " << trans << "\n";

            is_error = is_error || test_task_based_lu_factorization<double, GeneratorTestDoubleSymmetric>(trans, n1, n2, epsilon);
        }

        for (auto UPLO : {'L', 'U'}) {
            std::cout << "task based Cholesky factorization test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", UPLO = " << UPLO << "\n";

            is_error = is_error || test_task_based_cholesky_factorization<double, GeneratorTestDoubleSymmetric>(UPLO, n1, n2, epsilon);
        }
    }

    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "At least one test_task_based_factorization case failed."); // LCOV_EXCL_LINE
        return 1;

    } else {
        std::cout << "SUCCESS: All test_task_based_factorization cases passed." << std::endl;
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    return 0;
}
