#include "../test_hmatrix_task_based.hpp"   // for test_hmatrix_task_based
#include <algorithm>                        // for max
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplexSymm...
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    for (auto epsilon : {1e-3}) {
        for (auto n1 : {500}) {
            for (auto n2 : {500}) {
                for (auto transa : {'N'}) {
                    for (auto block_tree_consistency : {false}) {

                        // Non symmetric case
                        char trans_sym = 'N';
                        std::cout << "task based hmatrix product: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", trans_sym = " << trans_sym << ", transa = " << transa << ", block_tree_consistency = " << block_tree_consistency << "\n";

                        TestCaseProduct<double, GeneratorTestDouble> test_case(transa, 'N', n1, n2, 1, 1, 2);
                        is_error = is_error || test_hmatrix_task_based<double, GeneratorTestDouble, TestCaseProduct<double, GeneratorTestDouble>>(test_case, 'N', transa, epsilon, block_tree_consistency);

                        // Symmetric case
                        if (n1 == n2 && block_tree_consistency) {
                            for (auto UPLO : {'L'}) {
                                trans_sym = 'S';

                                std::cout << "task based symmetric hmatrix product: " << "epsilon = " << epsilon << ", n1 = n2 = " << n1 << ", trans_sym = " << trans_sym << ", transa = " << transa << ", block_tree_consistency = " << block_tree_consistency << "\n";

                                TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric> sym_test_case(n1, 1, 2, 'L', 'S', UPLO);
                                is_error = is_error || test_hmatrix_task_based<double, GeneratorTestDouble, TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric>>(sym_test_case, 'S', transa, epsilon, block_tree_consistency, UPLO);
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    if (is_error) {
        std::cerr << "ERROR: At least one test_hmatrix_task_based or test_symmetric_hmatrix_task_based case failed." << std::endl;
        return 1;

    } else {
        std::cout << "SUCCESS: All test_hmatrix_task_based and test_symmetric_hmatrix_task_based cases passed." << std::endl;
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    return 0;
}
