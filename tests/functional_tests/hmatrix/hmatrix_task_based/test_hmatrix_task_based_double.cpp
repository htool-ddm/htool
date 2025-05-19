#include "../test_hmatrix_task_based.hpp"   // for test_hmatrix_task_based
#include <algorithm>                        // for max
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplexSymm...
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    for (auto epsilon : {1e-6}) {
        for (auto n1 : {200, 400}) {
            for (auto n2 : {200, 400}) {
                for (auto transa : {'N', 'T'}) {
                    for (auto block_tree_consistency : {false}) {

                        std::cout << "task based hmatrix product: " << epsilon << " " << n1 << " " << n2 << " " << transa << " " << block_tree_consistency << "\n";

                        is_error = is_error || test_hmatrix_task_based<double, GeneratorTestDouble>(transa, n1, n2, epsilon, block_tree_consistency);

                        if (n1 == n2 && block_tree_consistency) {
                            for (auto UPLO : {'L'}) {
                                std::cout << "task based symmetric matrix product: " << n1 << " " << epsilon << " " << UPLO << "\n";

                                is_error = is_error || test_symmetric_hmatrix_task_based<double, GeneratorTestDoubleSymmetric>(transa, n1, epsilon, UPLO);
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
