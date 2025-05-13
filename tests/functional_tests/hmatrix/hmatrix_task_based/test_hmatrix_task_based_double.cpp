#include "../test_hmatrix_task_based.hpp"   // for test_hmatrix_task_based
#include <algorithm>                        // for max
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplexSymm...
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    // for (auto nr : {200, 400}) {
    //     for (auto nc : {200, 400}) {
    //         for (auto epsilon : {1e-14, 1e-6}) {
    //             for (auto transa : {'N', 'T'}) {
    //                 std::cout << "Case : nr =" << nr << ", nc =  " << nc << ", epsilon = " << epsilon << ", transa = " << transa << "\n";

    //                 is_error = is_error || test_hmatrix_task_based<double, GeneratorTestDoubleSymmetric>(nr, nc, transa, 'N', epsilon);

    //                 // if (nr == nc && block_tree_consistency) {
    //                 //     for (auto UPLO : {'U', 'L'}) {
    //                 //         std::cout << UPLO << "\n";
    //                 //         is_error = is_error || test_hmatrix_task_based<double, GeneratorTestDoubleSymmetric>(nr, nc, use_local_cluster, 'S', UPLO, epsilon, use_dense_block_generator, block_tree_consistency);
    //                 //     }
    //                 // }
    //             }
    //         }
    //     }
    // }

    for (auto epsilon : {1e-6, 1e-10}) {
        for (auto n1 : {200, 400}) {
            for (auto n2 : {200, 400}) {
                for (auto transa : {'N', 'T'}) {
                    std::cout << "task based hmatrix product: " << epsilon << " " << n1 << " " << n2 << " " << transa << " " << "\n";

                    is_error = is_error || test_hmatrix_task_based<double, GeneratorTestDouble>(transa, n1, n2, epsilon);
                }
            }
            // for (auto side : {'L', 'R'}) {
            //     for (auto UPLO : {'U', 'L'}) {
            //         const double margin = 0;
            //         std::cout << "symmetric matrix product: " << n1 << " " << n3 << " " << side << " " << UPLO << " " << epsilon << "\n";
            //         is_error = is_error || test_symmetric_hmatrix_product<double, GeneratorTestDoubleSymmetric>(n1, n3, side, UPLO, epsilon, margin);
            //     }
            // }
        }
    }

    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    if (is_error) {
        std::cerr << "ERROR: At least one test_hmatrix_task_based case failed." << std::endl;
        return 1;

    } else {
        std::cout << "SUCCESS: All test_hmatrix_task_based cases passed." << std::endl;
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    return 0;
}
