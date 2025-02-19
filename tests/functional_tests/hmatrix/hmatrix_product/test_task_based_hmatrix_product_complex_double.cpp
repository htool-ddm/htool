#include "../test_task_based_hmatrix_hmatrix_product.hpp"
#include "../test_task_based_hmatrix_vector_product.hpp"
#include <algorithm>                        // for max
#include <complex>                          // for complex, abs, operator-
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplexSymm...
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    char sym = 'N';
    for (auto epsilon : {1e-4, 1e-8}) {
        for (auto n1 : {1000}) {
            for (auto n2 : {1000, 1100}) {
                for (auto transa : {'N', 'T'}) {
                    for (auto block_tree_consistency : {true}) {

                        // Non symmetric case
                        sym = 'N';

                        std::cout << "task based hmatrix test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", sym = " << sym << ", transa = " << transa << ", block_tree_consistency = " << block_tree_consistency << "\n";

                        TestCaseProduct<std::complex<double>, GeneratorTestComplexSymmetric> test_case(transa, 'N', n1, n2, 1, 1, 2);

                        is_error = is_error || test_task_based_hmatrix_vector_product<std::complex<double>, GeneratorTestComplexSymmetric, TestCaseProduct<std::complex<double>, GeneratorTestComplexSymmetric>>(test_case, sym, transa, epsilon, block_tree_consistency);

                        // Symmetric case
                        if (n1 == n2 && block_tree_consistency) {
                            for (auto UPLO : {'L'}) {
                                sym = 'S';

                                std::cout << "task based symmetric hmatrix test case: " << "epsilon = " << epsilon << ", n1 = n2 = " << n1 << ", sym = " << sym << ", transa = " << transa << ", UPLO = " << UPLO << ", block_tree_consistency = " << block_tree_consistency << "\n";

                                TestCaseSymmetricProduct<std::complex<double>, GeneratorTestComplexSymmetric> sym_test_case(n1, n2, 2, 'L', sym, UPLO);

                                is_error = is_error || test_task_based_hmatrix_vector_product<std::complex<double>, GeneratorTestComplexSymmetric, TestCaseSymmetricProduct<std::complex<double>, GeneratorTestComplexSymmetric>>(sym_test_case, sym, transa, epsilon, block_tree_consistency, UPLO);
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    if (is_error) {
        std::cerr << "ERROR: At least one test_task_based_hmatrix_vector_product case failed." << std::endl;
        return 1;

    } else {
        std::cout << "SUCCESS: All test_task_based_hmatrix_vector_product cases passed." << std::endl;
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    for (auto epsilon : {1e-6}) {
        for (auto n1 : {1000}) {
            for (auto n3 : {800}) {
                for (auto transa : {'N', 'T', 'C'}) {
                    for (auto transb : {'N', 'T', 'C'}) {
                        for (auto n2 : {900}) {

                            // Non symmetric case
                            std::cout << "task based hmatrix test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", n3 = " << n3 << ", sym = " << 'N' << ", transa = " << transa << ", transb = " << transb << "\n";

                            TestCaseProduct<std::complex<double>, GeneratorTestComplexSymmetric> test_case(transa, transb, n1, n2, n3, 1, 1);

                            is_error = is_error || test_task_based_hmatrix_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric, TestCaseProduct<std::complex<double>, GeneratorTestComplexSymmetric>>(test_case, 'N', transa, transb, epsilon, true);
                        }
                    }
                }

                // Symmetric case WIP ToDO: fix the test case side = 'R'
                for (auto side : {'L', 'R'}) {
                    for (auto UPLO : {'U', 'L'}) {
                        std::cout << "task based symmetric hmatrix test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n3 = " << n3 << ", sym = " << 'S' << ", side = " << side << ", UPLO = " << UPLO << "\n";

                        TestCaseSymmetricProduct<std::complex<double>, GeneratorTestComplexSymmetric> test_case(n1, n3, 2, side, 'S', UPLO);

                        is_error = is_error || test_task_based_hmatrix_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric, TestCaseSymmetricProduct<std::complex<double>, GeneratorTestComplexSymmetric>>(test_case, 'S', 'N', 'N', epsilon, true);

                        std::cout << "task based hermitian hmatrix test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n3 = " << n3 << ", sym = " << 'S' << ", side = " << side << ", UPLO = " << UPLO << "\n";

                        TestCaseSymmetricProduct<std::complex<double>, GeneratorTestComplexHermitian> test_case_herm(n1, n3, 2, side, 'S', UPLO);
                        is_error = is_error || test_task_based_hmatrix_hmatrix_product<std::complex<double>, GeneratorTestComplexHermitian, TestCaseSymmetricProduct<std::complex<double>, GeneratorTestComplexHermitian>>(test_case_herm, 'S', 'N', 'N', epsilon, true);
                    }
                }
            }
        }
    }

    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    if (is_error) {
        std::cerr << "ERROR: At least one test_task_based_hmatrix_hmatrix_product case failed." << std::endl;
        return 1;

    } else {
        std::cout << "SUCCESS: All test_task_based_hmatrix_hmatrix_product cases passed." << std::endl;
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;

    return 0;
}
