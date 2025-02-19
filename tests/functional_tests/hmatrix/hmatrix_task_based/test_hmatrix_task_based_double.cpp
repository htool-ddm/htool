#include "../test_hmatrix_task_based.hpp"   // for test_hmatrix_task_based
#include <algorithm>                        // for max
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplexSymm...
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    ///////////////////////////////////////////////////////////////////////////////
    // Tests for hmatrix vector product
    ///////////////////////////////////////////////////////////////////////////////
    if (false) {
        char sym = 'N';
        for (auto epsilon : {1e-4}) {
            for (auto n1 : {2000}) {
                for (auto n2 : {2100}) {
                    for (auto transa : {'N', 'T'}) {
                        for (auto block_tree_consistency : {true}) {

                            // Non symmetric case
                            sym = 'N';

                            std::cout << "task based hmatrix test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", sym = " << sym << ", transa = " << transa << ", block_tree_consistency = " << block_tree_consistency << "\n";

                            TestCaseProduct<double, GeneratorTestDouble> test_case(transa, 'N', n1, n2, 1, 1, 2);

                            is_error = is_error || test_task_based_hmatrix_vector_product<double, GeneratorTestDouble, TestCaseProduct<double, GeneratorTestDouble>>(test_case, sym, transa, epsilon, block_tree_consistency);

                            // Symmetric case
                            if (n1 == n2 && block_tree_consistency) {
                                for (auto UPLO : {'L'}) {
                                    sym = 'S';

                                    std::cout << "task based symmetric hmatrix test case: " << "epsilon = " << epsilon << ", n1 = n2 = " << n1 << ", sym = " << sym << ", transa = " << transa << ", UPLO = " << UPLO << ", block_tree_consistency = " << block_tree_consistency << "\n";

                                    TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric> sym_test_case(n1, n2, 2, 'L', sym, UPLO);

                                    is_error = is_error || test_task_based_hmatrix_vector_product<double, GeneratorTestDouble, TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric>>(sym_test_case, sym, transa, epsilon, block_tree_consistency, UPLO); // with symmetry

                                    // is_error = is_error || test_task_based_hmatrix_vector_product<double, GeneratorTestDouble, TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric>>(sym_test_case, 'N', transa, epsilon, block_tree_consistency, 'N'); // without symmetry
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
    } // end of tests for hmatrix vector product

    ///////////////////////////////////////////////////////////////////////////////
    // Tests for hmatrix hmatrix product
    ///////////////////////////////////////////////////////////////////////////////
    if (false) {
        for (auto epsilon : {1e-6}) {
            for (auto n1 : {1000}) {
                for (auto n3 : {800}) {
                    for (auto transa : {'N', 'T'}) {
                        for (auto transb : {'N', 'T'}) {
                            for (auto n2 : {900}) {

                                // Non symmetric case
                                std::cout << "task based hmatrix test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", n3 = " << n3 << ", sym = " << 'N' << ", transa = " << transa << ", transb = " << transb << "\n";

                                TestCaseProduct<double, GeneratorTestDoubleSymmetric> test_case(transa, transb, n1, n2, n3, 1, 1);

                                is_error = is_error || test_task_based_hmatrix_hmatrix_product<double, GeneratorTestDoubleSymmetric, TestCaseProduct<double, GeneratorTestDoubleSymmetric>>(test_case, 'N', transa, transb, epsilon, true);
                            }
                        }
                    }

                    // Symmetric case WIP ToDO: fix the test case side = 'R'
                    for (auto side : {'L', 'R'}) {
                        for (auto UPLO : {'U', 'L'}) {
                            std::cout << "task based symmetric hmatrix test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n3 = " << n3 << ", sym = " << 'S' << ", side = " << side << ", UPLO = " << UPLO << "\n";

                            TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric> test_case(n1, n3, 2, side, 'S', UPLO);

                            is_error = is_error || test_task_based_hmatrix_hmatrix_product<double, GeneratorTestDoubleSymmetric, TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric>>(test_case, 'S', 'N', 'N', epsilon, true);
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
    } // end of tests for hmatrix hmatrix product

    ///////////////////////////////////////////////////////////////////////////////
    // Tests for hmatrix triangular solve
    ///////////////////////////////////////////////////////////////////////////////
    if (false) {

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
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Tests for hmatrix factorization
    ///////////////////////////////////////////////////////////////////////////////
    if (true) {
        const int n1 = 1100;
        const int n2 = 1000;

        for (auto epsilon : {1e-3, 1e-8}) {
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
            std::cerr << "ERROR: At least one test_task_based_lu_factorization case failed." << std::endl;
            return 1;

        } else {
            std::cout << "SUCCESS: All test_task_based_lu_factorization cases passed." << std::endl;
        }
        std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    }

    return 0;
}
