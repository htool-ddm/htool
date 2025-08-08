#include "../test_task_based_hmatrix_dependencies.hpp"
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_test.hpp>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;

    char sym = 'N';
    for (auto epsilon : {1e-4, 1e-8}) {
        for (auto n1 : {1000}) {
            for (auto n2 : {1000, 1100}) {
                for (auto block_tree_consistency : {true}) {

                    // Non symmetric case
                    sym = 'N';

                    std::cout << "task based hmatrix build test case: " << "epsilon = " << epsilon << ", n1 = " << n1 << ", n2 = " << n2 << ", sym = " << sym << ", block_tree_consistency = " << block_tree_consistency << "\n";

                    TestCaseProduct<double, GeneratorTestDouble> test_case('N', 'N', n1, n2, 1, 1, 2);

                    is_error = is_error || test_task_based_hmatrix_dependencies<double, GeneratorTestDouble, TestCaseProduct<double, GeneratorTestDouble>>(test_case, sym, epsilon, block_tree_consistency);

                    // Symmetric case
                    if (n1 == n2 && block_tree_consistency) {
                        for (auto UPLO : {'L'}) {
                            sym = 'S';

                            std::cout << "task based symmetric hmatrix build test case: " << "epsilon = " << epsilon << ", n1 = n2 = " << n1 << ", sym = " << sym << ", UPLO = " << UPLO << ", block_tree_consistency = " << block_tree_consistency << "\n";

                            TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric> sym_test_case(n1, n2, 2, 'L', sym, UPLO);

                            is_error = is_error || test_task_based_hmatrix_dependencies<double, GeneratorTestDouble, TestCaseSymmetricProduct<double, GeneratorTestDoubleSymmetric>>(sym_test_case, sym, epsilon, block_tree_consistency, UPLO); // with symmetry
                        }
                    }
                }
            }
        }
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << std::endl;
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "At least one test_task_based_hmatrix_build case failed."); // LCOV_EXCL_LINE
        return 1;

    } else {
        std::cout << "SUCCESS: All test_task_based_hmatrix_build cases passed." << std::endl;
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++\n"
              << std::endl;

    return 0;
}
