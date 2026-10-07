#include "../test_hmatrix_factorization.hpp"
#include "htool/hmatrix/execution_policies.hpp"
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error       = false;
    const double margin = 1;
    const int n2        = 100;

    for (auto n1 : {10, 1000}) { // 10 to deal with trivial case
        for (auto epsilon : {1e-3, 1e-6}) {
            for (auto trans : {'N', 'T'}) {
                is_error = is_error || test_hmatrix_lu<htool::omp_task_policy<double> &&, double, GeneratorTestDoubleSymmetric>(omp_task_policy<double>{}, trans, n1, n2, epsilon, margin);
            }
            for (auto UPLO : {'L', 'U'}) {
                for (bool full_storage : {false, true}) {
                    is_error = is_error || test_hmatrix_cholesky<htool::omp_task_policy<double> &&, double, GeneratorTestDoubleSymmetric>(omp_task_policy<double>{}, UPLO, full_storage, n1, n2, epsilon, margin);
                }
            }
        }
    }

    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << '\n';
    if (is_error) {
        return 1;

    } else {
        std::cout << "SUCCESS: All test_task_based_factorization cases passed." << '\n';
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << '\n';

    return 0;
}
