#include "../test_hmatrix_factorization.hpp" // for test_hmatrix_cholesky
#include "htool/hmatrix/execution_policies.hpp"
#include <complex>                          // for complex, operator==
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplex...
#include <initializer_list>                 // for initializer_list

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error       = false;
    const int n1        = 500;
    const double margin = 1;
    const int n2        = 100;

    for (auto epsilon : {1e-3, 1e-6, 1e-10}) {
        for (auto trans : {'N', 'T'}) {
            is_error = is_error || test_hmatrix_lu<htool::omp_task_policy<std::complex<double>> &&, std::complex<double>, GeneratorTestComplexHermitian>(omp_task_policy<std::complex<double>>{}, trans, n1, n2, epsilon, margin);
        }
        for (auto UPLO : {'L', 'U'}) {
            for (bool full_storage : {false, true}) {
                is_error = is_error || test_hmatrix_cholesky<htool::omp_task_policy<std::complex<double>> &&, std::complex<double>, GeneratorTestComplexHermitian>(omp_task_policy<std::complex<double>>{}, UPLO, full_storage, n1, n2, epsilon, margin);
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
