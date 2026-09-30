#include "../test_hmatrix_factorization.hpp" // for test_hmatrix_cholesky
#include <complex>                           // for complex, operator==
#include <htool/testing/generator_test.hpp>  // for GeneratorTestComplex...
#include <initializer_list>                  // for initializer_list

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error       = false;
    const int n1        = 500;
    const double margin = 1;
    const int n2        = 100;

    for (auto epsilon : {1e-3, 1e-6, 1e-10}) {
        for (auto trans : {'N', 'T'}) {
            is_error = is_error || test_hmatrix_lu<std::complex<double>, GeneratorTestComplexHermitian>(trans, n1, n2, epsilon, margin);
        }
        for (auto UPLO : {'L', 'U'}) {
            for (bool full_storage : {false, true}) {
                is_error = is_error || test_hmatrix_cholesky<std::complex<double>, GeneratorTestComplexHermitian>(UPLO, full_storage, n1, n2, epsilon, margin);
            }
            for (bool full_storage : {false, true}) {
                is_error = is_error || test_hmatrix_ldlt<std::complex<double>, GeneratorTestComplexSymmetric>(UPLO, 'S', full_storage, n1, n2, epsilon, margin);
            }
            for (bool full_storage : {false, true}) {
                is_error = is_error || test_hmatrix_ldlt<std::complex<double>, GeneratorTestComplexHermitian>(UPLO, 'H', full_storage, n1, n2, epsilon, margin);
            }
        }
    }
    for (auto UPLO : {'L', 'U'}) {
        for (auto side : {'L', 'R'}) {
            is_error = is_error || test_hmatrix_apply_ldlt_diagonal<std::complex<double>, GeneratorTestComplexSymmetric>('S', side, UPLO, n1);
            is_error = is_error || test_hmatrix_apply_ldlt_diagonal<std::complex<double>, GeneratorTestComplexHermitian>('H', side, UPLO, n1);
        }
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
