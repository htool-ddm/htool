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
            is_error = is_error || test_hmatrix_cholesky<std::complex<double>, GeneratorTestComplexHermitian>(UPLO, n1, n2, epsilon, margin);
        }
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
