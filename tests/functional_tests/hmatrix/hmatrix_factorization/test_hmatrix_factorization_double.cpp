#include "../test_hmatrix_factorization.hpp"
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error       = false;
    const int n1        = 1000;
    const double margin = 1;
    const int n2        = 100;

    for (auto epsilon : {1e-3, 1e-6}) {
        for (auto trans : {'N', 'T'}) {
            is_error = is_error || test_hmatrix_lu<double, GeneratorTestDoubleSymmetric>(trans, n1, n2, epsilon, margin);
        }
        for (auto UPLO : {'L', 'U'}) {
            is_error = is_error || test_hmatrix_cholesky<double, GeneratorTestDoubleSymmetric>(UPLO, n1, n2, epsilon, margin);
        }
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
