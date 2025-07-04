#include "../test_matrix_factorization.hpp" // for test_matrix_cholesky
#include <complex>                          // for complex, abs
#include <initializer_list>                 // for initializer_list

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error            = false;
    const int number_of_rows = 200;

    for (auto number_of_rhs : {1, 100}) {
        for (auto operation : {'N', 'T', 'C'}) {
            // Square matrix
            is_error = is_error || test_matrix_lu<std::complex<double>>(operation, number_of_rows, number_of_rhs);
        }

        is_error = is_error || test_matrix_cholesky<std::complex<double>>('N', number_of_rows, number_of_rhs, 'H', 'U');
        is_error = is_error || test_matrix_cholesky<std::complex<double>>('N', number_of_rows, number_of_rhs, 'H', 'L');
        is_error = is_error || test_matrix_symmetric_ldlt<std::complex<double>>('N', number_of_rows, number_of_rhs, 'U');
        is_error = is_error || test_matrix_symmetric_ldlt<std::complex<double>>('N', number_of_rows, number_of_rhs, 'L');
        is_error = is_error || test_matrix_hermitian_ldlt<std::complex<double>>('N', number_of_rows, number_of_rhs, 'U');
        is_error = is_error || test_matrix_hermitian_ldlt<std::complex<double>>('N', number_of_rows, number_of_rhs, 'L');
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
