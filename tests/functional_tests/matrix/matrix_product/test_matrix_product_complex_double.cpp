#include "../test_matrix_product.hpp"

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error                         = false;
    const int number_of_rows              = 200;
    const int number_of_rows_increased    = 400;
    const int number_of_columns           = 200;
    const int number_of_columns_increased = 400;

    for (auto number_of_rhs : {1, 5}) {
        // Square matrix
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'N', 'N', 'N');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'T', 'N', 'N');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'C', 'N', 'N');

        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'N', 'S', 'U');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'T', 'S', 'U');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'C', 'S', 'U');

        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'N', 'S', 'L');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'T', 'S', 'L');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'C', 'S', 'L');

        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'N', 'H', 'U');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'T', 'H', 'U');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'C', 'H', 'U');

        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'N', 'H', 'L');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'T', 'H', 'L');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns, number_of_rhs, 'C', 'H', 'L');

        // Rectangle
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows_increased, number_of_columns, number_of_rhs, 'N', 'N', 'N');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows_increased, number_of_columns, number_of_rhs, 'T', 'N', 'N');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows_increased, number_of_columns, number_of_rhs, 'C', 'N', 'N');

        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns_increased, number_of_rhs, 'N', 'N', 'N');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns_increased, number_of_rhs, 'T', 'N', 'N');
        is_error = is_error || test_matrix_product<std::complex<double>>(number_of_rows, number_of_columns_increased, number_of_rhs, 'C', 'N', 'N');
    }
    if (is_error) {
        return 1;
    }
    return 0;
}
