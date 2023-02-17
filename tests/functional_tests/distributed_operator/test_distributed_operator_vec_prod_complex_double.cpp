#include "test_distributed_operator.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool is_error                         = false;
    const int number_of_rows              = 200;
    const int number_of_rows_increased    = 400;
    const int number_of_columns           = 200;
    const int number_of_columns_increased = 400;

    // Square matrix
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns, 1, true, 'N', 'N', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns, 1, false, 'N', 'N', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, true, 'H', 'L', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, false, 'H', 'L', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, true, 'H', 'U', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, false, 'H', 'U', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'L', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'L', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'U', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'U', 'N', false);

    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns, 1, true, 'N', 'N', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns, 1, false, 'N', 'N', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, true, 'H', 'L', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, false, 'H', 'L', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, true, 'H', 'U', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, 1, false, 'H', 'U', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'L', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'L', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'U', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'U', 'N', true);

    // Rectangular matrix
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, 1, true, 'N', 'N', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, 1, false, 'N', 'N', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, 1, true, 'N', 'N', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, 1, false, 'N', 'N', 'N', true);

    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, 1, true, 'N', 'N', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, 1, false, 'N', 'N', 'N', false);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, 1, true, 'N', 'N', 'N', true);
    is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, 1, false, 'N', 'N', 'N', true);
    MPI_Finalize();

    if (is_error) {
        return 1;
    }
    return 0;
}
