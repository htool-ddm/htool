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
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, true, 'N', 'N', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, false, 'N', 'N', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'L', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'L', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'U', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'U', 'T', false);

    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, true, 'N', 'N', 'T', true);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, false, 'N', 'N', 'T', true);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'L', 'T', true);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'L', 'T', true);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, true, 'S', 'U', 'T', true);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, 1, false, 'S', 'U', 'T', true);

    // Rectangular matrix
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows_increased, number_of_columns, 1, true, 'N', 'N', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows_increased, number_of_columns, 1, false, 'N', 'N', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows_increased, number_of_columns, 1, true, 'N', 'N', 'T', true);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows_increased, number_of_columns, 1, false, 'N', 'N', 'T', true);

    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns_increased, 1, true, 'N', 'N', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns_increased, 1, false, 'N', 'N', 'T', false);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns_increased, 1, true, 'N', 'N', 'T', true);
    is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns_increased, 1, false, 'N', 'N', 'T', true);
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
