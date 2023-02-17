#include "../test_hmatrix_product.hpp"
#include <htool/hmatrix/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    bool is_error                         = false;
    const int number_of_rows              = 200;
    const int number_of_rows_increased    = 400;
    const int number_of_columns           = 200;
    const int number_of_columns_increased = 400;
    const int number_of_rhs               = 5;
    const double margin                   = 0;

    for (auto use_local_cluster : {true}) {
        for (auto epsilon : {1e-14, 1e-6}) {

            // Square matrix
            is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, use_local_cluster, 'N', 'N', 'N', epsilon);
            // is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, number_of_rhs, true, 'N', 'N', 'N');
            // is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, use_local_cluster, 'T', 'N', 'N', epsilon);
            // is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, number_of_rhs, true, 'T', 'N', 'N');

            // is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, true, 'N', 'S', 'L');
            // is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, true, 'T', 'S', 'L');
            // is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, true, 'N', 'S', 'U');
            // is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(number_of_rows, number_of_columns, 1, true, 'T', 'S', 'U');
        }
    }
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
