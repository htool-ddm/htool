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

    for (auto use_local_cluster : {true, false}) {
        for (auto epsilon : {1e-14, 1e-6}) {
            for (auto number_of_rhs : {1, 5}) {
                for (auto operation : {'N', 'T'}) {
                    // Square matrix
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, number_of_rhs, use_local_cluster, operation, 'N', 'N', epsilon);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, number_of_rhs, use_local_cluster, operation, 'S', 'L', epsilon);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, number_of_rhs, use_local_cluster, operation, 'S', 'U', epsilon);

                    // Rectangle matrix
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, number_of_rhs, use_local_cluster, operation, 'N', 'N', epsilon);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, number_of_rhs, use_local_cluster, operation, 'N', 'N', epsilon);
                }

                for (auto operation : {'N', 'C'}) {

                    // Square matrix
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, number_of_rhs, use_local_cluster, operation, 'N', 'N', epsilon);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, number_of_rhs, use_local_cluster, operation, 'H', 'L', epsilon);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, number_of_rhs, use_local_cluster, operation, 'H', 'U', epsilon);

                    // Rectangle matrix
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, number_of_rhs, use_local_cluster, operation, 'N', 'N', epsilon);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, number_of_rhs, use_local_cluster, operation, 'N', 'N', epsilon);
                }
            }
        }
    }
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
