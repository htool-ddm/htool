#include "../test_lrmat_product.hpp"
#include <htool/hmatrix/lrmat/sympartialACA.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    bool is_error                         = false;
    const int number_of_rows              = 200;
    const int number_of_rows_increased    = 400;
    const int number_of_columns           = 200;
    const int number_of_columns_increased = 400;
    const int number_of_rhs               = 5;
    const double margin                   = 0;

    // Square matrix
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, 1, 'N', margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, number_of_rhs, 'N', margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, 1, 'T', margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, number_of_rhs, 'T', margin);

    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, 1, 'N', margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, number_of_rhs, 'N', margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, 1, 'T', margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, number_of_rhs, 'T', margin);

    // Rectangle matrix
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows_increased, number_of_columns, 1, 'N', margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows_increased, number_of_columns, number_of_rhs, 'N', margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns_increased, 1, 'N', margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns_increased, number_of_rhs, 'N', margin);

    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows_increased, number_of_columns, 1, 'N', margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows_increased, number_of_columns, number_of_rhs, 'N', margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns_increased, 1, 'N', margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns_increased, number_of_rhs, 'N', margin);

    if (is_error) {
        return 1;
    }
    return 0;
}
