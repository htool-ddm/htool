#include "../test_lrmat_product.hpp"
#include <htool/hmatrix/lrmat/sympartialACA.hpp>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error                         = false;
    const int number_of_rows              = 200;
    const int number_of_rows_increased    = 400;
    const int number_of_columns           = 200;
    const int number_of_columns_increased = 400;
    const int number_of_rhs               = 5;
    const double margin                   = 0;

    //
    sympartialACA<double> compressor;
    sympartialACA<std::complex<double>> complex_compressor;

    // Square matrix
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, 1, 'N', compressor, margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, number_of_rhs, 'N', compressor, margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, 1, 'T', compressor, margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns, number_of_rhs, 'T', compressor, margin);

    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, 1, 'N', complex_compressor, margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, number_of_rhs, 'N', complex_compressor, margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, 1, 'T', complex_compressor, margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns, number_of_rhs, 'T', complex_compressor, margin);

    // Rectangle matrix
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows_increased, number_of_columns, 1, 'N', compressor, margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows_increased, number_of_columns, number_of_rhs, 'N', compressor, margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns_increased, 1, 'N', compressor, margin);
    is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, sympartialACA<double>>(number_of_rows, number_of_columns_increased, number_of_rhs, 'N', compressor, margin);

    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows_increased, number_of_columns, 1, 'N', complex_compressor, margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows_increased, number_of_columns, number_of_rhs, 'N', complex_compressor, margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns_increased, 1, 'N', complex_compressor, margin);
    is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, sympartialACA<std::complex<double>>>(number_of_rows, number_of_columns_increased, number_of_rhs, 'N', complex_compressor, margin);

    if (is_error) {
        return 1;
    }
    return 0;
}
