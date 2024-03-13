#include "../test_lrmat_product.hpp"
#include <htool/hmatrix/lrmat/fullACA.hpp>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error                                 = false;
    const int n1                                  = 200;
    const int n1_increased                        = 400;
    const int n2                                  = 200;
    const int n2_increased                        = 400;
    const int n3                                  = 100;
    const double additional_compression_tolerance = 0;
    const std::array<double, 4> additional_lrmat_sum_tolerances{1., 1., 1., 1.};

    for (auto epsilon : {1e-6, 1e-10}) {
        for (auto operation : {'N', 'T'}) {
            std::cout << epsilon << " " << operation << "\n";
            // Square matrix
            is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, fullACA<double>>(operation, 'N', n1, n2, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);
            is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, fullACA<std::complex<double>>>(operation, 'N', n1, n2, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);

            // Rectangle matrix
            is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, fullACA<double>>(operation, 'N', n1_increased, n2, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);
            is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, fullACA<double>>(operation, 'N', n1, n2_increased, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);

            is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, fullACA<std::complex<double>>>(operation, 'N', n1_increased, n2, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);
            is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, fullACA<std::complex<double>>>(operation, 'N', n1, n2_increased, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);
        }
    }
    if (is_error) {
        return 1;
    }
    return 0;
}
