#include "../test_lrmat_product.hpp"        // for test_lrmat_product
#include <algorithm>                        // for copy
#include <array>                            // for array
#include <complex>                          // for complex, operator*, oper...
#include <htool/hmatrix/lrmat/SVD.hpp>      // for SVD
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplex
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error                                 = false;
    const double additional_compression_tolerance = 0;
    const std::array<double, 4> additional_lrmat_sum_tolerances{10., 10., 10., 10.};

    for (auto epsilon : {1e-6, 1e-10}) {
        for (auto n1 : {200, 400}) {
            for (auto n3 : {100}) {
                for (auto n2 : {200, 400}) {
                    for (auto transa : {'N', 'T', 'C'}) {
                        for (auto transb : {'N', 'T', 'C'}) {
                            std::cout << epsilon << " " << n1 << " " << n2 << " " << n3 << " " << transa << " " << transb << "\n";
                            is_error = is_error || test_lrmat_product<std::complex<double>, GeneratorTestComplex, SVD<std::complex<double>>>(transa, transb, n1, n2, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);
                        }
                    }
                }
            }
        }
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
