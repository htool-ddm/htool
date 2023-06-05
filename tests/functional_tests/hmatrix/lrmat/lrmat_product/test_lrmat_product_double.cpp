#include "../test_lrmat_product.hpp"
#include <array>
#include <htool/hmatrix/lrmat/SVD.hpp> // for SVD
#include <htool/testing/generator_test.hpp>
#include <initializer_list>
#include <iostream>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error                                 = false;
    const double additional_compression_tolerance = 0;
    const std::array<double, 4> additional_lrmat_sum_tolerances{1., 1., 1., 1.};

    for (auto epsilon : {1e-6, 1e-10}) {
        for (auto n1 : {200, 400}) {
            for (auto n3 : {100}) {
                for (auto n2 : {200, 400}) {
                    for (auto transa : {'N', 'T'}) {
                        for (auto transb : {'N', 'T'}) {
                            std::cout << epsilon << " " << n1 << " " << n2 << " " << n3 << " " << transa << " " << transb << "\n";
                            is_error = is_error || test_lrmat_product<double, GeneratorTestDouble, SVD<double>>(transa, transb, n1, n2, n3, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances);
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
