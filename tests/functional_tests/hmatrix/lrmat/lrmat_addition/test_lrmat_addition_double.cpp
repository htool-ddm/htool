#include "../test_lrmat_lrmat_addition.hpp" // for test_lrmat_lrmat_addition
#include <htool/hmatrix/lrmat/SVD.hpp>      // for SVD
#include <htool/testing/generator_test.hpp> // for GeneratorTestDouble
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits

int main(int, char const *[]) {
    bool is_error = false;
    // const int greater_size = 400;
    // const int lower_size   = 200;
    const double margin = 10;

    for (auto &epsilon : {1e-6, 1e-10}) {
        for (auto &n1 : {200, 400}) {
            for (auto &n2 : {200, 400}) {
                std::cout << epsilon << " " << n1 << " " << n2 << "\n";
                is_error = is_error || test_lrmat_lrmat_addition<double, GeneratorTestDouble, SVD<double>>(n1, n2, epsilon, margin);
            }
        }
    }

    if (is_error) {
        return 1;
    }

    return 0;
}
