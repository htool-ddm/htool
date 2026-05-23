#include "test_lrmat_build_BEMHCA.hpp"
#include <iostream>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error    = false;
    double tolerance = 1.;
    for (auto epsilon : {1e-4, 1e-8}) {
        std::cout << epsilon << " triangle\n";
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 3>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 2>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<std::complex<double>, 3>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<std::complex<double>, 2>(epsilon, 10, tolerance);

        std::cout << epsilon << " segment\n";
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 2>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 3>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<std::complex<double>, 2>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<std::complex<double>, 3>(epsilon, 10, tolerance);
    }

    return is_error;
}
