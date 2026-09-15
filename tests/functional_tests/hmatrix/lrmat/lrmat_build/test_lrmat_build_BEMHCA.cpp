#include "test_lrmat_build_BEMHCA.hpp"
#include <iostream>

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error    = false;
    double tolerance = 1.;
    for (auto epsilon : {1e-4, 1e-8}) {
        std::cout << epsilon << " triangle in 2D with P0\n";
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 2, KernelType::Constant, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 2, KernelType::Laplace, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<std::complex<double>, 2, KernelType::Helmholtz, ElementType::P0>(epsilon, 10, tolerance);

        std::cout << epsilon << " triangle in 2D with P1\n";
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 2, KernelType::Constant, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 2, KernelType::Laplace, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<std::complex<double>, 2, KernelType::Helmholtz, ElementType::P1>(epsilon, 10, tolerance);

        std::cout << epsilon << " triangle in 3D with P0\n";
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 3, KernelType::Constant, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 3, KernelType::Laplace, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<std::complex<double>, 3, KernelType::Helmholtz, ElementType::P0>(epsilon, 10, tolerance);

        std::cout << epsilon << " triangle in 3D with P1\n";
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 3, KernelType::Constant, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<double, 3, KernelType::Laplace, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_triangle<std::complex<double>, 3, KernelType::Helmholtz, ElementType::P1>(epsilon, 10, tolerance);

        std::cout << epsilon << " segment in 2D with P0\n";
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 2, KernelType::Constant, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 2, KernelType::Laplace, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<std::complex<double>, 2, KernelType::Helmholtz, ElementType::P0>(epsilon, 10, tolerance);

        std::cout << epsilon << " segment in 2D with P1\n";
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 2, KernelType::Constant, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 2, KernelType::Laplace, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<std::complex<double>, 2, KernelType::Helmholtz, ElementType::P1>(epsilon, 10, tolerance);

        std::cout << epsilon << " segment en 3D with P0\n";
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 3, KernelType::Constant, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 3, KernelType::Laplace, ElementType::P0>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<std::complex<double>, 3, KernelType::Helmholtz, ElementType::P0>(epsilon, 10, tolerance);

        std::cout << epsilon << " segment en 3D with P1\n";
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 3, KernelType::Constant, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<double, 3, KernelType::Laplace, ElementType::P1>(epsilon, 10, tolerance);
        is_error = is_error || test_lrmat_build_BEMHCA_segment<std::complex<double>, 3, KernelType::Helmholtz, ElementType::P1>(epsilon, 10, tolerance);
    }

    return is_error;
}
