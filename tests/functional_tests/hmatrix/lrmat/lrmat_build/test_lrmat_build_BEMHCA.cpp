#include "test_lrmat_build_BEMHCA.hpp"
#include <htool/hmatrix/lrmat/interpolation.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp> // for LowRank...
#include <iostream>                      // for basic_o...
#include <vector>                        // for vector

using namespace std;
using namespace htool;

int main(int, char *[]) {

    auto kernel = std::function([](std::array<double, 3> *target_points, int Nx, std::array<double, 3> *source_points, int Ny, double *mat) {
        for (int i = 0; i < Nx; i++) {
            for (int j = 0; j < Ny; j++) {
                mat[i + j * Nx] = 1. / (4 * M_PI * std::sqrt(std::inner_product(target_points[i].begin(), target_points[i].end(), source_points[j].begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
            }
        }
    });

    std::vector<double> target_points{0, 0, 0, 0, 1, 0, 1, 1, 0};
    std::vector<int> target_elements_to_points{0, 1, 2};
    std::map<int, std::vector<int>> target_dofs_to_elements;
    target_dofs_to_elements[0] = {0};
    std::vector<int> target_permutation(6);
    std::iota(target_permutation.begin(), target_permutation.end(), 0);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon = 0.0001;

    bool test = 0;
    for (int idist = 0; idist < ndistance; idist++) {
        std::vector<double> source_points{0, 0, 0 + distance[idist], 0, 1, 0 + distance[idist], 1, 1, 0 + distance[idist]};
        std::vector<int> source_elements_to_points{0, 1, 2};
        std::map<int, std::vector<int>> source_dofs_to_elements;
        source_dofs_to_elements[0] = {0};
        std::vector<int> source_permutation(6);
        std::iota(source_permutation.begin(), source_permutation.end(), 0);

        BEMHCA<double, double, 3> compressor(kernel, make_p0_basis_on_triangle<double, double, 3>(), target_dofs_to_elements, target_elements_to_points.data(), 3, target_points.data(), target_points.size(), target_permutation.data(), make_p0_basis_on_triangle<double, double, 3>(), source_dofs_to_elements, source_elements_to_points.data(), 3, source_points.data(), source_points.size(), source_permutation.data());
    }
    cout << "test : " << test << endl;

    return test;
}
