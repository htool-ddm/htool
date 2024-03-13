
#ifndef HTOOL_TESTING_PARTITION_HPP
#define HTOOL_TESTING_PARTITION_HPP

#include "../matrix/matrix.hpp"
#include "../misc/evp.hpp"

namespace htool {

template <typename CoordinatePrecision>
void test_partition(int spatial_dimension, int number_of_points, std::vector<CoordinatePrecision> &coordinates, int partition_size, std::vector<int> &partition) {
    // Compute largest extent
    std::vector<CoordinatePrecision> direction(spatial_dimension, 0);
    Matrix<CoordinatePrecision> cov(spatial_dimension, spatial_dimension);
    std::vector<int> permutation(number_of_points, 0);
    std::iota(permutation.begin(), permutation.end(), int(0));

    for (int j = 0; j < number_of_points; j++) {
        std::vector<CoordinatePrecision> u(spatial_dimension, 0);
        for (int p = 0; p < spatial_dimension; p++) {
            u[p] = coordinates[spatial_dimension * j + p]; // zero is the center
        }

        for (int p = 0; p < spatial_dimension; p++) {
            for (int q = 0; q < spatial_dimension; q++) {
                cov(p, q) += u[p] * u[q];
            }
        }
    }
    if (spatial_dimension == 2) {
        direction = solve_EVP_2(cov);
    } else if (spatial_dimension == 3) {
        direction = solve_EVP_3(cov);
    }

    // sort
    std::sort(permutation.begin(), permutation.end(), [&](int a, int b) {
        CoordinatePrecision c = std::inner_product(coordinates.data() + spatial_dimension * a, coordinates.data() + spatial_dimension * (1 + a), direction.data(), CoordinatePrecision(0));
        CoordinatePrecision d = std::inner_product(coordinates.data() + spatial_dimension * b, coordinates.data() + spatial_dimension * (1 + b), direction.data(), CoordinatePrecision(0));
        return c < d;
    });

    // permute
    std::vector<CoordinatePrecision> coordinates_perm(number_of_points * spatial_dimension, 0);
    for (int i = 0; i < permutation.size(); i++) {
        for (int p = 0; p < spatial_dimension; p++) {
            coordinates_perm[i * spatial_dimension + p] = coordinates[permutation[i] * spatial_dimension + p];
        }
    }
    coordinates = coordinates_perm;

    // split
    partition.clear();
    int size_numbering = number_of_points / partition_size;
    int count_size     = 0;
    for (int p = 0; p < partition_size - 1; p++) {
        partition.emplace_back(count_size);
        partition.emplace_back(size_numbering);
        count_size += size_numbering;
    }
    partition.emplace_back(count_size);
    partition.emplace_back(number_of_points - count_size);
}
} // namespace htool
#endif
