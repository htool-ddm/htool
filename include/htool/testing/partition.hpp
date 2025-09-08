
#ifndef HTOOL_TESTING_PARTITION_HPP
#define HTOOL_TESTING_PARTITION_HPP

#include "../matrix/matrix.hpp" // for Matrix
#include "../misc/evp.hpp"      // for solve_EVP_2, solve_EVP_3
#include <algorithm>            // for copy, max, sort
#include <numeric>              // for iota, inner_product
#include <vector>               // for vector

namespace htool {

template <typename CoordinatePrecision>
std::vector<int> test_global_partition(int spatial_dimension, int number_of_points, const std::vector<CoordinatePrecision> &coordinates, int partition_size) {
    // Compute largest extent
    Matrix<CoordinatePrecision> direction(spatial_dimension, spatial_dimension);
    std::vector<CoordinatePrecision> weigths;
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
        std::tie(direction, weigths) = solve_EVP_2(cov);
    } else if (spatial_dimension == 3) {
        std::tie(direction, weigths) = solve_EVP_3(cov);
    }

    // sort
    std::sort(permutation.begin(), permutation.end(), [&](int a, int b) {
        CoordinatePrecision c = std::inner_product(coordinates.data() + spatial_dimension * a, coordinates.data() + spatial_dimension * (1 + a), direction.data(), CoordinatePrecision(0));
        CoordinatePrecision d = std::inner_product(coordinates.data() + spatial_dimension * b, coordinates.data() + spatial_dimension * (1 + b), direction.data(), CoordinatePrecision(0));
        return c < d;
    });

    std::vector<int> partition(number_of_points);
    int size_numbering = number_of_points / partition_size;
    int count_size     = 0;
    for (int p = 0; p < partition_size - 1; p++) {
        for (int i = count_size; i < count_size + size_numbering; i++) {
            partition[permutation[i]] = p;
        }
        count_size += size_numbering;
    }
    for (int i = count_size; i < number_of_points; i++) {
        partition[permutation[i]] = partition_size - 1;
    }
    return partition;
}

template <typename CoordinatePrecision>
std::vector<int> test_local_partition(int spatial_dimension, int number_of_points, std::vector<CoordinatePrecision> &coordinates, int partition_size) {
    // Compute largest extent
    Matrix<CoordinatePrecision> direction(spatial_dimension, spatial_dimension);
    std::vector<CoordinatePrecision> weigths;
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
        std::tie(direction, weigths) = solve_EVP_2(cov);
    } else if (spatial_dimension == 3) {
        std::tie(direction, weigths) = solve_EVP_3(cov);
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
    std::vector<int> partition;
    int size_numbering = number_of_points / partition_size;
    int count_size     = 0;
    for (int p = 0; p < partition_size - 1; p++) {
        partition.emplace_back(count_size);
        partition.emplace_back(size_numbering);
        count_size += size_numbering;
    }
    partition.emplace_back(count_size);
    partition.emplace_back(number_of_points - count_size);
    return partition;
}

} // namespace htool
#endif
