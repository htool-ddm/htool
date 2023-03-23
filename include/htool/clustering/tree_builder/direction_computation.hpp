#ifndef HTOOL_CLUSTERING_TREE_BUILDER_DIRECTION_COMPUTATION_HPP
#define HTOOL_CLUSTERING_TREE_BUILDER_DIRECTION_COMPUTATION_HPP

#include "../../misc/evp.hpp"
#include "../cluster_node.hpp"

namespace htool {

template <typename T>
class ComputeLargestExtent {
  public:
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
        if (spatial_dimension != 2 && spatial_dimension != 3) {
            throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
        }

        Matrix<T> cov(spatial_dimension, spatial_dimension);
        std::vector<T> direction(spatial_dimension, 0);

        for (int j = 0; j < cluster->get_size(); j++) {
            std::vector<T> u(spatial_dimension, 0);
            for (int p = 0; p < spatial_dimension; p++) {
                u[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p] - cluster->get_center()[p];
            }

            for (int p = 0; p < spatial_dimension; p++) {
                for (int q = 0; q < spatial_dimension; q++) {
                    cov(p, q) += weights[permutation[j + cluster->get_offset()]] * u[p] * u[q];
                }
            }
        }
        if (spatial_dimension == 2) {
            direction = solve_EVP_2(cov);
        } else if (spatial_dimension == 3) {
            direction = solve_EVP_3(cov);
        } else {
            throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
        }
        return direction;
    }
};

template <typename T>
class ComputeBoundingBox {
  public:
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {

        // min max for each axis
        std::vector<T> min_point(spatial_dimension, std::numeric_limits<T>::max());
        std::vector<T> max_point(spatial_dimension, std::numeric_limits<T>::min());
        for (int j = 0; j < cluster->get_size(); j++) {
            std::vector<T> u(spatial_dimension, 0);
            for (int p = 0; p < spatial_dimension; p++) {
                if (min_point[p] > coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p]) {
                    min_point[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p];
                }
                if (max_point[p] < coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p]) {
                    max_point[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p];
                }
                u[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p] - cluster->get_center()[p];
            }
        }

        // Direction of largest extent
        T max_distance(std::numeric_limits<T>::min());
        int dir_axis = 0;
        for (int p = 0; p < spatial_dimension; p++) {
            if (max_distance < max_point[p] - min_point[p]) {
                max_distance = max_point[p] - min_point[p];
                dir_axis     = p;
            }
        }
        std::vector<T> direction(spatial_dimension, 0);
        direction[dir_axis] = 1;
        return direction;
    }
};

} // namespace htool

#endif
