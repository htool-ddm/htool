#ifndef HTOOL_CLUSTERING_PARTITIONING_HPP
#define HTOOL_CLUSTERING_PARTITIONING_HPP

#include "../../basic_types/vector.hpp"           // for Matrix
#include "../../matrix/matrix.hpp"                // for vector operations
#include "../../matrix/utils.hpp"                 // for get_col
#include "../../misc/evp.hpp"                     // for solve_EVP_2, solve_EVP_3
#include "../interfaces/virtual_partitioning.hpp" // for VirtualPartitioning
#include <tuple>
namespace htool {

template <typename CoordinatePrecision, typename ComputationDirectionPolicy, typename SplittingPolicy>
class Partitioning : public VirtualPartitioning<CoordinatePrecision> {
  public:
    std::vector<std::pair<int, int>> compute_partitioning(Cluster<CoordinatePrecision> &current_cluster, int spatial_dimension, const CoordinatePrecision *coordinates, const CoordinatePrecision *const radii, const CoordinatePrecision *const weights, int number_of_partitions) override {

        std::vector<int> &permutation = current_cluster.get_permutation();
        auto current_offset           = current_cluster.get_offset();
        auto current_size             = current_cluster.get_size();

        // Compute directions
        Matrix<CoordinatePrecision> directions;
        directions = ComputationDirectionPolicy::compute_direction(current_cluster, spatial_dimension, coordinates, radii, weights);

        // If number of partitions corresponds to 2^d, we do binary/quadtree/octree/...
        if (number_of_partitions == pow(2, spatial_dimension)) {
            std::vector<int> dimensions(spatial_dimension);
            std::iota(dimensions.begin(), dimensions.end(), int(0));
            std::stack<std::tuple<int, int, int>> stack;
            stack.push(std::make_tuple(current_cluster.get_offset(), current_cluster.get_size(), 0));
            std::vector<std::pair<int, int>> result;
            while (!stack.empty()) {
                auto tmp_partition = stack.top();
                stack.pop();
                auto tmp_offset    = std::get<0>(tmp_partition);
                auto tmp_size      = std::get<1>(tmp_partition);
                auto tmp_dimension = std::get<2>(tmp_partition);

                std::vector<CoordinatePrecision> direction = get_col(directions, tmp_dimension);

                std::sort(permutation.begin() + tmp_offset, permutation.begin() + tmp_offset + tmp_size, [&](int a, int b) {
                    CoordinatePrecision c = std::inner_product(coordinates + spatial_dimension * a, coordinates + spatial_dimension * (1 + a), direction.data(), CoordinatePrecision(0));
                    CoordinatePrecision d = std::inner_product(coordinates + spatial_dimension * b, coordinates + spatial_dimension * (1 + b), direction.data(), CoordinatePrecision(0));
                    return c < d;
                });

                auto tmp = SplittingPolicy::splitting(tmp_offset, tmp_size, spatial_dimension, coordinates, current_cluster.get_permutation(), direction, 2);

                if ((tmp_dimension < spatial_dimension - 1) and tmp.size() == 2) {
                    stack.push(std::make_tuple(tmp[1].first, tmp[1].second, tmp_dimension + 1));
                    stack.push(std::make_tuple(tmp[0].first, tmp[0].second, tmp_dimension + 1));
                } else if ((tmp_dimension == spatial_dimension - 1) and tmp.size() == 2) {
                    result.insert(result.end(), tmp.begin(), tmp.end());
                } else {
                    break;
                }
            }

            if (result.size() == number_of_partitions) {
                std::sort(result.begin(), result.end(), [](auto a, auto b) { return a.first < b.first; });
                return result;
            }
        }

        // Sort along main direction
        std::sort(permutation.begin() + current_offset, permutation.begin() + current_offset + current_size, [&](int a, int b) {
            CoordinatePrecision c = std::inner_product(coordinates + spatial_dimension * a, coordinates + spatial_dimension * (1 + a), directions.data(), CoordinatePrecision(0));
            CoordinatePrecision d = std::inner_product(coordinates + spatial_dimension * b, coordinates + spatial_dimension * (1 + b), directions.data(), CoordinatePrecision(0));
            return c < d;
        });

        // Split along main direction
        return SplittingPolicy::splitting(current_cluster.get_offset(), current_cluster.get_size(), spatial_dimension, coordinates, current_cluster.get_permutation(), get_col(directions, 0), number_of_partitions);
    }
};

template <typename T>
class ComputeLargestExtent {
  public:
    static Matrix<T> compute_direction(const Cluster<T> &cluster, int spatial_dimension, const T *const coordinates, const T *const, const T *const weights) {
        if (spatial_dimension != 2 && spatial_dimension != 3) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
        }

        const std::vector<int> &permutation = cluster.get_permutation();
        Matrix<T> cov(spatial_dimension, spatial_dimension);
        Matrix<T> directions(spatial_dimension, spatial_dimension);

        for (int j = 0; j < cluster.get_size(); j++) {
            std::vector<T> u(spatial_dimension, 0);
            for (int p = 0; p < spatial_dimension; p++) {
                u[p] = coordinates[spatial_dimension * permutation[j + cluster.get_offset()] + p] - cluster.get_center()[p];
            }

            for (int p = 0; p < spatial_dimension; p++) {
                for (int q = 0; q < spatial_dimension; q++) {
                    cov(p, q) += weights[permutation[j + cluster.get_offset()]] * u[p] * u[q];
                }
            }
        }
        if (spatial_dimension == 2) {
            directions = solve_EVP_2(cov);
        } else if (spatial_dimension == 3) {
            directions = solve_EVP_3(cov);
        }
        return directions;
    }
};

template <typename T>
class ComputeBoundingBox {
  public:
    static Matrix<T> compute_direction(const Cluster<T> &cluster, int spatial_dimension, const T *const coordinates, const T *const, const T *const) {

        const std::vector<int> &permutation = cluster.get_permutation();

        // min max for each axis
        std::vector<T> min_point(spatial_dimension, std::numeric_limits<T>::max());
        std::vector<T> max_point(spatial_dimension, std::numeric_limits<T>::min());
        for (int j = 0; j < cluster.get_size(); j++) {
            std::vector<T> u(spatial_dimension, 0);
            for (int p = 0; p < spatial_dimension; p++) {
                if (min_point[p] > coordinates[spatial_dimension * permutation[j + cluster.get_offset()] + p]) {
                    min_point[p] = coordinates[spatial_dimension * permutation[j + cluster.get_offset()] + p];
                }
                if (max_point[p] < coordinates[spatial_dimension * permutation[j + cluster.get_offset()] + p]) {
                    max_point[p] = coordinates[spatial_dimension * permutation[j + cluster.get_offset()] + p];
                }
                u[p] = coordinates[spatial_dimension * permutation[j + cluster.get_offset()] + p] - cluster.get_center()[p];
            }
        }

        // Direction of largest extent
        std::vector<int> indexes(spatial_dimension);
        std::iota(indexes.begin(), indexes.end(), int(0));
        std::sort(indexes.begin(), indexes.end(), [&max_point, &min_point](int a, int b) { return (max_point[a] - min_point[a]) < (max_point[b] - min_point[b]); });
        Matrix<T> directions(spatial_dimension, spatial_dimension);
        for (int dim = 0; dim < spatial_dimension; dim++) {
            directions(indexes[spatial_dimension - 1 - dim], dim) = 1;
        }
        return directions;
    }
};

template <typename T>
class RegularSplitting {
  public:
    static std::vector<std::pair<int, int>> splitting(int offset, int size, int, const T *const, const std::vector<int> &, const std::vector<T> &, int number_of_partition) {

        std::vector<std::pair<int, int>> current_partition(number_of_partition);

        // Children
        int children_size = int(size / number_of_partition);

        for (int p = 0; p < number_of_partition - 1; p++) {
            current_partition[p] = std::pair<int, int>(offset + children_size * p, children_size);
        }
        current_partition.back() = std::pair<int, int>(offset + children_size * (number_of_partition - 1), size - children_size * (number_of_partition - 1));

        return current_partition;
    }
};

template <typename T>
class GeometricSplitting {
  public:
    static std::vector<std::pair<int, int>> splitting(int offset, int size, int spatial_dimension, const T *const coordinates, const std::vector<int> &permutation, const std::vector<T> &direction, int number_of_partition) {

        // Geometry of current cluster
        std::vector<std::vector<int>> numbering(number_of_partition);
        std::vector<std::pair<int, int>> current_partition;

        if (size > number_of_partition) { // otherwise it won't be possible anyway
            current_partition.resize(number_of_partition);
            std::vector<T> first_point = std::vector<T>(coordinates + spatial_dimension * permutation[offset], coordinates + spatial_dimension * (1 + permutation[offset]));
            std::vector<T> last_point  = std::vector<T>(coordinates + spatial_dimension * permutation[offset + size - 1], coordinates + spatial_dimension * (1 + permutation[offset + size - 1]));

            T geometric_distance      = dprod(direction, last_point - first_point);
            T children_geometric_size = geometric_distance / number_of_partition;

            auto count = permutation.begin() + offset;
            std::vector<int> offsets(number_of_partition, 0);
            std::vector<int> sizes(number_of_partition, 0);

            for (int p = 0; p < number_of_partition - 1; p++) {
                auto result = std::find_if(count, permutation.begin() + offset + size, [&](int a) {
                    return dprod(direction, std::vector<T>(coordinates + spatial_dimension * a, coordinates + spatial_dimension * (1 + a)) - first_point) > children_geometric_size;
                });
                if (result != permutation.end()) {
                    offsets[p]  = count - permutation.begin();
                    sizes[p]    = (result - permutation.begin()) - (count - permutation.begin());
                    count       = result;
                    first_point = std::vector<T>(coordinates + spatial_dimension * (*result), coordinates + spatial_dimension * (*result + 1));
                } else {
                    offsets[p] = 0;
                    sizes[p]   = 0;
                    break;
                }
            }
            offsets.back() = (count - permutation.begin());
            sizes.back()   = size - std::accumulate(sizes.begin(), sizes.end() - 1, 0);
            for (int p = 0; p < number_of_partition; p++) {
                current_partition[p] = std::pair<int, int>(offsets[p], sizes[p]);
            }
        }
        return current_partition;
    }
};

} // namespace htool

#endif
