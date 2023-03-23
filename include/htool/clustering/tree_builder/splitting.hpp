#ifndef HTOOL_CLUSTERING_TREE_BUILDER_SPLITTING_HPP
#define HTOOL_CLUSTERING_TREE_BUILDER_SPLITTING_HPP

#include "../cluster_node.hpp"

namespace htool {

template <typename T>
class RegularSplitting {
  public:
    void splitting(Cluster<T> *curr_cluster, const std::vector<int> &permutation, int spatial_dimension, const T *x, int current_number_of_children, int minclustersize, const std::vector<T> &dir, std::vector<std::pair<int, int>> &current_partition) {

        int offset = curr_cluster->get_offset();
        int size   = curr_cluster->get_size();
        current_partition.resize(current_number_of_children);

        // Children
        int children_size = int(size / current_number_of_children);
        if (children_size > minclustersize) {
            for (int p = 0; p < current_number_of_children - 1; p++) {
                current_partition[p] = std::pair<int, int>(curr_cluster->get_offset() + children_size * p, children_size);
            }
            current_partition.back() = std::pair<int, int>(curr_cluster->get_offset() + children_size * (current_number_of_children - 1), size - children_size * (current_number_of_children - 1));
        } else {
            current_partition.clear();
        }
    }
};

template <typename T>
class GeometricSplitting {
    RegularSplitting<T> regular_splitting;

  public:
    void splitting(Cluster<T> *curr_cluster, const std::vector<int> &permutation, int spatial_dimension, const T *x, int current_number_of_children, int minclustersize, const std::vector<T> &dir, std::vector<std::pair<int, int>> &current_partition) {

        // Geometry of current cluster
        int offset = curr_cluster->get_offset();
        int size   = curr_cluster->get_size();
        std::vector<std::vector<int>> numbering(current_number_of_children);
        current_partition.resize(current_number_of_children);

        if (size > current_number_of_children) { // otherwise it won't be possible anyway

            std::vector<T> first_point = std::vector<T>(x + spatial_dimension * permutation[offset], x + spatial_dimension * (1 + permutation[offset]));
            std::vector<T> last_point  = std::vector<T>(x + spatial_dimension * permutation[offset + size - 1], x + spatial_dimension * (1 + permutation[offset + size - 1]));

            T geometric_distance      = dprod(dir, last_point - first_point);
            T children_geometric_size = geometric_distance / current_number_of_children;

            auto count = permutation.begin() + offset;
            std::vector<int> offsets(current_number_of_children, 0);
            std::vector<int> sizes(current_number_of_children, 0);

            for (int p = 0; p < current_number_of_children - 1; p++) {
                auto result = std::find_if(count, permutation.begin() + offset + size, [&](int a) {
                    return dprod(dir, std::vector<T>(x + spatial_dimension * a, x + spatial_dimension * (1 + a)) - first_point) > children_geometric_size;
                });
                if (result != permutation.end()) {
                    offsets[p]  = count - permutation.begin();
                    sizes[p]    = (result - permutation.begin()) - (count - permutation.begin());
                    count       = result;
                    first_point = std::vector<T>(x + spatial_dimension * (*result), x + spatial_dimension * (*result + 1));
                } else {
                    offsets[p] = 0;
                    sizes[p]   = 0;
                    break;
                }
            }
            offsets.back() = (count - permutation.begin());
            sizes.back()   = size - std::accumulate(sizes.begin(), sizes.end() - 1, 0);

            if (std::all_of(sizes.begin(), sizes.end(), [&](int a) { return a > minclustersize; })) {
                for (int p = 0; p < current_number_of_children; p++) {
                    current_partition[p] = std::pair<int, int>(offsets[p], sizes[p]);
                }
            } else {
                regular_splitting.splitting(curr_cluster, permutation, spatial_dimension, x, current_number_of_children, minclustersize, dir, current_partition);
            }
        } else {
            current_partition.clear();
        }
    }
};

template <typename T>
class GeometricSplittingTest {
    RegularSplitting<T> regular_splitting;
    int m_test = 0;

  public:
    GeometricSplittingTest() {}
    GeometricSplittingTest(int test) : m_test(test) {}

    void splitting(Cluster<T> *curr_cluster, const std::vector<int> &permutation, int spatial_dimension, const T *x, int current_number_of_children, int minclustersize, const std::vector<T> &dir, std::vector<std::pair<int, int>> &current_partition) {
        std::cout << m_test << "\n";

        // Geometry of current cluster
        int offset = curr_cluster->get_offset();
        int size   = curr_cluster->get_size();
        std::vector<std::vector<int>> numbering(current_number_of_children);
        current_partition.resize(current_number_of_children);

        if (size > current_number_of_children) { // otherwise it won't be possible anyway

            std::vector<T> first_point = std::vector<T>(x + spatial_dimension * permutation[offset], x + spatial_dimension * (1 + permutation[offset]));
            std::vector<T> last_point  = std::vector<T>(x + spatial_dimension * permutation[offset + size - 1], x + spatial_dimension * (1 + permutation[offset + size - 1]));

            T geometric_distance      = dprod(dir, last_point - first_point);
            T children_geometric_size = geometric_distance / current_number_of_children;

            auto count = permutation.begin() + offset;
            std::vector<int> offsets(current_number_of_children, 0);
            std::vector<int> sizes(current_number_of_children, 0);

            for (int p = 0; p < current_number_of_children - 1; p++) {
                auto result = std::find_if(count, permutation.begin() + offset + size, [&](int a) {
                    return dprod(dir, std::vector<T>(x + spatial_dimension * a, x + spatial_dimension * (1 + a)) - first_point) > children_geometric_size;
                });
                if (result != permutation.end()) {
                    offsets[p]  = count - permutation.begin();
                    sizes[p]    = (result - permutation.begin()) - (count - permutation.begin());
                    count       = result;
                    first_point = std::vector<T>(x + spatial_dimension * (*result), x + spatial_dimension * (*result + 1));
                } else {
                    offsets[p] = 0;
                    sizes[p]   = 0;
                    break;
                }
            }
            offsets.back() = (count - permutation.begin());
            sizes.back()   = size - std::accumulate(sizes.begin(), sizes.end() - 1, 0);

            if (std::all_of(sizes.begin(), sizes.end(), [&](int a) { return a > minclustersize; })) {
                for (int p = 0; p < current_number_of_children; p++) {
                    current_partition[p] = std::pair<int, int>(offsets[p], sizes[p]);
                }
            } else {
                regular_splitting.splitting(curr_cluster, permutation, spatial_dimension, x, current_number_of_children, minclustersize, dir, current_partition);
            }
        } else {
            current_partition.clear();
        }
    }
};

} // namespace htool

#endif
