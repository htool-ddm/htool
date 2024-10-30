#ifndef HTOOL_CLUSTERING_TREE_BUILDER_HPP
#define HTOOL_CLUSTERING_TREE_BUILDER_HPP

#include "../../basic_types/vector.hpp" // for norm2
#include "../../misc/logger.hpp"        // for Logger, LogLevel
#include "../cluster_node.hpp"          // for Cluster
#include "direction_computation.hpp"
#include "splitting.hpp"
#include <algorithm>  // for sort, transform
#include <cmath>      // for pow, log, floor
#include <deque>      // for deque
#include <functional> // for _1, bind, multiplies
#include <memory>     // for shared_ptr, make_shared
#include <numeric>    // for iota, inner_product
#include <stack>      // for stack
#include <string>     // for basic_string
#include <utility>    // for pair
#include <vector>     // for vector

namespace htool {

template <typename T>
class ClusterTreeBuilder {

    int m_minclustersize{10};
    enum PartitionType { PowerOfNumberOfChildren,
                         Given,
                         Simple };

    std::vector<T> compute_center(int spatial_dimension, const T *coordinates, const T *weights, int offset, int size, const int *permutation = nullptr) const;
    T compute_radius(int spatial_dimension, const T *coordinates, const T *radii, std::vector<T> center, int offset, int size, const int *permutation = nullptr) const;

    std::shared_ptr<VirtualDirectionComputationStrategy<T>> m_direction_computation_strategy = std::make_shared<ComputeLargestExtent<T>>();
    std::shared_ptr<VirtualSplittingStrategy<T>> m_splitting_strategy                        = std::make_shared<RegularSplitting<T>>();

  public:
    ClusterTreeBuilder() {}

    void set_direction_computation_strategy(std::shared_ptr<VirtualDirectionComputationStrategy<T>> direction_computation_strategy) { m_direction_computation_strategy = direction_computation_strategy; }

    void set_splitting_strategy(std::shared_ptr<VirtualSplittingStrategy<T>> splitting_strategy) { m_splitting_strategy = splitting_strategy; }

    void set_minclustersize(int minclustersize) { m_minclustersize = minclustersize; }

    Cluster<T> create_cluster_tree(int number_of_points, int spatial_dimension, const T *coordinates, const T *radii, const T *weights, int number_of_children, int size_of_partition, const int *partition) const;

    Cluster<T> create_cluster_tree(int number_of_points, int spatial_dimension, const T *coordinates, int number_of_children, int size_of_partition) const { return create_cluster_tree(number_of_points, spatial_dimension, coordinates, nullptr, nullptr, number_of_children, size_of_partition, nullptr); }

    Cluster<T> create_cluster_tree(int number_of_points, int spatial_dimension, const T *coordinates, int number_of_children, int size_of_partition, const int *partition) const { return create_cluster_tree(number_of_points, spatial_dimension, coordinates, nullptr, nullptr, number_of_children, size_of_partition, partition); }
};

template <typename T>
Cluster<T> ClusterTreeBuilder<T>::create_cluster_tree(int number_of_points, int spatial_dimension, const T *coordinates, const T *radii, const T *weights, int number_of_children, int size_partition, const int *partition) const {

    std::vector<std::pair<int, int>> m_partition{};
    PartitionType partition_type{Simple};

    // default values
    std::vector<T> default_radii{};
    std::vector<T> default_weights{};
    if (radii == nullptr) {
        default_radii.resize(number_of_points, 0);
        radii = default_radii.data();
    }
    if (weights == nullptr) {
        default_weights.resize(number_of_points, 1);
        weights = default_weights.data();
    }

    // Intialization of root
    std::vector<T> center = compute_center(spatial_dimension, coordinates, weights, 0, number_of_points);
    T radius              = compute_radius(spatial_dimension, coordinates, radii, center, 0, number_of_points);

    Cluster<T> root_cluster(radius, center, -1, 0, number_of_points);
    std::vector<int> &permutation = root_cluster.get_permutation();
    root_cluster.set_minclustersize(m_minclustersize);

    // Taking care of partition initialisation
    std::stack<Cluster<T> *> cluster_stack(std::deque<Cluster<T> *>{&root_cluster});
    int depth_of_partition;
    int number_of_children_on_partition_level = size_partition;
    int additional_children_on_last_partition = 0;
    if (partition != nullptr) {
        partition_type     = Given;
        depth_of_partition = 1;
        cluster_stack.pop();
        bool is_child_on_partition = true;
        root_cluster.set_is_permutation_local(true);
        for (int p = 0; p < size_partition; p++) {
            center                           = compute_center(spatial_dimension, coordinates, weights, partition[2 * p], partition[2 * p + 1], permutation.data());
            radius                           = compute_radius(spatial_dimension, coordinates, radii, center, partition[2 * p], partition[2 * p + 1], permutation.data());
            Cluster<T> *cluster_on_partition = root_cluster.add_child(radius, center, p, partition[2 * p], partition[2 * p + 1], p, is_child_on_partition);
            cluster_stack.push(cluster_on_partition);
        }
    } else {
        partition_type = Simple;
        if (size_partition >= number_of_children) {
            depth_of_partition                    = static_cast<int>(floor(log(size_partition) / log(number_of_children)));
            number_of_children_on_partition_level = number_of_children;
            if (size_partition != std::pow(number_of_children, depth_of_partition)) {
                Logger::get_instance().log(LogLevel::WARNING, "The given size for the partition is not a power of the number of children in the cluster tree.");
                additional_children_on_last_partition = size_partition - std::pow(number_of_children, depth_of_partition);
            }
        } else {
            depth_of_partition = 1;
        }
    }

    // Recursive build
    std::vector<std::pair<int, int>> current_splitting(number_of_children);

    while (!cluster_stack.empty()) {
        Cluster<T> *current_cluster = cluster_stack.top();
        cluster_stack.pop();
        auto current_offset            = current_cluster->get_offset();
        auto current_size              = current_cluster->get_size();
        int current_number_of_children = ((current_cluster->get_depth() == depth_of_partition - 1) && (partition_type == Simple)) ? number_of_children_on_partition_level : number_of_children;

        if ((current_cluster->get_depth() == depth_of_partition - 1) && (partition_type == Simple) && current_cluster->get_counter() == std::pow(number_of_children, current_cluster->get_depth()) - 1) {
            current_number_of_children += additional_children_on_last_partition;
        }

        // Direction of largest extent
        std::vector<T> direction(spatial_dimension, 0);
        direction = m_direction_computation_strategy->compute_direction(current_cluster, permutation, spatial_dimension, coordinates, radii, weights);

        // Sort along direction
        std::sort(permutation.begin() + current_offset, permutation.begin() + current_offset + current_size, [&](int a, int b) {
            T c = std::inner_product(coordinates + spatial_dimension * a, coordinates + spatial_dimension * (1 + a), direction.data(), T(0));
            T d = std::inner_product(coordinates + spatial_dimension * b, coordinates + spatial_dimension * (1 + b), direction.data(), T(0));
            return c < d;
        });

        // Compute numbering
        m_splitting_strategy->splitting(current_cluster, permutation, spatial_dimension, coordinates, current_number_of_children, m_minclustersize, direction, current_splitting);
        std::vector<Cluster<T> *> children;

        if (current_splitting.size() > 0) {
            bool is_child_on_partition = false;
            for (int p = 0; p < current_splitting.size(); p++) {
                center = compute_center(spatial_dimension, coordinates, weights, current_splitting[p].first, current_splitting[p].second, permutation.data());
                radius = compute_radius(spatial_dimension, coordinates, radii, center, current_splitting[p].first, current_splitting[p].second, permutation.data());

                int rank_of_child    = current_cluster->get_rank();
                int counter_of_child = current_cluster->get_counter() * current_number_of_children + p;
                if ((current_cluster->get_depth() == depth_of_partition - 1) && (partition_type == Simple)) {
                    rank_of_child         = current_cluster->get_counter() * number_of_children_on_partition_level + p;
                    counter_of_child      = rank_of_child;
                    is_child_on_partition = true;
                }

                children.emplace_back(current_cluster->add_child(radius, center, rank_of_child, current_splitting[p].first, current_splitting[p].second, counter_of_child, is_child_on_partition));
            }

            // Recursivity
            for (auto &child : children) {
                cluster_stack.push(child);
            }
        } else if (current_cluster->get_rank() < 0) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "Cluster tree reached maximal depth, but not enough children to define a partition"); // LCOV_EXCL_LINE
        } else {
            current_cluster->set_maximal_depth(std::max(current_cluster->get_maximal_depth(), current_cluster->get_depth()));
            current_cluster->set_minimal_depth(std::min(current_cluster->get_minimal_depth(), current_cluster->get_depth()));
        }

        if (partition_type == PartitionType::Simple && current_cluster->get_depth() == depth_of_partition - 1) {
            for (const auto &child : children) {
                m_partition.emplace_back(child->get_offset(), child->get_size());
            }
        }
    }

    return root_cluster;
}

template <typename T>
std::vector<T> ClusterTreeBuilder<T>::compute_center(int spatial_dimension, const T *coordinates, const T *weights, int offset, int size, const int *current_permutation) const {
    std::vector<T> center(spatial_dimension, 0);

    bool is_first_permutation = (current_permutation == nullptr);
    std::vector<int> first_permutation(is_first_permutation ? size : 0, 0);
    std::iota(first_permutation.begin(), first_permutation.end(), int(0));
    const int *permutation = is_first_permutation ? first_permutation.data() : current_permutation;

    // Mass of the cluster
    T total_weight = 0;
    for (int j = 0; j < size; j++) {
        total_weight += weights[permutation[j + offset]];
    }

    // Center of the cluster
    for (int j = 0; j < size; j++) {
        for (int p = 0; p < spatial_dimension; p++) {
            center[p] += weights[permutation[j + offset]] * coordinates[spatial_dimension * permutation[j + offset] + p];
        }
    }

    std::transform(center.begin(), center.end(), center.begin(), std::bind(std::multiplies<T>(), std::placeholders::_1, static_cast<T>(1.) / total_weight));
    return center;
}

template <typename T>
T ClusterTreeBuilder<T>::compute_radius(int spatial_dimension, const T *coordinates, const T *radii, std::vector<T> center, int offset, int size, const int *current_permutation) const {
    T radius = 0;

    bool is_first_permutation = (current_permutation == nullptr);
    std::vector<int> first_permutation(is_first_permutation ? size : 0, 0);
    std::iota(first_permutation.begin(), first_permutation.end(), int(0));
    const int *permutation = is_first_permutation ? first_permutation.data() : current_permutation;

    for (int j = 0; j < size; j++) {
        std::vector<T> u(spatial_dimension, 0);
        for (int p = 0; p < spatial_dimension; p++) {
            u[p] = coordinates[spatial_dimension * permutation[j + offset] + p] - center[p];
        }

        radius = std::max(radius, norm2(u) + radii[permutation[j + offset]]);
    }
    return radius;
}

} // namespace htool

#endif
