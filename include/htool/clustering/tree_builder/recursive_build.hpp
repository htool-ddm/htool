#ifndef HTOOL_CLUSTERING_TREE_BUILDER_HPP
#define HTOOL_CLUSTERING_TREE_BUILDER_HPP

#include "../../misc/evp.hpp"
#include "../cluster_node.hpp"

#include <stack>

namespace htool {

template <typename T, class DirectionComputationStrategy, class SplittingStrategy>
class ClusterTreeBuilder {

    int m_number_of_points;
    int m_spatial_dimension;
    const T *m_coordinates;
    const T *m_radii;
    const T *m_weights;
    int m_number_of_children;
    int m_size_partition;

    std::vector<std::pair<int, int>> m_partition{};
    int m_minclustersize{10};
    enum PartitionType { PowerOfNumberOfChildren,
                         Given,
                         Simple };
    PartitionType m_partition_type{Simple};

    std::vector<T> compute_center(int offset, int size, const int *permutation = nullptr) const;
    T compute_radius(std::vector<T> center, int offset, int size, const int *permutation = nullptr) const;

    DirectionComputationStrategy m_direction_computation_strategy;
    SplittingStrategy m_splitting_strategy;

  public:
    ClusterTreeBuilder(int number_of_points, int spatial_dimension, const T *coordinates, const T *radii, const T *weights, int number_of_children, int size_partition, const DirectionComputationStrategy &direction_computation_strategy = DirectionComputationStrategy(), const SplittingStrategy &splitting_strategy = SplittingStrategy()) : m_number_of_points(number_of_points), m_spatial_dimension(spatial_dimension), m_coordinates(coordinates), m_radii(radii), m_weights(weights), m_number_of_children(number_of_children), m_size_partition(size_partition), m_direction_computation_strategy(direction_computation_strategy), m_splitting_strategy(splitting_strategy) {}

    ClusterTreeBuilder(int number_of_points, int spatial_dimension, const T *coordinates, int number_of_children, int size_partition, const DirectionComputationStrategy &direction_computation_strategy = DirectionComputationStrategy(), const SplittingStrategy &splitting_strategy = SplittingStrategy()) : ClusterTreeBuilder(number_of_points, spatial_dimension, coordinates, nullptr, nullptr, number_of_children, size_partition, direction_computation_strategy, splitting_strategy) {}

    void set_partition(const std::vector<std::pair<int, int>> &partition) {
        m_partition_type = Given;
        m_size_partition = partition.size();
        m_partition      = partition;
    }
    void set_partition(int size_partition, const int *partition) {
        m_partition_type = Given;
        m_size_partition = size_partition;
        m_partition.resize(size_partition);
        for (int p = 0; p < size_partition; p++) {
            m_partition[p].first  = partition[2 * p];
            m_partition[p].second = partition[2 * p + 1];
        }
    }

    void set_direction_computation_strategy(const DirectionComputationStrategy &direction_computation_strategy) { m_direction_computation_strategy = direction_computation_strategy; }

    void set_splitting_strategy(const SplittingStrategy &splitting_strategy) { m_splitting_strategy = splitting_strategy; }

    void set_minclustersize(int minclustersize) { m_minclustersize = minclustersize; };
    Cluster<T> create_cluster_tree();
};

template <typename T, class DirectionComputationStrategy, class SplittingStrategy>
Cluster<T> ClusterTreeBuilder<T, DirectionComputationStrategy, SplittingStrategy>::create_cluster_tree() {

    // default values
    std::vector<T> default_radii{};
    std::vector<T> default_weights{};
    if (m_radii == nullptr) {
        default_radii.resize(m_number_of_points, 0);
        m_radii = default_radii.data();
    }
    if (m_weights == nullptr) {
        default_weights.resize(m_number_of_points, 1);
        m_weights = default_weights.data();
    }

    // Intialization of root
    std::vector<T> center = compute_center(0, m_number_of_points);
    T radius              = compute_radius(center, 0, m_number_of_points);
    // ClusterTree<T> cluster_tree(m_minclustersize);
    // Cluster<T> *root_cluster = cluster_tree.add_root(radius, center, -1, 0, m_number_of_points, 0);
    Cluster<T> root_cluster(radius, center, -1, 0, m_number_of_points);
    // Cluster<T> root(radius, center, -1, 0, m_number_of_points, 0);
    // root.set_minclustersize(m_minclustersize);
    // std::shared_ptr<std::vector<int>> permutation_ptr = std::make_shared<std::vector<int>>(m_number_of_points, 0);
    // std::vector<int> &permutation = cluster_tree.get_permutation();
    std::vector<int> &permutation = root_cluster.get_permutation();
    // std::iota(permutation.begin(), permutation.end(), int(0));
    // root.set_permutation(permutation_ptr);

    // Taking care of partition initialisation
    int depth_of_partition = 1;
    std::stack<Cluster<T> *> cluster_stack(std::deque<Cluster<T> *>{&root_cluster});

    if (m_partition_type == Given) {
        cluster_stack.pop();
        bool is_child_on_partition = true;
        root_cluster.set_is_permutation_local(true);
        for (int p = 0; p < m_partition.size(); p++) {
            center                           = compute_center(m_partition[p].first, m_partition[p].second, permutation.data());
            radius                           = compute_radius(center, m_partition[p].first, m_partition[p].second, permutation.data());
            Cluster<T> *cluster_on_partition = root_cluster.add_child(radius, center, p, m_partition[p].first, m_partition[p].second, p, is_child_on_partition);
            cluster_stack.push(cluster_on_partition);
        }
    }
    // else {
    //     int test = m_size_partition;
    //     int power_of_number_of_children{0};
    //     while (test % m_number_of_children == 0) {
    //         test = std::div(test, m_number_of_children).quot;
    //         power_of_number_of_children++;
    //     }
    //     bool is_size_partition_power_of_number_of_children{test == 1};

    //     if (is_size_partition_power_of_number_of_children) {

    //         m_partition_type   = PowerOfNumberOfChildren;
    //         depth_of_partition = power_of_number_of_children;
    //         std::cout << "Power " << depth_of_partition << "\n";
    //     }
    // }

    // Recursive build
    std::vector<std::pair<int, int>> current_splitting(m_number_of_children);

    while (!cluster_stack.empty()) {
        Cluster<T> *current_cluster = cluster_stack.top();
        cluster_stack.pop();
        auto current_offset            = current_cluster->get_offset();
        auto current_size              = current_cluster->get_size();
        int current_number_of_children = ((current_cluster->get_depth() == 0) && (m_partition_type == Simple)) ? m_size_partition : m_number_of_children;

        // Direction of largest extent
        std::vector<T> direction(m_spatial_dimension, 0);
        direction = m_direction_computation_strategy.compute_direction(current_cluster, permutation, m_spatial_dimension, m_coordinates, m_radii, m_weights);

        // Sort along direction
        std::sort(permutation.begin() + current_offset, permutation.begin() + current_offset + current_size, [&](int a, int b) {
            T c = std::inner_product(m_coordinates + m_spatial_dimension * a, m_coordinates + m_spatial_dimension * (1 + a), direction.data(), T(0));
            T d = std::inner_product(m_coordinates + m_spatial_dimension * b, m_coordinates + m_spatial_dimension * (1 + b), direction.data(), T(0));
            return c < d;
        });

        // Compute numbering
        m_splitting_strategy.splitting(current_cluster, permutation, m_spatial_dimension, m_coordinates, current_number_of_children, m_minclustersize, direction, current_splitting);
        std::vector<Cluster<T> *> children;

        if (current_splitting.size() > 0) {
            bool is_child_on_partition = false;
            for (int p = 0; p < current_splitting.size(); p++) {
                center = compute_center(current_splitting[p].first, current_splitting[p].second, permutation.data());
                radius = compute_radius(center, current_splitting[p].first, current_splitting[p].second, permutation.data());

                int rank_of_child = current_cluster->get_rank();
                if ((current_cluster->get_depth() == 0) && (m_partition_type == Simple)) {
                    rank_of_child         = p;
                    is_child_on_partition = true;
                }
                // else if ((current_cluster->get_depth() == depth_of_partition - 1) && (m_partition_type == PowerOfNumberOfChildren)) {
                //     rank_of_child = current_cluster->get_counter() * current_number_of_children + p;
                // }
                children.emplace_back(current_cluster->add_child(radius, center, rank_of_child, current_splitting[p].first, current_splitting[p].second, current_cluster->get_counter() * current_number_of_children + p, is_child_on_partition));
            }

            // Recursivity
            for (auto &child : children) {
                cluster_stack.push(child);
            }
        } else if (current_cluster->get_rank() < 0) {
            throw std::logic_error("[Htool error] Cluster tree reached maximal depth, but not enough children to define a partition."); // LCOV_EXCL_LINE
        } else {
            current_cluster->set_maximal_depth(std::max(current_cluster->get_maximal_depth(), current_cluster->get_depth()));
            current_cluster->set_minimal_depth(std::min(current_cluster->get_minimal_depth(), current_cluster->get_depth()));
        }
    }

    return root_cluster;
};

template <typename T, class DirectionComputationStrategy, class SplittingStrategy>
std::vector<T> ClusterTreeBuilder<T, DirectionComputationStrategy, SplittingStrategy>::compute_center(int offset, int size, const int *current_permutation) const {
    std::vector<T> center(m_spatial_dimension, 0);

    bool is_first_permutation = (current_permutation == nullptr);
    std::vector<int> first_permutation(is_first_permutation ? size : 0, 0);
    std::iota(first_permutation.begin(), first_permutation.end(), int(0));
    const int *permutation = is_first_permutation ? first_permutation.data() : current_permutation;

    // Mass of the cluster
    T total_weight = 0;
    for (int j = 0; j < size; j++) {
        // std::cout << j + current_offset << "\n";
        total_weight += m_weights[permutation[j + offset]];
    }

    // Center of the cluster
    for (int j = 0; j < size; j++) {
        for (int p = 0; p < m_spatial_dimension; p++) {
            center[p] += m_weights[permutation[j + offset]] * m_coordinates[m_spatial_dimension * permutation[j + offset] + p];
        }
    }
    std::transform(center.begin(), center.end(), center.begin(), std::bind(std::multiplies<T>(), std::placeholders::_1, 1. / total_weight));
    return center;
}

template <typename T, class DirectionComputationStrategy, class SplittingStrategy>
T ClusterTreeBuilder<T, DirectionComputationStrategy, SplittingStrategy>::compute_radius(std::vector<T> center, int offset, int size, const int *current_permutation) const {
    T radius = 0;

    bool is_first_permutation = (current_permutation == nullptr);
    std::vector<int> first_permutation(is_first_permutation ? size : 0, 0);
    std::iota(first_permutation.begin(), first_permutation.end(), int(0));
    const int *permutation = is_first_permutation ? first_permutation.data() : current_permutation;

    for (int j = 0; j < size; j++) {
        std::vector<T> u(m_spatial_dimension, 0);
        for (int p = 0; p < m_spatial_dimension; p++) {
            u[p] = m_coordinates[m_spatial_dimension * permutation[j + offset] + p] - center[p];
        }

        radius = std::max(radius, norm2(u) + m_radii[permutation[j + offset]]);
    }
    return radius;
}

} // namespace htool

#endif
