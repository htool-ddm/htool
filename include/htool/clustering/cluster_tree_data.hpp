#ifndef HTOOL_CLUSTERING_CLUSTER_TREE_DATA_HPP
#define HTOOL_CLUSTERING_CLUSTER_TREE_DATA_HPP

#include <limits>
#include <vector>

namespace htool {

template <class CoordinatePrecision>
class Cluster;

template <typename CoordinatePrecision>
struct ClusterTreeData {
    // Parameters
    unsigned int m_minclustersize{10}; // minimal number of geometric point in a cluster

    // Information
    unsigned int m_max_depth{std::numeric_limits<unsigned int>::min()}; // maximum depth of the tree
    unsigned int m_min_depth{std::numeric_limits<unsigned int>::max()}; // minimum depth of the tree
    std::shared_ptr<std::vector<int>> m_permutation{nullptr};           // permutation from htool numbering to user numbering
    bool m_is_permutation_local{false};

    // Nodes
    const Cluster<CoordinatePrecision> *m_root_cluster{nullptr};
    std::vector<const Cluster<CoordinatePrecision> *> m_clusters_on_partition{};
};

} // namespace htool
#endif
