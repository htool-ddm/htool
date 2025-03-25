#ifndef HTOOL_CLUSTERING_CLUSTER_NODE_HPP
#define HTOOL_CLUSTERING_CLUSTER_NODE_HPP

#include "../basic_types/tree.hpp" // for TreeNode
#include "../misc/logger.hpp"      // for Logger, LogLevel, LogLevel::ERROR
#include "../misc/misc.hpp"        // for underlying_type
#include "cluster_tree_data.hpp"
#include <cstddef> // for size_t
#include <memory>  // for make_shared, operator==
#include <numeric> // for iota
#include <string>  // for basic_string
#include <vector>  // for vector

namespace htool {

template <typename CoordinatesPrecision>
class Cluster : public TreeNode<Cluster<CoordinatesPrecision>, ClusterTreeData<CoordinatesPrecision>> {
  protected:
    CoordinatesPrecision m_radius{0};             // radius of the current cluster
    std::vector<CoordinatesPrecision> m_center{}; // center of the current cluster
    int m_rank;                                   // Rank for dofs of the current cluster
    int m_offset;                                 // Offset of the current cluster in the global numbering
    int m_size;                                   // number of geometric points
    int m_counter;                                // numbering of the nodes level-wise

  public:
    // Root constructor
    Cluster(CoordinatesPrecision radius, std::vector<CoordinatesPrecision> &center, int rank, int offset, int size) : TreeNode<Cluster, ClusterTreeData<CoordinatesPrecision>>(), m_radius(radius), m_center(center), m_rank(rank), m_offset(offset), m_size(size), m_counter(0) {
        this->m_tree_data->m_permutation = std::make_shared<std::vector<int>>(m_size);
        std::iota(this->m_tree_data->m_permutation->begin(), this->m_tree_data->m_permutation->end(), int(0));
        this->m_tree_data->m_root_cluster = this;
    }

    // Child constructor
    Cluster(const Cluster &parent, CoordinatesPrecision radius, std::vector<CoordinatesPrecision> &center, int rank, int offset, int size, int counter, bool is_on_partition) : TreeNode<Cluster, ClusterTreeData<CoordinatesPrecision>>(parent), m_radius(radius), m_center(center), m_rank(rank), m_offset(offset), m_size(size), m_counter(counter) {
        if (is_on_partition) {
            if (rank + 1 > this->m_tree_data->m_clusters_on_partition.size()) {
                this->m_tree_data->m_clusters_on_partition.resize(rank + 1, nullptr);
            }
            this->m_tree_data->m_clusters_on_partition[rank] = this;
        }
    }

    // no copy
    Cluster(const Cluster &)                       = delete;
    Cluster &operator=(const Cluster &)            = delete;
    Cluster(Cluster &&cluster) noexcept            = default;
    Cluster &operator=(Cluster &&cluster) noexcept = default;
    virtual ~Cluster()                             = default;

    // Cluster getters
    const CoordinatesPrecision &get_radius() const { return m_radius; }
    const std::vector<CoordinatesPrecision> &get_center() const { return m_center; }
    int get_rank() const { return m_rank; }
    int get_offset() const { return m_offset; }
    int get_size() const { return m_size; }
    int get_counter() const { return m_counter; }

    // Test properties
    bool is_permutation_local() const { return this->m_tree_data->m_is_permutation_local; }

    // Cluster tree getters
    unsigned int get_maximal_depth() const { return this->m_tree_data->m_max_depth; }
    unsigned int get_minimal_depth() const { return this->m_tree_data->m_min_depth; }
    unsigned int get_maximal_leaf_size() const { return this->m_tree_data->m_maximal_leaf_size; }
    const std::vector<const Cluster<CoordinatesPrecision> *> &get_clusters_on_partition() const { return this->m_tree_data->m_clusters_on_partition; }
    const Cluster<CoordinatesPrecision> &get_cluster_on_partition(size_t index) const {
        return *this->m_tree_data->m_clusters_on_partition[index];
    }
    const Cluster<CoordinatesPrecision> &get_root_cluster() const { return *this->m_tree_data->m_root_cluster; }
    const std::vector<int> &get_permutation() const { return *(this->m_tree_data->m_permutation); }
    std::vector<int> &get_permutation() { return *(this->m_tree_data->m_permutation); }

    // Cluster tree setters
    void set_is_permutation_local(bool is_permutation_local) { this->m_tree_data->m_is_permutation_local = is_permutation_local; }
    void set_minimal_depth(unsigned int minimal_depth) { this->m_tree_data->m_min_depth = minimal_depth; }
    void set_maximal_depth(unsigned int maximal_depth) { this->m_tree_data->m_max_depth = maximal_depth; }
    void set_maximal_leaf_size(unsigned int maximal_leaf_size) { this->m_tree_data->m_maximal_leaf_size = maximal_leaf_size; }

    // Operator overloading
    bool operator==(const Cluster<CoordinatesPrecision> &rhs) const { return this->get_offset() == rhs.get_offset() && this->get_size() == rhs.get_size() && this->m_tree_data == rhs.m_tree_data && this->get_depth() == rhs.get_depth() && this->get_counter() == rhs.get_counter(); }
};

template <typename CoordinatesPrecision>
bool is_cluster_on_partition(const Cluster<CoordinatesPrecision> &cluster) {
    return cluster.get_depth() == cluster.get_clusters_on_partition()[0]->get_depth();
}

template <typename CoordinatesPrecision>
bool left_cluster_contains_right_cluster(const Cluster<CoordinatesPrecision> &cluster1, const Cluster<CoordinatesPrecision> &cluster2) {

    if (cluster1.get_offset() <= cluster2.get_offset() && cluster1.get_size() + cluster1.get_offset() >= cluster2.get_size() + cluster2.get_offset()) {
        return true;
    }
    return false;
}

// Permutations
template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
void root_cluster_to_global(const Cluster<CoordinatesPrecision> &root_cluster, const CoefficientPrecision *const in, CoefficientPrecision *const out) {
    if (!root_cluster.is_root()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation needs root cluster"); // LCOV_EXCL_LINE
    }
    const auto &permutation = root_cluster.get_permutation();
    for (int i = 0; i < root_cluster.get_size(); i++) {
        out[permutation[i + root_cluster.get_offset()] - root_cluster.get_offset()] = in[i];
    }
}

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
void global_to_root_cluster(const Cluster<CoordinatesPrecision> &root_cluster, const CoefficientPrecision *const in, CoefficientPrecision *const out) {
    if (!root_cluster.is_root()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation needs root cluster"); // LCOV_EXCL_LINE
    }
    const auto &permutation = root_cluster.get_permutation();
    for (int i = 0; i < root_cluster.get_size(); i++) {
        out[i] = in[permutation[i + root_cluster.get_offset()] - root_cluster.get_offset()];
    }
}

// Local permutations
template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
void local_cluster_to_local(const Cluster<CoordinatesPrecision> &cluster, int index, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!cluster.is_permutation_local()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation is not local to partition, local_cluster_to_local cannot be used"); // LCOV_EXCL_LINE
    }
    const auto &permutation                                   = cluster.get_permutation();
    const Cluster<CoordinatesPrecision> *cluster_on_partition = cluster.get_clusters_on_partition()[index];

    for (int i = 0; i < cluster_on_partition->get_size(); i++) {
        out[permutation[cluster_on_partition->get_offset() + i] - cluster_on_partition->get_offset()] = in[i];
    }
}
template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
void local_to_local_cluster(const Cluster<CoordinatesPrecision> &cluster, int index, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!cluster.is_permutation_local()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation is not local to partition, local_cluster_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    const auto &permutation                                   = cluster.get_permutation();
    const Cluster<CoordinatesPrecision> *cluster_on_partition = cluster.get_clusters_on_partition()[index];

    for (int i = 0; i < cluster_on_partition->get_size(); i++) {
        out[i] = in[permutation[cluster_on_partition->get_offset() + i] - cluster_on_partition->get_offset()];
    }
}

// Local permutations
template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
void cluster_to_user(const Cluster<CoordinatesPrecision> &cluster, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!cluster.is_root() && !is_cluster_on_partition(cluster)) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Cluster is neither root nor local, permutation is not stable."); // LCOV_EXCL_LINE
    }
    if (is_cluster_on_partition(cluster) && !cluster.is_permutation_local()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Cluster is local, but permutation is not, so permutation is not stable."); // LCOV_EXCL_LINE
    }
    const auto &permutation = cluster.get_permutation();
    for (int i = 0; i < cluster.get_size(); i++) {
        out[permutation[cluster.get_offset() + i] - cluster.get_offset()] = in[i];
    }
}
template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
void user_to_cluster(const Cluster<CoordinatesPrecision> &cluster, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!cluster.is_root() && !is_cluster_on_partition(cluster)) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Cluster is neither root nor local, permutation is not stable."); // LCOV_EXCL_LINE
    }
    if (is_cluster_on_partition(cluster) && !cluster.is_permutation_local()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Cluster is local, but permutation is not, so permutation is not stable."); // LCOV_EXCL_LINE
    }

    const auto &permutation = cluster.get_permutation();
    for (int i = 0; i < cluster.get_size(); i++) {
        out[i] = in[permutation[cluster.get_offset() + i] - cluster.get_offset()];
    }
}

} // namespace htool
#endif
