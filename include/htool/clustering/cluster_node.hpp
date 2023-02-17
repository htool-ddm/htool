#ifndef HTOOL_CLUSTERING_CLUSTER_NODE_HPP
#define HTOOL_CLUSTERING_CLUSTER_NODE_HPP

#include "../basic_types/tree.hpp"
#include "../misc/user.hpp"
#include "cluster_tree_data.hpp"
#include <functional>
#include <memory>
#include <numeric>
#include <stack>
#include <vector>

namespace htool {

template <typename CoordinatesPrecision>
class ClusterTreeData;
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
    }

    // Child constructor
    Cluster(const Cluster &parent, CoordinatesPrecision radius, std::vector<CoordinatesPrecision> &center, int rank, int offset, int size, int counter, bool is_on_partition) : TreeNode<Cluster, ClusterTreeData<CoordinatesPrecision>>(parent), m_radius(radius), m_center(center), m_rank(rank), m_offset(offset), m_size(size), m_counter(counter) {
        if (is_on_partition) {
            this->m_tree_data->m_clusters_on_partition.push_back(this);
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
    unsigned int get_minclustersize() const { return this->m_tree_data->m_minclustersize; }
    const std::vector<const Cluster<CoordinatesPrecision> *> &get_clusters_on_partition() const { return this->m_tree_data->m_clusters_on_partition; }
    const std::vector<int> &get_permutation() const { return *(this->m_tree_data->m_permutation); }
    std::vector<int> &get_permutation() { return *(this->m_tree_data->m_permutation); }

    // Cluster tree setters
    void set_is_permutation_local(bool is_permutation_local) { this->m_tree_data->m_is_permutation_local = is_permutation_local; }
    void set_minimal_depth(unsigned int minimal_depth) { this->m_tree_data->m_min_depth = minimal_depth; }
    void set_maximal_depth(unsigned int maximal_depth) { this->m_tree_data->m_max_depth = maximal_depth; }
    void set_minclustersize(unsigned int minclustersize) { this->m_tree_data->m_minclustersize = minclustersize; }

    // Operator overloading
    bool operator==(const Cluster<CoordinatesPrecision> &rhs) const { return this->get_offset() == rhs.get_offset() && this->get_size() == rhs.get_size() && this->m_tree_data == rhs.m_tree_data; }
};

template <typename CoordinatesPrecision>
bool is_cluster_on_partition(const Cluster<CoordinatesPrecision> &cluster) {
    return cluster.get_depth() == cluster.get_clusters_on_partition()[0]->get_depth();
}

template <typename CoordinatesPrecision>
Cluster<CoordinatesPrecision> clone_cluster_tree_from_partition(const Cluster<CoordinatesPrecision> &cluster, int index) {
    if (!cluster.is_permutation_local()) {
        throw std::logic_error("[Htool error] Permutation is not local to partition, cluster on partition cannot be cloned."); // LCOV_EXCL_LINE
    }
    const Cluster<CoordinatesPrecision> &cluster_on_partition = *cluster.get_clusters_on_partition()[index];

    // Initialisation of new root own properties
    CoordinatesPrecision radius              = cluster_on_partition.get_radius();
    std::vector<CoordinatesPrecision> center = cluster_on_partition.get_center();
    int rank                                 = cluster_on_partition.get_rank();
    int offset                               = cluster_on_partition.get_offset();
    int size                                 = cluster_on_partition.get_size();
    int depth                                = 0;
    int counter                              = 0;

    // Build new cluster tree
    Cluster<CoordinatesPrecision> new_root_cluster(radius, center, rank, offset, size);
    new_root_cluster.set_maximal_depth(cluster.get_maximal_depth());
    new_root_cluster.set_maximal_depth(cluster.get_maximal_depth() - cluster_on_partition.get_depth());
    new_root_cluster.set_minimal_depth(cluster.get_minimal_depth() - cluster_on_partition.get_depth());
    new_root_cluster.set_is_permutation_local(cluster.is_permutation_local());
    // Cluster<CoordinatesPrecision> *new_root = new_cluster_tree.add_root(radius, center, rank, offset, size, counter);
    new_root_cluster.get_permutation() = cluster.get_permutation();

    int counter_offset = cluster_on_partition.get_counter();

    // Recursivity
    std::stack<const Cluster<CoordinatesPrecision> *> old_cluster_stack;
    std::stack<Cluster<CoordinatesPrecision> *> new_cluster_stack;
    new_cluster_stack.push(&new_root_cluster);
    old_cluster_stack.push(&cluster_on_partition);

    while (!new_cluster_stack.empty()) {
        const Cluster<CoordinatesPrecision> *current_old_cluster = old_cluster_stack.top();
        Cluster<CoordinatesPrecision> *current_new_cluster       = new_cluster_stack.top();
        old_cluster_stack.pop();
        new_cluster_stack.pop();

        for (const auto &child : current_old_cluster->get_children()) {
            // Copy
            radius  = child->get_radius();
            center  = child->get_center();
            rank    = child->get_rank();
            offset  = child->get_offset();
            size    = child->get_size();
            counter = child->get_counter() - counter_offset;

            old_cluster_stack.push(child.get());
            new_cluster_stack.push(current_new_cluster->add_child(radius, center, rank, offset, size, counter, false));
        }
    }

    return new_root_cluster;
}

// Permutations
template <typename CoefficientPrecision, typename CoordinatesPrecision = CoefficientPrecision>
void root_cluster_to_global(const Cluster<CoordinatesPrecision> &root_cluster, const CoefficientPrecision *const in, CoefficientPrecision *const out) {
    if (!root_cluster.is_root()) {
        throw std::logic_error("[Htool error] Permutation needs root cluster."); // LCOV_EXCL_LINE
    }
    const auto &permutation = root_cluster.get_permutation();
    for (int i = 0; i < root_cluster.get_size(); i++) {
        out[permutation[i + root_cluster.get_offset()] - root_cluster.get_offset()] = in[i];
    }
}

template <typename CoefficientPrecision, typename CoordinatesPrecision = CoefficientPrecision>
void global_to_root_cluster(const Cluster<CoordinatesPrecision> &root_cluster, const CoefficientPrecision *const in, CoefficientPrecision *const out) {
    if (!root_cluster.is_root()) {
        throw std::logic_error("[Htool error] Permutation needs root cluster."); // LCOV_EXCL_LINE
    }
    const auto &permutation = root_cluster.get_permutation();
    for (int i = 0; i < root_cluster.get_size(); i++) {
        out[i] = in[permutation[i + root_cluster.get_offset()] - root_cluster.get_offset()];
    }
}

// Local permutations
template <typename CoefficientPrecision, typename CoordinatesPrecision = CoefficientPrecision>
void local_cluster_to_local(const Cluster<CoordinatesPrecision> &cluster, int index, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!cluster.is_permutation_local()) {
        throw std::logic_error("[Htool error] Permutation is not local to partition, local_cluster_to_local cannot be used"); // LCOV_EXCL_LINE
    }
    const auto &permutation                                   = cluster.get_permutation();
    const Cluster<CoordinatesPrecision> *cluster_on_partition = cluster.get_clusters_on_partition()[index];

    for (int i = 0; i < cluster_on_partition->get_size(); i++) {
        out[permutation[cluster_on_partition->get_offset() + i] - cluster_on_partition->get_offset()] = in[i];
    }
}
template <typename CoefficientPrecision, typename CoordinatesPrecision = CoefficientPrecision>
void local_to_local_cluster(const Cluster<CoordinatesPrecision> &cluster, int index, const CoefficientPrecision *in, CoefficientPrecision *out) {
    if (!cluster.is_permutation_local()) {
        throw std::logic_error("[Htool error] Permutation is not local to partition, local_cluster_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    const auto &permutation                                   = cluster.get_permutation();
    const Cluster<CoordinatesPrecision> *cluster_on_partition = cluster.get_clusters_on_partition()[index];

    for (int i = 0; i < cluster_on_partition->get_size(); i++) {
        out[i] = in[permutation[cluster_on_partition->get_offset() + i] - cluster_on_partition->get_offset()];
    }
}

} // namespace htool
#endif
