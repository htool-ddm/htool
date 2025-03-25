#ifndef HTOOL_DISTRIBUTED_OPERATOR_PARTITION_FROM_CLUSTER_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_PARTITION_FROM_CLUSTER_HPP

#include "../../clustering/cluster_node.hpp"   // for global_to_root_cluster
#include "../../misc/misc.hpp"                 // for underlying_type
#include "../interfaces/virtual_partition.hpp" // for IPartition

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class PartitionFromCluster final : public VirtualPartition<CoefficientPrecision> {
    const Cluster<CoordinatePrecision> &m_root_cluster;

  public:
    explicit PartitionFromCluster(const Cluster<CoordinatePrecision> &root_cluster) : m_root_cluster(root_cluster) {}

    PartitionFromCluster(const PartitionFromCluster &)                                    = default;
    PartitionFromCluster &operator=(const PartitionFromCluster &)                         = default;
    PartitionFromCluster(PartitionFromCluster &&PartitionFromCluster) noexcept            = default;
    PartitionFromCluster &operator=(PartitionFromCluster &&PartitionFromCluster) noexcept = default;

    int get_size_of_partition(int subdomain_number) const { return m_root_cluster.get_clusters_on_partition()[subdomain_number]->get_size(); }
    int get_offset_of_partition(int subdomain_number) const { return m_root_cluster.get_clusters_on_partition()[subdomain_number]->get_offset(); }

    int get_global_size() const { return m_root_cluster.get_size(); }

    void global_to_partition_numbering(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
        global_to_root_cluster(m_root_cluster, in, out);
    }
    void partition_to_global_numbering(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
        root_cluster_to_global(m_root_cluster, in, out);
    }

    void local_to_local_partition_numbering(int subdomain_number, const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
        local_to_local_cluster(m_root_cluster, subdomain_number, in, out);
    }
    void local_partition_to_local_numbering(int subdomain_number, const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
        local_cluster_to_local(m_root_cluster, subdomain_number, in, out);
    }

    bool is_renumbering_local() const { return m_root_cluster.is_permutation_local(); }
};
} // namespace htool

#endif
