#ifndef HTOOL_CLUSTERING_VIRTUAL_PARTITIONING_HPP
#define HTOOL_CLUSTERING_VIRTUAL_PARTITIONING_HPP

#include "../cluster_node.hpp" // for Cluster

namespace htool {

template <typename CoordinatePrecision>
class VirtualPartitioning {
  public:
    virtual std::vector<std::pair<int, int>> compute_partitioning(Cluster<CoordinatePrecision> &current_cluster, int spatial_dimension, const CoordinatePrecision *coordinates, const CoordinatePrecision *const radii, const CoordinatePrecision *const weights, int number_of_partitions) = 0;

    virtual ~VirtualPartitioning() {}
};

} // namespace htool

#endif
