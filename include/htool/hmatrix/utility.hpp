#ifndef HTOOL_HMATRIX_BUILDER_HPP
#define HTOOL_HMATRIX_BUILDER_HPP

#include "../clustering/cluster_node.hpp"
#include "../clustering/tree_builder/tree_builder.hpp"
#include "hmatrix.hpp"
#include "tree_builder/tree_builder.hpp"

namespace htool {

template <typename CoefficientsPrecision, typename CoordinatesPrecision = htool::underlying_type<CoefficientsPrecision>>
class HMatrixBuilder {
  public:
    Cluster<CoordinatesPrecision> target_cluster;
    Cluster<CoordinatesPrecision> source_cluster; // use an optional source cluster

    /// @brief
    /// @param target_number_of_points
    /// @param target_spatial_dimension
    /// @param target_coordinates
    /// @param target_cluster_tree_builder
    /// @param source_number_of_points
    /// @param source_spatial_dimension
    /// @param source_coordinates
    /// @param source_cluster_tree_builder
    HMatrixBuilder(int target_number_of_points, int target_spatial_dimension, const CoordinatesPrecision *target_coordinates, const ClusterTreeBuilder<CoordinatesPrecision> *target_cluster_tree_builder, int source_number_of_points, int source_spatial_dimension, const CoordinatesPrecision *source_coordinates, const ClusterTreeBuilder<CoordinatesPrecision> *source_cluster_tree_builder) : target_cluster(target_cluster_tree_builder == nullptr ? ClusterTreeBuilder<CoordinatesPrecision>().create_cluster_tree(target_number_of_points, target_spatial_dimension, target_coordinates, 2, 2) : target_cluster_tree_builder->create_cluster_tree(target_number_of_points, target_spatial_dimension, target_coordinates, 2, 2)), source_cluster(source_cluster_tree_builder == nullptr ? ClusterTreeBuilder<CoordinatesPrecision>().create_cluster_tree(source_number_of_points, source_spatial_dimension, source_coordinates, 2, 2) : source_cluster_tree_builder->create_cluster_tree(source_number_of_points, source_spatial_dimension, source_coordinates, 2, 2)) {
    }

    /// @brief
    /// @param target_number_of_points
    /// @param target_spatial_dimension
    /// @param target_coordinates
    /// @param source_number_of_points
    /// @param source_spatial_dimension
    /// @param source_coordinates
    HMatrixBuilder(int target_number_of_points, int target_spatial_dimension, const CoordinatesPrecision *target_coordinates, int source_number_of_points, int source_spatial_dimension, const CoordinatesPrecision *source_coordinates) : HMatrixBuilder(target_number_of_points, target_spatial_dimension, target_coordinates, nullptr, source_number_of_points, source_spatial_dimension, source_coordinates, nullptr) {
    }

    /// @brief
    /// @param number_of_points
    /// @param spatial_dimension
    /// @param coordinates
    HMatrixBuilder(int number_of_points, int spatial_dimension, const CoordinatesPrecision *coordinates) : HMatrixBuilder(number_of_points, spatial_dimension, coordinates, nullptr, number_of_points, spatial_dimension, coordinates, nullptr) {
    }

    /// @brief
    /// @param generator
    /// @param hmatrix_tree_builder
    /// @return
    HMatrix<CoefficientsPrecision, CoordinatesPrecision> build(const VirtualGenerator<CoefficientsPrecision> &generator, const HMatrixTreeBuilder<CoefficientsPrecision, CoordinatesPrecision> &hmatrix_tree_builder) {
        return hmatrix_tree_builder.build(generator, target_cluster, source_cluster);
    }
};
} // namespace htool
#endif
